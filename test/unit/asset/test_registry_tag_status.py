# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""steps/registry-tag-status.sh: does an image tag exist on its registry?

It prints `present`, `absent` or `unknown` (exit 0, 1, 2). `publish-image` refuses
to overwrite a `present` tag, and `publish-step` refuses an image that is not
`present`, so `unknown` must never be reported as either answer. Runs the real
script with fake `skopeo` and `curl` first on PATH; the fake curl plays a
registry's v2 API (status codes and a Bearer token challenge).
"""

import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "steps" / "registry-tag-status.sh"
REF = "quay.io/org/gb-steps:train-foo-skypilot-v1"
_MANIFEST_URL = "https://quay.io/v2/org/gb-steps/manifests/train-foo-skypilot-v1"

# Answers HEAD on the manifest with $FAKE_MANIFEST_CODE, the token endpoint with
# a token, and records every call's argv. With $FAKE_CHALLENGE set, an
# unauthenticated manifest HEAD gets a 401 and a Bearer challenge first.
# Credentials and the Authorization header arrive as curl config on stdin
# (`--config -`), never in argv; the fake reads them from there.
_FAKE_CURL = """#!/bin/sh
echo "$*" >> "$FAKE_LOG/curl.calls"
url=""; auth=""; head=""; out=""; cfg=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    -D) head="$2"; shift 2 ;;
    --config) [ "$2" = "-" ] && cfg="$(cat)"; shift 2 ;;
    -w|--max-time|-H) shift 2 ;;
    http*) url="$1"; shift ;;
    *) shift ;;
  esac
done
case "$cfg" in
  *'header = "Authorization: '*) auth="Authorization: $(printf '%s' "$cfg" | sed -n 's/^header = "Authorization: \\(.*\\)"$/\\1/p')" ;;
  *'user = "'*) auth="basic:$(printf '%s' "$cfg" | sed -n 's/^user = "\\(.*\\)"$/\\1/p')" ;;
esac
[ -n "${FAKE_CURL_FAIL:-}" ] && exit 7
case "$url" in
  *"/token"*|*"/auth"*)
    [ -n "$auth" ] && echo "$auth" > "$FAKE_LOG/token.auth"
    [ -n "$out" ] && printf '{"token": "tok123"}' > "$out"
    printf '%s' "${FAKE_TOKEN_CODE:-200}" ;;
  *)
    if [ -n "${FAKE_CHALLENGE:-}" ] && [ "$auth" != "Authorization: Bearer tok123" ]; then
      [ -n "$head" ] && printf 'HTTP/2 401\\r\\nwww-authenticate: Bearer realm="https://quay.io/v2/auth",service="quay.io",scope="repository:org/gb-steps:pull"\\r\\n\\r\\n' > "$head"
      printf '401'
    else
      printf '%s' "${FAKE_MANIFEST_CODE:-200}"
    fi ;;
esac
"""

_FAKE_SKOPEO = """#!/bin/sh
echo "$*" >> "$FAKE_LOG/skopeo.calls"
[ -n "${FAKE_SKOPEO_ERR:-}" ] && echo "$FAKE_SKOPEO_ERR" >&2
exit "${FAKE_SKOPEO_RC:-0}"
"""


@pytest.fixture
def fakes(tmp_path: pathlib.Path) -> pathlib.Path:
    """A bin dir with a fake curl (and the system tools the script needs)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text(_FAKE_CURL)
    (bin_dir / "curl").chmod(0o755)
    for tool in ("sh", "sed", "tr", "cat", "mktemp", "rm", "grep", "printf"):
        found = shutil.which(tool)
        if found and not (bin_dir / tool).exists():
            (bin_dir / tool).symlink_to(found)
    return bin_dir


def _add_skopeo(fakes: pathlib.Path) -> None:
    (fakes / "skopeo").write_text(_FAKE_SKOPEO)
    (fakes / "skopeo").chmod(0o755)


def _status(fakes: pathlib.Path, ref: str = REF, **env: str):
    e = {"PATH": str(fakes), "FAKE_LOG": str(fakes), "HOME": str(fakes)}
    e.update(env)
    proc = subprocess.run(
        ["sh", str(SCRIPT), ref],
        env=e,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return proc.stdout.strip(), proc.returncode, proc.stderr


def _calls(fakes: pathlib.Path, tool: str) -> list[str]:
    log = fakes / f"{tool}.calls"
    return log.read_text().splitlines() if log.exists() else []


class TestCurl:
    def test_present(self, fakes):
        assert _status(fakes, FAKE_MANIFEST_CODE="200")[:2] == ("present", 0)
        assert _MANIFEST_URL in _calls(fakes, "curl")[0]

    def test_absent(self, fakes):
        assert _status(fakes, FAKE_MANIFEST_CODE="404")[:2] == ("absent", 1)

    @pytest.mark.parametrize("code", ["500", "403", "000", "429"])
    def test_other_answers_are_unknown(self, fakes, code):
        assert _status(fakes, FAKE_MANIFEST_CODE=code)[:2] == ("unknown", 2)

    def test_network_failure_is_unknown(self, fakes):
        assert _status(fakes, FAKE_CURL_FAIL="1")[:2] == ("unknown", 2)

    def test_bearer_challenge_is_followed(self, fakes):
        out, rc, _ = _status(fakes, FAKE_CHALLENGE="1", FAKE_MANIFEST_CODE="404")
        assert (out, rc) == ("absent", 1)
        token_call = [c for c in _calls(fakes, "curl") if "/v2/auth" in c]
        assert token_call and "service=quay.io" in token_call[0]
        assert "scope=repository:org/gb-steps:pull" in token_call[0]

    def test_token_request_uses_registry_credentials_when_set(self, fakes):
        _status(
            fakes,
            FAKE_CHALLENGE="1",
            REGISTRY_USER="org+robot",
            REGISTRY_PASSWORD="pw",
        )
        assert (fakes / "token.auth").read_text().strip() == "basic:org+robot:pw"

    def test_failed_token_request_is_unknown(self, fakes):
        out, rc, _ = _status(fakes, FAKE_CHALLENGE="1", FAKE_TOKEN_CODE="500")
        assert (out, rc) == ("unknown", 2)

    def test_default_tag_and_docker_hub(self, fakes):
        _status(fakes, "alpine")
        assert (
            "https://registry-1.docker.io/v2/library/alpine/manifests/latest"
            in _calls(fakes, "curl")[0]
        )

    def test_digest_reference(self, fakes):
        _status(fakes, "quay.io/org/img@sha256:abc")
        assert (
            "https://quay.io/v2/org/img/manifests/sha256:abc"
            in _calls(fakes, "curl")[0]
        )

    def test_registry_with_port(self, fakes):
        _status(fakes, "localhost:5000/org/img:t")
        assert (
            "https://localhost:5000/v2/org/img/manifests/t" in _calls(fakes, "curl")[0]
        )


class TestSkopeo:
    def test_preferred_when_installed(self, fakes):
        _add_skopeo(fakes)
        assert _status(fakes)[:2] == ("present", 0)
        assert _calls(fakes, "skopeo") == [f"inspect --raw docker://{REF}"]
        assert _calls(fakes, "curl") == []

    def test_manifest_unknown_is_absent(self, fakes):
        _add_skopeo(fakes)
        out = _status(
            fakes,
            FAKE_SKOPEO_RC="1",
            FAKE_SKOPEO_ERR="reading manifest v1: manifest unknown",
        )
        assert out[:2] == ("absent", 1)

    def test_other_errors_fall_back_to_curl(self, fakes):
        _add_skopeo(fakes)
        out = _status(
            fakes,
            FAKE_SKOPEO_RC="1",
            FAKE_SKOPEO_ERR="pinging container registry: connection refused",
            FAKE_MANIFEST_CODE="404",
        )
        assert out[:2] == ("absent", 1)
        assert _calls(fakes, "curl")


def test_usage_error_is_unknown(fakes):
    e = {"PATH": str(fakes), "FAKE_LOG": str(fakes)}
    proc = subprocess.run(
        ["sh", str(SCRIPT)], env=e, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 2


def test_the_password_is_not_on_curls_command_line_for_the_manifest(fakes):
    _status(fakes, FAKE_CHALLENGE="1", REGISTRY_USER="u", REGISTRY_PASSWORD="pw")
    manifest_calls = [c for c in _calls(fakes, "curl") if "/manifests/" in c]
    assert manifest_calls and all("pw" not in c for c in manifest_calls)
