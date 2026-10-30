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

"""steps/scripts/registry-tag-status.sh: does an image tag exist on its registry?

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
SCRIPT = REPO_ROOT / "steps" / "scripts" / "registry-tag-status.sh"
REF = "quay.io/org/steps/train/foo/skypilot:v1"
_MANIFEST_URL = "https://quay.io/v2/org/steps/train/foo/skypilot/manifests/v1"

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
  *"/tags/list"*)
    printf '%s' "${FAKE_TAGS_CODE:-200}" ;;
  *"/token"*|*"/auth"*)
    [ -n "$auth" ] && echo "$auth" > "$FAKE_LOG/token.auth"
    [ -n "$out" ] && printf '{"token": "tok123"}' > "$out"
    printf '%s' "${FAKE_TOKEN_CODE:-200}" ;;
  *)
    if [ -n "${FAKE_CHALLENGE:-}" ] && [ "$auth" != "Authorization: Bearer tok123" ]; then
      [ -n "$head" ] && printf 'HTTP/2 401\\r\\nwww-authenticate: Bearer realm="https://quay.io/v2/auth",service="quay.io",scope="repository:org/steps/train/foo/skypilot:pull"\\r\\n\\r\\n' > "$head"
      printf '401'
    else
      if [ -n "${FAKE_MANIFEST_BODY:-}" ] && [ -n "$out" ] && [ "$out" != /dev/null ]; then
        printf '%s' "$FAKE_MANIFEST_BODY" > "$out"
      fi
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
    for tool in ("sh", "sed", "tr", "cat", "mktemp", "rm", "grep", "printf", "base64"):
        found = shutil.which(tool)
        if found and not (bin_dir / tool).exists():
            (bin_dir / tool).symlink_to(found)
    return bin_dir


def _add_skopeo(fakes: pathlib.Path) -> None:
    (fakes / "skopeo").write_text(_FAKE_SKOPEO)
    (fakes / "skopeo").chmod(0o755)


def _status(fakes: pathlib.Path, *args: str, **env: str):
    """Run the script; ``args`` are its arguments (default: REF)."""
    if not args or args in (("--repository",), ("--config-digest",)):
        args = (*args, REF)
    e = {"PATH": str(fakes), "FAKE_LOG": str(fakes), "HOME": str(fakes)}
    e.update(env)
    proc = subprocess.run(
        ["sh", str(SCRIPT), *args],
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
        assert "scope=repository:org/steps/train/foo/skypilot:pull" in token_call[0]

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


def _auth_file(path: pathlib.Path, key: str, user: str, password: str) -> None:
    """Write a podman/docker auth file with one ``auths`` entry (base64 user:pw)."""
    import base64
    import json

    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"auths": {key: {"auth": token}}}))


class TestStoredLogin:
    """Without REGISTRY_USER/REGISTRY_PASSWORD, a stored `podman login` is used."""

    def _token_auth(self, fakes):
        f = fakes / "token.auth"
        return f.read_text().strip() if f.exists() else None

    def test_registry_auth_file_namespace_entry(self, fakes, tmp_path):
        af = tmp_path / "auth.json"
        _auth_file(af, "quay.io/org", "org+robot", "s3cr3t")
        _status(fakes, FAKE_CHALLENGE="1", REGISTRY_AUTH_FILE=str(af))
        assert self._token_auth(fakes) == "basic:org+robot:s3cr3t"

    def test_host_entry(self, fakes, tmp_path):
        af = tmp_path / "auth.json"
        _auth_file(af, "quay.io", "me", "pw")
        _status(fakes, FAKE_CHALLENGE="1", REGISTRY_AUTH_FILE=str(af))
        assert self._token_auth(fakes) == "basic:me:pw"

    def test_default_podman_auth_file_under_home(self, fakes):
        _auth_file(
            fakes / ".config" / "containers" / "auth.json", "quay.io/org", "u", "p"
        )
        _status(fakes, FAKE_CHALLENGE="1")
        assert self._token_auth(fakes) == "basic:u:p"

    def test_env_credentials_win_over_the_stored_login(self, fakes, tmp_path):
        af = tmp_path / "auth.json"
        _auth_file(af, "quay.io/org", "stored", "x")
        _status(
            fakes,
            FAKE_CHALLENGE="1",
            REGISTRY_AUTH_FILE=str(af),
            REGISTRY_USER="env",
            REGISTRY_PASSWORD="y",
        )
        assert self._token_auth(fakes) == "basic:env:y"

    def test_another_registry_entry_is_not_used(self, fakes, tmp_path):
        af = tmp_path / "auth.json"
        _auth_file(af, "ghcr.io/org", "u", "p")
        _status(fakes, FAKE_CHALLENGE="1", REGISTRY_AUTH_FILE=str(af))
        assert self._token_auth(fakes) is None

    def test_another_namespace_on_the_same_host_is_not_used(self, fakes, tmp_path):
        af = tmp_path / "auth.json"
        _auth_file(af, "quay.io/other", "u", "p")
        _status(fakes, FAKE_CHALLENGE="1", REGISTRY_AUTH_FILE=str(af))
        assert self._token_auth(fakes) is None

    def test_the_stored_password_is_not_on_curls_command_line(self, fakes, tmp_path):
        af = tmp_path / "auth.json"
        _auth_file(af, "quay.io/org", "u", "s3cr3t")
        _status(fakes, FAKE_CHALLENGE="1", REGISTRY_AUTH_FILE=str(af))
        assert all("s3cr3t" not in c for c in _calls(fakes, "curl"))


class TestRepository:
    """--repository: does the repository exist at all (tags/list), any tag."""

    def test_existing_repository(self, fakes):
        out, rc, _ = _status(fakes, "--repository", FAKE_TAGS_CODE="200")
        assert (out, rc) == ("present", 0)
        assert "https://quay.io/v2/org/steps/train/foo/skypilot/tags/list" in (
            _calls(fakes, "curl")[0]
        )

    def test_missing_repository(self, fakes):
        assert _status(fakes, "--repository", FAKE_TAGS_CODE="404")[:2] == (
            "absent",
            1,
        )

    @pytest.mark.parametrize("code", ["401", "403", "500"])
    def test_other_answers_are_unknown(self, fakes, code):
        assert _status(fakes, "--repository", FAKE_TAGS_CODE=code)[:2] == (
            "unknown",
            2,
        )

    def test_skopeo_is_not_used_for_repositories(self, fakes):
        _add_skopeo(fakes)
        _status(fakes, "--repository", FAKE_TAGS_CODE="200")
        assert _calls(fakes, "skopeo") == []


_OCI_MANIFEST = (
    '{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json",'
    '"config":{"mediaType":"application/vnd.oci.image.config.v1+json",'
    '"digest":"sha256:7f9e94997de914cb","size":1234},'
    '"layers":[{"mediaType":"application/vnd.oci.image.layer.v1.tar+gzip",'
    '"digest":"sha256:aaaa","size":1}]}'
)
_OCI_INDEX = (
    '{"schemaVersion":2,"mediaType":"application/vnd.oci.image.index.v1+json",'
    '"manifests":[{"digest":"sha256:bbbb","platform":{"architecture":"amd64"}}]}'
)


class TestConfigDigest:
    """--config-digest: the image's config digest (equal to its local image ID)."""

    def test_prints_the_config_digest(self, fakes):
        out, rc, _ = _status(fakes, "--config-digest", FAKE_MANIFEST_BODY=_OCI_MANIFEST)
        assert (out, rc) == ("sha256:7f9e94997de914cb", 0)

    def test_missing_tag_is_absent(self, fakes):
        out, rc, _ = _status(fakes, "--config-digest", FAKE_MANIFEST_CODE="404")
        assert (out, rc) == ("absent", 1)

    def test_an_index_has_no_single_config_and_is_unknown(self, fakes):
        out, rc, _ = _status(fakes, "--config-digest", FAKE_MANIFEST_BODY=_OCI_INDEX)
        assert (out, rc) == ("unknown", 2)

    def test_skopeo_is_not_used(self, fakes):
        _add_skopeo(fakes)
        _status(fakes, "--config-digest", FAKE_MANIFEST_BODY=_OCI_MANIFEST)
        assert _calls(fakes, "skopeo") == []


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
