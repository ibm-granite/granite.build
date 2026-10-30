# Copyright LLM.build Authors
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

"""Behaviour tests for the bundled ``src/dd_ollama.sh``.

Runs the real script with ``curl`` and the Ollama binary stubbed on PATH, so nothing is
downloaded and no model is served. Pins the parts the review found unguarded:

* the archive is installed only when its sha256 matches the published one, and a bad or
  failed download leaves nothing in the cache — no install, no temporary directory;
* a server that exits during startup (the port is taken) fails the run with that reason
  rather than trusting another job's server on the same port;
* ``run`` exits with the wrapped command's status and always stops its server.

Cluster-agnostic, Mode 1 only. Needs bash and the ``zstandard`` module (it builds the fake archive
and is what the script falls back to without zstd); skips without it.
"""

import hashlib
import io
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "src" / "dd_ollama.sh"

zstandard = pytest.importorskip(
    "zstandard",
    reason="needs the zstandard module (not in the repo .venv); run with the venv used for test_dd_run.py",
)
pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)

_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[
    os.uname().machine
]

# A fake `ollama` that logs like Ollama 0.40.7 (checked on the SLURM node): `serve`
# prints "Listening on <OLLAMA_HOST>" once bound and runs until killed. FAKE_SERVE_FAILS
# makes it print the real bind error and exit; FAKE_SERVE_DELAY delays that exit, which
# is how a real server on a taken port behaves while another job's server already
# answers. `show`/`pull` record themselves; the PID file lets a test check the stop.
_FAKE_OLLAMA = r"""#!/usr/bin/env bash
case "$1" in
  serve)
    if [ -n "${FAKE_SERVE_FAILS:-}" ]; then
      sleep "${FAKE_SERVE_DELAY:-0}"
      echo "Error: listen tcp $OLLAMA_HOST: bind: address already in use"; exit 1
    fi
    echo $$ > "$TRACE_DIR/serve.pid"
    echo "level=INFO msg=\"Listening on $OLLAMA_HOST (version 9.9.9)\""
    [ "${OLLAMA_NOPRUNE:-}" = 1 ] && echo NOPRUNE >> "$TRACE_DIR/calls"
    trap 'rm -f "$TRACE_DIR/serve.pid"; exit 0' TERM; while :; do sleep 0.1; done ;;
  show) [ -f "$TRACE_DIR/pulled" ] ;;
  pull) echo pull >> "$TRACE_DIR/calls"; touch "$TRACE_DIR/pulled" ;;
esac
"""


def _archive(tmp: pathlib.Path) -> bytes:
    """A tiny ollama-linux-<arch>.tar.zst: bin/ollama, a top-level lib, a CUDA lib."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data, mode in [
            ("bin/ollama", _FAKE_OLLAMA.encode(), 0o755),
            ("lib/ollama/libggml-base.so", b"cpu", 0o644),
            ("lib/ollama/cuda_v12/libcublas.so.12", b"gpu" * 100, 0o644),
        ]:
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    return zstandard.ZstdCompressor().compress(buf.getvalue())


def _setup(tmp_path, *, published_sha=None, curl_fails=False):
    """A PATH with stub curl serving the fake archive and its sha256sum.txt."""
    archive = _archive(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "archive").write_bytes(archive)
    sha = published_sha or hashlib.sha256(archive).hexdigest()
    (assets / "sha256sum.txt").write_text(f"{sha}  ./ollama-linux-{_ARCH}.tar.zst\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # curl: -o <file> <url> writes the archive; a sha256sum.txt URL prints the sums;
    # the readiness probe (-fs http://127.0.0.1...) succeeds. CURL_FAILS fails downloads.
    (bin_dir / "curl").write_text(rf"""#!/usr/bin/env bash
out=""; url=""
while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift 2;; -*) shift;; *) url="$1"; shift;; esac; done
case "$url" in
  http://127.0.0.1*) [ -n "${{OTHER_SERVER:-}}" ] || [ -f "{tmp_path}/serve.pid" ] ;;
  *sha256sum.txt) cat "{assets}/sha256sum.txt" ;;
  *) [ -z "${{CURL_FAILS:-}}" ] || exit 22; cp "{assets}/archive" "$out" ;;
esac
""")
    (bin_dir / "curl").chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "TRACE_DIR": str(tmp_path),
        "OLLAMA_LOG": str(tmp_path / "serve.log"),
        # The interpreter running this test, which the zstandard skip check just
        # imported from. Not whatever python3 is first on PATH: with another venv
        # activated that python3 can lack zstandard, and the script's Python fallback then
        # fails although the test didn't skip (found when the repo .venv was active).
        "PYTHON": sys.executable,
    }
    if curl_fails:
        env["CURL_FAILS"] = "1"
    # Force the Python extractor, the path the local SLURM and kube nodes take: drop
    # any directory holding a real zstd from PATH (the stubs dir comes first).
    keep = [d for d in ("/usr/bin", "/bin") if not (pathlib.Path(d) / "zstd").exists()]
    if len(keep) < 2:
        pytest.skip(
            "zstd is in /usr/bin or /bin here; can't force the Python extractor"
        )
    return env


def _ollama(tmp_path, env, *args):
    cache = tmp_path / "cache"
    return (
        subprocess.run(
            [
                "bash",
                str(_SCRIPT),
                *args[:1],
                "--version",
                "9.9.9",
                "--model",
                "m:1",
                "--cache-dir",
                str(cache),
                "--port",
                "11434",
                *args[1:],
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp_path,
            timeout=60,
        ),
        cache,
    )


def test_install_verifies_and_keeps_cpu_libs_only(tmp_path):
    env = _setup(tmp_path)
    proc, cache = _ollama(tmp_path, env, "install")
    assert proc.returncode == 0, proc.stderr
    install = cache / f"ollama-9.9.9-{_ARCH}"
    assert (install / "bin/ollama").is_file()
    assert (install / "lib/ollama/libggml-base.so").is_file()
    assert not (install / "lib/ollama/cuda_v12").exists()
    assert not list(cache.glob(".ollama-install.*"))
    assert (tmp_path / "calls").read_text().split() == ["NOPRUNE", "pull"]


def test_checksum_mismatch_installs_nothing(tmp_path):
    env = _setup(tmp_path, published_sha="0" * 64)
    proc, cache = _ollama(tmp_path, env, "install")
    assert proc.returncode != 0
    assert "failed its checksum" in proc.stderr
    assert not (cache / f"ollama-9.9.9-{_ARCH}").exists()
    assert not list(cache.glob(".ollama-install.*")), "temporary dir left in the cache"


def test_failed_download_leaves_no_temp_dir(tmp_path):
    env = _setup(tmp_path, curl_fails=True)
    proc, cache = _ollama(tmp_path, env, "install")
    assert proc.returncode != 0
    assert not list(cache.glob(".ollama-install.*")), "temporary dir left in the cache"


def test_second_install_reuses_cache(tmp_path):
    env = _setup(tmp_path)
    _ollama(tmp_path, env, "install")
    env["CURL_FAILS"] = "1"  # a re-download would now fail
    proc, _ = _ollama(tmp_path, env, "install")
    assert proc.returncode == 0, proc.stderr
    assert "already installed" in proc.stdout
    assert (tmp_path / "calls").read_text().split().count(
        "pull"
    ) == 1, "model pulled twice"


def _run_with_port_held_by_another_server(tmp_path, delay):
    env = _setup(tmp_path)
    _ollama(tmp_path, env, "install")
    env.update(OTHER_SERVER="1", FAKE_SERVE_FAILS="1", FAKE_SERVE_DELAY=str(delay))
    return _ollama(tmp_path, env, "run", "--", "bash", "-c", "echo GENERATED")[0]


@pytest.mark.parametrize("delay", [0, 0.5])
def test_taken_port_fails_instead_of_borrowing(tmp_path, delay):
    """Another server answers on the port; ours fails to bind, at once or after 0.5s.

    The 0.5s case is the race the second review reproduced: the old check accepted the
    other server's answer while ours was still alive.
    """
    proc = _run_with_port_held_by_another_server(tmp_path, delay)
    assert proc.returncode != 0
    assert "already in use" in proc.stderr
    assert "GENERATED" not in proc.stdout, "ran against another job's server"


def test_server_that_never_listens_is_not_trusted(tmp_path):
    """No "Listening on" line, but the port answers: still not our server."""
    env = _setup(tmp_path)
    _ollama(tmp_path, env, "install")
    env["OTHER_SERVER"] = "1"
    proc, _ = _ollama(tmp_path, env, "run", "--", "bash", "-c", "echo GENERATED")
    assert proc.returncode != 0 and "already in use" in proc.stderr
    assert "GENERATED" not in proc.stdout


@pytest.mark.parametrize("status", [0, 3])
def test_run_propagates_status_and_stops_server(tmp_path, status):
    env = _setup(tmp_path)
    _ollama(tmp_path, env, "install")
    proc, _ = _ollama(tmp_path, env, "run", "--", "bash", "-c", f"exit {status}")
    assert proc.returncode == status, proc.stderr
    assert not (tmp_path / "serve.pid").exists(), "server was not stopped"


def test_default_release_uses_the_pinned_hash_not_the_fetched_one(tmp_path):
    """For 0.40.7 the pinned hash wins: a tampered release whose published
    sha256sum.txt agrees with the tampered archive must still be refused."""
    env = _setup(tmp_path)  # sha256sum.txt agrees with the fake archive...
    cache = tmp_path / "cache"
    proc = subprocess.run(
        [
            "bash",
            str(_SCRIPT),
            "install",
            "--version",
            "0.40.7",
            "--model",
            "m:1",
            "--cache-dir",
            str(cache),
            "--port",
            "11434",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode != 0  # ...but it isn't the real 0.40.7 archive
    assert "failed its checksum" in proc.stderr
    assert not (cache / f"ollama-0.40.7-{_ARCH}").exists()


def test_unfetchable_checksum_gives_the_explanation(tmp_path):
    env = _setup(tmp_path)
    (tmp_path / "assets" / "sha256sum.txt").unlink()  # its fetch now fails
    bin_curl = tmp_path / "bin" / "curl"
    bin_curl.write_text(
        bin_curl.read_text().replace(
            '*sha256sum.txt) cat "', '*sha256sum.txt) exit 22; cat "'
        )
    )
    proc, cache = _ollama(tmp_path, env, "install")
    assert proc.returncode != 0
    assert "could not get a sha256" in proc.stderr
    assert not list(cache.glob(".ollama-install.*"))
