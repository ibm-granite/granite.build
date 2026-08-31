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

"""Behaviour tests for the bundled ``src/dpk_setup.sh``.

The install phase's shell now lives in a real file, so its contract can be
executed rather than pattern matched against a rendered template. ``pip`` and
``uv`` are stubbed on ``PATH`` and record their argv, which lets these tests pin
the parts that were previously only assertable as template text — and that carry
real, measured consequences:

* ``uv`` does the resolving and installing; ``pip`` only bootstraps ``uv``
  itself, which is absent on a bare launcher node.
* ``UV_CACHE_DIR`` is exported **before** ``uv venv``, and anchors at
  ``$GB_SHARED_WORKDIR`` when the environment provides one. It must share the
  venv's filesystem (or uv copies instead of hard-linking) and be stable across
  runs (or there is nothing to link from — a per-run cache measured *worse* than
  pip: venv 5.8G->5.5G but a fresh 6.2G cache each run).
* the install does not pass ``--no-cache-dir``, which would defeat linking.
* requirements arrive as argv, so a specifier containing ``[extra]`` needs no
  shell re-quoting.

Cluster-agnostic, so this sits at the root of the step's ``test/`` dir (Mode 1
only) and is not copied by ``make publish-step``.
"""

import pathlib
import shutil
import subprocess

import pytest

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _STEP_DIR / "src" / "dpk_setup.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)

_STUB = """#!/usr/bin/env bash
{{
  echo "TOOL:{name}"
  for a in "$@"; do echo "ARG:$a"; done
  echo "UV_CACHE_DIR=${{UV_CACHE_DIR-<unset>}}"
  echo "END"
}} >> "$TRACE"
# `uv venv <dir>` must actually produce an activatable venv: the script sources
# it, and `set -e` would abort otherwise.
if [ "{name}" = "uv" ] && [ "$1" = "venv" ]; then
  mkdir -p "$2/bin" && : > "$2/bin/activate"
fi
"""


@pytest.fixture
def run_setup(tmp_path):
    """Run dpk_setup.sh with pip/uv stubbed, returning (proc, trace-entries)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    trace = tmp_path / "trace.txt"
    for name in ("pip", "uv"):
        stub = bin_dir / name
        stub.write_text(_STUB.format(name=name))
        stub.chmod(0o755)

    def _run(*args: str, env_extra: dict | None = None):
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "TRACE": str(trace),
        }
        env.update(env_extra or {})
        proc = subprocess.run(
            ["bash", str(_SCRIPT), *args],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env=env,
        )
        return proc, _parse(trace)

    _run.tmp_path = tmp_path  # type: ignore[attr-defined]
    return _run


def _parse(trace: pathlib.Path) -> list[dict]:
    """Parse the stub trace into [{tool, args, uv_cache_dir}, ...] in call order."""
    if not trace.exists():
        return []
    calls: list[dict] = []
    cur: dict = {}
    for line in trace.read_text().splitlines():
        if line.startswith("TOOL:"):
            cur = {"tool": line[len("TOOL:") :], "args": [], "uv_cache_dir": None}
        elif line.startswith("ARG:"):
            cur["args"].append(line[len("ARG:") :])
        elif line.startswith("UV_CACHE_DIR="):
            cur["uv_cache_dir"] = line[len("UV_CACHE_DIR=") :]
        elif line == "END":
            calls.append(cur)
    return calls


_BASE = ("--venv", "./venv", "--index-url", "https://pypi.org/simple")


def _call(calls: list[dict], tool: str, first_arg: str | None = None) -> dict | None:
    for c in calls:
        if c["tool"] == tool and (first_arg is None or c["args"][:1] == [first_arg]):
            return c
    return None


class TestScriptIsValidShell:
    def test_parses_under_bash(self):
        assert subprocess.run(["bash", "-n", str(_SCRIPT)]).returncode == 0

    @pytest.mark.skipif(
        shutil.which("shellcheck") is None, reason="shellcheck not installed"
    )
    def test_shellcheck_is_clean(self):
        """Now possible at all, because the shell is a file rather than YAML."""
        proc = subprocess.run(
            ["shellcheck", str(_SCRIPT)], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stdout


class TestRequiredOptions:
    @pytest.mark.parametrize(
        "missing,args",
        [
            ("--venv", ("--index-url", "https://x")),
            ("--index-url", ("--venv", "./venv")),
        ],
    )
    def test_missing_required_option_fails(self, run_setup, missing, args):
        proc, calls = run_setup(*args)
        assert proc.returncode != 0
        assert missing in proc.stderr
        assert calls == []


class TestUvIsTheInstaller:
    def test_pip_only_bootstraps_uv(self, run_setup):
        proc, calls = run_setup(*_BASE, "--", "somepkg==1.0")
        assert proc.returncode == 0, proc.stderr
        pip_calls = [c for c in calls if c["tool"] == "pip"]
        assert len(pip_calls) == 1
        assert pip_calls[0]["args"][-1] == "uv"
        assert "--no-cache-dir" in pip_calls[0]["args"]

    def test_uv_does_the_install(self, run_setup):
        proc, calls = run_setup(*_BASE, "--", "somepkg==1.0")
        install = _call(calls, "uv", "pip")
        assert install is not None
        assert install["args"][:2] == ["pip", "install"]
        assert "somepkg==1.0" in install["args"]

    def test_install_keeps_its_cache(self, run_setup):
        """--no-cache-dir on the uv install would defeat hard-linking outright."""
        _, calls = run_setup(*_BASE, "--", "somepkg==1.0")
        install = _call(calls, "uv", "pip")
        assert "--no-cache-dir" not in install["args"]

    def test_index_url_is_forwarded(self, run_setup):
        _, calls = run_setup(
            "--venv",
            "./venv",
            "--index-url",
            "https://mirror.example/simple",
            "--",
            "somepkg",
        )
        install = _call(calls, "uv", "pip")
        idx = install["args"].index("--index-url")
        assert install["args"][idx + 1] == "https://mirror.example/simple"


class TestUvCacheDir:
    def test_cache_is_exported_before_the_venv_is_created(self, run_setup):
        """Ordering matters: uv reads UV_CACHE_DIR when it builds the venv."""
        _, calls = run_setup(*_BASE, "--", "somepkg")
        venv_call = _call(calls, "uv", "venv")
        assert venv_call is not None
        assert venv_call["uv_cache_dir"] not in (None, "<unset>")

    def test_cache_anchors_at_shared_workdir_when_present(self, run_setup):
        """Stable across runs, and on the same filesystem as the venv."""
        _, calls = run_setup(
            *_BASE, "--", "somepkg", env_extra={"GB_SHARED_WORKDIR": "/shared"}
        )
        assert _call(calls, "uv", "venv")["uv_cache_dir"] == "/shared/.uv-cache"

    def test_cache_falls_back_to_cwd_without_a_shared_workdir(self, run_setup):
        """e.g. aws, where each step is its own instance anyway."""
        _, calls = run_setup(*_BASE, "--", "somepkg")
        expected = f"{run_setup.tmp_path}/.uv-cache"
        assert _call(calls, "uv", "venv")["uv_cache_dir"] == expected

    def test_install_sees_the_same_cache(self, run_setup):
        _, calls = run_setup(
            *_BASE, "--", "somepkg", env_extra={"GB_SHARED_WORKDIR": "/shared"}
        )
        assert _call(calls, "uv", "pip")["uv_cache_dir"] == "/shared/.uv-cache"


class TestVenvCreation:
    def test_venv_is_created_at_the_requested_path(self, run_setup):
        proc, calls = run_setup(*_BASE, "--", "somepkg")
        assert proc.returncode == 0, proc.stderr
        assert _call(calls, "uv", "venv")["args"] == ["venv", "./venv"]

    def test_venv_is_created_before_the_install(self, run_setup):
        """The install must land INSIDE the venv, so ordering is the contract."""
        _, calls = run_setup(*_BASE, "--", "somepkg")
        order = [(c["tool"], c["args"][0]) for c in calls]
        assert order.index(("uv", "venv")) < order.index(("uv", "pip"))


class TestRequirements:
    def test_bracketed_extra_arrives_as_one_argument(self, run_setup):
        """The "[extra]" would be a glob candidate if it were not passed as argv."""
        req = "data-prep-toolkit-transforms[pii-redactor]==1.1.8"
        _, calls = run_setup(*_BASE, "--", req)
        assert _call(calls, "uv", "pip")["args"][-1] == req

    def test_several_requirements_are_all_forwarded(self, run_setup):
        _, calls = run_setup(*_BASE, "--", "dpk==1.1.8", "pyarrow", "numpy<1.29")
        args = _call(calls, "uv", "pip")["args"]
        assert args[-3:] == ["dpk==1.1.8", "pyarrow", "numpy<1.29"]

    def test_no_requirements_still_creates_the_venv_and_skips_the_install(
        self, run_setup
    ):
        """`uv pip install` with no packages is an error; a bare venv is valid."""
        proc, calls = run_setup(*_BASE, "--")
        assert proc.returncode == 0, proc.stderr
        assert _call(calls, "uv", "venv") is not None
        assert _call(calls, "uv", "pip") is None

    def test_requirement_with_spaces_stays_one_argument(self, run_setup):
        _, calls = run_setup(*_BASE, "--", "pkg @ file:///some path/x.whl")
        assert _call(calls, "uv", "pip")["args"][-1] == "pkg @ file:///some path/x.whl"


class TestFailurePropagation:
    def test_failed_install_fails_the_script(self, run_setup, tmp_path):
        """set -e: a failed install must fail setup, not leave a broken venv."""
        failing = tmp_path / "bin" / "uv"
        failing.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "venv" ]; then mkdir -p "$2/bin" && : > "$2/bin/activate";'
            " exit 0; fi\nexit 7\n"
        )
        failing.chmod(0o755)
        proc, _ = run_setup(*_BASE, "--", "somepkg")
        assert proc.returncode == 7

    def test_failed_uv_bootstrap_fails_the_script(self, run_setup, tmp_path):
        failing = tmp_path / "bin" / "pip"
        failing.write_text("#!/usr/bin/env bash\nexit 5\n")
        failing.chmod(0o755)
        proc, _ = run_setup(*_BASE, "--", "somepkg")
        assert proc.returncode == 5
