# SPDX-License-Identifier: Apache-2.0
"""Render tests for the `eval/livecodebench-score` step's step-template.yaml.

Cluster-agnostic, so this sits at the root of the step's ``test/`` dir (Mode 1
only) and is not copied by ``make publish-step``.

The rendered ``run`` block is executed with a stub ``python`` on PATH that records
its argv and environment, so the assertions are on what bash actually hands the
bundled scorer, not on rendered text. The stub stands in for the sandbox wrapper too
(``src/lcb_sandbox.py``): it records the wrapper's paths and runs the command after
``--``, so the tests run on hosts without Landlock. ``TestSandboxForReal`` runs the
real wrapper where the host has it.
"""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import yaml

jinja2 = pytest.importorskip("jinja2", reason="jinja2 renders the step template")

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = _STEP_DIR / "step-template.yaml"
# Relative on purpose: run must make them absolute before it cd's into the checkout.
# _run creates them under the test's working directory.
_COMPLETIONS = "inputs/completions"
_DATASET = "inputs/dataset"


@pytest.fixture(scope="module")
def template() -> dict:
    return yaml.safe_load(_TEMPLATE.read_text())


@pytest.fixture(scope="module")
def defaults(template) -> dict:
    return dict(template["config"]["lcb_score_config"])


@pytest.fixture(scope="module")
def launcher(template) -> dict:
    return template["environment_configs"]["Skypilot"]["launchers"]["lcb_score"][
        "config"
    ]


def _cfg(defaults: dict, **over) -> dict:
    """The USAGE.md example config, with overrides."""
    base = dict(
        defaults,
        release_version="release_v6",
        start_date="2025-04-05",
        end_date="2025-04-05",
    )
    base.update(over)
    return base


def _render(source: str, cfg: dict) -> str:
    return jinja2.Template(source, undefined=jinja2.StrictUndefined).render(
        config={"lcb_score_config": cfg},
        bindings={
            "completions": {"binding": {"path": _COMPLETIONS}},
            "dataset": {"binding": {"path": _DATASET}},
        },
    )


def _bash_ok(script: str) -> bool:
    if shutil.which("bash") is None:  # pragma: no cover - bash is present in CI
        pytest.skip("bash not available")
    return subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


# What the rendered blocks run besides the stubs, for a PATH with nothing else on it.
_TOOLS = (
    "bash",
    "env",
    "mkdir",
    "rm",
    "sed",
    "true",
    "ls",
    "grep",
    "cat",
    "id",
    "wc",
)

_SANDBOX_PY = str(_STEP_DIR / "src" / "lcb_sandbox.py")

# The top of the stub python, when it's asked to run the sandbox wrapper: "ok"
# records the wrapper's arguments and runs the command after --; "broken" fails the
# way the wrapper does on a kernel without Landlock; "real" runs the real wrapper with
# the host's python, which restricts itself and execs the command (the stub again).
_SANDBOX_STUBS = {
    "ok": (
        'if [ "${1:-}" = "$SRC_DIR_EXPECTED/lcb_sandbox.py" ]; then shift\n'
        '  w=(); while [ "$1" != -- ]; do w+=("$1"); shift; done; shift\n'
        '  echo "SANDBOX:${w[*]}"; exec "$@"\n'
        "fi\n"
    ),
    "broken": (
        'if [ "${1:-}" = "$SRC_DIR_EXPECTED/lcb_sandbox.py" ]; then\n'
        '  echo "eval/livecodebench-score: ERROR sandbox is on but this node can\'t apply it (this kernel has no Landlock); refusing" >&2\n'
        "  exit 1\n"
        "fi\n"
    ),
    "real": (
        'if [ "${1:-}" = "$SRC_DIR_EXPECTED/lcb_sandbox.py" ]; then\n'
        '  exec "$REAL_PYTHON" "$@"\n'
        "fi\n"
    ),
}


def _tools_dir(cwd: pathlib.Path) -> pathlib.Path:
    """A directory with only the basic tools, so PATH can leave out the host's own
    ``python`` (Debian's python-is-python3, RHEL's /usr/bin/python)."""
    tools = cwd / "tools"
    tools.mkdir()
    for name in _TOOLS:
        found = shutil.which(name)
        if found:
            (tools / name).symlink_to(found)
    return tools


def _run(
    rendered: str,
    cwd: pathlib.Path,
    env_over: dict | None = None,
    sandbox: str = "ok",
    stub_body: str | None = None,
) -> subprocess.CompletedProcess:
    """Execute a rendered `run` block with a stub `python` on PATH.

    The stub prints ``ARG:<word>`` per argument, plus the working directory and the
    environment variables the step must set or clear (or runs ``stub_body``).
    ``sandbox`` picks what the stub does when the step runs the sandbox wrapper: a key
    of ``_SANDBOX_STUBS``. Apart from the stub, PATH holds only ``_TOOLS``.
    """
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available")
    bin_dir = cwd / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        # Paths written in, not passed in the environment: env -i starts the wrapper.
        + _SANDBOX_STUBS[sandbox]
        .replace("$SRC_DIR_EXPECTED", str(cwd / "src"))
        .replace("$REAL_PYTHON", sys.executable)
        + (
            stub_body
            or 'for a in "$@"; do echo "ARG:$a"; done\n'
            'echo "CWD:$(pwd)"\n'
            'echo "ENV:HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-}"\n'
            'echo "ENV:HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-}"\n'
            'echo "ENV:PYTHONPATH=${PYTHONPATH:-}"\n'
            'env | sed "s/^/FULLENV:/"\n'
        )
    )
    stub.chmod(0o755)
    tools = _tools_dir(cwd)
    (cwd / "src").mkdir()
    if sandbox == "real":
        shutil.copy(_SANDBOX_PY, cwd / "src")
    (cwd / "venv" / "bin").mkdir(parents=True)
    (cwd / "venv" / "bin" / "activate").write_text("")
    (cwd / "LiveCodeBench").mkdir()
    for d in (_COMPLETIONS, _DATASET):
        (cwd / d).mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    env["PATH"] = f"{bin_dir}:{tools}"
    for k, v in (env_over or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return subprocess.run(
        ["bash", "-c", rendered], capture_output=True, text=True, cwd=cwd, env=env
    )


def _run_setup_until_uv(setup: str, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    """Run a rendered `setup` with stub git and pip, so it stops at the uv bootstrap."""
    bin_dir = cwd / "setup-bin"
    bin_dir.mkdir()
    for name, body in (("git", "exit 0"), ("pip", 'echo "PIP_RAN:$*"; exit 1')):
        (bin_dir / name).write_text(f"#!/usr/bin/env bash\n{body}\n")
        (bin_dir / name).chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    return subprocess.run(
        ["bash", "-c", setup], capture_output=True, text=True, cwd=cwd, env=env
    )


def _lines(proc: subprocess.CompletedProcess, prefix: str) -> list[str]:
    assert proc.returncode == 0, f"rendered block failed: {proc.stderr}"
    return [
        line[len(prefix) :]
        for line in proc.stdout.splitlines()
        if line.startswith(prefix)
    ]


def _opt(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


class TestStepContract:
    def test_is_a_custom_step(self, template):
        assert template["name"] == "livecodebench-score"
        assert template["type"] == "custom"

    def test_no_image_ref_token(self):
        assert "${IMAGE_REF}" not in _TEMPLATE.read_text()

    def test_serves_every_skypilot_endpoint(self, template):
        skypilot = template["environment_configs"]["Skypilot"]
        assert (
            "subtypes" not in skypilot and skypilot["default_launcher"] == "lcb_score"
        )

    def test_declares_completions_and_dataset_and_a_results_output(self, template):
        assert template["inputs"]["allow_unknown"] is True
        assert template["inputs"]["required"] == {
            "completions": {"type": "dataset", "accept": ["uri", "binding"]},
            "dataset": {"type": "dataset", "accept": ["uri", "binding"]},
        }
        assert "optional" not in template["inputs"]
        assert template["outputs"]["optional"]["results"]["type"] == "dataset"

    def test_required_release_has_no_default(self, defaults):
        assert defaults["release_version"] == ""

    def test_declares_no_secrets(self, template, launcher):
        """No HF_TOKEN by any route the SkyPilot launcher reads (see the template header)."""
        text = _TEMPLATE.read_text()
        assert "secret_names_to_use_as_env_variable" not in text
        assert "envs" not in launcher

    def test_lcb_commit_is_a_full_sha(self, defaults):
        sha = defaults["lcb_commit"]
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)

    def test_ships_src(self, launcher):
        assert launcher["file_mounts"] == {"src": "src"}
        for f in ("lcb_score.py", "lcb_dataset.py", "requirements-score.txt"):
            assert (_STEP_DIR / "src" / f).is_file()

    def test_pinned_requirements_are_exact_pins(self):
        reqs = [
            line.strip()
            for line in (_STEP_DIR / "src" / "requirements-score.txt")
            .read_text()
            .splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert reqs, "requirements-score.txt is empty"
        assert all("==" in r for r in reqs), [r for r in reqs if "==" not in r]
        names = {r.split("==")[0].lower() for r in reqs}
        assert {"datasets", "anthropic", "numpy"} <= names
        assert not names & {
            "vllm",
            "torch",
            "triton",
            "xformers",
        }, "GPU stack must not be in the CPU pins"


class TestImage:
    def test_default_image_is_python_312_slim(self, defaults, launcher):
        assert defaults["image"] == "python:3.12-slim" and defaults["install"] is True
        assert _render(launcher["image_id"], defaults) == "docker:python:3.12-slim"

    def test_empty_image_runs_on_the_bare_node(self, defaults, launcher):
        assert _render(launcher["image_id"], _cfg(defaults, image="")) == ""


class TestSetup:
    def test_installs_pinned_lcb_without_its_deps(self, defaults, launcher):
        setup = _render(launcher["setup"], _cfg(defaults))
        assert _bash_ok(setup)
        assert f"'{defaults['lcb_commit']}'" in setup
        assert "-r ./src/requirements-score.txt" in setup
        assert 'uv pip install --no-deps -e "$LCB_DIR"' in setup
        assert (
            "--index-url 'https://download.pytorch.org/whl/cpu'" in setup
            and "'torch==2.15.0'" in setup
        )
        assert 'UV_CACHE_DIR="$PWD/.uv-cache"' in setup
        assert "GB_SHARED_WORKDIR" not in setup

    def test_torch_comes_before_the_rest(self, defaults, launcher):
        setup = _render(launcher["setup"], _cfg(defaults))
        assert setup.index("torch==2.15.0") < setup.index("requirements-score.txt")

    def test_install_false_installs_nothing(self, defaults, launcher):
        setup = _render(launcher["setup"], _cfg(defaults, install=False))
        assert _bash_ok(setup) and "uv" not in setup and "git" not in setup

    def test_uv_is_pinned_and_comes_from_the_package_index(self, defaults, launcher):
        mirror = "https://mirror.example/simple"
        setup = _render(launcher["setup"], _cfg(defaults, pip_index_url=mirror))
        assert f"--index-url '{mirror}' \\\n  \"uv==$UV_VERSION\"" in setup
        assert "UV_VERSION='0.13.6'" in setup
        assert "UV_VERSION='0.13.7'" in _render(
            launcher["setup"], _cfg(defaults, uv_version="0.13.7")
        )

    def test_uv_bootstrap_hides_no_errors(self, defaults, launcher):
        """Ask pip whether it knows --break-system-packages; don't try and fall back."""
        setup = _render(launcher["setup"], _cfg(defaults))
        assert "pip install --help" in setup
        boot = [
            l for l in setup.splitlines() if "pip install" in l and "uv pip" not in l
        ]
        assert boot and not any("2>/dev/null" in l for l in boot if "--help" not in l)

    def test_empty_uv_version_fails_with_a_message(self, defaults, launcher, tmp_path):
        setup = _render(launcher["setup"], _cfg(defaults, uv_version=""))
        proc = _run_setup_until_uv(setup, tmp_path)
        assert proc.returncode == 1
        assert "uv_version is empty" in proc.stderr
        assert "PIP_RAN" not in proc.stdout

    def test_venv_is_recreated_on_a_retry(self, defaults, launcher):
        """A retried step reuses the workdir; uv refuses to create over an existing venv."""
        setup = _render(launcher["setup"], _cfg(defaults))
        assert "uv venv --clear --managed-python --python 3.12 ./venv" in setup
        assert setup.count("uv venv") == 1


class TestRun:
    def test_parses(self, defaults, launcher):
        assert _bash_ok(_render(launcher["run"], _cfg(defaults)))

    def test_hands_the_scorer_the_bindings_and_window(
        self, defaults, launcher, tmp_path
    ):
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path)
        argv = _lines(proc, "ARG:")
        assert argv[0] == str(tmp_path / "src" / "lcb_score.py")
        assert _opt(argv, "--completions-dir") == str(tmp_path / _COMPLETIONS)
        assert _opt(argv, "--dataset-dir") == str(tmp_path / _DATASET)
        assert _opt(argv, "--release-version") == "release_v6"
        assert (
            _opt(argv, "--start-date") == "2025-04-05"
            and _opt(argv, "--end-date") == "2025-04-05"
        )
        assert (
            _opt(argv, "--timeout") == "6"
            and _opt(argv, "--num-process-evaluate") == "0"
        )
        assert _opt(argv, "--output-dir") == str(tmp_path / "output")

    def test_runs_offline_from_inside_the_checkout(self, defaults, launcher, tmp_path):
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path)
        assert _lines(proc, "CWD:") == [str(tmp_path / "LiveCodeBench")]
        env = _lines(proc, "ENV:")
        assert "HF_HUB_OFFLINE=1" in env and "HF_DATASETS_OFFLINE=1" in env
        assert f"PYTHONPATH={tmp_path / 'src'}" in env

    def test_empty_dates_are_passed_empty(self, defaults, launcher, tmp_path):
        proc = _run(
            _render(launcher["run"], _cfg(defaults, start_date="", end_date="")),
            tmp_path,
        )
        argv = _lines(proc, "ARG:")
        assert _opt(argv, "--start-date") == "" and _opt(argv, "--end-date") == ""

    def test_the_scorer_prints_the_marker_with_the_logical_path(
        self, defaults, launcher
    ):
        """Nothing runs after the exec, so the scorer prints the artifact marker. It
        keeps symlinks in the path (absolute(), not resolve()), as `pwd` and every
        other step do, so a symlinked mount reports the same path."""
        run = _render(launcher["run"], _cfg(defaults))
        assert 'exec env -i "${SCORE_ENV[@]}"' in run
        assert "GB_ARTIFACT_ID" not in run
        src = (_STEP_DIR / "src" / "lcb_score.py").read_text()
        assert "GB_ARTIFACT_ID:{ARTIFACT_ID} GB_ARTIFACT_PATH:{out.absolute()}" in src
        assert ".resolve()}" not in src

    def test_cd_is_safe_with_cdpath(self, defaults, launcher, tmp_path):
        """With CDPATH set, cd prints the directory; the captured path must stay clean."""
        rendered = _render(launcher["run"], _cfg(defaults))
        proc = _run(rendered, tmp_path, {"CDPATH": str(tmp_path)})
        assert proc.returncode == 0, proc.stderr
        argv = _lines(proc, "ARG:")
        assert _opt(argv, "--completions-dir") == str(tmp_path / _COMPLETIONS)
        assert _opt(argv, "--output-dir") == str(tmp_path / "output")

    def test_missing_python_fails_with_a_message(self, defaults, launcher, tmp_path):
        """PATH holds only the basic tools, so this holds on a host with its own
        /usr/bin/python too."""
        rendered = _render(launcher["run"], _cfg(defaults, install=False))
        _run(rendered, tmp_path)  # sets up the inputs, stubs and tools directory
        (tmp_path / "bin" / "python").unlink()
        proc = subprocess.run(
            ["bash", "-c", rendered],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={"PATH": f"{tmp_path / 'bin'}:{tmp_path / 'tools'}"},
        )
        assert proc.returncode == 1
        assert "no python on PATH" in proc.stderr
        assert "ARG:" not in proc.stdout

    @pytest.mark.parametrize("var", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
    def test_drops_an_inherited_hf_token(self, defaults, launcher, tmp_path, var):
        """A token from the job environment is dropped, with a warning, before the scorer runs."""
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path, {var: "hf_x"})
        assert proc.returncode == 0, proc.stderr
        assert f"WARNING {var} was set" in proc.stderr
        assert "kept out of the scorer's environment" in proc.stderr
        assert "hf_x" not in proc.stdout
        assert any(line.startswith("ARG:") for line in proc.stdout.splitlines())

    def test_scorer_gets_only_the_allowlisted_environment(
        self, defaults, launcher, tmp_path
    ):
        """Anything else the job environment carries (a login profile's exports, a
        declared secret) must not reach the scorer, and HOME must be empty."""
        planted = {
            "BV_SSH_PRIVATE_KEY": "secret-key",
            "AWS_SECRET_ACCESS_KEY": "secret-aws",
            "SOME_TOKEN": "secret-other",
            "OMP_NUM_THREADS": "4",
            "LD_LIBRARY_PATH": "/opt/py/lib",
        }
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path, planted)
        assert proc.returncode == 0, proc.stderr
        env = dict(
            line.split("=", 1) for line in _lines(proc, "FULLENV:") if "=" in line
        )
        assert not {"BV_SSH_PRIVATE_KEY", "AWS_SECRET_ACCESS_KEY", "SOME_TOKEN"} & set(
            env
        )
        assert "secret-" not in proc.stdout
        # The thread count is kept, so the scorer stays on the job's CPUs.
        assert env["OMP_NUM_THREADS"] == "4"
        # The loader's path is kept too: a module-loaded or image Python can need it
        # to start. The locale is passed only if the job set one, never forced.
        assert env["LD_LIBRARY_PATH"] == "/opt/py/lib"
        allowed = {
            "PATH",
            "HOME",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "LD_LIBRARY_PATH",
            "LIBRARY_PATH",
            "MODULEPATH",
            "LOADEDMODULES",
            "_LMFILES_",
            "CONDA_PREFIX",
            "PYTHONPATH",
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "VIRTUAL_ENV",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "SLURM_CPUS_PER_TASK",
            "SLURM_CPUS_ON_NODE",
        }
        # bash itself adds a few (PWD, SHLVL, _) when it runs the stub.
        assert set(env) - allowed <= {"PWD", "OLDPWD", "SHLVL", "_"}
        home = pathlib.Path(env["HOME"])
        assert home == tmp_path / ".score-home" and home.is_dir()
        assert not any(home.iterdir())

    def test_locale_is_not_forced(self, defaults, launcher, tmp_path):
        """A job with no locale gets none: forcing one the node lacks gives warnings."""
        rendered = _render(launcher["run"], _cfg(defaults))
        proc = _run(rendered, tmp_path, dict.fromkeys(("LANG", "LC_ALL", "LC_CTYPE")))
        assert proc.returncode == 0, proc.stderr
        got = dict(l.split("=", 1) for l in _lines(proc, "FULLENV:") if "=" in l)
        assert not {"LANG", "LC_ALL", "LC_CTYPE"} & set(got)

    def test_relative_inputs_reach_the_scorer_absolute(
        self, defaults, launcher, tmp_path
    ):
        """run cd's into the checkout, so relative bindings must be made absolute first."""
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path)
        argv = _lines(proc, "ARG:")
        for flag, rel in (
            ("--completions-dir", _COMPLETIONS),
            ("--dataset-dir", _DATASET),
        ):
            assert _opt(argv, flag) == str(tmp_path / rel)
            assert pathlib.Path(_opt(argv, flag)).is_absolute()

    def test_refuses_a_missing_input_dir(self, defaults, launcher, tmp_path):
        rendered = _render(launcher["run"], _cfg(defaults)).replace(
            _COMPLETIONS, "inputs/nowhere", 1
        )
        proc = _run(rendered, tmp_path)
        assert proc.returncode == 1
        assert (
            "input directory inputs/nowhere (COMPLETIONS_DIR) does not exist"
            in proc.stderr
        )
        assert "ARG:" not in proc.stdout

    def test_refuses_an_empty_release(self, defaults, launcher, tmp_path):
        proc = _run(
            _render(launcher["run"], _cfg(defaults, release_version="")), tmp_path
        )
        assert proc.returncode == 1 and "release_version is required" in proc.stderr

    def test_quotes_author_values(self, defaults, launcher, tmp_path):
        nasty = "release_v6'; touch PWNED; echo '"
        proc = _run(
            _render(launcher["run"], _cfg(defaults, release_version=nasty)), tmp_path
        )
        assert _opt(_lines(proc, "ARG:"), "--release-version") == nasty
        assert (
            not (tmp_path / "PWNED").exists()
            and not (tmp_path / "LiveCodeBench" / "PWNED").exists()
        )


class TestSandbox:
    def test_is_on_by_default(self, defaults):
        assert defaults["sandbox"] is True

    def test_scorer_runs_under_the_wrapper_with_only_the_steps_paths(
        self, defaults, launcher, tmp_path
    ):
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path)
        (wrapper_args,) = _lines(proc, "SANDBOX:")
        words = wrapper_args.split()
        reads = {words[i + 1] for i, w in enumerate(words) if w == "--read"}
        writes = {words[i + 1] for i, w in enumerate(words) if w == "--write"}
        assert reads == {
            str(tmp_path / "src"),
            str(tmp_path / _COMPLETIONS),
            str(tmp_path / _DATASET),
            str(tmp_path / "LiveCodeBench"),
        }
        # Only the step's own scratch and output directories: not the workdir
        # itself, HOME, or the shared /tmp.
        assert writes == {
            str(tmp_path / "output"),
            str(tmp_path / ".hf-cache"),
            str(tmp_path / ".score-home"),
            str(tmp_path / ".score-tmp"),
        }
        # The wrapper starts the scorer itself, inside env -i's clean environment.
        argv = _lines(proc, "ARG:")
        assert argv[0] == str(tmp_path / "src" / "lcb_score.py")

    def test_scorer_tmpdir_is_its_own(self, defaults, launcher, tmp_path):
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path)
        env = dict(l.split("=", 1) for l in _lines(proc, "FULLENV:") if "=" in l)
        assert env["TMPDIR"] == str(tmp_path / ".score-tmp")
        assert (tmp_path / ".score-tmp").is_dir()

    def test_fails_where_the_kernel_cant_apply_it(self, defaults, launcher, tmp_path):
        proc = _run(
            _render(launcher["run"], _cfg(defaults)), tmp_path, sandbox="broken"
        )
        assert proc.returncode == 1
        assert "this kernel has no Landlock" in proc.stderr
        assert "ARG:" not in proc.stdout

    def test_off_runs_without_the_wrapper_and_warns(self, defaults, launcher, tmp_path):
        proc = _run(
            _render(launcher["run"], _cfg(defaults, sandbox=False)),
            tmp_path,
            sandbox="broken",
        )
        assert proc.returncode == 0, proc.stderr
        assert "WARNING sandbox is off" in proc.stderr
        assert not _lines(proc, "SANDBOX:") and _lines(proc, "ARG:")


def _landlock_works() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    probe = "import lcb_sandbox, sys; sys.exit(lcb_sandbox.abi_version() < lcb_sandbox.MIN_ABI)"
    return (
        subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            env=dict(os.environ, PYTHONPATH=str(_STEP_DIR / "src")),
        ).returncode
        == 0
    )


@pytest.mark.skipif(
    not _landlock_works(), reason="this host has no Landlock, or an ABI before 4"
)
class TestSandboxForReal:
    """The real wrapper: what generated code would see from inside the scorer."""

    # Runs where the generated code would: reads every process's environment,
    # checks for a planted secret, and tries files and the network outside the step.
    _PROBE = (
        "n=0; d=0; for p in /proc/[0-9]*; do\n"
        '  if grep -q -a PLANTED_SECRET= "$p/environ" 2>/dev/null; then n=$((n+1))\n'
        '  elif ! cat "$p/environ" >/dev/null 2>&1; then d=$((d+1)); fi\n'
        "done\n"
        'echo "LEAKS:$n"\n'
        'echo "DENIED:$d"\n'
        'if cat "$OUTSIDE/secret" >/dev/null 2>&1; then echo "READ:ok"; else echo "READ:refused"; fi\n'
        'if (echo x > "$OUTSIDE/new") 2>/dev/null; then echo "WRITE:ok"; else echo "WRITE:refused"; fi\n'
        'if (echo x > "$OUTPUT/ok") 2>/dev/null; then echo "OUT:ok"; else echo "OUT:refused"; fi\n'
        # Nothing listens on port 1: "Connection refused" unsandboxed, "Permission
        # denied" when Landlock blocks the connect.
        'e="$( (exec 3<>/dev/tcp/127.0.0.1/1) 2>&1 )"\n'
        'case "$e" in *"Permission denied"*) echo "NET:denied";; *) echo "NET:other";; esac\n'
    )

    def test_no_process_environment_no_outside_files_and_no_tcp(
        self, defaults, launcher, tmp_path
    ):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret").write_text("key")
        probe = self._PROBE.replace("$OUTSIDE", str(outside)).replace(
            "$OUTPUT", str(tmp_path / "output")
        )
        proc = _run(
            _render(launcher["run"], _cfg(defaults)),
            tmp_path,
            {"PLANTED_SECRET": "s3cret", "HF_TOKEN": "hf_planted"},
            sandbox="real",
            stub_body=probe,
        )
        assert proc.returncode == 0, proc.stderr
        assert "sandbox on (Landlock ABI" in proc.stderr
        assert _lines(proc, "LEAKS:") == ["0"]
        # This shell and the test's own processes, at least, hold the secret.
        assert int(_lines(proc, "DENIED:")[0]) >= 2
        assert _lines(proc, "READ:") == ["refused"]
        assert _lines(proc, "WRITE:") == ["refused"]
        assert _lines(proc, "OUT:") == ["ok"]
        assert _lines(proc, "NET:") == ["denied"]
