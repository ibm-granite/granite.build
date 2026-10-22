# SPDX-License-Identifier: Apache-2.0
"""Render tests for the `eval/livecodebench-score` step's step-template.yaml.

Cluster-agnostic, so this sits at the root of the step's ``test/`` dir (Mode 1
only) and is not copied by ``make publish-step``.

The rendered ``run`` block is executed with a stub ``python`` on PATH that records
its argv and environment, so the assertions are on what bash actually hands the
bundled scorer, not on rendered text.
"""

import os
import pathlib
import shutil
import subprocess

import pytest
import yaml

jinja2 = pytest.importorskip("jinja2", reason="jinja2 renders the step template")

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = _STEP_DIR / "step-template.yaml"
_COMPLETIONS = "/workdir/inputs/completions"
_DATASET = "/workdir/inputs/dataset"


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


def _run(
    rendered: str, cwd: pathlib.Path, env_over: dict | None = None
) -> subprocess.CompletedProcess:
    """Execute a rendered `run` block with a stub `python` on PATH.

    The stub prints ``ARG:<word>`` per argument, plus the working directory and the
    environment variables the step must set or clear.
    """
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available")
    bin_dir = cwd / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do echo "ARG:$a"; done\n'
        'echo "CWD:$(pwd)"\n'
        'echo "ENV:HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-}"\n'
        'echo "ENV:HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-}"\n'
        'echo "ENV:PYTHONPATH=${PYTHONPATH:-}"\n'
    )
    stub.chmod(0o755)
    (cwd / "venv" / "bin").mkdir(parents=True)
    (cwd / "venv" / "bin" / "activate").write_text("")
    (cwd / "LiveCodeBench").mkdir()
    env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env.update(env_over or {})
    return subprocess.run(
        ["bash", "-c", rendered], capture_output=True, text=True, cwd=cwd, env=env
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
            and "'torch==2.6.0'" in setup
        )
        assert 'UV_CACHE_DIR="$PWD/.uv-cache"' in setup
        assert "GB_SHARED_WORKDIR" not in setup

    def test_torch_comes_before_the_rest(self, defaults, launcher):
        setup = _render(launcher["setup"], _cfg(defaults))
        assert setup.index("torch==2.6.0") < setup.index("requirements-score.txt")

    def test_install_false_installs_nothing(self, defaults, launcher):
        setup = _render(launcher["setup"], _cfg(defaults, install=False))
        assert _bash_ok(setup) and "uv" not in setup and "git" not in setup


class TestRun:
    def test_parses(self, defaults, launcher):
        assert _bash_ok(_render(launcher["run"], _cfg(defaults)))

    def test_hands_the_scorer_the_bindings_and_window(
        self, defaults, launcher, tmp_path
    ):
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path)
        argv = _lines(proc, "ARG:")
        assert argv[0] == str(tmp_path / "src" / "lcb_score.py")
        assert _opt(argv, "--completions-dir") == _COMPLETIONS
        assert _opt(argv, "--dataset-dir") == _DATASET
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

    def test_marker_is_absolute_and_starts_the_line(self, defaults, launcher, tmp_path):
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert (
            f"GB_ARTIFACT_ID:results GB_ARTIFACT_PATH:{tmp_path / 'output'}"
            in proc.stdout.splitlines()
        )

    def test_refuses_to_run_with_an_hf_token(self, defaults, launcher, tmp_path):
        proc = _run(
            _render(launcher["run"], _cfg(defaults)), tmp_path, {"HF_TOKEN": "hf_x"}
        )
        assert proc.returncode == 1 and "HF_TOKEN is set" in proc.stderr
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
