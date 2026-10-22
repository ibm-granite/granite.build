# SPDX-License-Identifier: Apache-2.0
"""Render tests for the `eval/livecodebench-generate` step's step-template.yaml.

Cluster-agnostic, so this sits at the root of the step's ``test/`` dir (Mode 1
only) and is not copied by ``make publish-step``.

The rendered ``run`` block is executed with a stub ``python`` on PATH that records
its argv, so the assertions are on what bash hands the bundled script.
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
_MODEL = "/workdir/inputs/model"
_DATASET = "/workdir/inputs/dataset"


@pytest.fixture(scope="module")
def template() -> dict:
    return yaml.safe_load(_TEMPLATE.read_text())


@pytest.fixture(scope="module")
def defaults(template) -> dict:
    return dict(template["config"]["lcb_generate_config"])


@pytest.fixture(scope="module")
def launcher(template) -> dict:
    return template["environment_configs"]["Skypilot"]["launchers"]["lcb_generate"][
        "config"
    ]


def _cfg(defaults: dict, **over) -> dict:
    """The USAGE.md example config, with overrides."""
    base = dict(
        defaults,
        release_version="release_v6",
        start_date="2025-04-05",
        end_date="2025-04-05",
        max_tokens=8192,
        enable_thinking="false",
    )
    base.update(over)
    return base


def _render(source: str, cfg: dict) -> str:
    return jinja2.Template(source, undefined=jinja2.StrictUndefined).render(
        config={"lcb_generate_config": cfg},
        bindings={
            "model": {"binding": {"path": _MODEL}},
            "dataset": {"binding": {"path": _DATASET}},
        },
    )


def _bash_ok(script: str) -> bool:
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available")
    return subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


def _run(rendered: str, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available")
    bin_dir = cwd / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text(
        '#!/usr/bin/env bash\nfor a in "$@"; do echo "ARG:$a"; done\n'
        'echo "CWD:$(pwd)"\necho "ENV:HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-}"\n'
        'echo "ENV:VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-}"\n'
    )
    stub.chmod(0o755)
    (cwd / "venv" / "bin").mkdir(parents=True)
    (cwd / "venv" / "bin" / "activate").write_text("")
    (cwd / "LiveCodeBench").mkdir()
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
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
        assert template["name"] == "livecodebench-generate"
        assert template["type"] == "custom"

    def test_no_image_ref_token(self):
        assert "${IMAGE_REF}" not in _TEMPLATE.read_text()

    def test_serves_every_skypilot_endpoint(self, template):
        skypilot = template["environment_configs"]["Skypilot"]
        assert (
            "subtypes" not in skypilot
            and skypilot["default_launcher"] == "lcb_generate"
        )

    def test_declares_model_and_dataset_and_a_completions_output(self, template):
        assert template["inputs"]["allow_unknown"] is True
        assert template["inputs"]["required"] == {
            "model": {"type": "model", "accept": ["uri", "binding"]},
            "dataset": {"type": "dataset", "accept": ["uri", "binding"]},
        }
        assert "optional" not in template["inputs"]
        assert template["outputs"]["optional"]["completions"]["type"] == "dataset"

    def test_the_model_is_not_a_config_key(self, defaults):
        assert "model" not in defaults

    def test_required_keys_have_no_usable_default(self, defaults):
        assert defaults["release_version"] == "" and defaults["max_tokens"] == 0

    def test_lcb_commit_matches_the_score_step(self, defaults):
        score = yaml.safe_load(
            (
                _STEP_DIR.parents[1]
                / "livecodebench-score"
                / "skypilot"
                / "step-template.yaml"
            ).read_text()
        )
        assert (
            defaults["lcb_commit"] == score["config"]["lcb_score_config"]["lcb_commit"]
        )

    def test_ships_src(self, launcher):
        assert launcher["file_mounts"] == {"src": "src"}
        for f in ("lcb_generate.py", "lcb_dataset.py", "requirements-generate.txt"):
            assert (_STEP_DIR / "src" / f).is_file()


class TestImage:
    def test_default_image_is_python_312_slim(self, defaults, launcher):
        assert defaults["image"] == "python:3.12-slim" and defaults["install"] is True
        assert _render(launcher["image_id"], defaults) == "docker:python:3.12-slim"

    def test_empty_image_runs_on_the_bare_node(self, defaults, launcher):
        assert _render(launcher["image_id"], _cfg(defaults, image="")) == ""


class TestSetup:
    def test_installs_vllm_and_lcb_without_its_deps(self, defaults, launcher):
        setup = _render(launcher["setup"], _cfg(defaults))
        assert _bash_ok(setup)
        assert (
            "'vllm==0.32.0'" in setup and "-r ./src/requirements-generate.txt" in setup
        )
        assert 'uv pip install --no-deps -e "$LCB_DIR"' in setup
        assert f"'{defaults['lcb_commit']}'" in setup
        assert "GB_SHARED_WORKDIR" not in setup

    def test_install_false_installs_nothing(self, defaults, launcher):
        setup = _render(launcher["setup"], _cfg(defaults, install=False))
        assert _bash_ok(setup) and "uv" not in setup and "git" not in setup


class TestRun:
    def test_parses(self, defaults, launcher):
        assert _bash_ok(_render(launcher["run"], _cfg(defaults)))

    def test_hands_the_script_the_bindings_and_settings(
        self, defaults, launcher, tmp_path
    ):
        argv = _lines(_run(_render(launcher["run"], _cfg(defaults)), tmp_path), "ARG:")
        assert argv[0] == str(tmp_path / "src" / "lcb_generate.py")
        expected = {
            "--model-dir": _MODEL,
            "--dataset-dir": _DATASET,
            "--output-dir": str(tmp_path / "output"),
            "--release-version": "release_v6",
            "--start-date": "2025-04-05",
            "--end-date": "2025-04-05",
            "--n": "1",
            "--temperature": "0.2",
            "--top-p": "0.95",
            "--max-tokens": "8192",
            "--enable-thinking": "false",
            "--tensor-parallel-size": "1",
            "--dtype": "bfloat16",
            "--gpu-memory-utilization": "0",
            "--max-model-len": "0",
            "--lcb-commit": defaults["lcb_commit"],
        }
        assert {k: _opt(argv, k) for k in expected} == expected
        assert "--trust-remote-code" not in argv

    def test_trust_remote_code_flag(self, defaults, launcher, tmp_path):
        argv = _lines(
            _run(
                _render(launcher["run"], _cfg(defaults, trust_remote_code=True)),
                tmp_path,
            ),
            "ARG:",
        )
        assert "--trust-remote-code" in argv

    def test_runs_offline_from_inside_the_checkout(self, defaults, launcher, tmp_path):
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path)
        assert _lines(proc, "CWD:") == [str(tmp_path / "LiveCodeBench")]
        # The FlashInfer sampler is off, so vLLM never JIT-compiles with the node's nvcc.
        assert _lines(proc, "ENV:") == [
            "HF_HUB_OFFLINE=1",
            "VLLM_USE_FLASHINFER_SAMPLER=0",
        ]

    def test_marker_is_absolute_and_starts_the_line(self, defaults, launcher, tmp_path):
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert (
            f"GB_ARTIFACT_ID:completions GB_ARTIFACT_PATH:{tmp_path / 'output'}"
            in proc.stdout.splitlines()
        )

    def test_refuses_an_empty_release(self, defaults, launcher, tmp_path):
        proc = _run(
            _render(launcher["run"], _cfg(defaults, release_version="")), tmp_path
        )
        assert proc.returncode == 1 and "release_version is required" in proc.stderr

    @pytest.mark.parametrize("bad", [0, "", "lots", "-5"])
    def test_refuses_a_missing_or_bad_max_tokens(
        self, defaults, launcher, tmp_path, bad
    ):
        proc = _run(_render(launcher["run"], _cfg(defaults, max_tokens=bad)), tmp_path)
        assert proc.returncode == 1 and "max_tokens is required" in proc.stderr
        assert "ARG:" not in proc.stdout

    def test_quotes_author_values(self, defaults, launcher, tmp_path):
        nasty = "2025-04-05'; touch PWNED; echo '"
        argv = _lines(
            _run(_render(launcher["run"], _cfg(defaults, start_date=nasty)), tmp_path),
            "ARG:",
        )
        assert _opt(argv, "--start-date") == nasty
        assert (
            not (tmp_path / "PWNED").exists()
            and not (tmp_path / "LiveCodeBench" / "PWNED").exists()
        )
