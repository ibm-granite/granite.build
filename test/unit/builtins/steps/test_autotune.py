"""Structural unit tests for the `autotune` step asset (bash + docker copies)."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from gbserver.utils.template import fill_template

REPO_ROOT = Path(__file__).resolve().parents[4]
BASH_STEP = REPO_ROOT / "configurations/assets/environments/bash/steps/autotune"
DOCKER_STEP = REPO_ROOT / "configurations/assets/environments/docker/steps/autotune"
RUN_PY = BASH_STEP / "bash_scripts/autotune/run.py"
COMMAND_SH = BASH_STEP / "bash_scripts/autotune/command.sh"
SAMPLES = REPO_ROOT / "samples/autotune"

SAMPLE_AUTOTUNE_CONFIG = {
    "training_config": {"tuning_algorithm": {"default": "lora"}},
    "tuners_config": {"lora": {"title": "LoRA"}},
}


def _load_run_module():
    spec = importlib.util.spec_from_file_location("autotune_run", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _base_env(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "finance_train.jsonl").write_text("{}\n")
    (ds / "finance_validation.jsonl").write_text("{}\n")
    cfg = tmp_path / "autotune_config.yaml"
    cfg.write_text("training_config:\n  tuning_algorithm:\n    default: lora\n")
    out = tmp_path / "out"
    out.mkdir()
    return {
        "LLMB_BASH_INPUT_MODEL": "/models/granite",
        "LLMB_BASH_INPUT_DATASET_FILES": str(ds),
        "AUTOTUNE_CONFIG_FILE": str(cfg),
        "LLMB_BASH_OUTPUT_DIR": str(out),
        "JOB_ID": "job-123",
    }


class TestRunPyDriver:
    def test_build_argv_core_flags(self, tmp_path):
        mod = _load_run_module()
        argv = mod.build_argv(_base_env(tmp_path))
        assert argv[0] == "main.py"
        assert "--model_name_or_path" in argv
        assert argv[argv.index("--model_name_or_path") + 1] == "/models/granite"
        assert "--output_dir" in argv
        assert "--job_id" in argv and argv[argv.index("--job_id") + 1] == "job-123"
        # train/validation resolved by glob
        assert any(a.endswith("finance_train.jsonl") for a in argv)
        assert any(a.endswith("finance_validation.jsonl") for a in argv)

    def test_algo_default_read_from_config(self, tmp_path):
        mod = _load_run_module()
        argv = mod.build_argv(_base_env(tmp_path))
        assert argv[argv.index("--tuning_algo") + 1] == "lora"

    def test_bool_flags_added_when_true(self, tmp_path):
        mod = _load_run_module()
        env = _base_env(tmp_path)
        env.update({"NO_AUTOTUNE": "true", "CLEANUP": "1", "SAVE_HISTORY": "yes"})
        argv = mod.build_argv(env)
        assert (
            "--no_autotune" in argv and "--cleanup" in argv and "--save_history" in argv
        )

    def test_missing_required_input_exits(self, tmp_path):
        mod = _load_run_module()
        env = _base_env(tmp_path)
        del env["LLMB_BASH_INPUT_MODEL"]
        with pytest.raises(SystemExit):
            mod.build_argv(env)

    def test_artifact_marker_format_and_column0(self):
        mod = _load_run_module()
        marker = mod.artifact_marker("/abs/out")
        assert marker == "LLMB_ARTIFACT_ID:custom LLMB_ARTIFACT_PATH:/abs/out"
        assert marker.startswith("LLMB_ARTIFACT_ID:")  # column 0, no leading space

    def test_run_py_has_no_jinja_tokens(self):
        text = RUN_PY.read_text()
        for tok in ("{{", "{%", "{#"):
            assert tok not in text, f"run.py must not contain Jinja token {tok!r}"


class TestCommandShRender:
    def _render(self, config):
        return fill_template(COMMAND_SH.read_text(), {"config": config}, strict=True)

    def test_renders_with_inline_config(self):
        out = self._render({"autotune-config": SAMPLE_AUTOTUNE_CONFIG})
        assert "base64 -d" in out  # materialization branch taken
        assert "AUTOTUNE_CONFIG_FILE" in out
        assert "run.py" in out  # execs the driver

    def test_base64_roundtrip_is_valid_yaml(self):
        import base64
        import re

        import yaml

        out = self._render({"autotune-config": SAMPLE_AUTOTUNE_CONFIG})
        m = re.search(r"echo '([A-Za-z0-9+/=]+)' \| base64 -d", out)
        assert m, "expected an `echo '<b64>' | base64 -d` line"
        decoded = base64.b64decode(m.group(1)).decode()
        assert (
            yaml.safe_load(decoded)["training_config"]["tuning_algorithm"]["default"]
            == "lora"
        )

    def test_fallback_branch_when_no_inline_config(self):
        out = self._render({})  # no autotune-config key
        assert "LLMB_BASH_INPUT_HPO_CONFIG" in out  # fallback path rendered
        assert "base64 -d" not in out  # inline branch not taken

    def test_leads_path_with_python_dir(self):
        out = self._render({"autotune-config": SAMPLE_AUTOTUNE_CONFIG})
        assert "LLMB_BASH_PYTHON_DIR" in out

    def test_command_sh_is_executable(self):
        import os

        assert os.access(COMMAND_SH, os.X_OK), "command.sh must be committed with +x"

    def test_installs_real_fm_tune_extra_for_ray(self):
        # fm-tune's `ray` dep (imported unconditionally by main.py) lives in the
        # [core]/[full] extras, NOT the base package. fm-tune has no [mlx] extra,
        # so the venv install must use a real, param-driven extra defaulting to core.
        out = self._render({"autotune-config": SAMPLE_AUTOTUNE_CONFIG})
        assert "[mlx]" not in out  # fm-tune has no mlx extra
        assert "FM_TUNE_EXTRA" in out  # extra is param-driven
        assert "${FM_TUNE_EXTRA:-core}" in out  # defaults to core (provides ray)
        assert (
            "${FM_TUNE_ROOT}[${FM_TUNE_EXTRA}]" in out
        )  # extra applied to the editable install

    def test_clones_fm_tune_when_root_is_a_git_url(self):
        # FM_TUNE_ROOT may be a git remote (ssh/https/*.git), not just a local
        # checkout: command.sh detects a URL, clones it once to a local dir, and
        # exports FM_TUNE_ROOT at the clone so the editable install and run.py's
        # cwd both operate on a real tree. FM_TUNE_REF (optional) pins branch/tag.
        out = self._render({"autotune-config": SAMPLE_AUTOTUNE_CONFIG})
        assert "git clone" in out  # clone branch present
        assert "*://*" in out  # URL detection pattern
        assert "FM_TUNE_REF" in out  # optional branch/tag pin
        assert "export FM_TUNE_ROOT=" in out  # repoint to the clone
        # clone must precede the editable install that consumes FM_TUNE_ROOT
        assert out.index("git clone") < out.index("${FM_TUNE_ROOT}[")


class TestBashStepYaml:
    def _load(self):
        return yaml.safe_load((BASH_STEP / "step.yaml").read_text())

    def test_identity_and_artifact_policy(self):
        cfg = self._load()
        assert cfg["name"] == "autotune"
        assert cfg["type"] == "custom"
        assert cfg["config"]["bash"]["skip_finding_output_artifacts"] is True

    def test_io_schema(self):
        cfg = self._load()
        req = cfg["inputs"]["required"]
        assert req["model"]["type"] == "model"
        assert req["dataset_files"]["type"] == "dataset"
        assert cfg["inputs"]["optional"]["hpo_config"]["type"] == "fileset"
        assert cfg["outputs"]["optional"]["custom"]["type"] == "model"

    def test_bash_launcher_and_monitor(self):
        cfg = self._load()
        bash = cfg["environment_configs"]["Bash"]
        assert bash["launchers"]["autotune"]["type"] == "nohup"
        assert bash["launchers"]["autotune"]["monitors"] == ["log_monitor"]
        assert bash["monitors"]["log_monitor"]["ref"] == "space://monitors/bash"

    def test_script_dir_matches_step_name(self):
        assert (BASH_STEP / "bash_scripts/autotune/command.sh").exists()


class TestDockerStep:
    def _load(self):
        return yaml.safe_load((DOCKER_STEP / "step.yaml").read_text())

    def test_scripts_are_byte_identical_to_bash(self):
        for rel in ("bash_scripts/autotune/command.sh", "bash_scripts/autotune/run.py"):
            assert (DOCKER_STEP / rel).read_bytes() == (
                BASH_STEP / rel
            ).read_bytes(), rel

    def test_docker_launcher_and_monitor(self):
        cfg = self._load()
        d = cfg["environment_configs"]["Docker"]
        launcher = d["launchers"]["autotune"]
        assert launcher["type"] == "docker"
        assert "command.sh" in launcher["config"]["command"]
        assert launcher["config"]["image"]
        assert d["monitors"]["docker_log"]["ref"] == "space://monitors/docker"

    def test_docker_env_wires_inputs_and_backend(self):
        cfg = self._load()
        env = cfg["environment_configs"]["Docker"]["launchers"]["autotune"]["config"][
            "env"
        ]
        assert "bindings.model.binding.path" in env["LLMB_BASH_INPUT_MODEL"]
        assert (
            "bindings.dataset_files.binding.path"
            in env["LLMB_BASH_INPUT_DATASET_FILES"]
        )
        assert env["BASH_BUILD_VENV"] == "false"
        assert env["BACKEND"] == "torch"

    def test_command_sh_copy_is_executable(self):
        import os

        assert os.access(DOCKER_STEP / "bash_scripts/autotune/command.sh", os.X_OK)


class TestReferenceBuilds:
    def test_bash_build_shape(self):
        b = yaml.safe_load((SAMPLES / "build.bash.yaml").read_text())["granite.build"]
        target = b["targets"]["custom"]
        assert "bash" in target["environment_uri"]
        step = target["steps"][0]
        assert step["step_uri"] == "space://steps/autotune"
        assert target["inputs"]["model"]["uri"].startswith("hf:")
        assert "dataset_files" in target["inputs"]
        assert "autotune-config" in step["config"]
        env = step["config"]["bash"]["env"]
        assert env["BACKEND"] == "torch"
        assert env["FM_TUNE_EXTRA"] == "core"  # extra that provides ray
        assert "FM_TUNE_ROOT" in env

    def test_k8s_build_uses_custom_code_and_files_to_create(self):
        b = yaml.safe_load((SAMPLES / "build.k8s.yaml").read_text())["granite.build"]
        step = b["targets"]["custom"]["steps"][0]
        assert step["step_uri"] == "space://steps/custom_code"
        ftc = step["config"]["gb"]["files_to_create"]
        # a [{filename: configKey}] entry mapping the tmp path to the autotune-config section
        assert any(v == "autotune-config" for entry in ftc for v in entry.values())
        cfg_path = [
            k for entry in ftc for k, v in entry.items() if v == "autotune-config"
        ][0]
        assert "autotune-config" in step["config"]
        assert cfg_path in step["config"]["custom_code_config"]["start_command"]
        assert (
            "--config_file " + cfg_path
            in step["config"]["custom_code_config"]["start_command"]
        )
