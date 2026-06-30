"""Unit tests for the openinstruct-rl (GRPO) step asset.

openinstruct-rl is a declarative step.yaml under
``configurations/assets/environments/skypilot/lsf/steps/openinstruct-rl/``,
companion to openinstruct-sft. These tests load the shipped asset, render its
``run:`` block via the same Jinja renderer production uses
(``gbserver.utils.template.fill_template``), and assert the GRPO command,
quoting, boolean-flag toggling, service-env exports, and that the monitor's
NEWARTIFACT regex matches the line the run script emits.
"""

import re
from pathlib import Path

import yaml

from gbserver.utils.template import fill_template

REPO_ROOT = Path(__file__).resolve().parents[4]
RL_STEP_YAML = (
    REPO_ROOT
    / "configurations/assets/environments/skypilot/lsf/steps/openinstruct-rl/step.yaml"
)


def _load() -> dict:
    return yaml.safe_load(RL_STEP_YAML.read_text())


class TestOpeninstructRlStep:
    def test_step_yaml_exists(self):
        assert RL_STEP_YAML.exists(), f"{RL_STEP_YAML} does not exist"

    def test_step_yaml_identity_and_outputs(self):
        cfg = _load()
        assert cfg["name"] == "openinstruct-rl"
        assert cfg["type"] == "training"
        assert "rl_config" in cfg["config"]
        assert cfg["outputs"]["optional"]["checkpoint"]["type"] == "model"
        # numeric hyperparameters are intentionally quoted strings (avoid YAML
        # float coercion; they are only interpolated into a shell command)
        assert isinstance(cfg["config"]["rl_config"]["learning_rate"], str)
        assert isinstance(cfg["config"]["rl_config"]["beta"], str)


SAMPLE_CONFIG = {
    "config": {
        "rl_config": {
            **_load()["config"]["rl_config"],  # start from shipped defaults
            "exp_name": "gb-ifrl-test",
            "run_name": "ifrl-run-0",
            "model_name_or_path": "/proj/models/granite4-350m",
            "dataset_mixer": '{"ai2-adapt-dev/rlvr_gsm8k_zs": 1.0}',
            "dataset_eval_mixer": '{"ai2-adapt-dev/rlvr_gsm8k_zs": 16}',
            "output_dir": "/proj/runs/ifrl/checkpoints",
            "checkpoint_state_dir": "/proj/runs/ifrl/state",
            "learning_rate": "5e-7",
            "rm_server_url": "http://rm-server:8000",
        }
    }
}


def _render_run() -> str:
    cfg = _load()
    run_block = cfg["environment_configs"]["Skypilot"]["launchers"]["rl"]["config"][
        "run"
    ]
    return fill_template(run_block, SAMPLE_CONFIG, strict=False)


class TestOpeninstructRlRun:
    def test_launcher_shape(self):
        cfg = _load()
        rl = cfg["environment_configs"]["Skypilot"]["launchers"]["rl"]
        assert rl["type"] == "skypilot"
        assert rl["monitors"] == ["skypilot_monitor"]
        assert "open-instruct-3:0.1.0-conda" in rl["config"]["image_id"]

    def test_command_uses_grpo_fast_with_values(self):
        run = _render_run()
        assert "open_instruct.grpo_fast" in run
        assert "--exp_name gb-ifrl-test" in run
        assert "--learning_rate 5e-7" in run
        assert "--output_dir /proj/runs/ifrl/checkpoints" in run
        assert "--model_name_or_path /proj/models/granite4-350m" in run

    def test_heredoc_is_quoted_and_self_contained(self):
        run = _render_run()
        assert "<< 'GRPO_CMD'" in run
        assert "PYTHON_BIN=" in run
        assert "PYTHONPATH=/stage:/stage/open_instruct" in run

    def test_stop_strings_and_dataset_mixer_quoting(self):
        run = _render_run()
        assert "--dataset_mixer '{\"ai2-adapt-dev/rlvr_gsm8k_zs\": 1.0}'" in run
        assert '--stop_strings "<|end_of_text|>"' in run
        assert '"<|im_end|>"' in run

    def test_home_exported_for_nltk_data(self):
        # The image's NLTK data (punkt_tab, needed by verifiable rewards) lives
        # under /stage/nltk_data, found only via $HOME/nltk_data. The SkyPilot
        # LSF backend sets HOME=/, so the run block must export HOME=/stage
        # before launching the trainer (matches gbansible). Must be outside the
        # quoted heredoc so it propagates to the cmd.sh child process.
        run = _render_run()
        assert "export HOME=/stage" in run
        assert run.index("export HOME=/stage") < run.index("<< 'GRPO_CMD'")

    def test_boolean_flags_toggle(self):
        run = _render_run()
        assert "--gradient_checkpointing" in run
        assert "--add_general_reward true" in run
        assert "--set_weight_decay_on_bias_and_norm false" in run
        assert "--with_tracking false" in run

    def test_service_env_present_when_set(self):
        run = _render_run()
        assert 'RM_SERVER_URL="http://rm-server:8000"' in run

    def test_emits_artifact_line(self):
        run = _render_run()
        emitted = (
            "LLMB_ARTIFACT_ID:checkpoint LLMB_ARTIFACT_PATH:/proj/runs/ifrl/checkpoints"
        )
        assert emitted in run

    def test_service_env_absent_when_unset(self):
        # code_server_url defaults to "" → the {% if %} guard omits the export
        run = _render_run()
        assert "CODE_SERVER_URL" not in run

    def test_boolean_flags_flipped_branch(self):
        """Exercise the conditional branches the default config never renders:
        with_tracking on, and the three off-by-default flags toggled."""
        flipped = {
            "config": {
                "rl_config": {
                    **_load()["config"]["rl_config"],
                    "exp_name": "gb-ifrl-test",
                    "output_dir": "/proj/runs/ifrl/checkpoints",
                    "with_tracking": True,
                    "set_weight_decay_on_bias_and_norm": True,
                    "additive_format_reward": True,
                    "filter_zero_advantage": True,
                }
            }
        }
        cfg = _load()
        run_block = cfg["environment_configs"]["Skypilot"]["launchers"]["rl"]["config"][
            "run"
        ]
        run = fill_template(run_block, flipped, strict=False)
        # with_tracking true → bare flag, and the explicit-false lines are gone
        assert "--with_tracking false" not in run
        assert "--with_tracking" in run
        assert "--set_weight_decay_on_bias_and_norm false" not in run
        assert "--additive_format_reward false" not in run
        assert "--filter_zero_advantage false" not in run

    def test_boolean_flags_string_values(self):
        """Recipe build params arrive via $${...} substitution as STRINGS, not
        YAML booleans. A bare Jinja `if` treats the non-empty string "false" as
        truthy, which would (a) emit a bare --with_tracking (parsed True by
        HfArgumentParser → grpo_fast runs wandb.login and crashes) and (b) drop
        the explicit `--X false` flags. The template must compare against a
        truthy set so string "false" is handled correctly."""
        string_cfg = {
            "config": {
                "rl_config": {
                    **_load()["config"]["rl_config"],
                    "exp_name": "gb-ifrl-test",
                    "output_dir": "/proj/runs/ifrl/checkpoints",
                    # mirror quoted $${...} substitution: all strings
                    "with_tracking": "false",
                    "gradient_checkpointing": "true",
                    "add_general_reward": "true",
                    "set_weight_decay_on_bias_and_norm": "false",
                    "additive_format_reward": "false",
                    "filter_zero_advantage": "false",
                }
            }
        }
        cfg = _load()
        run_block = cfg["environment_configs"]["Skypilot"]["launchers"]["rl"]["config"][
            "run"
        ]
        run = fill_template(run_block, string_cfg, strict=False)
        # "false" string must NOT produce a bare --with_tracking (would be True)
        assert "--with_tracking false" in run
        # the explicit-false flags must still be emitted (not dropped)
        assert "--set_weight_decay_on_bias_and_norm false" in run
        assert "--additive_format_reward false" in run
        assert "--filter_zero_advantage false" in run
        # "true" strings still emit their flags
        assert "--gradient_checkpointing" in run
        assert "--add_general_reward true" in run


class TestOpeninstructRlMonitor:
    def test_monitor_defined(self):
        cfg = _load()
        mons = cfg["environment_configs"]["Skypilot"]["monitors"]
        assert mons["skypilot_monitor"]["type"] == "skypilot_monitor"

    def test_newartifact_regex_matches_emitted_line(self):
        """The monitor's NEWARTIFACT line_regex must match the exact line the
        run script emits — this ties emitter and monitor together."""
        cfg = _load()
        events = cfg["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"][
            "config"
        ]["event_configs"]
        newart = next(
            e for e in events if e["event_type"] == "NEWARTIFACT_IN_ENVIRONMENT_EVENT"
        )
        emitted = (
            "LLMB_ARTIFACT_ID:checkpoint "
            "LLMB_ARTIFACT_PATH:/proj/runs/ifrl/checkpoints"
        )
        assert re.search(newart["line_regex"], emitted)
        path_field = next(
            f for f in newart["event_fields"] if f["field_name"] == "path"
        )
        m = re.search(path_field["field_regex"], emitted)
        assert m and m.group(0) == "/proj/runs/ifrl/checkpoints"

    def test_progress_regex_matches_metric_line(self):
        cfg = _load()
        events = cfg["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"][
            "config"
        ]["event_configs"]
        status = next(e for e in events if e["event_type"] == "WORKLOAD_STATUS_EVENT")
        assert re.search(status["line_regex"], "episode: 128 | training_step: 4")


def test_asset_loads_and_run_block_renders():
    """Mirror production's load path for an asset step.yaml.

    Asset step.yaml files are loaded with plain ``yaml.safe_load``
    (``targetstep.py``), which drops comments; templated *values* such as the
    ``run:`` block are rendered later, per-value, via ``fill_template``. (Only
    the gbstep ``step_default.yaml`` fallback is whole-file Jinja-rendered.)
    This guards both halves: the raw file parses as YAML, and its run block
    renders to valid, non-empty bash without leaving unresolved ``{{ }}``
    config placeholders.
    """
    # 1. Raw file parses as PyYAML sees it (comments dropped, no Jinja).
    cfg = yaml.safe_load(RL_STEP_YAML.read_text())
    assert cfg["name"] == "openinstruct-rl"

    # 2. The run-block value renders cleanly with a representative config.
    run = _render_run()
    assert "open_instruct.grpo_fast" in run
    assert "--exp_name gb-ifrl-test" in run
    # no unresolved config placeholders survived rendering
    assert "{{ config.rl_config" not in run
