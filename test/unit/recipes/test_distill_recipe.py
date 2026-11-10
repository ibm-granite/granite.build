#!/usr/bin/env python3

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

"""recipes/distill/distill: one distillation recipe, three lambda modes, a smoke layer.

The recipe is rendered the way `gb build start` renders it, through
gbcli.utils.buildutil.apply_parameters, with its parameters layered: the recipe's
parameters.yaml, then a site file (here site.example.yaml), then optionally
smoke.yaml, then --param overrides. Most of what is asserted is agreement between
values a person would otherwise keep in step by hand: the lambda mode and the
rollout server, the ladder and the trainer's save grid, the recipe's config keys
and the steps' schemas.
"""

import pathlib
import re

import jinja2
import pytest
import yaml
from jinja2 import meta

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_DISTILL = _REPO / "recipes" / "distill"
_RECIPE = _DISTILL / "distill"
_BUILD = _RECIPE / "build.yaml"
_PARAMS = _RECIPE / "parameters.yaml"
_SMOKE = _RECIPE / "smoke.yaml"
_SITE = _DISTILL / "site.example.yaml"
_STEPS = _REPO / "steps"
_SKYPILOT_STEPS = _REPO / "configurations" / "assets" / "environments" / "skypilot" / "steps"

_MODES = ("off-policy", "ramp", "on-policy")

# step_uri -> the file whose `config` is that step's schema.
_STEP_SCHEMAS = {
    "space://steps/train/distill/data-mixture": _STEPS
    / "train/distill/data-mixture/skypilot/step-template.yaml",
    "space://steps/train/distill/data-prep": _STEPS
    / "train/distill/data-prep/skypilot/step-template.yaml",
    "space://steps/train/distill/data-validate": _STEPS
    / "train/distill/data-validate/skypilot/step-template.yaml",
    "space://steps/train/distill/tokenizer-align": _STEPS
    / "train/distill/tokenizer-align/skypilot/step-template.yaml",
    "space://steps/train/distill/student-sft": _STEPS
    / "train/distill/student-sft/skypilot/step-template.yaml",
    "space://steps/train/distill/student-rollout-generator": _STEPS
    / "train/distill/student-rollout-generator/skypilot/step-template.yaml",
    "space://steps/train/distill/logit-distill": _STEPS
    / "train/distill/logit-distill/skypilot/step-template.yaml",
    "space://steps/train/distill/teacher-divergence": _STEPS
    / "train/distill/teacher-divergence/skypilot/step-template.yaml",
    "space://steps/train/distill/repetition-check": _STEPS
    / "train/distill/repetition-check/skypilot/step-template.yaml",
    "space://steps/misc/hf-export": _STEPS
    / "misc/hf-export/skypilot/step-template.yaml",
    "space://steps/skypilot-teardown": _SKYPILOT_STEPS / "skypilot-teardown/step.yaml",
    "space://steps/sage-eval": _SKYPILOT_STEPS / "sage-eval/step.yaml",
    "space://steps/bfcl-eval": _SKYPILOT_STEPS / "bfcl-eval/step.yaml",
    "space://steps/sage-export": _SKYPILOT_STEPS / "sage-export/step.yaml",
    "space://steps/bfcl-export": _SKYPILOT_STEPS / "bfcl-export/step.yaml",
    "space://steps/combined-export": _SKYPILOT_STEPS / "combined-export/step.yaml",
}

# Top-level step config keys every SkyPilot step reads through the shared monitor or
# the launcher, not through its own schema.
_GENERIC_CONFIG = {
    "poll_interval_seconds",
    "log_retrieval_interval_seconds",
    "log_retrieval_mode",
    "compute_config",
    "launcher_config",
}

_BENCHMARK = re.compile(r"^(eval-.+-ck\d+|export-(sage|bfcl)-ck\d+|export-combined)$")


# ─── Rendering ────────────────────────────────────────────────────────────────


def _layered(*layers):
    """Merge parameter files in order, later layers replacing a key whole.

    Mirrors the layered --parameters-path of ibm-granite/granite.build#663 (the
    recipe's parameters.yaml first, then each --parameters-path in order), which is
    not on this branch; --param overrides are applied after this by apply_parameters.
    """
    merged = {}
    for layer in (_PARAMS, *layers):
        merged.update(get_params_from_file(str(layer)))
    return merged


def _render_text(tmp_path, *layers, **params):
    """Render with the site file and the given extra layers; params arrive as
    strings, exactly as `--param KEY=VALUE` delivers them."""
    contents = _BUILD.read_text(encoding="utf-8")
    cli = [f"{k}={v}" for k, v in params.items()]
    return apply_parameters(contents, cli, _layered(_SITE, *layers), str(tmp_path))


def _render(tmp_path, *layers, **params):
    return yaml.safe_load(_render_text(tmp_path, *layers, **params))


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def _step(rendered, target):
    (step,) = _targets(rendered)[target]["steps"]
    return step


def _config(rendered, target):
    return _step(rendered, target)["config"]


def _gold(rendered):
    return _config(rendered, "train")["gold_config"]


def _yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ladder(params):
    return [int(n) for n in str(params["CKPT_LADDER"]).replace(" ", "").split(",")]


# ─── Every mode renders ───────────────────────────────────────────────────────


@pytest.mark.parametrize("smoke", [False, True], ids=["full", "smoke"])
@pytest.mark.parametrize("mode", _MODES)
def test_every_mode_renders(tmp_path, mode, smoke):
    layers = (_SMOKE,) if smoke else ()
    text = _render_text(tmp_path, *layers, RUN_NAME="t", LMBDA_MODE=mode)
    assert "$${" not in text and "<%" not in text
    rendered = yaml.safe_load(text)
    assert isinstance(_targets(rendered), dict)


def test_defaults_are_off_policy():
    assert _layered()["LMBDA_MODE"] == "off-policy"


# ─── LMBDA_MODE ───────────────────────────────────────────────────────────────


def test_off_policy(tmp_path):
    rendered = _render(tmp_path, LMBDA_MODE="off-policy")
    targets = _targets(rendered)
    assert "rollout-server" not in targets
    assert "teardown" not in targets
    gold = _gold(rendered)
    assert gold["lmbda"] == 0.0
    assert gold["vllm_num_servers"] == 0
    assert "vllm" not in targets["train"]["inputs"]
    # The on-policy-only keys the renderer refuses off-policy stay unrendered.
    for key in ("lmbda_schedule", "min_completion_length", "lmbda_ramp_steps"):
        assert key not in gold


def test_ramp(tmp_path):
    rendered = _render(tmp_path, LMBDA_MODE="ramp")
    targets = _targets(rendered)
    assert "rollout-server" in targets and "teardown" in targets
    gold = _gold(rendered)
    assert gold["lmbda"] == 1.0
    assert gold["lmbda_schedule"] == "linear"
    assert gold["lmbda_init"] == 0.0
    assert "lmbda_ramp_start_step" in gold and "lmbda_ramp_steps" in gold
    assert targets["train"]["inputs"]["vllm"] == {"binding": "rollout-server.vllm_url"}


def test_on_policy(tmp_path):
    rendered = _render(tmp_path, LMBDA_MODE="on-policy")
    targets = _targets(rendered)
    assert "rollout-server" in targets and "teardown" in targets
    gold = _gold(rendered)
    assert gold["lmbda"] == 1.0
    assert gold["lmbda_schedule"] == "constant"
    # The renderer refuses the ramp window under a constant schedule.
    assert "lmbda_ramp_steps" not in gold
    assert "vllm" in targets["train"]["inputs"]


def test_fractional_lmbda_needs_the_server(tmp_path):
    rendered = _render(tmp_path, LMBDA_MODE="on-policy", LMBDA="0.5")
    targets = _targets(rendered)
    assert "rollout-server" in targets and "teardown" in targets
    gold = _gold(rendered)
    assert gold["lmbda"] == 0.5
    assert gold["lmbda_schedule"] == "constant"


def test_fractional_lmbda_ends_the_ramp(tmp_path):
    gold = _gold(_render(tmp_path, LMBDA_MODE="ramp", LMBDA="0.75"))
    assert (gold["lmbda"], gold["lmbda_schedule"]) == (0.75, "linear")


@pytest.mark.parametrize(
    "params, needle",
    [
        ({"LMBDA_MODE": "onpolicy"}, "LMBDA_MODE"),
        ({"LMBDA_MODE": ""}, "LMBDA_MODE"),
        ({"LMBDA_MODE": "off-policy", "LMBDA": "0.5"}, "LMBDA_MODE"),
        ({"LMBDA_MODE": "on-policy", "LMBDA": "0"}, "LMBDA"),
        ({"LMBDA_MODE": "on-policy", "LMBDA": "1.5"}, "LMBDA"),
    ],
)
def test_bad_lambda_settings_fail_to_render(tmp_path, params, needle):
    with pytest.raises(jinja2.exceptions.UndefinedError) as err:
        _render_text(tmp_path, **params)
    assert needle in str(err.value)


@pytest.mark.parametrize(
    "params, needle",
    [
        ({"LMBDA_MODE": "off-policy", "MIN_COMPLETION_LENGTH": "16"}, "MIN_COMPLETION"),
        ({"LMBDA_MODE": "off-policy", "LMBDA_RAMP_START_STEP": "100"}, "LMBDA_RAMP"),
        ({"LMBDA_MODE": "off-policy", "LMBDA_RAMP_STEPS": "100"}, "LMBDA_RAMP"),
        ({"LMBDA_MODE": "on-policy", "LMBDA_RAMP_START_STEP": "100"}, "LMBDA_RAMP"),
        ({"LMBDA_MODE": "on-policy", "LMBDA_RAMP_STEPS": "100"}, "LMBDA_RAMP"),
    ],
)
def test_settings_a_mode_would_drop_fail_to_render(tmp_path, params, needle):
    """A key the mode does not render would otherwise be silently ignored."""
    with pytest.raises(jinja2.exceptions.UndefinedError) as err:
        _render_text(tmp_path, **params)
    assert needle in str(err.value)


@pytest.mark.parametrize("mode", _MODES)
def test_zero_and_empty_drop_settings_render(tmp_path, mode):
    """0 is never a setting the mode drops, and empty is the floor's default."""
    _render(
        tmp_path,
        LMBDA_MODE=mode,
        MIN_COMPLETION_LENGTH="0",
        LMBDA_RAMP_START_STEP="0",
        LMBDA_RAMP_STEPS="0",
    )
    _render(tmp_path, LMBDA_MODE=mode, MIN_COMPLETION_LENGTH="")


@pytest.mark.parametrize("mode", ["ramp", "on-policy"])
def test_generation_floor_defaults_to_16_online(tmp_path, mode):
    assert _gold(_render(tmp_path, LMBDA_MODE=mode))["min_completion_length"] == 16
    gold = _gold(_render(tmp_path, LMBDA_MODE=mode, MIN_COMPLETION_LENGTH="0"))
    assert gold["min_completion_length"] == 0


def test_ramp_window_reaches_the_trainer(tmp_path):
    gold = _gold(
        _render(
            tmp_path,
            LMBDA_MODE="ramp",
            LMBDA_RAMP_START_STEP="2000",
            LMBDA_RAMP_STEPS="2500",
        )
    )
    assert (gold["lmbda_ramp_start_step"], gold["lmbda_ramp_steps"]) == (2000, 2500)


@pytest.mark.parametrize("mode", _MODES)
def test_the_server_is_never_carved_out_of_the_trainer(tmp_path, mode):
    """vllm_num_servers > 0 without the `vllm` input would split the trainer's own
    allocation. The renderer requires 1 alongside a bound external server, so the
    count may only be non-zero when the input is bound."""
    rendered = _render(tmp_path, LMBDA_MODE=mode)
    servers = _gold(rendered)["vllm_num_servers"]
    bound = "vllm" in _targets(rendered)["train"]["inputs"]
    assert servers == (1 if bound else 0)


@pytest.mark.parametrize("mode", ["ramp", "on-policy"])
def test_rollout_server_serves_the_trained_student(tmp_path, mode):
    for sft in ("false", "true"):
        targets = _targets(_render(tmp_path, LMBDA_MODE=mode, INCLUDE_SFT=sft))
        assert (
            targets["rollout-server"]["inputs"]["model"]
            == targets["train"]["inputs"]["student"]
        )
        assert targets["teardown"]["inputs"]["trained"] == {
            "binding": "train.checkpoint"
        }
        cfg = _config({"granite.build": {"targets": targets}}, "rollout-server")
        assert (
            cfg["vllm_config"]["max_model_len"]
            >= _gold({"granite.build": {"targets": targets}})["max_length"]
        )


@pytest.mark.parametrize("smoke", [False, True], ids=["full", "smoke"])
@pytest.mark.parametrize("mode", ["ramp", "on-policy"])
def test_trainer_and_server_share_one_stop_file(tmp_path, mode, smoke):
    """The trainer's rank 0 writes the stop file when it fails; the server polls
    the same path and stops. One path, under the build's shared directory, so
    both allocations see it and no other build's file is ever read."""
    layers = (_SMOKE,) if smoke else ()
    rendered = _render(tmp_path, *layers, LMBDA_MODE=mode, RUN_NAME="arm-x")
    trainer = _config(rendered, "train")["rollout_stop_file"]
    server = _config(rendered, "rollout-server")["vllm_config"]["stop_file"]
    assert trainer and trainer == server
    assert "{{ run_metadata.build_id }}" in trainer
    root = _layered(_SITE)["WORKDIR_ROOT"].rstrip("/")
    assert trainer.startswith(f"{root}/arm-x/{{{{ run_metadata.build_id }}}}/")


def test_server_lifetime_cap_is_twelve_hours(tmp_path):
    """The stop file handles a trainer that fails; the cap is only the backstop
    for a rank 0 killed outright, so it need not cover a whole run. Smoke's own
    cap stays the smaller."""
    assert int(_layered()["VLLM_MAX_LIFETIME_SECONDS"]) == 43200
    assert int(_yaml(_SMOKE)["VLLM_MAX_LIFETIME_SECONDS"]) < 43200
    cfg = _config(_render(tmp_path, LMBDA_MODE="ramp"), "rollout-server")
    assert cfg["vllm_config"]["max_lifetime_seconds"] == 43200


def test_off_policy_sets_no_stop_file(tmp_path):
    cfg = _config(_render(tmp_path, LMBDA_MODE="off-policy"), "train")
    assert not cfg.get("rollout_stop_file")


def test_on_policy_mem_uris_carry_the_run_name(tmp_path):
    outputs = _targets(_render(tmp_path, LMBDA_MODE="ramp", RUN_NAME="arm-x"))[
        "rollout-server"
    ]["outputs"]
    assert {o["uri"] for o in outputs.values()} == {
        "mem://distill-rollout-arm-x",
        "mem://distill-rollout-cluster-arm-x",
    }


# ─── Step URIs ────────────────────────────────────────────────────────────────


def _all_renders(tmp_path):
    for mode in _MODES:
        yield _render(tmp_path, LMBDA_MODE=mode)
        yield _render(tmp_path, _SMOKE, LMBDA_MODE=mode)
    yield _render(tmp_path, INCLUDE_SFT="true", CORPUS_DIR="/c")
    yield _render(tmp_path, SCREEN_EVALS="olmes-gsm8k,bfcl", FULL_EVAL_RUNGS="8000")


def test_every_step_uri_is_expected(tmp_path):
    raw = _BUILD.read_text(encoding="utf-8")
    uris = set(re.findall(r"step_uri:\s*\"?([^\s\"]+)", raw))
    assert uris == set(_STEP_SCHEMAS)
    assert not any(u.startswith("space://steps/distill/") for u in uris)
    for rendered in _all_renders(tmp_path):
        for target in _targets(rendered):
            assert _step(rendered, target)["step_uri"] in _STEP_SCHEMAS


def test_every_expected_step_exists():
    for uri, schema in _STEP_SCHEMAS.items():
        assert schema.is_file(), uri


# ─── The ladder ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("smoke", [False, True], ids=["full", "smoke"])
def test_one_export_per_rung(tmp_path, smoke):
    layers = (_SMOKE,) if smoke else ()
    params = _layered(_SITE, *layers)
    ladder = _ladder(params)
    targets = _targets(_render(tmp_path, *layers))
    exports = sorted(int(t[7:]) for t in targets if re.fullmatch(r"export-\d+", t))
    assert exports == ladder
    assert ladder[-1] == int(params["GOLD_MAX_STEPS"])
    assert all(n % int(params["GOLD_SAVE_STEPS"]) == 0 for n in ladder)
    saves = -(-int(params["GOLD_MAX_STEPS"]) // int(params["GOLD_SAVE_STEPS"]))
    assert int(params["GOLD_SAVE_TOTAL_LIMIT"]) >= saves >= len(ladder)
    for n in ladder:
        assert targets[f"export-{n}"]["inputs"]["train_output"] == {
            "binding": f"train.checkpoint_{n}"
        }
        assert targets[f"divergence-{n}"]["inputs"]["student"] == {
            "binding": f"export-{n}.hf_model"
        }
    rungs = _config({"granite.build": {"targets": targets}}, "repetition-check")[
        "gen_smoke_config"
    ]["rungs"]
    assert [int(r.split(":", 1)[0]) for r in rungs] == ladder


@pytest.mark.parametrize(
    "params",
    [
        {"CKPT_LADDER": "1000,2000"},  # last rung short of GOLD_MAX_STEPS
        {"CKPT_LADDER": "1000,2500,8000"},  # 2500 is off the save grid
        {"GOLD_MAX_STEPS": "0"},  # epoch-bounded: the last rung is unknowable
        {"GOLD_SAVE_TOTAL_LIMIT": "2"},  # would evict rungs still to be exported
    ],
)
def test_inconsistent_ladder_fails_to_render(tmp_path, params):
    with pytest.raises(jinja2.exceptions.UndefinedError) as err:
        _render_text(tmp_path, **params)
    assert "CKPT_LADDER" in str(err.value) or "GOLD_" in str(err.value)


def test_rotation_counts_every_save_not_only_the_rungs(tmp_path):
    """HF Trainer saves every GOLD_SAVE_STEPS and, after each save, deletes the
    oldest checkpoint-* beyond save_total_limit (protecting only the newest; this
    recipe has no best checkpoint). Saves between rungs count, so a limit that
    covers the rungs alone can still delete one before its export reads it."""
    # 4 rungs, 8 saves (every 1000 to 8000): a limit of 4 covers the rungs only.
    with pytest.raises(jinja2.exceptions.UndefinedError) as err:
        _render_text(
            tmp_path, CKPT_LADDER="2000,4000,6000,8000", GOLD_SAVE_TOTAL_LIMIT="7"
        )
    assert "GOLD_SAVE_TOTAL_LIMIT" in str(err.value)
    # The boundary: exactly one slot per save.
    _render(tmp_path, CKPT_LADDER="2000,4000,6000,8000", GOLD_SAVE_TOTAL_LIMIT="8")
    # 0 turns rotation off (HF deletes nothing at a limit <= 0).
    _render(tmp_path, GOLD_SAVE_TOTAL_LIMIT="0")


def test_rotation_bound_ends_at_stop_at_step(tmp_path):
    # 4500 / 250 = 18 saves, whatever GOLD_MAX_STEPS is.
    params = dict(
        GOLD_MAX_STEPS="8150",
        STOP_AT_STEP="4500",
        GOLD_SAVE_STEPS="250",
        CKPT_LADDER="2250,2500,3000,3500,4000,4500",
    )
    with pytest.raises(jinja2.exceptions.UndefinedError):
        _render_text(tmp_path, GOLD_SAVE_TOTAL_LIMIT="17", **params)
    _render(tmp_path, GOLD_SAVE_TOTAL_LIMIT="18", **params)
    # A resume does not relax it: the start step is unknown at render time.
    with pytest.raises(jinja2.exceptions.UndefinedError):
        _render_text(
            tmp_path,
            GOLD_SAVE_TOTAL_LIMIT="17",
            RESUME_FROM_CHECKPOINT_DIR="/x/run_node2",
            RESUME_EMIT_SEEDED="false",
            **params,
        )


@pytest.mark.parametrize("smoke", [False, True], ids=["full", "smoke"])
def test_default_save_limit_covers_every_save(tmp_path, smoke):
    layers = (_SMOKE,) if smoke else ()
    params = _layered(_SITE, *layers)
    saves = -(-int(params["GOLD_MAX_STEPS"]) // int(params["GOLD_SAVE_STEPS"]))
    assert int(params["GOLD_SAVE_TOTAL_LIMIT"]) >= saves
    _render(tmp_path, *layers)


def test_stop_at_step_ends_the_ladder(tmp_path):
    targets = _targets(_render(tmp_path, STOP_AT_STEP="4000", CKPT_LADDER="2000,4000"))
    assert _gold({"granite.build": {"targets": targets}})["stop_at_step"] == 4000
    with pytest.raises(jinja2.exceptions.UndefinedError):
        _render_text(tmp_path, STOP_AT_STEP="4000")


def test_repetition_check_reports_and_never_gates(tmp_path):
    cfg = _config(_render(tmp_path), "repetition-check")["gen_smoke_config"]
    assert cfg["gate_final_rung"] is False


# ─── Resume ───────────────────────────────────────────────────────────────────


def test_train_resume_contract(tmp_path):
    rendered = _render(tmp_path)
    train = _targets(rendered)["train"]
    cfg = train["steps"][0]["config"]
    assert "{{ run_metadata.build_id }}" in cfg["train_dir"]
    assert cfg["train_dir"].endswith("/train")
    assert cfg["require_progress"] is True
    assert cfg["emit_checkpoint_artifacts"] is True
    assert cfg["resume_from_checkpoint_dir"] == ""
    for n in _ladder(_layered(_SITE)):
        assert train["outputs"][f"checkpoint_{n}"]["uri"] == "env://{{ binding.path }}"
    retries = rendered["granite.build"]["retries"]
    assert retries == {"max_retries": 3, "target_reuse_enabled": True}


def test_resume_dir_passes_through(tmp_path):
    cfg = _config(_render(tmp_path, RESUME_FROM_CHECKPOINT_DIR="/x/run_node2"), "train")
    assert cfg["resume_from_checkpoint_dir"] == "/x/run_node2"


# ─── The smoke layer ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", _MODES)
def test_smoke_layer(tmp_path, mode):
    site = _yaml(_SITE)
    rendered = _render(tmp_path, _SMOKE, LMBDA_MODE=mode)
    targets = _targets(rendered)
    cfg = _config(rendered, "train")
    assert cfg["compute_config"] == {"num_nodes": 1, "num_gpus_per_node": 2}
    assert cfg["launcher_config"]["resources"]["accelerators"].endswith(":2")
    assert sorted(t for t in targets if re.fullmatch(r"export-\d+", t)) == [
        "export-1",
        "export-2",
    ]
    assert _gold(rendered)["max_steps"] == 2
    assert _gold(rendered)["save_steps"] == 1
    (source,) = [v["uri"] for v in targets["sources"]["inputs"].values()]
    assert source == site["SMOKE_DATASET_URI"]
    assert not [t for t in targets if _BENCHMARK.match(t)]
    corpus = _config(rendered, "corpus")["corpus_config"]
    assert (corpus["max_examples"], corpus["eval_fraction"]) == (64, 0.25)
    assert _config(rendered, "divergence-2")["eval_config"]["max_samples"] == 8
    if mode != "off-policy":
        server = _config(rendered, "rollout-server")
        assert server["compute_config"]["num_gpus_per_node"] == 1


def test_full_run_mixes_the_site_splits(tmp_path):
    site = _yaml(_SITE)
    targets = _targets(_render(tmp_path))
    uris = [v["uri"] for v in targets["sources"]["inputs"].values()]
    root = site["DATASET_URI"].rstrip("/")
    assert uris == [f"{root}/{s}" for s in site["DATASET_SPLITS"]]
    cfg = targets["sources"]["steps"][0]["config"]["sources_config"]
    assert len(cfg["sources"]) == len(uris)
    assert all(s.startswith("{{ bindings.source_") for s in cfg["sources"])
    assert targets["corpus"]["inputs"]["source_dataset"] == {
        "binding": "sources.corpus_source"
    }


def test_a_single_file_dataset_is_used_as_is(tmp_path):
    site = _layered(_SITE)
    site["DATASET_SPLITS"] = []
    contents = _BUILD.read_text(encoding="utf-8")
    targets = _targets(
        yaml.safe_load(apply_parameters(contents, [], site, str(tmp_path)))
    )
    uris = [v["uri"] for v in targets["sources"]["inputs"].values()]
    assert uris == [site["DATASET_URI"]]


# ─── Benchmark evals ──────────────────────────────────────────────────────────


def test_benchmark_evals_are_off_by_default(tmp_path):
    assert not [t for t in _targets(_render(tmp_path)) if _BENCHMARK.match(t)]


def test_screening_and_full_battery(tmp_path):
    params = _layered(_SITE)
    ladder = _ladder(params)
    last = ladder[-1]
    screen = ["olmes-gsm8k", "salad-bench", "bfcl"]
    targets = _targets(
        _render(tmp_path, SCREEN_EVALS=",".join(screen), FULL_EVAL_RUNGS=str(last))
    )
    catalog = [e["name"] for e in params["EVAL_CATALOG"]]
    for n in ladder:
        expected = catalog if n == last else ["olmes-gsm8k", "salad-bench"]
        got = sorted(
            t[5 : -len(f"-ck{n}")]
            for t in targets
            if t.startswith("eval-")
            and t.endswith(f"-ck{n}")
            and t != f"eval-bfcl-ck{n}"
        )
        assert got == sorted(expected), n
        assert f"eval-bfcl-ck{n}" in targets
        assert f"export-sage-ck{n}" in targets and f"export-bfcl-ck{n}" in targets
        assert targets[f"eval-olmes-gsm8k-ck{n}"]["inputs"]["model"] == {
            "binding": f"export-{n}.hf_model"
        }
    gates = targets["export-combined"]["inputs"]
    assert len(gates) == 2 * len(ladder)


def test_a_rung_without_evals_gets_no_exporter(tmp_path):
    ladder = _ladder(_layered(_SITE))
    targets = _targets(_render(tmp_path, FULL_EVAL_RUNGS=str(ladder[-1])))
    assert [t for t in targets if t.startswith("export-sage-ck")] == [
        f"export-sage-ck{ladder[-1]}"
    ]


def test_eval_images_come_from_the_site_registry(tmp_path):
    site = _yaml(_SITE)
    rendered = _render(tmp_path, SCREEN_EVALS="olmes-gsm8k,bfcl")
    targets = _targets(rendered)
    images = [
        _config(rendered, t)["sage_eval_config"]["image_id"]
        for t in targets
        if t.startswith("eval-olmes")
    ]
    images += [
        _config(rendered, t)["launcher_config"]["image_id"]
        for t in targets
        if t.startswith(("eval-bfcl", "export-"))
        and "image_id" in _config(rendered, t)["launcher_config"]
    ]
    assert images
    assert all(i.startswith(f"docker:{site['EVAL_IMAGE_REGISTRY']}/") for i in images)


def test_bcb_url_reaches_bigcodebench(tmp_path):
    site = _yaml(_SITE)
    ladder = _ladder(_layered(_SITE))
    rendered = _render(tmp_path, FULL_EVAL_RUNGS=str(ladder[-1]))
    env = _config(rendered, f"eval-olmes-bigcodebench-ck{ladder[-1]}")[
        "sage_eval_config"
    ]["extra_env"]
    assert env["OE_EVAL_BCB_API_URL"] == site["OE_EVAL_BCB_API_URL"]


@pytest.mark.parametrize(
    "params",
    [{"SCREEN_EVALS": "no-such-eval"}, {"FULL_EVAL_RUNGS": "1234"}],
)
def test_unknown_eval_names_fail_to_render(tmp_path, params):
    with pytest.raises(jinja2.exceptions.UndefinedError):
        _render_text(tmp_path, **params)


# ─── Corpus reuse and the SFT arm ─────────────────────────────────────────────


def test_corpus_dir_reuses_a_corpus(tmp_path):
    targets = _targets(_render(tmp_path, CORPUS_DIR="/prev/corpus/"))
    assert "sources" not in targets and "corpus" not in targets
    assert "corpus-validate" in targets
    cfg = targets["corpus-validate"]["steps"][0]["config"]["pin_check_config"]
    assert cfg["corpus_dir"] == "/prev/corpus"
    train = targets["train"]["inputs"]
    assert train["corpus"] == {
        "uri": "env:///prev/corpus/train.jsonl",
        "type": "dataset",
    }
    assert train["corpus_pin"] == {"binding": "corpus-validate.pin_check"}
    div = targets["divergence-baseline"]["steps"][0]["config"]["eval_config"]
    assert div["corpus"] == "/prev/corpus/eval.jsonl"
    for target in targets.values():
        for value in target.get("inputs", {}).values():
            assert value.get("binding", "").split(".")[0] not in ("corpus", "sources")


def test_without_corpus_dir_no_validation(tmp_path):
    targets = _targets(_render(tmp_path))
    assert "corpus-validate" not in targets
    assert targets["train"]["inputs"]["corpus"] == {"binding": "corpus.corpus"}


@pytest.mark.parametrize("value", ["true", "True", "yes", "1"])
def test_include_sft(tmp_path, value):
    targets = _targets(_render(tmp_path, INCLUDE_SFT=value))
    assert "train-sft" in targets
    assert targets["train"]["inputs"]["student"] == {"binding": "train-sft.checkpoint"}


@pytest.mark.parametrize("value", ["false", "False", "no", "0"])
def test_include_sft_off(tmp_path, value):
    targets = _targets(_render(tmp_path, INCLUDE_SFT=value))
    assert "train-sft" not in targets
    assert targets["train"]["inputs"]["student"] == {
        "binding": "align.retagged_student"
    }


# ─── Scheduler neutrality and the site/experiment split ───────────────────────


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def test_launchers_are_scheduler_neutral(tmp_path):
    allowed = {"cluster", "zone", "accelerators", "memory"}
    for rendered in _all_renders(tmp_path):
        for target in _targets(rendered):
            launcher = _config(rendered, target).get("launcher_config", {})
            keys = {k for k, _ in _walk(launcher)}
            assert "sbatch_options" not in keys, target
            assert "partition" not in keys, target
            assert set(launcher) <= {"resources", "image_id"}, target
            assert set(launcher.get("resources", {})) <= allowed, target


def test_every_target_uses_the_site_environment(tmp_path):
    site = _yaml(_SITE)
    for rendered in _all_renders(tmp_path):
        for target in _targets(rendered).values():
            assert target["environment_uri"] == site["ENVIRONMENT_URI"]


def _cli_env():
    return jinja2.Environment(
        variable_start_string="$${",
        variable_end_string="}",
        block_start_string="<%",
        block_end_string="%>",
    )


def test_every_value_is_defined():
    source = _BUILD.read_text(encoding="utf-8")
    undeclared = meta.find_undeclared_variables(_cli_env().parse(source))
    # The deliberate render failures: an undefined name that says what is wrong.
    undeclared = {v for v in undeclared if "_must_" not in v and "_needs_" not in v}
    defined = set(_layered(_SITE))
    assert undeclared <= defined, sorted(undeclared - defined)


def test_every_parameter_is_read():
    source = _BUILD.read_text(encoding="utf-8")
    undeclared = meta.find_undeclared_variables(_cli_env().parse(source))
    for layer in (_PARAMS, _SMOKE):
        unread = set(_yaml(layer)) - undeclared
        assert not unread, (layer.name, sorted(unread))


def test_site_and_experiment_are_separate():
    site = set(_yaml(_SITE))
    for layer in (_PARAMS, _SMOKE):
        assert not set(_yaml(layer)) & site, layer.name


def test_smoke_only_overrides_recipe_knobs():
    assert set(_yaml(_SMOKE)) <= set(_yaml(_PARAMS))


def test_no_site_values_in_recipe_files():
    """No path, host, partition or accelerator model in the recipe's own layers."""
    bad = re.compile(r"(^|[\s\"'])/[a-z]|://|\b[AH]100\b|\bH200\b|\.org\b|\.com\b")
    for layer in (_PARAMS, _SMOKE):
        for key, value in _yaml(layer).items():
            if key == "EVAL_CATALOG":
                continue
            assert not bad.search(str(value)), (layer.name, key, value)


# ─── Config keys exist in each step's schema ──────────────────────────────────


def _schema(uri):
    return _yaml(_STEP_SCHEMAS[uri]).get("config", {})


def test_config_keys_exist_in_step_schemas(tmp_path):
    for rendered in _all_renders(tmp_path):
        for target in _targets(rendered):
            step = _step(rendered, target)
            schema = _schema(step["step_uri"])
            for key, value in step["config"].items():
                if key in _GENERIC_CONFIG:
                    continue
                assert key in schema, (target, key)
                if isinstance(value, dict) and isinstance(schema[key], dict):
                    extra = set(value) - set(schema[key])
                    assert not extra, (target, key, sorted(extra))


def test_gold_config_keys_are_logit_distill_keys(tmp_path):
    schema = _schema("space://steps/train/distill/logit-distill")["gold_config"]
    for mode in _MODES:
        gold = _gold(_render(tmp_path, LMBDA_MODE=mode))
        assert set(gold) <= set(schema), sorted(set(gold) - set(schema))
