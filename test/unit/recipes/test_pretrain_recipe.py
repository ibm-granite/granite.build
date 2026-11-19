#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""recipes/pretrain/pretrain: pretrain an lm-engine model from scratch.

The recipe is rendered the way ``gb build start`` renders it, through
gbcli.utils.buildutil.apply_parameters, with its parameters layered: the recipe's
parameters.yaml, then a site file, then --param overrides (ibm-granite/granite.build#665).

The site file is NOT in this repository -- it carries the cluster context, the PVC,
the image refs and the dataset and tokenizer paths. parameters.yaml holds the
experiment and nothing else, so the tests supply a placeholder site layer built from
``_SITE_KEYS`` and assert that the two key sets stay disjoint. Task 19 of the plan
replaces the placeholder with ``recipes/pretrain/site.example.yaml``.

What is asserted is agreement between values a person would otherwise keep in step by
hand: the 88 lm-engine knobs against ``render_config.DEFAULT_KNOBS``, the rung outputs
against the rung parameters, and the rendered config keys against the steps' schemas.
"""

import importlib.util
import pathlib
import re

import jinja2
import pytest
import yaml
from jinja2 import meta

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PRETRAIN = _REPO / "recipes" / "pretrain"
_RECIPE = _PRETRAIN / "pretrain"
_BUILD = _RECIPE / "build.yaml"
_PARAMS = _RECIPE / "parameters.yaml"
_STEPS = _REPO / "steps"
_RENDER_CONFIG = _STEPS / "train/lm-engine/pretrain/skypilot/src/render_config.py"

# step_uri -> the file whose `config` is that step's schema. The export and lmeval
# steps are listed although no target uses them yet: tasks 15 and 16 of the plan add
# the export fan-out and the eval fan-out, and test_every_expected_step_exists keeps
# their schemas from moving away under them in the meantime.
_STEP_SCHEMAS = {
    "space://steps/train/lm-engine/pretrain": _STEPS
    / "train/lm-engine/pretrain/skypilot/step-template.yaml",
    "space://steps/train/lm-engine/export": _STEPS
    / "train/lm-engine/export/skypilot/step-template.yaml",
    "space://steps/eval/lmeval": _STEPS / "eval/lmeval/skypilot/step-template.yaml",
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

# Free-form mappings: the schema spells them `{}` on purpose, so the key-by-key
# comparison test_every_recipe_config_key_is_in_step_schema makes for nested
# mappings cannot apply to them. `lme` has its own, stricter test below.
_FREE_FORM_CONFIG = {"lme", "nccl_env", "expected"}

# Everything the site file supplies, with placeholder values. Deliberately NOT in
# parameters.yaml: a cluster identifier, a PVC path, a registry ref or a dataset path
# in this repository is a leak, and a key defined in both layers hides the site's
# value behind the recipe's default.
_SITE_KEYS = {
    "ENVIRONMENT_URI": "space://environments/skypilot-local/example",
    "WORKDIR_ROOT": "/shared/pretrain",
    "LME_IMAGE": "registry.example.org/my-team/lm-engine:0.1.0",
    "DATASET_PATH": "/shared/data/nemotron-full",
    "TOKENIZER_NAME": "/shared/tokenizers/granite-4",
    "COS_BUCKET": "example-bucket",
    "COS_ROOT": "tables/example",
    "COS_CACHE_DIR": "/cos-cache",
    "GPU_ACCELERATOR": "H100",
    "NCCL_ENV": {"NCCL_SOCKET_IFNAME": "eth0"},
}

# The knobs the site file sets, so parameters.yaml carries no path at all.
_SITE_KNOBS = {
    "dataset_path",
    "tokenizer_name",
    "cos_bucket",
    "cos_root",
    "cos_cache_dir",
}


# ─── Rendering ────────────────────────────────────────────────────────────────


def _render_config():
    """render_config.py, imported by path: it is a step's source, not an installed module."""
    spec = importlib.util.spec_from_file_location(
        "_pretrain_render_config", _RENDER_CONFIG
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _site(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(_SITE_KEYS), encoding="utf-8")
    return path


def _layered(tmp_path, *layers):
    """Merge parameter files in order, later layers replacing a key whole.

    Mirrors the layered --param-file of ibm-granite/granite.build#665 (the recipe's
    parameters.yaml first, then each --param-file in order); --param overrides are
    applied after this by apply_parameters.
    """
    merged = {}
    for layer in (_PARAMS, _site(tmp_path), *layers):
        merged.update(get_params_from_file(str(layer)))
    return merged


def _render_text(tmp_path, *layers, **params):
    """Render with the site layer; params arrive as strings, exactly as
    `--param KEY=VALUE` delivers them."""
    cli = [f"{k}={v}" for k, v in params.items()]
    return apply_parameters(
        _BUILD.read_text(encoding="utf-8"),
        cli,
        _layered(tmp_path, *layers),
        str(tmp_path),
    )


def _render(tmp_path, *layers, **params):
    return yaml.safe_load(_render_text(tmp_path, *layers, **params))


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def _step(rendered, target):
    (step,) = _targets(rendered)[target]["steps"]
    return step


def _config(rendered, target):
    return _step(rendered, target)["config"]


def _yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _schema(uri):
    return _yaml(_STEP_SCHEMAS[uri]).get("config", {})


def _norm(value):
    """None and "" both mean "unset" to render_config._is_set; a YAML round trip
    through a recipe parameter turns one into the other."""
    return "" if value is None else value


# ─── The train-only slice renders ─────────────────────────────────────────────


def test_train_only_renders(tmp_path):
    text = _render_text(tmp_path, RUN_NAME="t")
    assert "$${" not in text and "<%" not in text
    rendered = yaml.safe_load(text)
    assert set(_targets(rendered)) == {"data-check", "train"}


def test_data_check_runs_the_pretrain_step_on_cpu(tmp_path):
    rendered = _render(tmp_path)
    step = _step(rendered, "data-check")
    assert step["step_uri"] == "space://steps/train/lm-engine/pretrain"
    config = step["config"]
    assert config["phase"] == "data-check"
    # A CPU pod: no accelerators anywhere in the launcher or the compute floor.
    assert "num_gpus_per_node" not in config["compute_config"]
    assert "accelerators" not in config["launcher_config"]["resources"]


def test_train_is_a_multi_node_gpu_target(tmp_path):
    params = _layered(tmp_path)
    config = _config(_render(tmp_path), "train")
    assert config["phase"] == "train"
    assert config["compute_config"]["num_nodes"] == int(params["NUM_NODES"])
    assert config["compute_config"]["num_gpus_per_node"] == int(params["GPUS_PER_NODE"])
    assert config["launcher_config"]["resources"]["accelerators"] == (
        f"{_SITE_KEYS['GPU_ACCELERATOR']}:{params['GPUS_PER_NODE']}"
    )


def test_train_is_ordered_after_data_check(tmp_path):
    """Ordering only: no artifact flows, but a bad corpus must not cost an allocation."""
    train = _targets(_render(tmp_path))["train"]
    bindings = {
        name: spec["binding"]
        for name, spec in train["inputs"].items()
        if "binding" in spec
    }
    assert any(b.startswith("data-check.") for b in bindings.values()), bindings


# ─── The rungs ────────────────────────────────────────────────────────────────


def test_train_has_no_rung_outputs_when_no_rungs(tmp_path):
    params = _layered(tmp_path)
    assert str(params["EXPORT_RUNGS"]).strip() == ""
    assert str(params["EVAL_RUNGS"]).strip() == ""
    targets = _targets(_render(tmp_path))
    assert set(targets["train"]["outputs"]) == {"checkpoint"}
    config = _config(_render(tmp_path), "train")
    assert config["emit_checkpoint_artifacts"] is False
    assert str(config["ckpt_rungs"]) == ""


def test_rung_outputs_are_the_union_of_export_and_eval_rungs(tmp_path):
    rendered = _render(tmp_path, EXPORT_RUNGS="20000,10000", EVAL_RUNGS="30000,10000")
    targets = _targets(rendered)
    assert set(targets["train"]["outputs"]) == {
        "checkpoint",
        "checkpoint_10000",
        "checkpoint_20000",
        "checkpoint_30000",
    }
    config = _config(rendered, "train")
    assert config["emit_checkpoint_artifacts"] is True
    # Ascending and de-duplicated: the watcher walks the list in order.
    assert str(config["ckpt_rungs"]) == "10000,20000,30000"
    for name, spec in targets["train"]["outputs"].items():
        assert spec["type"] == "model", name


def test_eval_rungs_alone_still_emit_checkpoints(tmp_path):
    config = _config(_render(tmp_path, EVAL_RUNGS="10000"), "train")
    assert config["emit_checkpoint_artifacts"] is True
    assert str(config["ckpt_rungs"]) == "10000"


# ─── Render-time validation (the $${ } trick tasks 15/16 build on) ────────────


@pytest.mark.parametrize(
    "params",
    [
        # A rung the trainer never writes is not a late error but a target that
        # never dispatches.
        {"EXPORT_RUNGS": "10001"},
        {"EVAL_RUNGS": "10000", "SAVE_INTERVAL": "3000"},
        # Past the end of the run.
        {"EXPORT_RUNGS": "130000"},
        {"EXPORT_RUNGS": "0"},
        # A full-eval rung that is not evaluated at all is a typo.
        {"EVAL_RUNGS": "10000", "FULL_EVAL_RUNGS": "20000"},
        {"FULL_EVAL_RUNGS": "10000"},
    ],
)
def test_a_bad_rung_fails_the_render(tmp_path, params):
    with pytest.raises(Exception):
        _render_text(tmp_path, **params)


def test_good_rungs_render(tmp_path):
    _render_text(
        tmp_path,
        EXPORT_RUNGS="10000,120000",
        EVAL_RUNGS="120000",
        FULL_EVAL_RUNGS="120000",
    )


# ─── The 88 lm-engine knobs ───────────────────────────────────────────────────


@pytest.mark.parametrize("target", ["data-check", "train"])
def test_lme_keys_are_exactly_default_knobs(tmp_path, target):
    """The knob table is the contract. This is the test that keeps it honest from
    either side: a knob added to render_config.DEFAULT_KNOBS and not to the recipe
    silently trains on the step's default, and a key the recipe invents is dropped."""
    knobs = _render_config().DEFAULT_KNOBS
    lme = _config(_render(tmp_path), target)["lme"]
    assert set(lme) == set(knobs), {
        "missing": sorted(set(knobs) - set(lme)),
        "extra": sorted(set(lme) - set(knobs)),
    }
    assert len(knobs) == 88


def test_knobs_reach_step(tmp_path):
    params = _layered(tmp_path)
    lme = _config(_render(tmp_path), "train")["lme"]
    assert lme["m_width"] == params["M_WIDTH"]
    assert lme["hidden_size"] == params["HIDDEN_SIZE"]
    assert lme["sequence_length"] == params["SEQ_LEN"]
    assert lme["gradient_accumulation_steps"] == params["GRAD_ACCUM"]
    assert lme["num_training_steps"] == params["NUM_TRAINING_STEPS"]
    assert lme["experiment"] == params["EXP"]
    assert lme["seed"] == params["SEED"]
    assert lme["dataset_path"] == _SITE_KEYS["DATASET_PATH"]
    assert lme["tokenizer_name"] == _SITE_KEYS["TOKENIZER_NAME"]


def test_knob_defaults_are_the_350m_mup_values(tmp_path):
    """Every knob the site file does not own renders at its DEFAULT_KNOBS value, so
    the recipe's defaults are the 350M muP model and not a re-typed copy of it."""
    knobs = _render_config().DEFAULT_KNOBS
    lme = _config(_render(tmp_path), "train")["lme"]
    differ = {
        key: (_norm(knobs[key]), _norm(lme[key]))
        for key in knobs
        if key not in _SITE_KNOBS and _norm(knobs[key]) != _norm(lme[key])
    }
    assert not differ, differ


def test_render_config_validate_passes_on_the_defaults(tmp_path):
    """A bad default is caught here, not on a cluster an hour into a queue."""
    module = _render_config()
    params = _layered(tmp_path)
    lme = _config(_render(tmp_path), "train")["lme"]
    errors = module.validate(
        lme,
        num_nodes=int(params["NUM_NODES"]),
        gpus_per_node=int(params["GPUS_PER_NODE"]),
    )
    assert errors == []


# ─── Step schemas ─────────────────────────────────────────────────────────────


def test_every_expected_step_exists():
    for uri, schema in _STEP_SCHEMAS.items():
        assert schema.is_file(), uri


def test_every_step_uri_is_expected(tmp_path):
    raw = _BUILD.read_text(encoding="utf-8")
    uris = set(re.findall(r"step_uri:\s*\"?([^\s\"]+)", raw))
    # A subset, not equality: tasks 15 and 16 add the export and eval targets.
    assert uris
    assert uris <= set(_STEP_SCHEMAS), sorted(uris - set(_STEP_SCHEMAS))
    for target in _targets(_render(tmp_path)):
        assert _step(_render(tmp_path), target)["step_uri"] in _STEP_SCHEMAS


def test_every_recipe_config_key_is_in_step_schema(tmp_path):
    for params in ({}, {"EXPORT_RUNGS": "10000", "EVAL_RUNGS": "120000"}):
        rendered = _render(tmp_path, **params)
        for target in _targets(rendered):
            step = _step(rendered, target)
            schema = _schema(step["step_uri"])
            for key, value in step["config"].items():
                if key in _GENERIC_CONFIG:
                    continue
                assert key in schema, (target, key)
                if (
                    key not in _FREE_FORM_CONFIG
                    and isinstance(value, dict)
                    and isinstance(schema[key], dict)
                ):
                    extra = set(value) - set(schema[key])
                    assert not extra, (target, key, sorted(extra))


# ─── parameters.yaml and build.yaml agree ─────────────────────────────────────


def _template_names():
    source = _BUILD.read_text(encoding="utf-8")
    env = jinja2.Environment(
        variable_start_string="$${",
        variable_end_string="}",
        block_start_string="<%",
        block_end_string="%>",
    )
    return meta.find_undeclared_variables(env.parse(source))


def test_every_template_name_resolves(tmp_path):
    """No accidentally undefined name: apply_parameters renders with StrictUndefined,
    so one would fail the build at submission -- but only on the branch that uses it.

    The exception is deliberate: a render-time check reports itself by referencing a
    name that cannot exist, which is how the renderer fails with a message. Those are
    spelled `..._must_...` and are the only undefined names allowed."""
    known = set(_layered(tmp_path)) | {"run_metadata", "binding", "unique_hash"}
    undefined = _template_names() - known
    assert all("_must_" in name for name in undefined), sorted(undefined)
    # ... and every one of them is reachable, i.e. the trick works at all.
    assert undefined


def test_every_parameter_is_referenced_by_build_yaml():
    """An unused parameter is a lie to the user."""
    raw = _BUILD.read_text(encoding="utf-8")
    unused = [
        key
        for key in _yaml(_PARAMS)
        if not re.search(rf"(?<![A-Z_]){re.escape(key)}(?![A-Z_])", raw)
    ]
    assert not unused, unused


def test_run_dir_keeps_builds_apart():
    """A cross-build URI collision drops artifacts silently: the producing target
    still reports SUCCESS and nothing downstream dispatches."""
    (run_dir,) = re.findall(
        r"set RUN_DIR = (.+)$", _BUILD.read_text(encoding="utf-8"), re.M
    )
    assert "WORKDIR_ROOT" in run_dir
    assert "EXP" in run_dir
    assert "RUN_NAME" in run_dir


def test_every_target_uses_the_environment_uri_parameter():
    uris = re.findall(
        r"^\s*environment_uri:\s*(.+?)\s*$", _BUILD.read_text(encoding="utf-8"), re.M
    )
    assert uris
    assert set(uris) == {'"$${ENVIRONMENT_URI}"'}


# ─── No site value in the recipe ──────────────────────────────────────────────


def test_parameters_yaml_defines_no_site_key():
    """A key in both layers hides the site's value behind the recipe's default.
    Task 19's site.example.yaml inherits this check."""
    assert not set(_yaml(_PARAMS)) & set(_SITE_KEYS)


def test_parameters_yaml_leaks_no_site_value():
    """Absolute paths, URIs, registries, hostnames and accelerator models all name a
    site. parameters.yaml is the experiment; the site file is the site."""
    bad = re.compile(
        r"(^|[\s\"'])/[a-z]"
        r"|://"
        r"|\bvela\b|\bdipc\b|dmf-|multi-nic-network|roce|cil15|us\.icr\.io"
        r"|gb-read-write|mlx5_"
        r"|\b[AH]100\b|\bH200\b|\.org\b|\.com\b",
        re.I,
    )
    for key, value in _yaml(_PARAMS).items():
        assert not bad.search(str(value)), (key, value)


# ─── The export fan-out ───────────────────────────────────────────────────────

# Standard parametrization: the four muP knobs null together, and param_groups with
# them (render_config.validate refuses either half alone).
_SP = {
    "M_WIDTH": "",
    "M_EMB": "",
    "M_RESIDUAL": "",
    "ATTENTION_MULTIPLIER": "",
    "PARAM_GROUPS": "configs/param-groups/normal.yml",
}

# A mixture of experts, keyed off MLP_TYPE -- the only knob render_config._mlp_block
# consults to build a MoE block.
_MOE = {
    "MLP_TYPE": "MoE",
    "MOE_NUM_EXPERTS": "32",
    "MOE_NUM_EXPERTS_PER_TOK": "4",
    "MOE_SHARED_INTERMEDIATE_SIZE": "512",
}

_MUP_EXPECTED = {"m_emb", "m_residual", "m_width", "attention_multiplier"}

_PLAIN_EXPECTED = {
    "model_type",
    "rope_theta",
    "vocab_size",
    "tie_word_embeddings",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "sequence_length",
}


def _exports(rendered):
    return {
        name: target
        for name, target in _targets(rendered).items()
        if name.startswith("export-")
    }


def test_no_export_target_without_rungs(tmp_path):
    """The train-only slice has no export at all, not an export of nothing."""
    assert _exports(_render(tmp_path)) == {}


def test_one_export_target_per_rung_and_no_more(tmp_path):
    rendered = _render(tmp_path, EXPORT_RUNGS="20000,10000", EVAL_RUNGS="30000,10000")
    assert set(_exports(rendered)) == {
        "export-10000",
        "export-20000",
        "export-30000",
    }


def test_each_export_exports_its_own_rung(tmp_path):
    """The pairing, not just the count: an off-by-one here exports the wrong
    checkpoint and every number downstream is attributed to the wrong step."""
    rendered = _render(tmp_path, EXPORT_RUNGS="10000,20000,120000")
    exports = _exports(rendered)
    assert len(exports) == 3
    for name, target in exports.items():
        rung = name.split("-", 1)[1]
        bindings = [spec["binding"] for spec in target["inputs"].values()]
        assert bindings == [f"train.checkpoint_{rung}"], name
        config = _config(rendered, name)
        assert str(config["iteration"]) == rung, name
        # train_dir is the run's ONE training directory; the rung is `iteration`.
        assert config["train_dir"] == _config(rendered, "train")["train_dir"], name
        assert config["output_dir"].endswith(f"/export/ckpt_{rung}/hf"), name
        hf_model = target["outputs"]["hf_model"]
        assert hf_model["type"] == "model", name
        assert hf_model["uri"] == f"env://{config['output_dir']}", name


def test_export_output_uris_are_unique(tmp_path):
    """A URI two targets share is registered once: the second target still reports
    SUCCESS with an empty output list and its consumers wait forever."""
    rendered = _render(tmp_path, EXPORT_RUNGS="10000,20000", EVAL_RUNGS="30000")
    uris = [t["outputs"]["hf_model"]["uri"] for t in _exports(rendered).values()]
    dirs = [_config(rendered, name)["output_dir"] for name in _exports(rendered)]
    assert len(set(uris)) == len(uris) == 3, uris
    assert len(set(dirs)) == len(dirs) == 3, dirs
    assert _targets(rendered)["train"]["outputs"]["checkpoint"]["uri"] not in uris


def test_export_runs_the_export_step_on_one_cpu_pod(tmp_path):
    rendered = _render(tmp_path, EXPORT_RUNGS="10000")
    step = _step(rendered, "export-10000")
    assert step["step_uri"] == "space://steps/train/lm-engine/export"
    config = step["config"]
    assert config["image"] == _SITE_KEYS["LME_IMAGE"]
    assert config["tokenizer_path"] == _SITE_KEYS["TOKENIZER_NAME"]
    assert config["compute_config"]["num_nodes"] == 1
    assert "num_gpus_per_node" not in config["compute_config"]
    assert "accelerators" not in config["launcher_config"]["resources"]


def test_export_keeps_the_steps_parity_tolerance(tmp_path):
    """The measured max |dlogits| is ~7e-07; the step's default 0.05 is deliberate
    headroom, and a tighter invented value fails faithful exports."""
    config = _config(_render(tmp_path, EXPORT_RUNGS="10000"), "export-10000")
    schema = _schema("space://steps/train/lm-engine/export")
    assert float(config["parity_tolerance"]) == float(schema["parity_tolerance"])


# ─── `expected`: what the exported config.json must still say ─────────────────


def test_expected_keys_are_exactly_intended_for_mup(tmp_path):
    """`expected` is checked key by key and ONLY for the keys it carries, so an
    empty or thinned block silently disables the config check that caught a
    dropped rope_theta."""
    config = _config(_render(tmp_path, EXPORT_RUNGS="10000"), "export-10000")
    assert set(config["expected"]) == _PLAIN_EXPECTED | _MUP_EXPECTED


def test_expected_omits_the_mup_fields_for_llama(tmp_path):
    """_export_llama_config writes no multiplier at all (it asserts they are None),
    so pinning them would fail every standard-parametrization export."""
    config = _config(_render(tmp_path, EXPORT_RUNGS="10000", **_SP), "export-10000")
    assert set(config["expected"]) == _PLAIN_EXPECTED


def test_expected_carries_the_mup_fields_for_granitemoehybrid(tmp_path):
    config = _config(_render(tmp_path, EXPORT_RUNGS="10000", **_MOE), "export-10000")
    assert set(config["expected"]) == _PLAIN_EXPECTED | _MUP_EXPECTED


@pytest.mark.parametrize(
    "params", [dict(_MOE, **_SP), {"EXPORT_MODEL_TYPE": "granite"}]
)
def test_expected_pins_no_multiplier_for_a_standard_parametrization_run(
    tmp_path, params
):
    """A Granite config of an SP run says `1` BECAUSE nothing was trained -- the
    exporters substitute 1 for a null multiplier -- so pinning the null would fail
    a faithful export. What is pinned is what training set, never a placeholder."""
    config = _config(
        _render(tmp_path, EXPORT_RUNGS="10000", **dict(_SP, **params)), "export-10000"
    )
    assert set(config["expected"]) == _PLAIN_EXPECTED
    assert not set(config["expected"]) & _MUP_EXPECTED


def test_expected_values_are_the_knobs_that_were_trained(tmp_path):
    """Keyed by the pretrain step's knob names, which is what export_checks reads:
    `sequence_length` is the one rename, onto the config's
    max_position_embeddings."""
    rendered = _render(tmp_path, EXPORT_RUNGS="10000")
    lme = _config(rendered, "train")["lme"]
    expected = _config(rendered, "export-10000")["expected"]
    for knob in sorted(set(expected) - {"model_type"}):
        assert expected[knob] == lme[knob], knob
    assert expected["sequence_length"] == _layered(tmp_path)["SEQ_LEN"]
    assert expected["rope_theta"] == _layered(tmp_path)["ROPE_THETA"]
    assert expected["tie_word_embeddings"] is True


def test_expected_is_keyed_the_way_export_checks_reads_it(tmp_path):
    """The authority is the step's own module, not this recipe: `expected` is keyed
    by TRAINING knob names and export_checks owns the rename, so a recipe that keyed
    it by the Hugging Face field name would pin nothing at all."""
    source = (
        _STEPS / "train/lm-engine/export/skypilot/src/export_checks.py"
    ).read_text(encoding="utf-8")
    assert '"sequence_length": "max_position_embeddings"' in source
    expected = _config(_render(tmp_path, EXPORT_RUNGS="10000"), "export-10000")[
        "expected"
    ]
    assert "max_position_embeddings" not in expected
    knobs = set(_render_config().DEFAULT_KNOBS)
    assert set(expected) - {"model_type"} <= knobs, sorted(set(expected) - knobs)


# ─── EXPORT_MODEL_TYPE ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "params,model_type",
    [
        # Dense muP: a Granite config, the only dense one carrying the multipliers.
        ({}, "granite"),
        # Standard parametrization: llama, because a Granite config with
        # attention_multiplier null is refused upstream.
        (_SP, "llama"),
        # MoE, and the hybrid, both of which are granitemoehybrid.
        (_MOE, "granitemoehybrid"),
        ({"MAMBA_HYBRID": "true"}, "granitemoehybrid"),
        (dict(_MOE, **_SP), "granitemoehybrid"),
        # An explicit value wins over the resolution.
        ({"EXPORT_MODEL_TYPE": "llama"}, "llama"),
        ({"EXPORT_MODEL_TYPE": "granitemoehybrid"}, "granitemoehybrid"),
    ],
)
def test_export_model_type_auto_resolves_from_the_knobs(tmp_path, params, model_type):
    rendered = _render(tmp_path, EXPORT_RUNGS="10000", **params)
    config = _config(rendered, "export-10000")
    assert config["model_type"] == model_type
    # ... and it is pinned in `expected`, which is what catches a muP model that
    # came out as `llama` and so skipped every multiplier check.
    assert config["expected"]["model_type"] == model_type


def test_an_unknown_export_model_type_fails_the_render(tmp_path):
    with pytest.raises(
        Exception,
        match="EXPORT_MODEL_TYPE_must_be_auto_granite_granitemoehybrid_or_llama",
    ):
        _render_text(tmp_path, EXPORT_RUNGS="10000", EXPORT_MODEL_TYPE="gpt_bigcode")


def test_an_unknown_export_model_type_fails_even_train_only(tmp_path):
    """The check is in the prelude, so a typo is caught before a rung is added."""
    with pytest.raises(
        Exception,
        match="EXPORT_MODEL_TYPE_must_be_auto_granite_granitemoehybrid_or_llama",
    ):
        _render_text(tmp_path, EXPORT_MODEL_TYPE="granite4")


def test_sliding_window_with_rungs_fails_the_render(tmp_path):
    """There is no faithful Hugging Face target for per-layer sliding window, so a
    run that wants exports may not use it."""
    with pytest.raises(
        Exception,
        match="SLIDING_WINDOW_must_not_be_set_when_a_run_has_export_or_eval_rungs",
    ):
        _render_text(tmp_path, EXPORT_RUNGS="10000", SLIDING_WINDOW="4096")


def test_sliding_window_with_eval_rungs_fails_the_render(tmp_path):
    with pytest.raises(
        Exception,
        match="SLIDING_WINDOW_must_not_be_set_when_a_run_has_export_or_eval_rungs",
    ):
        _render_text(tmp_path, EVAL_RUNGS="10000", SLIDING_WINDOW="4096")


def test_sliding_window_is_fine_for_a_train_only_run(tmp_path):
    rendered = _render(tmp_path, SLIDING_WINDOW="4096")
    assert _exports(rendered) == {}
    assert _config(rendered, "train")["lme"]["sliding_window"] == "4096"
