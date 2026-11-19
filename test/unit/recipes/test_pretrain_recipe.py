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
