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

"""recipes/distill/distill-eval: benchmark evals of existing exports, no training.

Rendered as `gb build start` renders it, with the parameters layered: the recipe's
parameters.yaml, then a site file (site.example.yaml), then --param overrides.

The eval targets are shared with recipes/distill/distill: the CLI's renderer has no
template loader (an include fails with "no loader for this environment"), so the
two recipes carry the same text between marker comments. The tests hold both the
text and the rendered targets equal.
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
_SITE = _DISTILL / "site.example.yaml"
_RECIPE = _DISTILL / "distill-eval"
_BUILD = _RECIPE / "build.yaml"
_PARAMS = _RECIPE / "parameters.yaml"
_TRAIN_BUILD = _DISTILL / "distill" / "build.yaml"
_TRAIN_PARAMS = _DISTILL / "distill" / "parameters.yaml"
_SKYPILOT_STEPS = (
    _REPO / "configurations" / "assets" / "environments" / "skypilot" / "steps"
)

_STEP_SCHEMAS = {
    f"space://steps/{name}": _SKYPILOT_STEPS / name / "step.yaml"
    for name in (
        "sage-eval",
        "bfcl-eval",
        "sage-export",
        "bfcl-export",
        "combined-export",
    )
}
_GENERIC_CONFIG = {
    "poll_interval_seconds",
    "log_retrieval_interval_seconds",
    "log_retrieval_mode",
    "compute_config",
    "launcher_config",
}
_BENCHMARK = re.compile(r"^(eval-.+-ck\d+|export-(sage|bfcl)-ck\d+|export-combined)$")
_SHARED = re.compile(
    r"^[ \t]*# >>> (shared eval [a-z]+) \(keep identical to [^)]+\)\n(.*?)^[ \t]*# <<< \1\n",
    re.M | re.S,
)

_RUNGS = (1000, 2000, 3000)
_EXPORTS = ",".join(f"{n}=env:///shared/run/b/export-{n}" for n in _RUNGS)
_TWO = "1000=env:///shared/run/b/export-1000,3000=env:///shared/run/b/export-3000"


# ─── Rendering ────────────────────────────────────────────────────────────────


def _yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _layered(params_file=_PARAMS):
    """The recipe's parameters.yaml, then the site file (ibm-granite/granite.build#663's
    layering; --param overrides are applied after this by apply_parameters)."""
    merged = {}
    for layer in (params_file, _SITE):
        merged.update(get_params_from_file(str(layer)))
    return merged


def _render_text(tmp_path, build=_BUILD, params_file=_PARAMS, **params):
    contents = build.read_text(encoding="utf-8")
    cli = [f"{k}={v}" for k, v in params.items()]
    return apply_parameters(contents, cli, _layered(params_file), str(tmp_path))


def _render(tmp_path, **params):
    params.setdefault("EXPORTS", _EXPORTS)
    return yaml.safe_load(_render_text(tmp_path, **params))


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def _step(target):
    (step,) = target["steps"]
    return step


def _all_renders(tmp_path):
    yield _render(tmp_path, SCREEN_EVALS="olmes-gsm8k,bfcl")
    yield _render(tmp_path, SCREEN_EVALS="mgsm", FULL_EVAL_RUNGS="3000", RERUN_TAG="r1")
    yield _render(
        tmp_path, EXPORTS=_TWO, EVALS="1000:attaq,bfcl;3000:olmes-bigcodebench"
    )


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


# ─── It renders, and holds no training ────────────────────────────────────────


def test_renders_and_loads(tmp_path):
    text = _render_text(tmp_path, EXPORTS=_EXPORTS, SCREEN_EVALS="olmes-gsm8k,bfcl")
    assert "$${" not in text and "<%" not in text
    rendered = yaml.safe_load(text)
    assert rendered["granite.build"]["name"] == "distill-eval"
    assert _targets(rendered)


def test_retries_reuse_what_succeeded(tmp_path):
    retries = _render(tmp_path, SCREEN_EVALS="bfcl")["granite.build"]["retries"]
    assert retries["target_reuse_enabled"] is True
    # A preempted eval is retried; the default is above distill's.
    assert retries["max_retries"] == int(_yaml(_PARAMS)["MAX_RETRIES"]) >= 3


def test_no_training_targets(tmp_path):
    for rendered in _all_renders(tmp_path):
        for name, target in _targets(rendered).items():
            assert _BENCHMARK.match(name), name
            assert _step(target)["step_uri"] in _STEP_SCHEMAS, name
        names = set(_targets(rendered))
        for absent in ("train", "align", "corpus", "sources", "rollout-server"):
            assert absent not in names
        assert not [n for n in names if re.fullmatch(r"export-\d+|divergence-.*", n)]


def test_one_eval_set_per_rung(tmp_path):
    params = _layered()
    catalog = [e["name"] for e in params["EVAL_CATALOG"]]
    full = ",".join(str(n) for n in _RUNGS)
    targets = _targets(_render(tmp_path, FULL_EVAL_RUNGS=full))
    for n in _RUNGS:
        sage = sorted(
            t[5 : -len(f"-ck{n}")]
            for t in targets
            if t.startswith("eval-")
            and t.endswith(f"-ck{n}")
            and t != f"eval-bfcl-ck{n}"
        )
        assert sage == sorted(catalog), n
        assert f"eval-bfcl-ck{n}" in targets
        assert f"export-sage-ck{n}" in targets and f"export-bfcl-ck{n}" in targets
        for t in (f"eval-olmes-gsm8k-ck{n}", f"eval-bfcl-ck{n}"):
            assert targets[t]["inputs"]["model"] == {
                "uri": f"env:///shared/run/b/export-{n}",
                "type": "model",
            }
    assert len(targets["export-combined"]["inputs"]) == 2 * len(_RUNGS)


def test_screening_applies_to_every_rung(tmp_path):
    targets = _targets(
        _render(tmp_path, SCREEN_EVALS="salad-bench", FULL_EVAL_RUNGS="2000")
    )
    assert "eval-salad-bench-ck1000" in targets and "eval-salad-bench-ck3000" in targets
    assert "eval-bfcl-ck2000" in targets and "eval-bfcl-ck1000" not in targets
    assert "eval-attaq-ck2000" in targets and "eval-attaq-ck1000" not in targets


def test_evals_pick_per_rung(tmp_path):
    targets = _targets(
        _render(tmp_path, EXPORTS=_TWO, EVALS="1000:attaq,bfcl;3000:mgsm")
    )
    assert sorted(t for t in targets if t.startswith("eval-")) == [
        "eval-attaq-ck1000",
        "eval-bfcl-ck1000",
        "eval-mgsm-ck3000",
    ]
    assert sorted(targets["export-combined"]["inputs"]) == [
        "gate_bfcl_1000",
        "gate_sage_1000",
        "gate_sage_3000",
    ]


def test_hf_and_unordered_exports(tmp_path):
    targets = _targets(
        _render(
            tmp_path,
            EXPORTS="20=hf://org/model-b, 10=env:///m/a",
            SCREEN_EVALS="attaq",
        )
    )
    assert targets["eval-attaq-ck20"]["inputs"]["model"]["uri"] == "hf://org/model-b"
    assert targets["eval-attaq-ck10"]["inputs"]["model"]["uri"] == "env:///m/a"


@pytest.mark.parametrize(
    "params, needle",
    [
        ({"EXPORTS": ""}, "EXPORTS"),
        ({"EXPORTS": "1000"}, "EXPORTS"),
        ({"EXPORTS": "ck1=env:///a"}, "EXPORTS"),
        ({"EXPORTS": "0=env:///a"}, "EXPORTS"),
        ({"EXPORTS": "1000=/no/scheme"}, "EXPORTS"),
        ({"EXPORTS": "1000=env:///a,1000=env:///b"}, "EXPORTS"),
        ({"SCREEN_EVALS": "", "FULL_EVAL_RUNGS": ""}, "SCREEN_EVALS"),
        ({"SCREEN_EVALS": "no-such-eval"}, "SCREEN_EVALS"),
        ({"FULL_EVAL_RUNGS": "1234"}, "FULL_EVAL_RUNGS"),
        ({"EVALS": "1000:no-such-eval"}, "EVALS"),
        ({"EVALS": "4000:attaq"}, "EVALS"),
        ({"EVALS": "1000"}, "EVALS"),
        (
            {"EVALS": "1000:attaq;2000:attaq;3000:attaq", "SCREEN_EVALS": "mgsm"},
            "EVALS",
        ),
        (
            {"EVALS": "1000:attaq;2000:attaq;3000:attaq", "EVAL_CATEGORIES": "code"},
            "EVALS",
        ),
        # EXPORTS names 2000, EVALS gives it nothing.
        ({"EVALS": "1000:attaq;3000:attaq"}, "EXPORTS"),
        # A rung given but never evaluated is a wasted input, and likely a typo.
        ({"FULL_EVAL_RUNGS": "1000"}, "EXPORTS"),
        ({"SCREEN_EVALS": "attaq", "RERUN_TAG": "a/b"}, "RERUN_TAG"),
        ({"SCREEN_EVALS": "attaq", "RERUN_TAG": "a b"}, "RERUN_TAG"),
    ],
)
def test_bad_inputs_fail_to_render(tmp_path, params, needle):
    params.setdefault("EXPORTS", _EXPORTS)
    with pytest.raises(jinja2.exceptions.UndefinedError) as err:
        _render_text(tmp_path, **params)
    assert needle in str(err.value)


# ─── RERUN_TAG keeps every output fresh ───────────────────────────────────────


def _outputs(rendered):
    """Every output URI, and every config value that names where results land."""
    uris, places = [], []
    for target in _targets(rendered).values():
        uris += [o["uri"] for o in target.get("outputs", {}).values()]
        for key, value in _walk(_step(target)["config"]):
            if key in ("experiment", "output_csv"):
                places.append(value)
    return uris, places


def test_every_output_carries_the_rerun_tag(tmp_path):
    params = dict(SCREEN_EVALS="olmes-gsm8k,bfcl", FULL_EVAL_RUNGS="3000")
    uris, places = _outputs(_render(tmp_path, RERUN_TAG="rerun-20261110", **params))
    assert uris and places
    for value in uris + places:
        assert "rerun-20261110" in value, value
    # The step appends experiment to output_dir, so the base root stays the site's.
    site = _yaml(_SITE)
    for target in _targets(_render(tmp_path, RERUN_TAG="t1", **params)).values():
        cfg = _step(target)["config"]
        for sub in ("sage_eval_config", "bfcl_config"):
            if sub in cfg:
                assert cfg[sub]["output_dir"] in (
                    site["SAGE_OUTPUT_DIR"],
                    site["BFCL_OUTPUT_DIR"],
                )
                assert "t1" in cfg[sub]["experiment"]


def test_two_tags_give_disjoint_outputs(tmp_path):
    params = dict(SCREEN_EVALS="olmes-gsm8k,bfcl", FULL_EVAL_RUNGS="3000")
    a, _ = _outputs(_render(tmp_path, RERUN_TAG="rerun-a", **params))
    b, _ = _outputs(_render(tmp_path, RERUN_TAG="rerun-b", **params))
    untagged, _ = _outputs(_render(tmp_path, **params))
    assert len(a) == len(set(a)) == len(b)
    assert not set(a) & set(b)
    assert not set(a) & set(untagged)


def test_without_a_tag_the_experiment_is_run_names(tmp_path):
    targets = _targets(_render(tmp_path, RUN_NAME="r1", SCREEN_EVALS="attaq"))
    assert (
        _step(targets["eval-attaq-ck1000"])["config"]["sage_eval_config"]["experiment"]
        == "r1-ckpt_1000"
    )
    targets = _targets(
        _render(
            tmp_path, RUN_NAME="r1", EXPERIMENT="e", RERUN_TAG="t", SCREEN_EVALS="attaq"
        )
    )
    assert (
        _step(targets["eval-attaq-ck1000"])["config"]["sage_eval_config"]["experiment"]
        == "e-t-ckpt_1000"
    )


# ─── The same targets as distill ──────────────────────────────────────────────


def _shared_blocks(path):
    return {
        m.group(1): m.group(2)
        for m in _SHARED.finditer(path.read_text(encoding="utf-8"))
    }


def test_shared_blocks_are_textually_identical():
    ours, theirs = _shared_blocks(_BUILD), _shared_blocks(_TRAIN_BUILD)
    assert set(ours) == {"shared eval protocol", "shared eval targets"}
    assert set(theirs) == set(ours)
    for name, text in ours.items():
        assert text == theirs[name], name


def test_marker_comments_name_the_other_recipe():
    assert "keep identical to recipes/distill/distill/build.yaml" in _BUILD.read_text()
    assert "keep identical to recipes/distill/distill-eval/build.yaml" in (
        _TRAIN_BUILD.read_text()
    )


@pytest.mark.parametrize(
    "protocol",
    [
        {"SCREEN_EVALS": "olmes-gsm8k,salad-bench,bfcl", "FULL_EVAL_RUNGS": "3000"},
        {"FULL_EVAL_RUNGS": "1000,2000,3000"},
        {"SCREEN_EVALS": "bfcl,olmes-bigcodebench", "EVAL_CATEGORIES": "code,bfcl"},
    ],
)
def test_rendered_eval_targets_equal_distills(tmp_path, protocol):
    common = dict(RUN_NAME="run", EXPERIMENT="exp", **protocol)
    ladder = ",".join(str(n) for n in _RUNGS)
    train = yaml.safe_load(
        _render_text(
            tmp_path,
            build=_TRAIN_BUILD,
            params_file=_TRAIN_PARAMS,
            CKPT_LADDER=ladder,
            GOLD_MAX_STEPS=str(_RUNGS[-1]),
            **common,
        )
    )
    ours = _targets(_render(tmp_path, **common))
    theirs = {k: v for k, v in _targets(train).items() if _BENCHMARK.match(k)}
    assert set(ours) == set(theirs)
    for name, target in ours.items():
        other = theirs[name]
        if "model" in target["inputs"]:
            n = int(name.rsplit("-ck", 1)[1])
            assert other["inputs"]["model"] == {"binding": f"export-{n}.hf_model"}
            assert target["inputs"]["model"]["uri"].endswith(f"/export-{n}")
            target = dict(target, inputs={})
            other = dict(other, inputs={})
        assert target == other, name


def test_eval_knobs_agree_with_distill():
    """The eval settings both parameters.yaml carry are the same values, so a rung
    evaluated in either recipe is configured identically by default."""
    ours, theirs = _yaml(_PARAMS), _yaml(_TRAIN_PARAMS)
    deliberate = {"RUN_NAME", "MAX_RETRIES"}
    shared = (set(ours) & set(theirs)) - deliberate
    assert "EVAL_CATALOG" in shared and "SAGE_BATCH_SIZE" in shared
    for key in sorted(shared):
        assert ours[key] == theirs[key], key


def test_every_shared_block_knob_is_ours():
    """Every recipe knob the shared blocks read is in this recipe's parameters.yaml."""
    blocks = "".join(_shared_blocks(_BUILD).values())
    used = meta.find_undeclared_variables(_cli_env().parse(blocks))
    site = set(_yaml(_SITE))
    knobs = {v for v in used if v.isupper() and "_must_" not in v} - site
    # Set by each recipe's own preamble.
    knobs -= {"LADDER", "EXP", "RUNG_PICK"}
    assert knobs <= set(_yaml(_PARAMS)), sorted(knobs - set(_yaml(_PARAMS)))


# ─── Scheduler neutrality and the site/experiment split ───────────────────────


def test_launchers_are_scheduler_neutral(tmp_path):
    allowed = {"cluster", "zone", "accelerators", "memory"}
    for rendered in _all_renders(tmp_path):
        for name, target in _targets(rendered).items():
            launcher = _step(target)["config"].get("launcher_config", {})
            keys = {k for k, _ in _walk(launcher)}
            assert "sbatch_options" not in keys, name
            assert "partition" not in keys, name
            assert set(launcher) <= {"resources", "image_id"}, name
            assert set(launcher.get("resources", {})) <= allowed, name


def test_evals_run_on_the_site_eval_partition(tmp_path):
    site = _yaml(_SITE)
    for rendered in _all_renders(tmp_path):
        for name, target in _targets(rendered).items():
            zone = _step(target)["config"]["launcher_config"]["resources"]["zone"]
            want = "EVAL_PARTITION" if name.startswith("eval-") else "EXPORT_PARTITION"
            assert zone == site[want], name
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
    undeclared = {v for v in undeclared if "_must_" not in v and "_needs_" not in v}
    defined = set(_yaml(_PARAMS)) | set(_yaml(_SITE))
    assert undeclared <= defined, sorted(undeclared - defined)


def test_every_parameter_is_read():
    source = _BUILD.read_text(encoding="utf-8")
    undeclared = meta.find_undeclared_variables(_cli_env().parse(source))
    unread = set(_yaml(_PARAMS)) - undeclared
    assert not unread, sorted(unread)


def test_site_and_experiment_are_separate():
    assert not set(_yaml(_PARAMS)) & set(_yaml(_SITE))


def test_no_site_values_in_parameters():
    bad = re.compile(r"(^|[\s\"'])/[a-z]|://|\b[AH]100\b|\bH200\b|\.org\b|\.com\b")
    for key, value in _yaml(_PARAMS).items():
        if key == "EVAL_CATALOG":
            continue
        assert not bad.search(str(value)), (key, value)


def test_config_keys_exist_in_step_schemas(tmp_path):
    for rendered in _all_renders(tmp_path):
        for name, target in _targets(rendered).items():
            step = _step(target)
            schema = _yaml(_STEP_SCHEMAS[step["step_uri"]]).get("config", {})
            for key, value in step["config"].items():
                if key in _GENERIC_CONFIG:
                    continue
                assert key in schema, (name, key)
                if isinstance(value, dict) and isinstance(schema[key], dict):
                    extra = set(value) - set(schema[key])
                    assert not extra, (name, key, sorted(extra))
