#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""recipes/pretrain/pretrain-eval -- the eval half, pointed at existing exports.

The eval, roll-up and catalog-check text is the pretraining recipe's own, copied
between marker comments: `gb build start` uploads build.yaml alone, so a recipe
cannot yet ship a shared include (ibm-granite/granite.build#628). The first tests
here hold the copies byte-identical; the rest cover what only this recipe has --
the adopt targets that turn an existing export directory into the binding the
shared block expects.
"""

import difflib
import pathlib

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PRETRAIN = _REPO / "recipes" / "pretrain"
_TRAIN_BUILD = _PRETRAIN / "pretrain" / "build.yaml"
_BUILD = _PRETRAIN / "pretrain-eval" / "build.yaml"
_PARAMS = _PRETRAIN / "pretrain-eval" / "parameters.yaml"
_SITE = _PRETRAIN / "site.example.yaml"

_SHARED = [
    ("# ─── catalog (shared with pretrain-eval) ───", "# ─── end catalog ───"),
    ("# ─── evals (shared with pretrain-eval) ───", "# ─── end evals ───"),
]


def _block(path, start, end):
    text = path.read_text(encoding="utf-8")
    assert text.count(start) == 1 and text.count(end) == 1, path
    i = text.index(start)
    return text[i : text.index(end, i) + len(end)]


# ─── The copies stay identical ────────────────────────────────────────────────


@pytest.mark.parametrize("start,end", _SHARED, ids=["catalog", "evals"])
def test_the_shared_blocks_are_byte_identical(start, end):
    """One source, two files: a change made to one copy only is this failure. The
    eval block decides what is scored and where the results land, so two versions
    of it would mean a re-score that silently differs from the run it re-scores."""
    left = _block(_TRAIN_BUILD, start, end).splitlines()
    right = _block(_BUILD, start, end).splitlines()
    diff = "\n".join(
        difflib.unified_diff(left, right, "pretrain", "pretrain-eval", lineterm="")
    )
    assert not diff, f"the two copies differ; change both together:\n{diff}"


# ─── Rendering ────────────────────────────────────────────────────────────────


def _layered(tmp_path, *layers):
    merged = {}
    for layer in (_PARAMS, _SITE, *layers):
        merged.update(get_params_from_file(str(layer)))
    return merged


def _render_text(tmp_path, *layers, **params):
    cli = [f"{k}={v}" for k, v in params.items()]
    return apply_parameters(
        _BUILD.read_text(encoding="utf-8"),
        cli,
        _layered(tmp_path, *layers),
        str(tmp_path),
    )


def _render(tmp_path, **params):
    defaults = {
        "RUN_NAME": "e1",
        "EXPORT_DIR": "/shared/pretrain/exp/run1/export",
        "RUNGS": "10000,20000",
    }
    defaults.update(params)
    return yaml.safe_load(_render_text(tmp_path, **defaults))


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def _config(rendered, target):
    (step,) = _targets(rendered)[target]["steps"]
    return step["config"]


def test_the_graph_is_one_adopt_and_one_eval_per_rung(tmp_path):
    rendered = _render(tmp_path)
    assert set(_targets(rendered)) == {
        "export-10000",
        "export-20000",
        "eval-base-core-ck10000",
        "eval-base-core-ck20000",
        "eval-combined",
    }


def test_the_full_rung_runs_the_whole_catalog(tmp_path):
    rendered = _render(tmp_path, RUNGS="10000", FULL_EVAL_RUNGS="10000")
    catalog = yaml.safe_load(_PARAMS.read_text(encoding="utf-8"))["EVAL_CATALOG"]
    assert {t for t in _targets(rendered) if t.startswith("eval-ck")} == set()
    assert {f"eval-{e['name']}-ck10000" for e in catalog} <= set(_targets(rendered))


def test_no_rungs_fails_the_render(tmp_path):
    """A recipe whose only job is scoring rungs must not render to a build that
    scores nothing and reports SUCCESS. The guard is an undefined name whose text
    is the message, so the render raises (StrictUndefined) with it."""
    with pytest.raises(Exception, match="RUNGS_must_name_at_least_one_exported_rung"):
        _render_text(tmp_path, RUN_NAME="e1", EXPORT_DIR="/x", RUNGS="")


def test_a_full_rung_outside_rungs_fails_the_render(tmp_path):
    with pytest.raises(Exception, match="FULL_EVAL_RUNGS_must_name_rungs_of_RUNGS"):
        _render_text(
            tmp_path,
            RUN_NAME="e1",
            EXPORT_DIR="/x",
            RUNGS="10000",
            FULL_EVAL_RUNGS="999",
        )


# ─── The adopt targets ────────────────────────────────────────────────────────


def test_each_adopt_target_publishes_a_run_scoped_symlink(tmp_path):
    """The URI must be this build's own: two builds may not register one artifact
    URI, and re-registering the source export would make the second build report
    SUCCESS with an empty output list while every eval downstream waits forever."""
    rendered = _render(tmp_path, RUN_NAME="e1")
    out = _targets(rendered)["export-10000"]["outputs"]["hf_model"]
    assert out["type"] == "model"
    assert out["uri"].endswith("/e1/models/ckpt_10000/hf")
    assert "/shared/pretrain/exp/run1/export" not in out["uri"]


def test_the_adopt_command_links_the_source_export(tmp_path):
    command = _config(_render(tmp_path), "export-20000")["command_config"]["command"]
    assert "/shared/pretrain/exp/run1/export/ckpt_20000/hf" in command
    assert "ln -sfn" in command


def test_the_adopt_command_refuses_a_half_written_export(tmp_path):
    """A directory with a config and no weights is what a crashed export leaves;
    without this check it surfaces as a vLLM load error in an eval pod, with the
    rung's name nowhere in the message."""
    command = _config(_render(tmp_path), "export-10000")["command_config"]["command"]
    assert "config.json" in command
    assert ".safetensors" in command
    assert command.lstrip().startswith("set -euo pipefail")


def test_the_adopt_targets_cost_one_cpu(tmp_path):
    config = _config(_render(tmp_path), "export-10000")
    assert config["compute_config"]["num_nodes"] == 1
    assert "accelerators" not in config["launcher_config"]["resources"]


def test_the_adopt_targets_have_no_inputs(tmp_path):
    """Nothing in this build produces the export, so there is nothing to bind: the
    ordering edge the evals need is their own binding on hf_model."""
    assert "inputs" not in _targets(_render(tmp_path))["export-10000"]


# ─── The shared half, as rendered here ────────────────────────────────────────


def test_each_eval_reads_its_rung_through_the_binding(tmp_path):
    rendered = _render(tmp_path)
    target = _targets(rendered)["eval-base-core-ck10000"]
    assert target["inputs"]["hf_model"]["binding"] == "export-10000.hf_model"
    assert (
        _config(rendered, "eval-base-core-ck10000")["model_path"]
        == "{{ bindings.hf_model.binding.path }}"
    )


def test_the_roll_up_waits_for_every_eval(tmp_path):
    rendered = _render(tmp_path)
    bound = {
        spec["binding"]
        for spec in _targets(rendered)["eval-combined"]["inputs"].values()
    }
    assert bound == {
        "eval-base-core-ck10000.granite_results",
        "eval-base-core-ck20000.granite_results",
    }


def test_the_eval_context_length_is_the_trained_one(tmp_path):
    """SEQ_LEN is a parameter of this recipe too: vLLM's max_model_len must be what
    the exports were trained at, or prompts are invented or truncated."""
    rendered = _render(tmp_path, SEQ_LEN="4096")
    assert _config(rendered, "eval-base-core-ck10000")["max_model_len"] == 4096


# ─── The site boundary ────────────────────────────────────────────────────────


def test_no_key_is_both_a_site_key_and_a_parameter():
    params = set(yaml.safe_load(_PARAMS.read_text(encoding="utf-8")))
    site = set(yaml.safe_load(_SITE.read_text(encoding="utf-8")))
    both = params & site
    assert not both, sorted(both)


def test_the_shared_eval_knobs_match_the_training_recipe():
    """A catalog or a screen worked out on a ladder has to transfer to a re-score
    unchanged, so the knobs the shared block reads must be spelled the same and
    default the same in both recipes."""
    mine = yaml.safe_load(_PARAMS.read_text(encoding="utf-8"))
    theirs = yaml.safe_load(
        (_PRETRAIN / "pretrain" / "parameters.yaml").read_text(encoding="utf-8")
    )
    for key in (
        "EVAL_CATALOG",
        "SCREEN_EVALS",
        "EVAL_LIMIT",
        "EVAL_MEMORY",
        "REPORT_IMAGE",
        "REPORT_CPUS",
        "REPORT_MEMORY",
        "FULL_EVAL_RUNGS",
    ):
        assert mine[key] == theirs[key], key
