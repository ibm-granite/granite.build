# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the lm-engine config renderer.

The renderer is a port of gbansible's ``lm_engine_pretrain_job`` role
(``templates/pretrain_config.yml.j2`` + ``defaults/main.yml``), so the floor for
"correct" is: the granite-4 350M muP defaults must render to the same config that
role rendered for the ``nemotron-g4-350m-fused-500B`` experiment. That is
``test_render_matches_gbansible_golden``; every other test pins one validation
rule or one rendering branch the golden does not exercise.
"""

import copy
import json
import pathlib
import subprocess
import sys

import render_config as rc  # conftest.py puts src/ on sys.path
import yaml

# granite-4 350M muP defaults, from gbansible vars nemotron-g4-350m-fused-500B.yml
BASE = rc.DEFAULT_KNOBS

MUP_KNOBS = {
    "m_width": None,
    "m_emb": None,
    "m_residual": None,
    "attention_multiplier": None,
}


def errs(**over):
    k = copy.deepcopy(BASE)
    k.update(over)
    return rc.validate(k, num_nodes=4, gpus_per_node=8)


# ── validate ────────────────────────────────────────────────────────────────────


def test_defaults_valid():
    assert errs() == []


def test_partial_mup_rejected():
    assert any("muP" in e for e in errs(m_emb=None))


def test_mup_with_normal_param_groups_rejected():
    assert any(
        "param_groups" in e
        for e in errs(param_groups="configs/param-groups/normal.yml")
    )


def test_sp_model_valid():
    assert (
        errs(
            m_width=None,
            m_emb=None,
            m_residual=None,
            attention_multiplier=None,
            param_groups="configs/param-groups/normal.yml",
            lr="3e-4",
        )
        == []
    )


def test_sp_with_mup_param_groups_rejected():
    """Rule 2 in the other direction: no muP knobs, but mup.yml param groups."""
    assert any("param_groups" in e for e in errs(**MUP_KNOBS))


def test_attention_multiplier_method_must_be_null():
    """Rule 3: lm-engine rejects a method alongside an explicit multiplier."""
    assert any(
        "attention_multiplier_method" in e
        for e in errs(attention_multiplier_method="softmax_attention")
    )


def test_heads_kv():
    assert any("num_kv_heads" in e for e in errs(num_heads=10, num_kv_heads=4))


def test_logical_shards():
    assert any("logical_shards" in e for e in errs(logical_shards=100))


def test_num_workers_zero_rejected():
    assert any("num_workers" in e for e in errs(num_workers=0))


def test_num_workers_missing_rejected():
    k = copy.deepcopy(BASE)
    del k["num_workers"]
    assert any("num_workers" in e for e in rc.validate(k, num_nodes=4, gpus_per_node=8))


def test_save_interval_must_be_positive():
    assert any("save_interval" in e for e in errs(save_interval=0))


def test_save_interval_above_num_training_steps_rejected():
    assert any("save_interval" in e for e in errs(save_interval=200000))


def test_save_interval_need_not_divide_num_training_steps():
    """Rung alignment is the recipe's business, not the step's."""
    assert errs(save_interval=7000) == []


def test_unknown_file_type_rejected():
    assert any("file_type" in e for e in errs(file_type="jsonl"))


def test_arrow_needs_tokens_column():
    assert any("col_name" in e for e in errs(file_type="arrow", col_name="contents"))


def test_arrow_with_tokens_column_valid():
    assert errs(file_type="arrow", col_name="tokens") == []


# ── render ──────────────────────────────────────────────────────────────────────


def test_render_sets_save_and_no_load(tmp_path):
    cfg = rc.render(BASE, train_dir=str(tmp_path), load=None)
    assert cfg["save_args"]["save_path"] == str(tmp_path)
    assert "load_args" not in cfg


def test_render_mup_attention_method_null(tmp_path):
    cfg = rc.render(BASE, train_dir=str(tmp_path), load=None)
    blk = cfg["model_args"]["pretrained_config"]["sequence_mixer_blocks"][0]
    assert blk["attention_multiplier"] == BASE["attention_multiplier"]
    assert blk["attention_multiplier_method"] is None


def test_render_load_args(tmp_path):
    load = {"load_path": str(tmp_path), "load_rng_state": False}
    cfg = rc.render(BASE, train_dir=str(tmp_path), load=load)
    la = cfg["load_args"]
    assert la["load_path"] == str(tmp_path) and la["iteration"] is None
    assert (
        la["load_optimizer"]
        and la["load_lr_scheduler"]
        and la["load_starting_iteration"]
    )
    assert la["load_dataloader_state"] is False and la["load_rng_state"] is False


def test_render_does_not_mutate_knobs(tmp_path):
    before = copy.deepcopy(BASE)
    rc.render(BASE, train_dir=str(tmp_path), load=None)
    assert BASE == before


def test_render_matches_gbansible_golden(tmp_path):
    """The 350M defaults render to the same config gbansible rendered for
    nemotron-g4-350m-fused-500B.

    The fixture was captured once from the role itself, with jinja2 (plus an
    Ansible-compatible ``bool`` filter) over ``defaults/main.yml`` overridden by
    ``vars/lm_engine_pretrain/nemotron-g4-350m-fused-500B.yml``::

        v = yaml.safe_load(DEFAULTS_MAIN_YML)
        v.update(yaml.safe_load(VARS_350M_YML))
        v["experiment"] = "nemotron-g4-350m-fused-500B"
        v["lme_output_dir"] = "/TRAIN"
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.filters["bool"] = lambda x: str(x).strip().lower() in ("true","yes","on","1","y")
        out = env.from_string(PRETRAIN_CONFIG_YML_J2).render(**v)

    Four values were then edited by hand; each edit is commented in the fixture
    (two internal filesystem paths this repo must not ship as defaults, and two
    dotless scientific-notation scalars that YAML 1.1 reads as strings).
    """
    golden = yaml.safe_load(
        (
            pathlib.Path(__file__).parent / "fixtures" / "g4-350m-fused.golden.yml"
        ).read_text()
    )
    got = rc.render(BASE, train_dir="/TRAIN", load=None)
    assert got == golden


# ── CLI ─────────────────────────────────────────────────────────────────────────

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "render_config.py"


def _cli(*args):
    return subprocess.run(
        [sys.executable, str(_SRC), *args], capture_output=True, text=True
    )


def _knobs_json():
    return json.dumps(BASE)


def test_main_writes_config(tmp_path):
    out = tmp_path / "lme_config.yml"
    rc_code = rc.main(
        [
            "--knobs",
            _knobs_json(),
            "--train-dir",
            "/TRAIN",
            "--num-nodes",
            "8",
            "--gpus-per-node",
            "8",
            "--out",
            str(out),
        ]
    )
    assert rc_code == 0
    cfg = yaml.safe_load(out.read_text())
    assert cfg["save_args"]["save_path"] == "/TRAIN"
    assert "load_args" not in cfg


def test_main_load_inline_json(tmp_path):
    out = tmp_path / "c.yml"
    assert (
        rc.main(
            [
                "--knobs",
                _knobs_json(),
                "--train-dir",
                "/TRAIN",
                "--num-nodes",
                "8",
                "--gpus-per-node",
                "8",
                "--load",
                json.dumps({"load_path": "/TRAIN"}),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert yaml.safe_load(out.read_text())["load_args"]["load_path"] == "/TRAIN"


def test_main_decision_file(tmp_path):
    """run_pretrain.sh passes resume.py's decision file, not inline JSON."""
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps({"load": {"load_path": "/TRAIN"}, "seeded": False, "error": None})
    )
    out = tmp_path / "c.yml"
    assert (
        rc.main(
            [
                "--knobs",
                _knobs_json(),
                "--train-dir",
                "/TRAIN",
                "--num-nodes",
                "8",
                "--gpus-per-node",
                "8",
                "--decision",
                str(decision),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert yaml.safe_load(out.read_text())["load_args"]["load_path"] == "/TRAIN"


def test_main_decision_file_no_load(tmp_path):
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"load": None, "seeded": True, "error": None}))
    out = tmp_path / "c.yml"
    assert (
        rc.main(
            [
                "--knobs",
                _knobs_json(),
                "--train-dir",
                "/TRAIN",
                "--num-nodes",
                "8",
                "--gpus-per-node",
                "8",
                "--decision",
                str(decision),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert "load_args" not in yaml.safe_load(out.read_text())


def test_main_decision_error_is_fatal(tmp_path):
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {"load": None, "seeded": False, "error": "checkpoint tree is corrupt"}
        )
    )
    out = tmp_path / "c.yml"
    r = _cli(
        "--knobs",
        _knobs_json(),
        "--train-dir",
        "/TRAIN",
        "--num-nodes",
        "8",
        "--gpus-per-node",
        "8",
        "--decision",
        str(decision),
        "--out",
        str(out),
    )
    assert r.returncode != 0
    assert "checkpoint tree is corrupt" in r.stderr
    assert not out.exists()


def test_main_exits_2_with_every_validation_error(tmp_path):
    knobs = copy.deepcopy(BASE)
    knobs.update(num_heads=10, num_kv_heads=4, logical_shards=100)
    out = tmp_path / "c.yml"
    r = _cli(
        "--knobs",
        json.dumps(knobs),
        "--train-dir",
        "/TRAIN",
        "--num-nodes",
        "4",
        "--gpus-per-node",
        "8",
        "--out",
        str(out),
    )
    assert r.returncode == 2
    assert "num_kv_heads" in r.stderr and "logical_shards" in r.stderr
    assert not out.exists()


def test_main_rejects_load_and_decision_together(tmp_path):
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"load": None}))
    r = _cli(
        "--knobs",
        _knobs_json(),
        "--train-dir",
        "/TRAIN",
        "--num-nodes",
        "8",
        "--gpus-per-node",
        "8",
        "--load",
        "null",
        "--decision",
        str(decision),
        "--out",
        str(tmp_path / "c.yml"),
    )
    assert r.returncode != 0
