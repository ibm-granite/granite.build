# SPDX-License-Identifier: Apache-2.0

"""masking.check_config's last_message_only rule agrees with render_gold_config.py's.

check-config audits a gold config against the masking contract masking.json derives from the
chat template: the response_template scan. last_message_only: true replaces that scan with a
positional split, so it used to be reported as a problem on every config. That is right for
the arms that scan, and wrong for the two arms whose LOSS_ARMS entry REQUIRES it: uld (ULD's
teacher is the last assistant turn alone) and sampled_opd. For those, true is the contract and
false or unset is the problem, which is the config the renderer refuses.
"""

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest
import render_gold_config as rgc
import yaml
from gb_steps_post_training.distillation import masking

_TEMPLATE = "<|im_start|>assistant\n"
_DOC = {"response_template": _TEMPLATE}
_RENDER = Path(__file__).resolve().parent.parent / "src" / "render_gold_config.py"

_REQUIRING = sorted(
    arm
    for arm, spec in rgc.LOSS_ARMS.items()
    if spec.get("requires", {}).get("last_message_only") is True
)


def _lmo_problems(cfg):
    return [p for p in masking.check_config(_DOC, cfg) if "last_message_only" in p]


def _forms(arm):
    """The arm as a step gold_config names it (loss_arm) and as the renderer emits it (its
    switches; a rendered config has no loss_arm key)."""
    return [{"loss_arm": arm}, dict(rgc.LOSS_ARMS[arm]["flags"])]


def test_the_requiring_arms_are_uld_and_sampled_opd():
    """If the renderer's table gains or loses an arm requiring it, check_config's
    _arm_requiring_last_message_only must follow."""
    assert _REQUIRING == ["sampled_opd", "uld"]


@pytest.mark.parametrize("arm", _REQUIRING)
@pytest.mark.parametrize("lmo", [None, False], ids=["unset", "false"])
def test_a_requiring_arm_without_it_is_a_problem(arm, lmo):
    for form in _forms(arm):
        cfg = {"response_template": _TEMPLATE, **form}
        if lmo is not None:
            cfg["last_message_only"] = lmo
        (problem,) = _lmo_problems(cfg)
        assert f"loss_arm {arm} requires last_message_only: true" in problem


@pytest.mark.parametrize("arm", _REQUIRING)
@pytest.mark.parametrize("lmo", ["true", "True", 1], ids=["string", "capital", "int"])
def test_what_the_parsers_read_as_true_satisfies_it(arm, lmo):
    """render_gold_config.py's _bool (a step's gold_config block) and gold.py's TrlParser
    (a gold config given to gold.py) both read these as true, so neither form is a
    problem."""
    for form in _forms(arm):
        cfg = {"response_template": _TEMPLATE, "last_message_only": lmo, **form}
        assert _lmo_problems(cfg) == []


@pytest.mark.parametrize("arm", _REQUIRING)
def test_a_value_the_parser_refuses_is_a_problem(arm):
    named, switch = _forms(arm)
    (problem,) = _lmo_problems(
        {"response_template": _TEMPLATE, "last_message_only": "maybe", **named}
    )
    assert "render_gold_config.py does not read it as a boolean" in problem
    (problem,) = _lmo_problems(
        {"response_template": _TEMPLATE, "last_message_only": "maybe", **switch}
    )
    assert "gold.py's TrlParser does not read it as a boolean" in problem


@pytest.mark.parametrize("arm", _REQUIRING)
def test_each_form_is_read_by_its_own_parser(arm):
    """The two parsers differ at the edges: TrlParser reads "t" as true and refuses "on";
    render_gold_config.py's _bool reads "on" as true and refuses "t". A step's
    gold_config block (named arm) goes through the renderer, any other config through
    TrlParser."""
    named, switch = _forms(arm)
    ok = {"response_template": _TEMPLATE}
    assert _lmo_problems({**ok, **switch, "last_message_only": "t"}) == []
    assert _lmo_problems({**ok, **named, "last_message_only": "on"}) == []
    assert _lmo_problems({**ok, **switch, "last_message_only": "on"})
    assert _lmo_problems({**ok, **named, "last_message_only": "t"})


@pytest.mark.parametrize("arm", _REQUIRING)
def test_a_switch_given_as_a_string_is_the_switch(arm):
    """TrlParser reads `use_uld_loss: "true"` (or `use_sampled_opd_loss: "true"`) as
    the switch, so the arm's rule applies to it as to the boolean."""
    (switch,) = rgc.LOSS_ARMS[arm]["flags"]
    (problem,) = _lmo_problems({"response_template": _TEMPLATE, switch: "true"})
    assert f"loss_arm {arm} requires last_message_only: true" in problem
    cfg = {"response_template": _TEMPLATE, switch: "false", "last_message_only": True}
    (problem,) = _lmo_problems(cfg)
    assert "bypasses the response_template scan" in problem


_READER_VALUES = [True, False, "true", "True", "TRUE", "false", "False", "t", "y"]
_READER_VALUES += ["yes", "no", "n", "f", "on", "off", "1", "0", 1, 0, 2, "", "maybe"]


@pytest.mark.parametrize("value", _READER_VALUES, ids=repr)
def test_the_renderer_reader_is_render_gold_configs_bool(value):
    """As the step's template puts the value on the command line: str(value)."""
    try:
        want = rgc._bool(str(value))
    except Exception:
        want = None
    assert masking._renderer_reads(value) is want


@pytest.mark.parametrize("value", _READER_VALUES, ids=repr)
def test_the_trainer_reader_is_trl_parser(tmp_path, value):
    """Through gold.py's real parser, TrlParser over CustomGOLDConfig, from a YAML file."""
    pytest.importorskip("trl")
    want = _trl_parser_reads(tmp_path, {"last_message_only": value})
    want = None if want is None else bool(want["last_message_only"])
    assert masking._trainer_reads(value) is want


def _trl_parser_reads(tmp_path, fields):
    """`fields` written to a YAML config and parsed the way gold.py parses its config:
    the CustomGOLDConfig fields as parsed (its __post_init__ included), or None when the
    parser refuses a value (it exits) or __post_init__ raises."""
    from gb_steps_post_training.distillation.custom_gold_config import (
        CustomGOLDConfig,
    )
    from trl import TrlParser

    cfg = tmp_path / "gold.yaml"
    base = dict(output_dir=str(tmp_path / "out"), report_to=[], use_cpu=True)
    cfg.write_text(yaml.safe_dump({**base, **fields}))
    try:
        (args,) = TrlParser((CustomGOLDConfig,)).parse_args_and_config(
            ["--config", str(cfg)]
        )
    except (SystemExit, ValueError):
        return None
    return vars(args)


@pytest.mark.parametrize("arm", _REQUIRING)
def test_the_message_names_who_refuses_it(arm):
    """Named as loss_arm, the renderer refuses it; inferred from the switch alone (a config
    that never went through the renderer), only CustomGOLDConfig does."""
    named, switch = _forms(arm)
    (problem,) = _lmo_problems({"response_template": _TEMPLATE, **named})
    assert "render_gold_config.py refuses it at render and CustomGOLDConfig" in problem
    (problem,) = _lmo_problems({"response_template": _TEMPLATE, **switch})
    assert "render_gold_config.py" not in problem
    assert "CustomGOLDConfig refuses it at startup" in problem


_ULD_EXEMPT_MODES = [
    "uld_per_turn_regions",
    "uld_reuse_student_input",
    "uld_rebuild_student_input_with_teacher_template",
]


@pytest.mark.parametrize("mode", _ULD_EXEMPT_MODES)
def test_uld_modes_that_render_the_teacher_otherwise_are_exempt(mode):
    """The exemptions CustomGOLDConfig makes: per-turn regions require false, a teacher
    reusing the student's input needs no last-turn render, and a student rebuilt from the
    teacher's texts is supervised on the last turn alone whatever the collator labelled.
    For the switch form only, which is what reaches CustomGOLDConfig."""
    _, switch = _forms("uld")
    cfg = {"response_template": _TEMPLATE, mode: True, **switch}
    assert masking.check_config(_DOC, cfg) == []
    assert masking.check_config(_DOC, {**cfg, "last_message_only": False}) == []


@pytest.mark.parametrize("mode", _ULD_EXEMPT_MODES)
@pytest.mark.parametrize("lmo", [None, False, True], ids=["unset", "false", "true"])
def test_a_named_uld_arm_with_a_mode_is_a_problem(mode, lmo):
    """The step passes no uld_* key to its renderer, which has no uld_* option, so the
    mode is dropped and the run would start without it (test_the_step_drops_uld_keys).
    That is a problem whatever last_message_only is, and without true the renderer's own
    refusal is reported as well."""
    named, _ = _forms("uld")
    cfg = {"response_template": _TEMPLATE, mode: True, **named}
    if lmo is not None:
        cfg["last_message_only"] = lmo
    problems = masking.check_config(_DOC, cfg)
    dropped = [p for p in problems if f"would start without {mode}" in p]
    assert len(dropped) == 1
    assert "passes no uld_* key to render_gold_config.py" in dropped[0]
    lmo_problems = _lmo_problems(cfg)
    assert len(lmo_problems) == (0 if lmo is True else 1)
    assert "CustomGOLDConfig refuses it at startup." not in " ".join(dropped)


def test_the_step_drops_uld_keys():
    """What a named arm's uld_* mode comes to: the step's run script, rendered from a
    gold_config that sets every mode, passes none of them to render_gold_config.py, and
    the renderer has no option for one."""
    from jinja2 import Template

    step = yaml.safe_load((_RENDER.parent.parent / "step-template.yaml").read_text())
    script = step["environment_configs"]["Skypilot"]["launchers"]["gold"]["config"][
        "run"
    ]
    config = dict(step["config"])
    gold = {**config["gold_config"], "loss_arm": "uld", "last_message_only": True}
    config["gold_config"] = {**gold, **{m: True for m in _ULD_EXEMPT_MODES}}
    bindings = {
        "student": {"binding": {"path": "/models/student"}},
        "teacher": {"binding": {"path": "/models/teacher"}},
        "corpus": {"binding": {"path": "/data/corpus/train.jsonl"}},
    }
    rendered = Template(script).render(config=config, bindings=bindings)
    assert "render_gold_config.py" in rendered
    assert "uld_" not in rendered and "--uld" not in rendered
    with pytest.raises(SystemExit):
        rgc._parse_args(["--output", "/dev/null", "--uld-reuse-student-input", "true"])


# check-config against the renderer and CustomGOLDConfig, for every ULD form, mode and
# last_message_only value, booleans and the strings and integers the parsers also read. The
# verdict is whether the run would start AS CONFIGURED: a named arm goes through the step,
# which drops every uld_* mode, and through render_gold_config.py, which requires
# last_message_only: true; a switch-form config goes to gold.py's TrlParser and
# CustomGOLDConfig. Per-turn regions also need the hybrid loss (CustomGOLDConfig refuses them
# without), which is set so that only last_message_only varies.
_ULD_MODES = {
    "none": {},
    "rebuild": {"uld_rebuild_student_input_with_teacher_template": True},
    "reuse": {"uld_reuse_student_input": True},
    "per-turn": {"uld_per_turn_regions": True, "uld_use_hybrid_loss": True},
}
_LMO_VALUES = {
    "unset": None,
    "false": False,
    "true": True,
    "str-true": "true",
    "str-false": "false",
    "int-1": 1,
    "int-0": 0,
}
_TRUE = {"true", "str-true", "int-1"}
_ALL = set(_LMO_VALUES)
# (form, mode) -> the last_message_only values with which the run starts as configured.
_STARTS_WITH = {
    ("switch", "none"): _TRUE,
    ("switch", "rebuild"): _ALL,
    ("switch", "reuse"): _ALL,
    ("switch", "per-turn"): _ALL - _TRUE,
    ("named", "none"): _TRUE,
    ("named", "rebuild"): set(),  # the step drops the mode
    ("named", "reuse"): set(),
    ("named", "per-turn"): set(),
}
_ULD_MATRIX = [
    (form, mode, lmo)
    for form in ("switch", "named")
    for mode in _ULD_MODES
    for lmo in _LMO_VALUES
]


def _uld_cfg(form, mode, lmo):
    cfg = {"response_template": _TEMPLATE, **_ULD_MODES[mode]}
    cfg.update({"use_uld_loss": True} if form == "switch" else {"loss_arm": "uld"})
    if _LMO_VALUES[lmo] is not None:
        cfg["last_message_only"] = _LMO_VALUES[lmo]
    return cfg


@pytest.mark.parametrize("form, mode, lmo", _ULD_MATRIX)
def test_check_config_passes_exactly_the_uld_configs_that_start(form, mode, lmo):
    problems = masking.check_config(_DOC, _uld_cfg(form, mode, lmo))
    if lmo in _STARTS_WITH[(form, mode)]:
        assert problems == []
    else:
        assert problems


@pytest.mark.parametrize("form, mode, lmo", _ULD_MATRIX)
def test_the_table_is_what_the_renderer_and_custom_gold_config_decide(
    tmp_path, form, mode, lmo
):
    """_STARTS_WITH, checked against the real gates. A named arm: the step drops the mode
    (test_the_step_drops_uld_keys), so the run cannot start as configured with one; with
    none, render_gold_config.py renders `--loss-arm uld --last-message-only <value as the
    template renders it>`, and its config's CustomGOLDConfig fields go to TrlParser. A
    switch-form config goes to TrlParser as it is."""
    pytest.importorskip("trl")
    from gb_steps_post_training.distillation.custom_gold_config import (
        CustomGOLDConfig,
    )
    from jinja2 import Template

    value = _LMO_VALUES[lmo]
    if form == "switch":
        fields = {"use_uld_loss": True, **_ULD_MODES[mode]}
        if value is not None:
            fields["last_message_only"] = value
        starts = _trl_parser_reads(tmp_path, fields) is not None
    elif mode != "none":
        starts = False
    else:
        step = yaml.safe_load(
            (_RENDER.parent.parent / "step-template.yaml").read_text()
        )
        default = step["config"]["gold_config"]["last_message_only"]
        lmo_arg = Template("{{ v }}").render(v=default if value is None else value)
        out = tmp_path / "rendered.yaml"
        cmd = [sys.executable, str(_RENDER), "--output", str(out), "--total-nodes", "1"]
        cmd += ["--model-name-or-path", "/s", "--teacher-model-name-or-path", "/t"]
        cmd += ["--dataset-name", "/d.jsonl", "--response-template", _TEMPLATE]
        cmd += ["--loss-arm", "uld", "--last-message-only", lmo_arg]
        rendered = subprocess.run(cmd, capture_output=True, check=False)
        starts = rendered.returncode == 0
        if starts:
            names = {f.name for f in dataclasses.fields(CustomGOLDConfig)}
            cfg = yaml.safe_load(out.read_text())
            trainer = {k: v for k, v in cfg.items() if k in names}
            trainer.pop("output_dir", None)
            assert trainer["use_uld_loss"] is True
            starts = _trl_parser_reads(tmp_path, trainer) is not None
    assert starts is (lmo in _STARTS_WITH[(form, mode)])


@pytest.mark.parametrize("arm", _REQUIRING)
def test_a_requiring_arm_with_it_is_sound(arm):
    """The direction that changed: true is no longer reported for these arms."""
    for form in _forms(arm):
        cfg = {"response_template": _TEMPLATE, "last_message_only": True, **form}
        assert masking.check_config(_DOC, cfg) == []


@pytest.mark.parametrize(
    "arm", sorted(a for a in rgc.LOSS_ARMS if a not in _REQUIRING) + [""]
)
def test_every_other_arm_keeps_todays_rule(arm):
    """Unchanged for the scanning arms: true bypasses the scan masking.json describes, and
    false or unset is sound."""
    forms = _forms(arm) if arm else [{}]
    for form in forms:
        base = {"response_template": _TEMPLATE, **form}
        assert masking.check_config(_DOC, base) == []
        assert masking.check_config(_DOC, {**base, "last_message_only": False}) == []
        (problem,) = _lmo_problems({**base, "last_message_only": True})
        assert "bypasses the response_template scan" in problem


def _render(tmp_path, extra, nodes=1):
    out = tmp_path / "gold_config.yaml"
    cmd = [
        sys.executable,
        str(_RENDER),
        "--output",
        str(out),
        "--total-nodes",
        str(nodes),
    ]
    cmd += ["--model-name-or-path", "/s", "--teacher-model-name-or-path", "/t"]
    cmd += ["--dataset-name", "/d.jsonl", "--response-template", _TEMPLATE, *extra]
    subprocess.run(cmd, check=True, capture_output=True)
    return yaml.safe_load(out.read_text())


def test_what_the_renderer_emits_for_uld_passes(tmp_path):
    """End to end: the config the renderer emits for uld (off-policy, where last_message_only
    is emitted for that arm alone) is sound by check_config."""
    cfg = _render(tmp_path, ["--loss-arm", "uld", "--last-message-only", "true"])
    assert cfg["use_uld_loss"] is True and cfg["last_message_only"] is True
    assert masking.check_config(_DOC, cfg) == []


def test_what_the_renderer_emits_for_jsd_with_it_is_still_reported(tmp_path):
    """On-policy the renderer emits last_message_only for every arm; for a scanning arm
    true is still the bypass check_config reports."""
    cfg = _render(
        tmp_path,
        ["--loss-arm", "jsd", "--last-message-only", "true", "--vllm-num-servers", "1"],
        nodes=2,
    )
    assert cfg["last_message_only"] is True
    (problem,) = _lmo_problems(cfg)
    assert "bypasses" in problem
