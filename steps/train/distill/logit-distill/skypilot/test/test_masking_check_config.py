# SPDX-License-Identifier: Apache-2.0

"""masking.check_config's last_message_only rule agrees with render_gold_config.py's.

check-config audits a gold config against the masking contract masking.json derives from the
chat template: the response_template scan. last_message_only: true replaces that scan with a
positional split, so it used to be reported as a problem on every config. That is right for
the arms that scan, and wrong for the two arms whose LOSS_ARMS entry REQUIRES it: uld (ULD's
teacher is the last assistant turn alone) and sampled_opd. For those, true is the contract and
false or unset is the problem, which is the config the renderer refuses.
"""

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
@pytest.mark.parametrize("lmo", ["true", 1], ids=["string", "int"])
def test_only_the_boolean_true_satisfies_it(arm, lmo):
    """A YAML string or a 1 is not the boolean the trainer's parser needs; the check is
    `is not True`, not falsiness."""
    for form in _forms(arm):
        cfg = {"response_template": _TEMPLATE, "last_message_only": lmo, **form}
        (problem,) = _lmo_problems(cfg)
        assert f"it is {lmo!r}" in problem


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


@pytest.mark.parametrize("mode", ["uld_per_turn_regions", "uld_reuse_student_input"])
def test_uld_modes_that_render_the_teacher_otherwise_are_exempt(mode):
    """The exemptions CustomGOLDConfig makes: per-turn regions require false, and a
    teacher reusing the student's input needs no last-turn render."""
    for form in _forms("uld"):
        cfg = {"response_template": _TEMPLATE, mode: True, **form}
        assert masking.check_config(_DOC, cfg) == []
        assert masking.check_config(_DOC, {**cfg, "last_message_only": False}) == []


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
