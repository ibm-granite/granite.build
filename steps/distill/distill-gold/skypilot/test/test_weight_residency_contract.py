"""The residency preflight is ONE program, copied into three steps and asserted identical.

Same argument as test_source_contract.py makes for the source-delivery region, and the same
mechanism: three subtly different copies of a check whose whole value is a hard-won pair of
invariants (mmlsattr is authoritative both ways; allocation may only ever accuse) is exactly
the failure this avoids. A step cannot import from a sibling step -- each publishes its own
src/ -- so a copy is the only option, and a checked copy is the difference between that and
drift.

THIS step is the reference: its test_check_weight_residency.py asserts the module is
*correct*, and this file asserts the other copies are byte-identical to it. Discovery is by
scanning for the invocation in every step-template.yaml, so a fourth step that wires the
preflight is covered here the moment it lands, with no list to update. The reverse direction
-- a weight-reading step that quietly stops wiring it -- is what the explicit list in
test_the_three_weight_reading_steps_all_wire_it guards, because discovery alone would go
green by finding nothing.
"""

from pathlib import Path

import pytest

_STEPS_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE = "distill-gold"
_MODULE = "check_weight_residency.py"

# The steps that load model weights from a path a recipe supplies. distill-eval and
# distill-hf-export read weights too, but only ones a previous target in the same recipe
# just wrote -- those are resident by construction, and an eval that waits on a recall is
# not holding a training allocation.
# Each step names its config block after itself, so the switch's full key differs per step
# even though the wiring does not.
_CONFIG_BLOCK = {
    "distill-gold": "gold_config",
    "distill-sft": "sft_config",
    "distill-logit-precompute": "precompute_config",
}
_EXPECTED = set(_CONFIG_BLOCK)


def _templates():
    return {
        p.parts[-3]: p
        for p in sorted(_STEPS_ROOT.glob("*/skypilot/step-template.yaml"))
    }


def _wiring_it():
    """Every step whose run block invokes the preflight."""
    return {
        name: path for name, path in _templates().items() if _MODULE in path.read_text()
    }


def _module_of(name):
    return _STEPS_ROOT / name / "skypilot" / "src" / _MODULE


def test_the_reference_step_is_present():
    """Guards against this file passing vacuously if distill-gold is ever renamed."""
    assert _REFERENCE in _wiring_it()


def test_the_three_weight_reading_steps_all_wire_it():
    """The direction discovery cannot see: a step that drops the invocation disappears from
    _wiring_it() and every other test here stays green."""
    assert _EXPECTED <= set(_wiring_it()), sorted(_EXPECTED - set(_wiring_it()))


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_every_wiring_step_ships_the_module(name):
    """A template invoking ./src/check_weight_residency.py without the file present fails at
    runtime as rc 2 from python, inside the allocation, which is the opposite of the point.
    """
    assert _module_of(name).is_file()


@pytest.mark.parametrize("name", sorted(n for n in _wiring_it() if n != _REFERENCE))
def test_the_module_is_byte_identical_to_the_reference(name):
    expected = _module_of(_REFERENCE).read_bytes()
    assert _module_of(name).read_bytes() == expected, (
        f"{name}'s {_MODULE} has drifted from {_REFERENCE}'s. Copy it from the reference "
        "rather than editing it in place; the invariants in its docstring were measured, "
        "not reasoned out, and a local edit loses that."
    )


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_every_wiring_is_switchable_and_overridable(name):
    """Two properties, both load-bearing. The `{% if %}` is how a recipe turns the check off
    on a cluster where it cannot measure anything; --allow-offline is how a run proceeds when
    the operator already knows the recall is under way. A copy wired without them is a step
    that can only be fixed by editing the template."""
    text = _wiring_it()[name].read_text()
    assert "check_weight_residency %}" in text
    assert "--allow-offline" in text
    assert "allow_offline_weights %}" in text


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_the_only_guard_on_the_invocation_is_the_config_switch(name):
    """Deliberately every node, not rank 0: a refusal on rank 0 alone would leave the other
    ranks waiting on a rendezvous that is never coming, which surfaces as a rendezvous
    timeout rather than as a storage problem. Asserting the immediately preceding line IS the
    config `{% if %}` is how that stays true -- a rank test, a `[ -d ]`, or any other
    condition spliced in front of it would fail here."""
    lines = _wiring_it()[name].read_text().splitlines()
    index = next(
        i for i, line in enumerate(lines) if _MODULE in line and "#" not in line
    )
    switch = f"{{% if config.{_CONFIG_BLOCK[name]}.check_weight_residency %}}"
    assert lines[index - 1].strip() == switch


def test_distill_gold_checks_before_the_vllm_role_split():
    """distill-gold's run block `exec`s into run_vllm_serve on the server nodes, so
    anything spliced after that branch never runs there -- and a vLLM node loads the student
    too. This assertion is the reason the invocation sits where it does."""
    text = _templates()[_REFERENCE].read_text()
    assert text.index(_MODULE) < text.index(
        "gb_steps_post_training.distillation.run_vllm_serve"
    )
