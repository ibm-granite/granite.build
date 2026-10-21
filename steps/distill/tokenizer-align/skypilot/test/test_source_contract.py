"""The source-delivery contract is ONE contract, asserted mechanically.

Every ported distillation step delivers ``gb_steps_post_training.distillation`` the same
way: the same ``code_config`` block, and the same region at the top of its ``run:``. That
sameness is the whole point — six subtly different copies of credential handling and
PYTHONPATH resolution is exactly the failure this avoids — so it is asserted rather than
described.

THIS step is the reference. Its own test_step_template.py asserts the region is
*correct*; this file asserts every other ported step's copy is byte-identical to it. The
two together are what let each other step's suite skip re-asserting the twelve
properties: identical to a correct reference is correct.

Living here rather than in a shared conftest is deliberate — it is the reference step's
job to notice when a copy has drifted, and a test that scans its siblings finds a NEW
ported step automatically rather than waiting for someone to add it to a list.
"""

from pathlib import Path

import pytest

_STEPS_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE = "tokenizer-align"

_CC_BEGIN = "  # ─── Source delivery"
_CC_END = '    setup_command: ""'
_SR_BEGIN = "            # --- distill source delivery: BEGIN"
_SR_END = "            # --- distill source delivery: END"


# The steps under steps/distill/ that do not carry the contract. gold and vllm-server
# both get their trainer source baked into the shared GPU image
# (steps/distill/gpu/Dockerfile, COPY of steps/distill/src/) rather than cloned at
# job-launch time, so neither has a code_config/source-delivery region at all -- not a
# drifted copy of one. corpus-pin-check/corpus-sources/gen-smoke/probe clone nothing.
# Everything else under steps/distill/ is discovered, so a new step is checked with no
# edit here and has to be named in this set to be left out.
_NOT_PORTED = {
    "corpus-pin-check",
    "corpus-sources",
    "gen-smoke",
    "gold",
    "probe",
    "vllm-server",
}


def _templates():
    """Every ported distillation step's template, keyed by step directory name."""
    found = {}
    for path in sorted(_STEPS_ROOT.glob("*/skypilot/step-template.yaml")):
        if path.parts[-3] not in _NOT_PORTED:
            found[path.parts[-3]] = path
    return found


def _region(text, begin, end, *, inclusive_end):
    start = text.index(begin)
    stop = text.index(end, start) + (len(end) if inclusive_end else 0)
    return text[start:stop]


def _code_config(text):
    return _region(text, _CC_BEGIN, _CC_END, inclusive_end=True)


def _source_region(text):
    return _region(text, _SR_BEGIN, _SR_END, inclusive_end=True)


def test_the_reference_step_is_present():
    """Guards against this test passing vacuously if the reference is ever renamed."""
    assert _REFERENCE in _templates()


def test_at_least_one_other_step_is_compared():
    """A byte-identity test over a single file proves nothing. This fails while only the
    reference exists, so it turns into a real assertion the moment a second step lands —
    rather than sitting green and empty."""
    assert len(_templates()) >= 2, "only the reference step exists; nothing to compare"


# A ported step that runs multi-node needs its own clone guard (code_config.workdir is
# a path on the SHARED filesystem, GB_BUILD_WORKDIR, so concurrent ranks rm -rf/cloning
# the same path race each other -- measured on build 7e9995b3, where two ranks' clones
# into the same CODE_DIR crashed one of them inside git's own ref-transaction code), so
# its source-delivery region legitimately drifts from the single-node reference's. Empty
# today: gold (the one multi-node ported step) now gets its trainer baked into the
# shared GPU image instead of cloning at job-launch time, so it carries no
# source-delivery region to compare at all (see _NOT_PORTED above). Add a step here,
# rather than loosening the check for everyone, if a future multi-node step clones.
_MULTI_NODE_STEPS = set()

# A ported step with its own per-step Dockerfile/image would deliberately diverge in
# code_config.python (baked venv path vs. "" for a runtime-synced one). Empty today --
# add a step here, rather than loosening the check for everyone, if one ever owns its
# own image again.
_OWNS_ITS_OWN_IMAGE = set()


@pytest.mark.parametrize(
    "name", sorted(n for n in _templates() if n != _REFERENCE and n not in _OWNS_ITS_OWN_IMAGE)
)
def test_code_config_block_is_byte_identical_to_the_reference(name):
    templates = _templates()
    expected = _code_config(templates[_REFERENCE].read_text())
    actual = _code_config(templates[name].read_text())
    assert actual == expected, (
        f"{name}'s code_config block has drifted from {_REFERENCE}'s. "
        "Splice it from the reference rather than editing it in place."
    )


@pytest.mark.parametrize(
    "name",
    sorted(n for n in _templates() if n != _REFERENCE and n not in _MULTI_NODE_STEPS),
)
def test_source_delivery_region_is_byte_identical_to_the_reference(name):
    templates = _templates()
    expected = _source_region(templates[_REFERENCE].read_text())
    actual = _source_region(templates[name].read_text())
    assert actual == expected, (
        f"{name}'s source-delivery region has drifted from {_REFERENCE}'s. "
        "Splice it from the reference rather than editing it in place."
    )


@pytest.mark.parametrize("name", sorted(_templates()))
def test_every_ported_step_has_both_regions(name):
    """A step that grew a bespoke source path would otherwise fail with an obscure
    ValueError from .index() instead of saying what is wrong."""
    text = _templates()[name].read_text()
    for marker in (_CC_BEGIN, _CC_END, _SR_BEGIN, _SR_END):
        assert marker in text, f"{name} is missing the marker {marker!r}"


@pytest.mark.parametrize(
    "name", sorted(n for n in _templates() if n not in _OWNS_ITS_OWN_IMAGE)
)
def test_no_step_ships_a_dockerfile(name):
    """Every ported step except one listed in _OWNS_ITS_OWN_IMAGE is a non-image
    step: common.mk keys off the Dockerfile's ABSENCE, so adding one silently
    turns image/publish-image back on."""
    assert not (_templates()[name].parent / "Dockerfile").exists()


# ─── The pinned source ─────────────────────────────────────────────────────────
# Asserted here rather than in test_step_template.py because it is a property of the
# CONTRACT: all six steps must clone the same public repo at the same commit, and
# the value itself has to be one this project controls. The byte-identity tests
# above already guarantee sameness; these say what the shared value must BE.
#
# code_dir is deliberately empty on every ported step: there is no pre-staged
# checkout to pin, on BlueVela or anywhere else. Each run clones _PINNED_REPO at
# _PINNED_REF itself, which is what makes these steps runnable on any environment
# class that can reach github.com rather than only the one host that happened to
# have a /proj clone.

_PINNED_REPO = "https://github.com/laminair/gb-steps-distillation.git"
_PINNED_REF = "a5d59bc45524a8d75706e20d44ae1a254f273f23"


@pytest.mark.parametrize("name", sorted(_templates()))
def test_every_step_pins_the_project_controlled_checkout(name):
    """Not a /proj checkout. A pre-staged filesystem clone only exists on the one host
    it was staged on, which is exactly what made these six steps unrunnable anywhere
    else; a public repo/ref clones identically in any environment class."""
    text = _templates()[name].read_text(encoding="utf-8")
    assert (
        f'code_dir: ""' in text
    ), f"{name} pins a filesystem checkout, not a public repo"
    assert (
        f'repo: "{_PINNED_REPO}"' in text
    ), f"{name} does not clone the public source repo"


@pytest.mark.parametrize("name", sorted(_templates()))
def test_every_step_pins_a_full_commit(name):
    """A branch head makes two runs a week apart different runs while reporting the
    same provenance, and a prefix is not a commit."""
    text = _templates()[name].read_text(encoding="utf-8")
    assert f'expect_ref: "{_PINNED_REF}"' in text, f"{name} pins a different commit"
    assert (
        f'ref: "{_PINNED_REF}"' in text
    ), f"{name}'s clone ref disagrees with expect_ref"
    assert len(_PINNED_REF) == 40
