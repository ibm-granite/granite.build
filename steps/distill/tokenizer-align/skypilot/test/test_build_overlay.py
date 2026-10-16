"""
PORTED, not authored here. Upstream source of truth:
  repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
  path   steps/distill-tokenizer-align/test/test_build_overlay.py
  commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579

Two intentional divergences, both about path resolution rather than behaviour:
  - upstream's ``sys.path.insert(0, parents[3] / "src")`` is removed; conftest.py
    resolves the upstream package from GB_DISTILL_CODE_DIR, and parents[3] does not
    point at a checkout from inside granite.build.
  - the import of the package is therefore gated: see conftest.py.

Keep this a near-verbatim copy so re-syncing upstream stays a three-way merge.
"""

"""Unit tests for the overlay builder.

Covers the stdlib half -- classify() and build(). verify() defers its transformers and
tokenizers imports into the function body, so it is exercised against real Granite
tokenizers by an LSF job (see README.md) rather than mocked here: a mocked round-trip
would assert nothing about the property the overlay exists to guarantee.
"""

import json
import sys
from pathlib import Path

import pytest
from gb_steps_post_training.distillation.build_overlay import (  # noqa: E402
    CONFIG_KEYS_FORCED,
    OVERLAY_KEEP,
    SIDECARS,
    OverlayError,
    build,
    build_parser,
    classify,
    verify,
)


def _model_dir(
    root: Path,
    *,
    sidecars=("vocab.json", "merges.txt"),
    tokenizer_json=True,
    extra=("config.json", "model.safetensors"),
    tokenizer_class="GPT2Tokenizer",
) -> Path:
    d = root / "model"
    d.mkdir(parents=True)
    if tokenizer_json:
        (d / "tokenizer.json").write_text('{"added_tokens": []}')
    # tokenizer_class defaults to GPT2Tokenizer because that is what the real Granite
    # directories ship, and it is the single key measured to change segmentation.
    cfg = {"padding_side": "right"}
    if tokenizer_class is not None:
        cfg["tokenizer_class"] = tokenizer_class
    (d / "tokenizer_config.json").write_text(json.dumps(cfg))
    (d / "special_tokens_map.json").write_text("{}")
    for name in sidecars:
        (d / name).write_text("legacy")
    for name in extra:
        (d / name).write_text("x")
    return d


# ------------------------------------------------------------------ classify


def test_classify_separates_overlay_sidecar_and_other(tmp_path):
    parts = classify(_model_dir(tmp_path))
    assert "tokenizer.json" in parts["overlay"]
    assert "tokenizer_config.json" in parts["overlay"]
    assert set(parts["sidecar"]) == {"vocab.json", "merges.txt"}
    assert "model.safetensors" in parts["other"]
    assert "config.json" in parts["other"]


def test_classify_refuses_a_dir_with_no_fast_tokenizer(tmp_path):
    # The one case an overlay cannot fix: there is only a slow tokenizer to begin with.
    with pytest.raises(OverlayError, match="no tokenizer.json"):
        classify(_model_dir(tmp_path, tokenizer_json=False))


def test_classify_handles_a_dir_with_no_sidecars(tmp_path):
    parts = classify(_model_dir(tmp_path, sidecars=()))
    assert parts["sidecar"] == []


def test_every_sidecar_name_is_classified_as_a_sidecar(tmp_path):
    parts = classify(_model_dir(tmp_path, sidecars=SIDECARS))
    assert set(parts["sidecar"]) == set(SIDECARS)


def test_keep_and_sidecar_lists_are_disjoint():
    # A name in both lists would make the outcome depend on branch order in classify().
    assert not (set(OVERLAY_KEEP) & set(SIDECARS))


# ------------------------------------------------------------------ build


def test_build_copies_only_overlay_files(tmp_path):
    src = _model_dir(tmp_path)
    dest = tmp_path / "overlay"
    build(src, dest)
    got = {p.name for p in dest.iterdir()}
    assert got == {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    assert "model.safetensors" not in got


def test_build_excludes_the_sidecars_that_are_the_whole_point(tmp_path):
    dest = tmp_path / "overlay"
    build(_model_dir(tmp_path), dest)
    for name in SIDECARS:
        assert not (dest / name).exists(), f"{name} must never appear in an overlay"


def test_build_manifest_records_which_sidecars_were_excluded(tmp_path):
    # Recorded rather than merely omitted: knowing the source HAD sidecars is what tells
    # the next reader whether this source was actually exposed to the trap.
    m = build(
        _model_dir(tmp_path, sidecars=("vocab.json", "merges.txt")), tmp_path / "o"
    )
    assert set(m["sidecars_excluded"]) == {"vocab.json", "merges.txt"}
    assert "model.safetensors" in m["source_files_not_copied"]
    assert m["copy_mode"] == "copy"


def test_build_is_idempotent(tmp_path):
    src, dest = _model_dir(tmp_path), tmp_path / "overlay"
    build(src, dest)
    build(src, dest)  # must not raise FileExistsError
    assert (dest / "tokenizer.json").read_text() == '{"added_tokens": []}'


def test_build_refuses_an_unknown_copy_mode(tmp_path):
    with pytest.raises(OverlayError, match="copy_mode"):
        build(_model_dir(tmp_path), tmp_path / "o", copy_mode="symlink")


def test_build_hardlink_mode_produces_readable_identical_content(tmp_path):
    src, dest = _model_dir(tmp_path), tmp_path / "overlay"
    m = build(src, dest, copy_mode="hardlink")
    assert m["copy_mode"] == "hardlink"
    assert (dest / "tokenizer.json").read_text() == (src / "tokenizer.json").read_text()


def test_build_does_not_modify_the_source(tmp_path):
    src = _model_dir(tmp_path)
    before = sorted(p.name for p in src.iterdir())
    build(src, tmp_path / "overlay")
    assert sorted(p.name for p in src.iterdir()) == before


def test_build_overwrites_a_stale_overlay_file(tmp_path):
    # An overlay rebuilt after the source tokenizer changed must not keep the old bytes.
    src, dest = _model_dir(tmp_path), tmp_path / "overlay"
    build(src, dest)
    (src / "tokenizer.json").write_text('{"added_tokens": ["new"]}')
    build(src, dest)
    assert json.loads((dest / "tokenizer.json").read_text())["added_tokens"] == ["new"]


# ------------------------------------------------- the tokenizer_class rewrite
#
# These cover the defect that LSF job 1136957 exposed: build() used to copy
# tokenizer_config.json verbatim, so the overlay it produced for granite-4.1-3b-base
# still told transformers to construct a GPT2Tokenizer, which imposed GPT-2's plain
# ByteLevel pre_tokenizer over the model's Sequence[Split(regex),ByteLevel]. The overlay
# built cleanly and then failed its own verify().


def test_build_pins_tokenizer_class_in_the_overlay_config(tmp_path):
    dest = tmp_path / "overlay"
    build(_model_dir(tmp_path), dest)
    cfg = json.loads((dest / "tokenizer_config.json").read_text())
    assert cfg["tokenizer_class"] == "PreTrainedTokenizerFast", (
        "tokenizer_class was not pinned. Left as GPT2Tokenizer, transformers builds that "
        "class and imposes its pre_tokenizer over the one in tokenizer.json."
    )
    # Everything else must survive -- this is a targeted rewrite of one key.
    assert cfg["padding_side"] == "right"


def test_build_pins_the_key_even_when_the_source_omits_it(tmp_path):
    # Job 1137115: with a config.json declaring model_type: granite, an ABSENT
    # tokenizer_class is just as broken as an explicit GPT2Tokenizer one -- transformers
    # falls back to TOKENIZER_MAPPING_NAMES. So absent is not a safe state to leave alone;
    # the overlay must assert the value positively.
    m = build(_model_dir(tmp_path, tokenizer_class=None), tmp_path / "overlay")
    cfg = json.loads((tmp_path / "overlay" / "tokenizer_config.json").read_text())
    assert cfg["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert m["config_keys_forced"] == {
        "tokenizer_config.json": {
            "tokenizer_class": {"was": None, "now": "PreTrainedTokenizerFast"}
        }
    }


def test_build_records_forced_config_keys_in_the_manifest(tmp_path):
    m = build(_model_dir(tmp_path), tmp_path / "overlay")
    assert m["config_keys_forced"] == {
        "tokenizer_config.json": {
            "tokenizer_class": {
                "was": "GPT2Tokenizer",
                "now": "PreTrainedTokenizerFast",
            }
        }
    }


def test_build_manifest_reports_nothing_forced_when_the_value_was_already_right(
    tmp_path,
):
    # An empty dict, not a missing key: a reader must be able to tell "checked, already
    # correct" apart from "this manifest predates the check".
    m = build(
        _model_dir(tmp_path, tokenizer_class="PreTrainedTokenizerFast"),
        tmp_path / "overlay",
    )
    assert m["config_keys_forced"] == {}


def test_build_does_not_modify_the_source_config(tmp_path):
    # The source is a shared HF cache directory that other runs read. Rewriting the key
    # there would silently change every other consumer of that snapshot.
    src = _model_dir(tmp_path)
    build(src, tmp_path / "overlay")
    assert (
        json.loads((src / "tokenizer_config.json").read_text())["tokenizer_class"]
        == "GPT2Tokenizer"
    )


def test_hardlink_mode_still_rewrites_the_config(tmp_path):
    # A hardlinked config would be the SAME inode as the source's, so pinning the key
    # would mutate the source -- and not pinning it would reintroduce the defect.
    # The config must be a real, separate, rewritten file even in hardlink mode.
    src, dest = _model_dir(tmp_path), tmp_path / "overlay"
    build(src, dest, copy_mode="hardlink")
    assert (
        json.loads((dest / "tokenizer_config.json").read_text())["tokenizer_class"]
        == "PreTrainedTokenizerFast"
    )
    assert (
        json.loads((src / "tokenizer_config.json").read_text())["tokenizer_class"]
        == "GPT2Tokenizer"
    )
    assert (dest / "tokenizer_config.json").stat().st_ino != (
        src / "tokenizer_config.json"
    ).stat().st_ino


def test_config_rewrite_is_idempotent(tmp_path):
    src, dest = _model_dir(tmp_path), tmp_path / "overlay"
    build(src, dest)
    first = (dest / "tokenizer_config.json").read_bytes()
    build(src, dest)
    assert (dest / "tokenizer_config.json").read_bytes() == first


def test_tokenizer_class_is_pinned_to_the_fast_wrapper():
    # Guard on the constant itself. If someone empties CONFIG_KEYS_FORCED, every test
    # above still passes trivially against a fixture with no tokenizer_class -- so assert
    # the key that was actually measured to matter is named, with the measured-good value.
    # Deleting the key is NOT an acceptable substitute here: job 1137115 showed that with
    # a config.json present, an absent key resolves to GPT2Tokenizer anyway.
    assert CONFIG_KEYS_FORCED["tokenizer_class"] == "PreTrainedTokenizerFast"


# ------------------------------------------------------- require_chatml plumbing
#
# NOT a test of segmentation -- see the module docstring: a mocked round-trip asserts
# nothing about the property the overlay guarantees, and real tokenizers are exercised on
# BlueVela. What IS worth testing without transformers installed is the CONTROL FLOW of the
# flag, because getting it backwards fails a correct artifact (the pre-retag base student,
# whose vocabulary genuinely has no ChatML control tokens) or silently accepts a broken one.


class _FakeTok:
    """Minimal stand-in: agrees with the backend on every probe, and spells out the
    ChatML markers the way granite-4.1-3b-base actually does."""

    is_fast = True

    def __init__(self, pre_tokenizer, chatml_ids):
        self._pt = pre_tokenizer
        self._chatml = chatml_ids
        self.backend_tokenizer = self

    def to_str(self):
        return json.dumps({"pre_tokenizer": self._pt})

    def __len__(self):
        return 100352

    def __call__(self, text, add_special_tokens=False):
        if text in ("<|im_start|>", "<|im_end|>"):
            return {"input_ids": list(self._chatml)}
        return {"input_ids": _fake_backend_ids(None, text)}


def _fake_backend_ids(_path, text):
    return [ord(c) % 97 for c in text]


@pytest.fixture
def stub_transformers(monkeypatch):
    """Install a fake `transformers` and stub _backend_ids so verify() runs without deps."""
    import types

    from gb_steps_post_training.distillation import build_overlay as bo

    def _install(chatml_ids, pre_tokenizer={"type": "ByteLevel"}):
        mod = types.ModuleType("transformers")

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path, local_files_only=False):
                return _FakeTok(pre_tokenizer, chatml_ids)

        mod.AutoTokenizer = AutoTokenizer
        monkeypatch.setitem(sys.modules, "transformers", mod)
        monkeypatch.setattr(bo, "_backend_ids", _fake_backend_ids)

    return _install


def _overlay(tmp_path, pre_tokenizer={"type": "ByteLevel"}):
    d = tmp_path / "overlay"
    d.mkdir()
    (d / "tokenizer.json").write_text(json.dumps({"pre_tokenizer": pre_tokenizer}))
    return d


def test_verify_requires_single_id_chatml_by_default(tmp_path, stub_transformers):
    # Default True: a model that spells the markers out is a failure worth reporting,
    # because ChatML turn boundaries -- and therefore EOS -- are unrepresentable.
    stub_transformers(chatml_ids=[27, 91, 318, 5011, 91, 29])
    with pytest.raises(OverlayError, match="not 1"):
        verify(_overlay(tmp_path))


def test_verify_skips_the_chatml_assertion_when_not_required(
    tmp_path, stub_transformers
):
    # The pre-retag base student case. Same tokenizer, and now it must PASS.
    stub_transformers(chatml_ids=[27, 91, 318, 5011, 91, 29])
    lines = verify(_overlay(tmp_path), require_chatml=False)
    assert any("not required" in ln for ln in lines)


def test_verify_reports_single_id_control_tokens_when_present(
    tmp_path, stub_transformers
):
    stub_transformers(chatml_ids=[100256])
    lines = verify(_overlay(tmp_path))
    assert any("100256" in ln for ln in lines)


def test_verify_rejects_a_resolved_pre_tokenizer_that_is_not_the_one_on_disk(
    tmp_path, stub_transformers
):
    # The mechanism itself: tokenizer.json says Split+ByteLevel, transformers resolved
    # plain ByteLevel. This is what a surviving tokenizer_class looks like from inside.
    stub_transformers(chatml_ids=[100256], pre_tokenizer={"type": "ByteLevel"})
    on_disk = {
        "type": "Sequence",
        "pretokenizers": [{"type": "Split"}, {"type": "ByteLevel"}],
    }
    with pytest.raises(OverlayError, match="NOT the one in tokenizer.json"):
        verify(_overlay(tmp_path, pre_tokenizer=on_disk))


def test_verify_refuses_an_overlay_containing_a_sidecar(tmp_path, stub_transformers):
    stub_transformers(chatml_ids=[100256])
    d = _overlay(tmp_path)
    (d / "vocab.json").write_text("legacy")
    with pytest.raises(OverlayError, match="vocab.json"):
        verify(d)


def test_require_chatml_flag_is_exposed_and_defaults_to_true(tmp_path):
    # The step-template passes this flag unconditionally, so both spellings must parse.
    assert (
        build_parser().parse_args(["--source", "a", "--out", "b"]).require_chatml
        is True
    )
    assert (
        build_parser()
        .parse_args(["--source", "a", "--out", "b", "--no-require-chatml"])
        .require_chatml
        is False
    )


# ------------------------------------------------------ tokenizer identity
#
# These live in this step's suite because THIS step is what writes the identity:
# build_overlay records the source tokenizer's name and retag_student records the teacher's.
# distill-gold-train reads the same value through the same module and refuses a mismatch.
#
# The module exists at all because that guard was INERT: render_gold_config read
# `<model>/tokenizer_identity.json`, and nothing in the repo wrote it, so the read returned
# None and the comparison was skipped on every run it was meant to protect.

from gb_steps_post_training.distillation import tokenizer_identity as ti  # noqa: E402


def _tok(tmp_path, name="m", body='{"a": 1}'):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "tokenizer.json").write_text(body)
    return d


def test_recorded_name_wins_over_the_hash(tmp_path):
    d = _tok(tmp_path)
    ti.write(d, "granite-4.2-3b")
    assert ti.read(d) == "granite-4.2-3b"


def test_hash_is_the_fallback_when_nothing_recorded_an_identity(tmp_path):
    """Returning None here is what made the trainer's guard skippable for any tokenizer no
    step produced -- which includes every base model, i.e. every smoke run."""
    got = ti.read(_tok(tmp_path))
    assert got.startswith("sha256:") and len(got) == len("sha256:") + 16


def test_the_hash_fallback_actually_distinguishes_tokenizers(tmp_path):
    a = ti.read(_tok(tmp_path, "a", '{"vocab": 1}'))
    b = ti.read(_tok(tmp_path, "b", '{"vocab": 2}'))
    assert a != b


def test_the_hash_fallback_is_stable_for_identical_bytes(tmp_path):
    """Both sides of the comparison compute it independently; if it were not a pure function
    of the bytes, the guard would fire on tokenizers that are in fact the same."""
    assert ti.read(_tok(tmp_path, "a")) == ti.read(_tok(tmp_path, "b"))


def test_a_directory_with_no_tokenizer_is_None_not_an_error(tmp_path):
    """None means "not a tokenizer directory", which is a different thing from
    "unverifiable" and is the caller's business to report."""
    d = tmp_path / "empty"
    d.mkdir()
    assert ti.read(d) is None


def test_a_present_but_empty_identity_is_an_error_not_a_silent_hash(tmp_path):
    """Falling through to the hash here would compare two DIFFERENT identity schemes, which
    is how a guard starts comparing things that were never the same kind of thing."""
    d = _tok(tmp_path)
    (d / ti.IDENTITY_FILE).write_text('{"tokenizer_identity": ""}')
    with pytest.raises(ti.IdentityError, match="no non-empty"):
        ti.read(d)


def test_malformed_identity_json_names_the_file(tmp_path):
    d = _tok(tmp_path)
    (d / ti.IDENTITY_FILE).write_text("{not json")
    with pytest.raises(ti.IdentityError, match=ti.IDENTITY_FILE):
        ti.read(d)


def test_write_records_the_hash_alongside_for_provenance(tmp_path):
    d = _tok(tmp_path)
    ti.write(d, "some-name", produced_by="a-test")
    payload = json.loads((d / ti.IDENTITY_FILE).read_text())
    assert payload["tokenizer_identity"] == "some-name"
    assert payload["tokenizer_sha256"].startswith("sha256:")
    assert payload["produced_by"] == "a-test"


def test_write_refuses_an_empty_identity(tmp_path):
    with pytest.raises(ti.IdentityError):
        ti.write(_tok(tmp_path), "")


def test_derive_name_decodes_the_hf_cache_layout(tmp_path):
    """The default identity for a hub-resolved model. Without this it is a 40-char commit
    SHA, and a mismatch reads "8b6ac672... != a50b46ce..." -- measured, LSF job 1137785.
    """
    snap = (
        tmp_path
        / "hub"
        / "models--ibm-granite--granite-4.2-3b"
        / "snapshots"
        / "8b6ac672"
    )
    snap.mkdir(parents=True)
    assert ti.derive_name(snap) == "ibm-granite/granite-4.2-3b"


def test_derive_name_keeps_a_plain_directorys_basename(tmp_path):
    d = tmp_path / "retagged_student"
    d.mkdir()
    assert ti.derive_name(d) == "retagged_student"


def test_the_overlay_of_a_teacher_and_a_student_retagged_onto_it_agree(tmp_path):
    """The property that makes the guard correct rather than merely present: a retag copies
    the teacher's tokenizer.json byte-for-byte, so the teacher's overlay and the retagged
    student hold the SAME tokenizer and must compare EQUAL. If they did not, the trainer
    would refuse every correctly-wired run."""
    teacher = tmp_path / "hub" / "models--org--teacher" / "snapshots" / "abc"
    teacher.mkdir(parents=True)
    overlay, retagged = _tok(tmp_path, "overlay"), _tok(tmp_path, "retagged")
    ti.write(overlay, ti.derive_name(teacher))
    ti.write(retagged, ti.derive_name(teacher))
    assert ti.read(overlay) == ti.read(retagged) == "org/teacher"
