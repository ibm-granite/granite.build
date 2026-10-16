"""
PORTED, not authored here. Upstream source of truth:
  repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
  path   steps/distill-hf-export/test/test_export_hf_model.py
  commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579

Four divergences. The first is behavioural, the rest are about path resolution:
  - the `normalise_tokenizer_json` tests (and `_ckpt`'s `tokenizer_json` argument) cover a
    divergence this repo carries: clearing the trainer's live truncation state out of the
    published tokenizer.json. See the DIVERGENCE note in src/export_hf_model.py. Drop
    these when the fix lands upstream.
  - upstream's sys.path.insert is removed; conftest.py resolves both the step's own src/
    and the upstream package (from GB_DISTILL_CODE_DIR). Upstream's version inserted only
    parents[1]/src while export_hf_model imports the shared package at MODULE scope, so
    this suite ERRORED ON COLLECTION when run alone and passed only when a sibling suite
    had already patched sys.path. The conftest fixes that.
  - REAL_TEMPLATE now resolves to test-data/, where the real chat template is vendored.
    Upstream reached for parents[3]/templates/, which does not exist from inside
    granite.build. The template is vendored rather than stubbed because these tests assert
    against the REAL one on purpose — a stub would assert nothing about the property the
    thinking-policy rewrite has to preserve.

Keep this a near-verbatim copy so re-syncing upstream stays a three-way merge.
"""

"""Unit tests for distill-hf-export.

These import only stdlib + the module under test: `verify()` defers its transformers
import into the function body precisely so select/classify/normalise/export stay testable
on a CPU box with nothing installed. The transformers-dependent path is covered by the
real-checkpoint run recorded in README.md, not here.
"""

import json
import sys
from pathlib import Path

import pytest
from export_hf_model import (  # noqa: E402
    THINKING_POLICIES,
    ExportError,
    assert_config_portable,
    classify,
    diff_resolved_numeric,
    export,
    normalise_chat_template,
    normalise_model_config,
    normalise_tokenizer_config,
    normalise_tokenizer_json,
    published_config_needs_repair,
    select_checkpoint,
    tokenizer_pin_problems,
)

# The thinking-policy tests run against the REAL template, not a stub. `templates/
# chatml_granite_42_generation.jinja` is byte-identical (md5 40810c08) to the
# chat_template.jinja every granite-4.x checkpoint in data/distillation carries, so a test
# that passes here is a statement about the file the export will actually rewrite. A
# hand-written stub would let the five-site structure drift out from under the code that
# reasons about it and the tests would stay green.
REAL_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "test-data"
    / "chatml_granite_42_generation.jinja"
)


def _ckpt(
    root: Path,
    step: int,
    *,
    model_config=None,
    tokenizer_config=None,
    tokenizer_json=None,
    chat_template=None,
    extra=(),
    weights=True,
) -> Path:
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "config.json").write_text(
        "{}" if model_config is None else json.dumps(model_config, indent=2) + "\n"
    )
    if weights:
        (d / "model.safetensors").write_text("w")
    (d / "tokenizer.json").write_text(
        "{}" if tokenizer_json is None else json.dumps(tokenizer_json)
    )
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            tokenizer_config
            if tokenizer_config is not None
            else {"padding_side": "left", "is_local": True, "local_files_only": True}
        )
    )
    if chat_template is not None:
        (d / "chat_template.jinja").write_text(chat_template)
    (d / "trainer_state.json").write_text("{}")
    (d / "scheduler.pt").write_text("s")
    (d / "latest").write_text(f"global_step{step}")
    (d / f"global_step{step}").mkdir()
    for name in extra:
        if name.endswith("/"):
            (d / name.rstrip("/")).mkdir()
        else:
            (d / name).write_text("x")
    return d


# ------------------------------------------------------------------ select


def test_select_latest_uses_step_number_not_mtime(tmp_path):
    # A resumed run rewrites older checkpoints' mtimes, so mtime ordering can pick a
    # checkpoint that is not the furthest along. Make checkpoint-9 the NEWEST on disk and
    # assert checkpoint-100 still wins.
    _ckpt(tmp_path, 100)
    newer = _ckpt(tmp_path, 9)
    import os
    import time

    future = time.time() + 60
    os.utime(newer, (future, future))
    assert select_checkpoint(tmp_path).name == "checkpoint-100"


def test_select_explicit_name(tmp_path):
    _ckpt(tmp_path, 25)
    _ckpt(tmp_path, 50)
    assert select_checkpoint(tmp_path, "checkpoint-25").name == "checkpoint-25"


def test_select_missing_explicit_is_refused(tmp_path):
    _ckpt(tmp_path, 25)
    with pytest.raises(ExportError, match="does not exist"):
        select_checkpoint(tmp_path, "checkpoint-999")


def test_select_no_checkpoints_is_refused(tmp_path):
    with pytest.raises(ExportError, match="no checkpoint-"):
        select_checkpoint(tmp_path)


def test_select_non_numeric_checkpoint_is_refused(tmp_path):
    _ckpt(tmp_path, 25)
    (tmp_path / "checkpoint-best").mkdir()
    # Silently ignoring it could publish an older checkpoint than the operator meant.
    with pytest.raises(ExportError, match="cannot order"):
        select_checkpoint(tmp_path)


# ------------------------------------------------------------------ classify


def test_classify_splits_a_real_shaped_checkpoint(tmp_path):
    parts = classify(_ckpt(tmp_path, 25))
    assert "model.safetensors" in parts["keep"]
    assert "tokenizer_config.json" in parts["keep"]
    assert "global_step25" in parts["prune"]
    assert "trainer_state.json" in parts["prune"]
    assert parts["unknown"] == []


def test_classify_keeps_sharded_weights(tmp_path):
    d = _ckpt(
        tmp_path,
        25,
        weights=False,
        extra=(
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
        ),
    )
    parts = classify(d)
    assert "model-00001-of-00002.safetensors" in parts["keep"]
    assert "model.safetensors.index.json" in parts["keep"]
    assert parts["unknown"] == []


def test_classify_surfaces_unknown_entries(tmp_path):
    parts = classify(_ckpt(tmp_path, 25, extra=("mystery.bin", "surprise_dir/")))
    assert set(parts["unknown"]) == {"mystery.bin", "surprise_dir"}


# ------------------------------------------------------------------ normalise


def test_normalise_strips_load_kwargs_and_fixes_padding():
    raw = {
        "padding_side": "left",
        "is_local": True,
        "local_files_only": True,
        "eos_token": "<|im_end|>",
    }
    out, changes = normalise_tokenizer_config(raw)
    assert out["padding_side"] == "right"
    assert "is_local" not in out and "local_files_only" not in out
    assert out["eos_token"] == "<|im_end|>", "unrelated keys must survive untouched"
    assert len(changes) == 3


def test_normalise_rewrites_a_transformers_5_only_tokenizer_class():
    """A v5-authored tokenizer_config pins a class no v4 consumer can resolve.

    transformers 5 records `tokenizer_class: TokenizersBackend`, and a consumer on
    transformers 4 raises `ValueError: Tokenizer class TokenizersBackend does not
    exist or is not currently imported` before reading a single token. The
    distillation steps run transformers 5.8.0, so EVERY model this step publishes
    carries that pin — measured in build 30a99c4b, where bfcl-eval (a transformers
    4 image) died at base_oss_handler.py:109 on the exported model.

    PreTrainedTokenizerFast exists in both generations and loads tokenizer.json
    directly, so it is the portable spelling of the same tokenizer.
    """
    out, changes = normalise_tokenizer_config(
        {"tokenizer_class": "TokenizersBackend"}, padding_side="keep"
    )
    assert out["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert any(
        "tokenizer_class" in c for c in changes
    ), "the rewrite must be recorded in the manifest, like every other normalisation"


def test_normalise_leaves_any_other_tokenizer_class_alone():
    """Only the v5 backend name is rewritten. Guessing at an unfamiliar class would
    change which tokenizer a consumer instantiates, which is exactly the kind of
    silent substitution this step refuses elsewhere."""
    for name in ("PreTrainedTokenizerFast", "GPT2Tokenizer", "LlamaTokenizerFast"):
        out, changes = normalise_tokenizer_config(
            {"tokenizer_class": name}, padding_side="keep"
        )
        assert out["tokenizer_class"] == name
        assert changes == []


def test_normalise_does_not_invent_a_tokenizer_class():
    """Absent is not the same as wrong: with no tokenizer_class, transformers infers
    one from config.json, and adding a pin here would override that inference."""
    out, changes = normalise_tokenizer_config({"eos_token": "x"}, padding_side="keep")
    assert "tokenizer_class" not in out
    assert changes == []


def test_normalise_does_not_mutate_its_input():
    raw = {"padding_side": "left", "is_local": True}
    normalise_tokenizer_config(raw)
    assert raw == {"padding_side": "left", "is_local": True}


def test_normalise_keep_preserves_padding_side():
    out, changes = normalise_tokenizer_config(
        {"padding_side": "left"}, padding_side="keep"
    )
    assert out["padding_side"] == "left"
    assert changes == []


def test_normalise_reports_no_change_when_already_right():
    out, changes = normalise_tokenizer_config({"padding_side": "right"})
    assert out["padding_side"] == "right"
    assert changes == [], "a no-op must not be reported as a normalisation"


def test_normalise_rejects_bad_padding_side():
    with pytest.raises(ExportError, match="padding_side"):
        normalise_tokenizer_config({}, padding_side="sideways")


def test_normalise_adds_padding_side_when_absent():
    out, changes = normalise_tokenizer_config({"eos_token": "x"})
    assert out["padding_side"] == "right"
    assert len(changes) == 1


# ------------------------------------------------- normalise tokenizer.json


# A trained tokenizer.json carries `truncation` and `padding` as explicit nulls. The
# trainer's in-process tokenizer had truncation switched on, and `save_pretrained`
# serialises that RUNTIME state into the file, so the published model ships a budget
# nobody asked for. Observed on build df8512e0: max_length 466 in the exported
# tokenizer.json where align's was null.
_TRAINER_TRUNCATION = {
    "direction": "Right",
    "max_length": 466,
    "strategy": "LongestFirst",
    "stride": 0,
}


def test_normalise_tokenizer_json_clears_stale_truncation():
    out, changes = normalise_tokenizer_json({"truncation": dict(_TRAINER_TRUNCATION)})
    assert out["truncation"] is None
    assert len(changes) == 1
    assert "466" in changes[0], "the manifest must name the budget it removed"


def test_normalise_tokenizer_json_clears_stale_padding():
    padding = {"strategy": {"Fixed": 8}, "direction": "Right", "pad_id": 0}
    out, changes = normalise_tokenizer_json({"padding": padding})
    assert out["padding"] is None
    assert len(changes) == 1


def test_normalise_tokenizer_json_reports_no_change_when_already_clean():
    raw = {"truncation": None, "padding": None, "model": {"vocab": {}}}
    out, changes = normalise_tokenizer_json(raw)
    assert changes == [], "a no-op must not be reported as a normalisation"
    assert out == raw


def test_normalise_tokenizer_json_leaves_post_processor_alone():
    """The trainer also writes an identity TemplateProcessing where align had null.

    It adds no special tokens, so it changes nothing -- and a post_processor decides
    which specials get inserted at encode time. Rewriting one would be exactly the
    silent substitution this step refuses everywhere else.
    """
    pp = {"type": "TemplateProcessing", "single": [], "pair": [], "special_tokens": {}}
    out, changes = normalise_tokenizer_json({"post_processor": pp})
    assert out["post_processor"] == pp
    assert changes == []


def test_normalise_tokenizer_json_does_not_mutate_its_input():
    raw = {"truncation": dict(_TRAINER_TRUNCATION)}
    normalise_tokenizer_json(raw)
    assert raw["truncation"] == _TRAINER_TRUNCATION


def test_normalise_tokenizer_json_preserves_the_vocabulary():
    raw = {
        "truncation": dict(_TRAINER_TRUNCATION),
        "model": {"vocab": {"a": 0, "b": 1}, "merges": ["a b"]},
        "pre_tokenizer": {"type": "Sequence"},
        "added_tokens": [{"id": 0, "content": "a"}],
    }
    out, _ = normalise_tokenizer_json(raw)
    for key in ("model", "pre_tokenizer", "added_tokens"):
        assert out[key] == raw[key], f"{key} must survive untouched"


# ------------------------------------------------- normalise config.json
#
# These cover the RoPE relocation bug: transformers 5.8.0 writes
# `rope_parameters: {rope_theta: 10000000}` and no top-level `rope_theta`, so every
# transformers-4-schema consumer (vLLM 0.11.0, bfcl-eval) silently resolves the class
# default 10000.0 and serves a positionally scrambled model. Measured on both sides:
#
#   config                                    transformers 4.55.4 resolves rope_theta
#   align/retagged_student (4.57.6-authored)  10000000     <- healthy
#   export-N (5.8.0-authored)                 10000.0      <- 1000x too small
#   export-N + top-level rope_theta hoisted   10000000     <- the repair
#
# and under 5.8.0 a hoisted top-level `rope_theta` is inert: the attribute does not exist
# on that generation's config at all, so adding it changes nothing there.


def test_hoists_nested_rope_theta_to_top_level(tmp_path):
    """The bug, and the fix. A transformers-4 consumer reads the top-level field."""
    raw = {"rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}}
    out, changes = normalise_model_config(raw)
    assert out["rope_theta"] == 10000000
    assert any("rope_theta" in c for c in changes)


def test_hoisting_keeps_the_nested_dict(tmp_path):
    """Hoist, not move: transformers 5 reads the nested dict and must stay served."""
    raw = {"rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}}
    out, _ = normalise_model_config(raw)
    assert out["rope_parameters"] == {"rope_theta": 10000000, "rope_type": "default"}


def test_hoists_every_numeric_scalar_not_just_rope_theta():
    """The generalisation the fix exists for. A hard-coded list of one protects exactly
    one field and the next transformers reorganisation lands the same bug silently."""
    raw = {"rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.5}}
    out, _ = normalise_model_config(raw)
    assert out["rope_theta"] == 10000000
    assert out["partial_rotary_factor"] == 0.5


def test_does_not_hoist_non_numeric_scalars():
    """`rope_type: "default"` is not a transformers-4 top-level field; inventing one
    would change which RoPE implementation a v4 consumer selects."""
    raw = {"rope_parameters": {"rope_theta": 1e7, "rope_type": "default"}}
    out, _ = normalise_model_config(raw)
    assert "rope_type" not in out


def test_leaves_an_already_correct_config_alone():
    """align/retagged_student is written under 4.57.6 and is already right. Idempotent:
    re-running the export over a repaired dir must report no change."""
    raw = {"rope_theta": 10000000, "use_cache": True}
    out, changes = normalise_model_config(raw)
    assert out == raw
    assert changes == []


def test_refuses_when_top_level_disagrees_with_nested():
    """Two live values for one field. Picking one silently would publish a model served
    at a RoPE base the weights never saw, which is the whole defect being fixed."""
    raw = {"rope_theta": 10000.0, "rope_parameters": {"rope_theta": 10000000}}
    with pytest.raises(ExportError, match="rope_theta"):
        normalise_model_config(raw)


def test_restores_use_cache(tmp_path):
    """The trainer sets use_cache=False for gradient checkpointing; it is wrong to ship
    on an inference artifact."""
    raw = {"use_cache": False}
    out, changes = normalise_model_config(raw)
    assert out["use_cache"] is True
    assert any("use_cache" in c for c in changes)


def test_drops_a_non_finite_float_sentinel():
    """transformers 5 spells infinity as {"__float__": "Infinity"}, which is not JSON any
    other consumer parses into a float -- transformers 4.55.4 hands back a dict where a
    float belongs. Infinity has no strict-JSON spelling, so the portable move is to drop
    the key and let the consumer's own class default apply; verify() then asserts the
    resolved value did not change."""
    raw = {"time_step_limit": [0.0, {"__float__": "Infinity"}]}
    out, changes = normalise_model_config(raw)
    assert "time_step_limit" not in out
    assert any("time_step_limit" in c for c in changes)


def test_keeps_plain_float_neighbours_of_a_dropped_sentinel():
    """time_step_min/max are plain portable floats equal to the class default on both
    generations. Only the unrepresentable one goes."""
    raw = {
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
        "time_step_min": 0.001,
        "time_step_max": 0.1,
    }
    out, _ = normalise_model_config(raw)
    assert out["time_step_min"] == 0.001
    assert out["time_step_max"] == 0.1


# ---------------------------------------- the portability assertion


def test_portability_assertion_catches_an_unmirrored_nested_numeric():
    """THE PART THAT MATTERS. A resolved-vs-source comparison under one transformers
    cannot catch this bug -- 5.8.0 reads the nested dict and resolves both sides to
    10000000 either way. The invariant that does catch it is checkable in pure JSON:
    every numeric scalar in a nested parameters dict must be mirrored top-level."""
    with pytest.raises(ExportError, match="rope_theta"):
        assert_config_portable({"rope_parameters": {"rope_theta": 10000000}})


def test_portability_assertion_accepts_a_normalised_config():
    cfg, _ = normalise_model_config(
        {"rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}}
    )
    assert_config_portable(cfg)  # must not raise


def test_portability_assertion_rejects_a_surviving_sentinel():
    with pytest.raises(ExportError, match="__float__"):
        assert_config_portable({"time_step_limit": [0.0, {"__float__": "Infinity"}]})


def test_export_normalises_the_model_config_on_disk(tmp_path):
    """End to end: the written config.json is the repaired one, and the manifest says so."""
    src = _ckpt(
        tmp_path,
        7,
        model_config={
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
            "use_cache": False,
            "time_step_limit": [0.0, {"__float__": "Infinity"}],
        },
    )
    manifest = export(src, tmp_path / "out")
    cfg = json.loads((tmp_path / "out" / "config.json").read_text())
    assert cfg["rope_theta"] == 10000000
    assert cfg["use_cache"] is True
    assert "time_step_limit" not in cfg
    assert any("config.json" in c for c in manifest["normalisations"])
    # And the trainer's own config must be untouched -- the run stays resumable.
    assert "rope_theta" not in json.loads((src / "config.json").read_text())


def test_export_leaves_a_clean_config_byte_identical(tmp_path):
    """Same discipline as the tokenizer.json and chat-template paths: a rewritten but
    unchanged file would make `diff -r` against the checkpoint report a change this step
    did not make."""
    clean = {"rope_theta": 10000000, "use_cache": True}
    src = _ckpt(tmp_path, 8, model_config=clean)
    export(src, tmp_path / "out")
    assert (tmp_path / "out" / "config.json").read_bytes() == (
        src / "config.json"
    ).read_bytes()


# ------------------------------------------------------------------ export


def test_export_prunes_and_writes_manifest(tmp_path):
    src = _ckpt(tmp_path / "run", 25)
    dest = tmp_path / "out"
    manifest = export(src, dest)
    assert (dest / "model.safetensors").exists()
    assert not (dest / "global_step25").exists()
    assert not (dest / "trainer_state.json").exists()
    assert json.loads((dest / "export_manifest.json").read_text()) == manifest
    assert manifest["source_checkpoint"] == str(src)


def test_export_normalises_the_tokenizer_config_on_disk(tmp_path):
    dest = tmp_path / "out"
    export(_ckpt(tmp_path / "run", 25), dest)
    cfg = json.loads((dest / "tokenizer_config.json").read_text())
    assert cfg["padding_side"] == "right"
    assert "is_local" not in cfg and "local_files_only" not in cfg


def test_export_clears_runtime_truncation_and_records_it(tmp_path):
    src = _ckpt(
        tmp_path / "run",
        25,
        tokenizer_json={
            "truncation": dict(_TRAINER_TRUNCATION),
            "model": {"vocab": {}},
        },
    )
    dest = tmp_path / "out"
    manifest = export(src, dest)
    published = json.loads((dest / "tokenizer.json").read_text())
    assert published["truncation"] is None
    assert published["model"] == {"vocab": {}}
    assert any(
        "tokenizer.json" in c and "truncation" in c for c in manifest["normalisations"]
    )


def test_export_keeps_a_clean_tokenizer_json_byte_identical(tmp_path):
    """7 MB of vocabulary: rewriting it when nothing needs changing would reformat the
    whole file and make `diff -r` against the checkpoint report a change this step did
    not make. Same discipline as the chat template."""
    body = json.dumps({"truncation": None, "model": {"vocab": {"a": 0}}})
    src = _ckpt(tmp_path / "run", 25, tokenizer_json=json.loads(body))
    dest = tmp_path / "out"
    manifest = export(src, dest)
    assert (dest / "tokenizer.json").read_bytes() == (
        src / "tokenizer.json"
    ).read_bytes()
    assert not any("tokenizer.json" in c for c in manifest["normalisations"])


def test_export_refuses_unknown_entries_by_default(tmp_path):
    src = _ckpt(tmp_path / "run", 25, extra=("mystery.bin",))
    with pytest.raises(ExportError, match="unrecognised"):
        export(src, tmp_path / "out")


def test_export_allow_unknown_drops_and_records_them(tmp_path):
    src = _ckpt(tmp_path / "run", 25, extra=("mystery.bin",))
    dest = tmp_path / "out"
    manifest = export(src, dest, allow_unknown=True)
    assert not (dest / "mystery.bin").exists()
    assert manifest["dropped_unrecognised"] == ["mystery.bin"]


def test_export_refuses_a_checkpoint_with_no_hf_weights(tmp_path):
    # This is the one case where the step would genuinely need a converter: weights only
    # inside global_step*/, i.e. zero3_save_16bit_model was not set. It must say so
    # rather than publish a directory with no model in it.
    src = _ckpt(tmp_path / "run", 25, weights=False)
    with pytest.raises(ExportError, match="zero3_save_16bit_model"):
        export(src, tmp_path / "out")


def test_export_does_not_modify_the_source_checkpoint(tmp_path):
    src = _ckpt(tmp_path / "run", 25)
    before = sorted(p.name for p in src.iterdir())
    export(src, tmp_path / "out")
    assert sorted(p.name for p in src.iterdir()) == before
    # And the trainer's own tokenizer config must still say left -- the run stays resumable.
    assert (
        json.loads((src / "tokenizer_config.json").read_text())["padding_side"]
        == "left"
    )


# ------------------------------------------------------- chat template / thinking


def test_the_real_template_still_has_the_shape_this_code_reasons_about():
    """A premise check, not a behaviour test.

    Every claim in THINKING_POLICIES is about a specific structure in this file: one
    `enable_thinking` default, one `truncate_history_thinking` default, and five literal
    `<think></think>` sites of two different kinds. If a template bump changes any of those
    counts, the block comment becomes fiction and `default-off` may no longer be the right
    or the only edit. This fails first, and loudly, so the reasoning gets re-derived instead
    of the tests below quietly testing a template nobody has read.
    """
    text = REAL_TEMPLATE.read_text()
    assert text.count("<think></think>") == 5, (
        "the five sites THINKING_POLICIES enumerates (90 injector, 113 canonicaliser, "
        "119 injector, 147 canonicaliser, 193 generation prompt) are no longer five"
    )
    on = [
        ln
        for ln in text.splitlines()
        if ln.strip().startswith("{%- set enable_thinking =")
    ]
    assert len(on) == 1 and "else True" in on[0]
    trunc = [
        ln
        for ln in text.splitlines()
        if ln.strip().startswith("{%- set truncate_history_thinking =")
    ]
    assert len(trunc) == 1 and "else True" in trunc[0], (
        "truncate_history_thinking no longer defaults True, so sites 113/147 no longer fire "
        "on their own -- the argument for why deleting site 90 alone is inconsistent needs "
        "re-checking"
    )


def test_keep_is_byte_identical():
    text = REAL_TEMPLATE.read_text()
    out, changes = normalise_chat_template(text, chat_template_thinking="keep")
    assert out == text
    assert changes == []


def test_default_off_flips_exactly_one_line():
    text = REAL_TEMPLATE.read_text()
    out, changes = normalise_chat_template(text, chat_template_thinking="default-off")
    assert out != text
    assert len(changes) == 1 and "True -> False" in changes[0]
    # Exactly one line differs, and it is the default line.
    diff = [(a, b) for a, b in zip(text.splitlines(), out.splitlines()) if a != b]
    assert len(diff) == 1
    assert "else True" in diff[0][0] and "else False" in diff[0][1]
    # And nothing else moved: same line count, same number of empty think blocks.
    assert len(out.splitlines()) == len(text.splitlines())
    assert out.count("<think></think>") == text.count("<think></think>") == 5
    # The consequence the change is FOR: line 190's `{%- if enable_thinking %}` now takes the
    # else branch, so a caller that says nothing gets the non-reasoning generation prompt.
    assert "{%- if enable_thinking %}" in out


def test_default_off_is_idempotent_and_says_it_changed_nothing():
    once, _ = normalise_chat_template(
        REAL_TEMPLATE.read_text(), chat_template_thinking="default-off"
    )
    twice, changes = normalise_chat_template(once, chat_template_thinking="default-off")
    assert twice == once
    assert len(changes) == 1 and "already defaults to False" in changes[0]


def test_default_off_preserves_indentation_and_line_ending():
    src = "prefix\r\n    {%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}\r\nsuffix\r\n"
    out, _ = normalise_chat_template(src, chat_template_thinking="default-off")
    assert out.startswith("prefix\r\n    {%- set enable_thinking")
    assert out.endswith("%}\r\nsuffix\r\n")
    assert "\r\n" in out.splitlines(keepends=True)[1]


def test_missing_default_line_is_refused_not_guessed():
    with pytest.raises(ExportError, match="exactly one `enable_thinking` default line"):
        normalise_chat_template(
            "{{ 'no thinking default here' }}", chat_template_thinking="default-off"
        )


def test_duplicated_default_line_is_refused():
    ln = "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}"
    with pytest.raises(ExportError, match="found 2 defaulting True"):
        normalise_chat_template(f"{ln}\n{ln}\n", chat_template_thinking="default-off")


def test_unknown_policy_is_refused():
    with pytest.raises(ExportError, match="not one of"):
        normalise_chat_template("x", chat_template_thinking="strip-injection")
    assert "strip-injection" not in THINKING_POLICIES


def test_export_keeps_the_template_byte_identical_by_default(tmp_path):
    text = REAL_TEMPLATE.read_text()
    src = _ckpt(tmp_path / "run", 25, chat_template=text)
    man = export(src, tmp_path / "out")
    assert (tmp_path / "out" / "chat_template.jinja").read_text() == text
    assert man["chat_template_thinking_policy"] == "keep"
    assert not any("chat_template" in c for c in man["normalisations"])


def test_export_applies_default_off_and_records_it(tmp_path):
    text = REAL_TEMPLATE.read_text()
    src = _ckpt(tmp_path / "run", 25, chat_template=text)
    man = export(src, tmp_path / "out", chat_template_thinking="default-off")
    out = (tmp_path / "out" / "chat_template.jinja").read_text()
    assert "else False %}" in out
    assert out.count("<think></think>") == 5
    assert man["chat_template_thinking_policy"] == "default-off"
    assert any(
        c.startswith("chat_template.jinja: enable_thinking default True -> False")
        for c in man["normalisations"]
    ), man["normalisations"]
    # The source checkpoint is untouched: the run stays resumable and reproducible.
    assert (src / "chat_template.jinja").read_text() == text


def test_export_without_a_template_is_not_an_error(tmp_path):
    # chat_template.jinja is in KEEP_FILES but is not universal -- a checkpoint from a model
    # whose template lives inside tokenizer_config.json has none, and that is not a defect.
    src = _ckpt(tmp_path / "run", 25, chat_template=None)
    man = export(src, tmp_path / "out", chat_template_thinking="default-off")
    assert not (tmp_path / "out" / "chat_template.jinja").exists()
    assert man["chat_template_thinking_policy"] == "default-off"


# ------------------------------------- resolved-value drift (the second half of the guard)
#
# Kept as a pure function over two attribute dicts so it is testable with no transformers
# installed, which is the property the whole module is arranged around. `verify()` supplies
# the real `AutoConfig.to_dict()` pair.


def test_resolved_drift_accepts_a_key_dropped_to_its_class_default():
    """The sentinel drop is only safe because absence resolves to the same value. This is
    what checks that per export instead of trusting a comment: `time_step_limit` is gone
    from the written file, and both sides still RESOLVE (0.0, inf)."""
    inf = float("inf")
    assert (
        diff_resolved_numeric(
            {
                "rope_theta": 10000000,
                "time_step_min": 0.001,
                "time_step_limit": (0.0, inf),
            },
            {
                "rope_theta": 10000000,
                "time_step_min": 0.001,
                "time_step_limit": [0.0, inf],
            },
        )
        == []
    )


def test_resolved_drift_compares_numeric_sequences():
    """`time_step_limit` is the one key the export drops, and it resolves to a PAIR, so a
    scalar-only comparison skipped exactly the key it exists to check."""
    drift = diff_resolved_numeric(
        {"time_step_limit": (0.0, float("inf"))}, {"time_step_limit": [0.0, 10.0]}
    )
    assert len(drift) == 1
    assert "time_step_limit" in drift[0]
    drift = diff_resolved_numeric({"time_step_limit": (0.0, float("inf"))}, {})
    assert len(drift) == 1


def test_resolved_drift_catches_a_dropped_key_that_was_not_default():
    """The failure this guards: dropping or rewriting a key whose value was NOT the class
    default silently changes the published model."""
    drift = diff_resolved_numeric({"time_step_min": 0.005}, {"time_step_min": 0.001})
    assert len(drift) == 1
    assert "time_step_min" in drift[0]


def test_resolved_drift_ignores_non_numeric_and_new_keys():
    """Only numbers, and only fields the SOURCE had: this step adds top-level keys on
    purpose (the hoist), and that is not drift."""
    assert (
        diff_resolved_numeric(
            {"model_type": "granitemoehybrid", "use_cache": False},
            {
                "model_type": "granitemoehybrid",
                "use_cache": True,
                "rope_theta": 10000000,
            },
        )
        == []
    )


# --------------------------------------- the resume gate vs a changed export contract
#
# `export_state.expectation()` keys a skip on the CHECKPOINT and the policy flags, and has
# no component for the export's own normalisation contract. So when that contract gains a
# rule -- as it just did for config.json -- every already-published export still hashes
# equal and the resume gate SKIPs it, leaving the broken config in place forever. This is
# exactly how 38 export directories would have survived their own repair.


def test_published_config_needs_repair_flags_an_unmirrored_config(tmp_path):
    dest = tmp_path / "published"
    dest.mkdir()
    (dest / "config.json").write_text(
        json.dumps({"rope_parameters": {"rope_theta": 10000000}})
    )
    reason = published_config_needs_repair(dest)
    assert reason is not None and "rope_theta" in reason


def test_published_config_needs_repair_passes_a_repaired_config(tmp_path):
    dest = tmp_path / "published"
    dest.mkdir()
    (dest / "config.json").write_text(
        json.dumps(
            {"rope_theta": 10000000, "rope_parameters": {"rope_theta": 10000000}}
        )
    )
    assert published_config_needs_repair(dest) is None


def test_published_config_needs_repair_is_quiet_when_there_is_nothing_published(
    tmp_path,
):
    """No dest yet, or a dest with no config: not this function's business to complain --
    the resume gate is already going to run the export."""
    assert published_config_needs_repair(tmp_path / "nope") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert published_config_needs_repair(empty) is None


def test_published_config_needs_repair_survives_unreadable_json(tmp_path):
    """A truncated config must read as 'needs repair', never as an uncaught crash in the
    resume gate."""
    dest = tmp_path / "published"
    dest.mkdir()
    (dest / "config.json").write_text("{not json")
    assert published_config_needs_repair(dest) is not None


# ------------------------------------- tokenizer pin (the consumer's-eye segmentation check)
#
# Pure over two encoders for the same reason as diff_resolved_numeric: `verify()` supplies
# AutoTokenizer and tokenizers.Tokenizer, and this file must run with neither installed.


def _ids(text):
    return [ord(c) for c in text]


def test_tokenizer_pin_passes_when_both_encoders_agree():
    assert tokenizer_pin_problems("PreTrainedTokenizerFast", True, _ids, _ids) == []


def test_tokenizer_pin_catches_a_gpt2_class_that_ignored_the_pin():
    """The measured failure: AutoTokenizer honours `GPT2Tokenizer` and drops the trained
    Split. The class name alone is enough to refuse."""
    problems = tokenizer_pin_problems("GPT2TokenizerFast", True, _ids, _ids)
    assert len(problems) == 1
    assert "pin did not take" in problems[0]


def test_tokenizer_pin_catches_a_slow_tokenizer():
    problems = tokenizer_pin_problems("LlamaTokenizer", False, _ids, _ids)
    assert len(problems) == 1
    assert "slow" in problems[0]


def test_tokenizer_pin_catches_a_segmentation_that_differs():
    """A class name is a label. This is the property that matters: a probe whose ids
    differ from tokenizer.json's is reported, whatever the class is called."""

    def split_on_spaces(text):
        return [len(w) for w in text.split(" ")]

    problems = tokenizer_pin_problems(
        "PreTrainedTokenizerFast", True, split_on_spaces, _ids, probes=["a b", "ab"]
    )
    assert len(problems) == 2
    assert "'a b'" in problems[0]
