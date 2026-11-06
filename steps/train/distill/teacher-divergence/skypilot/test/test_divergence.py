"""
PORTED, not authored here. Upstream source of truth:
  repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
  path   steps/distill-eval/test/test_divergence.py
  commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579

Divergence, about path resolution rather than behaviour: upstream's sys.path.insert is
removed; conftest.py resolves both the step's own src/ and the upstream package from
GB_DISTILL_CODE_DIR.

Ten of these tests need torch and skip without it, which is upstream's design rather than
an accident: the metric arithmetic is worth testing against real tensors, and the image
that has torch is the cluster's, not CI's.

Keep this a near-verbatim copy so re-syncing upstream stays a three-way merge.
"""

"""Unit tests for the divergence module behind distill-eval.

WHAT IS TESTED HERE AND WHAT IS NOT, because the split is deliberate and the GPU smoke
(data/distillation/logs/divergence-smoke-exec.sh) is the other half. These tests cover the
decisions -- which records are measured, which pairings are refused, what the summary
reports -- on fakes, because those are exactly the paths where a real-model test tells you
only that the happy path works. They do NOT establish that the metric is right: two models
have to actually run for that, and three of this module's four real defects were found by
running them, not here.

torch is imported lazily by the functions that need it and is NOT a declared test
dependency, so those tests skip where it is absent (a laptop, `make test`) and run inside
the image, where the Dockerfile additionally asserts the ln2 bound at build time.
"""
import json
import math
import sys
from pathlib import Path

import pytest
from gb_steps_post_training.distillation import divergence as dv  # noqa: E402
from gb_steps_post_training.distillation import tokenizer_identity  # noqa: E402


# --------------------------------------------------------------------------------------
# A character-level tokenizer: one token per character, exact offsets. Chosen so the
# arithmetic under test is checkable by counting characters in the test itself -- with a
# real tokenizer the expected span is whatever the tokenizer says, which tests nothing.
# --------------------------------------------------------------------------------------
class CharTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        assert (
            tokenize is False
        ), "assistant_span must render to text, never tokenize=True"
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(
        self,
        text,
        truncation=False,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        ids = [ord(c) % 97 + 1 for c in text]
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        if return_tensors == "pt":
            import torch

            return {
                k: torch.tensor([v]) if k != "offset_mapping" else torch.tensor([v])
                for k, v in out.items()
            }
        return out


def _conv(user_len: int, answer_len: int) -> list[dict]:
    return [
        {"role": "user", "content": "u" * user_len},
        {"role": "assistant", "content": "a" * answer_len},
    ]


def _lengths(messages):
    tok = CharTokenizer()
    prompt = len(tok.apply_chat_template(messages[:-1], add_generation_prompt=True))
    full = len(tok.apply_chat_template(messages))
    return prompt, full


# --------------------------------------------------------------------------------------
# assistant_span: the three-way outcome that cost three GPU jobs to get straight.
# --------------------------------------------------------------------------------------
def test_span_is_complete_when_everything_fits():
    pytest.importorskip("torch")
    msgs = _conv(10, 40)
    prompt, full = _lengths(msgs)
    item = dv.assistant_span(msgs, CharTokenizer(), max_length=full + 100)
    assert item is not None
    assert item["truncated"] is False
    assert item["dropped_tokens"] == 0
    assert item["full_length"] == full
    # The mask covers exactly what the full rendering adds to the prompt.
    assert int(item["assistant_mask"].sum()) == full - prompt


def test_span_is_truncated_not_skipped_when_only_the_answer_overruns():
    """The case job 1138147 exposed, and the reason `truncated` exists at all.

    The prompt fits, so a record like this is NOT skipped -- it is measured on a prefix of
    its completion. A skip counter reports zero here and the run looks fully covered.
    """
    pytest.importorskip("torch")
    msgs = _conv(10, 400)
    prompt, full = _lengths(msgs)
    cut = prompt + 50
    assert cut < full
    item = dv.assistant_span(msgs, CharTokenizer(), max_length=cut)
    assert item is not None, "a fitting prompt must not be skipped"
    assert item["truncated"] is True
    assert item["dropped_tokens"] == full - cut
    assert int(item["input_ids"].shape[1]) == cut
    # 50 tokens of answer survived; the rest were never scored.
    assert int(item["assistant_mask"].sum()) == 50


def test_span_is_skipped_when_the_prompt_alone_fills_the_budget():
    pytest.importorskip("torch")
    msgs = _conv(400, 40)
    prompt, _ = _lengths(msgs)
    assert dv.assistant_span(msgs, CharTokenizer(), max_length=prompt) is None
    assert dv.assistant_span(msgs, CharTokenizer(), max_length=prompt - 10) is None


def test_span_requires_the_last_turn_to_be_the_assistants():
    pytest.importorskip("torch")
    msgs = [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "?"}]
    assert dv.assistant_span(msgs, CharTokenizer(), max_length=4096) is None
    assert dv.assistant_span([], CharTokenizer(), max_length=4096) is None


def test_span_falls_back_to_a_prompt_token_count_without_offsets():
    """The entropy copy in the scratchpad had ONLY this path; the jsd copy had both. Keeping
    the fallback working is what makes the two consolidated call sites measure one thing.
    """
    pytest.importorskip("torch")

    class NoOffsets(CharTokenizer):
        def __call__(self, text, **kwargs):
            kwargs["return_offsets_mapping"] = False
            return super().__call__(text, **kwargs)

    msgs = _conv(10, 40)
    prompt, full = _lengths(msgs)
    item = dv.assistant_span(msgs, NoOffsets(), max_length=full + 100)
    assert item is not None
    assert int(item["assistant_mask"].sum()) == full - prompt


# --------------------------------------------------------------------------------------
# reduce_logits: the arithmetic. The ln2 bound is the only assertion here that does not
# depend on knowing the answer in advance.
# --------------------------------------------------------------------------------------
def test_jsd_of_disjoint_distributions_is_ln2():
    torch = pytest.importorskip("torch")
    a = torch.tensor([[[20.0, -20.0]]])
    b = torch.tensor([[[-20.0, 20.0]]])
    assert abs(float(dv.reduce_logits(a, b, "jsd").mean()) - math.log(2)) < 1e-3


def test_jsd_of_identical_distributions_is_zero():
    torch = pytest.importorskip("torch")
    a = torch.tensor([[[1.0, 2.0, 3.0]]])
    assert float(dv.reduce_logits(a, a.clone(), "jsd").mean()) < 1e-6


def test_kld_and_rkld_are_the_two_directions():
    torch = pytest.importorskip("torch")
    # The two logit vectors must NOT be permutations of each other. This test first ran
    # with b = [0, 0, 3], which is [3, 0, 0] reordered, and it failed on the GPU (job
    # 1138506) asserting 0.0 > 1e-6 -- correctly, because for a permuted pair the forward
    # and reverse KL are equal by symmetry. The bug was the probe, not the reduction:
    # a pair like that cannot distinguish "both directions computed" from "one direction
    # computed twice", which is the whole point of the assertion below.
    a = torch.tensor([[[3.0, 0.0, 0.0]]])
    b = torch.tensor([[[0.0, 1.0, 2.0]]])
    kld = float(dv.reduce_logits(a, b, "kld").mean())
    rkld = float(dv.reduce_logits(a, b, "rkld").mean())
    # Not equal (1.905 vs 1.992 for this pair), and each is the other with the arguments
    # swapped. Equality would mean the direction never reached the reduction.
    assert abs(kld - rkld) > 1e-6
    assert abs(rkld - float(dv.reduce_logits(b, a, "kld").mean())) < 1e-5


def test_entropy_is_bounded_by_ln_vocab_and_zero_for_a_point_mass():
    torch = pytest.importorskip("torch")
    flat = torch.zeros(1, 1, 8)
    assert (
        abs(float(dv.reduce_logits(flat, None, "entropy").mean()) - math.log(8)) < 1e-5
    )
    peaked = torch.tensor([[[50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0]]])
    assert float(dv.reduce_logits(peaked, None, "entropy").mean()) < 1e-6


def test_unknown_metric_is_rejected():
    torch = pytest.importorskip("torch")
    a = torch.zeros(1, 1, 4)
    with pytest.raises((dv.MetricError, KeyError, ValueError)):
        dv.reduce_logits(a, a.clone(), "cosine")


# --------------------------------------------------------------------------------------
# check_tokenizers. The regression under test is the one this guard already had once, in
# both directions: a version that never fired, and a version that always did.
# --------------------------------------------------------------------------------------
def _model_dir(tmp_path, name, tokenizer_bytes):
    d = tmp_path / name
    d.mkdir()
    (d / "tokenizer.json").write_bytes(tokenizer_bytes)
    return d


def test_identical_tokenizer_json_is_accepted(tmp_path, monkeypatch):
    a = _model_dir(tmp_path, "student", b'{"model": "x"}')
    b = _model_dir(tmp_path, "teacher", b'{"model": "x"}')
    # read() is made to explode: this guard must not consult it. On the real pairing it
    # returns a NAME for the retagged student and a HASH for the teacher, which are never
    # equal, so a version built on it refused every correct pairing.
    monkeypatch.setattr(
        tokenizer_identity,
        "read",
        lambda *a, **k: pytest.fail("check_tokenizers must not use read()"),
    )
    result = dv.check_tokenizers(a, b)
    assert result["compared"] is True
    assert result["agreement"] == "identical_tokenizer_json"


def test_differing_bytes_but_agreeing_ids_is_recorded_not_refused(
    tmp_path, monkeypatch
):
    a = _model_dir(tmp_path, "student", b'{"model": "x"}')
    b = _model_dir(
        tmp_path, "teacher", b'{"model":  "x"}'
    )  # re-serialized, same behaviour
    monkeypatch.setattr(dv, "probe_disagreements", lambda *_: [])
    result = dv.check_tokenizers(a, b)
    assert result["agreement"] == "ids_agree_despite_differing_tokenizer_json"
    assert "note" in result


def test_disagreeing_ids_are_refused(tmp_path, monkeypatch):
    a = _model_dir(tmp_path, "student", b'{"model": "x"}')
    b = _model_dir(tmp_path, "teacher", b'{"model": "y"}')
    monkeypatch.setattr(
        dv,
        "probe_disagreements",
        lambda *_: [{"text": "<|im_end|>", "ids_a": [1], "ids_b": [2, 3]}],
    )
    with pytest.raises(dv.MetricError) as excinfo:
        dv.check_tokenizers(a, b)
    message = str(excinfo.value)
    assert (
        "--allow-tokenizer-mismatch" in message
    ), "the refusal must name its escape hatch"
    assert "<|im_end|>" in message, "the refusal must show WHICH text disagreed"


def test_allow_mismatch_computes_anyway_and_records_that_it_did(tmp_path, monkeypatch):
    """The case that could not live in the GPU harness. Testing the hatch needs two
    id-disagreeing models that BOTH render conversations; the real base student has no chat
    template, so fast_tokenizer refuses before the hatch is ever consulted (job 1138086).
    """
    a = _model_dir(tmp_path, "student", b'{"model": "x"}')
    b = _model_dir(tmp_path, "teacher", b'{"model": "y"}')
    monkeypatch.setattr(
        dv,
        "probe_disagreements",
        lambda *_: [{"text": "hi", "ids_a": [1], "ids_b": [2]}],
    )
    result = dv.check_tokenizers(a, b, allow_mismatch=True)
    assert result["agreement"] == "MISMATCH_ALLOWED"
    # The reason is carried into the manifest, so a number produced this way is not
    # indistinguishable later from one produced on an aligned pair.
    assert "do not agree on token ids" in result["note"]


def test_a_single_model_is_not_compared_against_anything(tmp_path):
    a = _model_dir(tmp_path, "student", b'{"model": "x"}')
    result = dv.check_tokenizers(a, None)
    assert result["compared"] is False
    assert "teacher_tokenizer" not in result


def test_a_directory_without_a_tokenizer_is_refused(tmp_path):
    empty = tmp_path / "not-a-model"
    empty.mkdir()
    with pytest.raises(dv.MetricError, match="tokenizer.json"):
        dv.check_tokenizers(empty, None)


def test_probe_strings_cover_the_boundaries_that_actually_broke(tmp_path):
    """PROBE_STRINGS is the behavioural fallback's entire resolution. If the ChatML control
    tokens leave it, the probe stops testing the one difference the retag exists to create
    and would report agreement on a pairing the metric cannot use."""
    joined = "".join(dv.PROBE_STRINGS)
    assert "<|im_start|>" in joined and "<|im_end|>" in joined
    assert any(
        s != s.strip() or "  " in s for s in dv.PROBE_STRINGS
    ), "whitespace probe"
    assert any(
        any(c.isdigit() for c in s) for s in dv.PROBE_STRINGS
    ), "digit-split probe"
    assert any("{" in s for s in dv.PROBE_STRINGS), "tool-call JSON probe"


# --------------------------------------------------------------------------------------
# load_jsonl / summarise
# --------------------------------------------------------------------------------------
def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        '{"messages": [{"role": "user", "content": "a"}]}\n\n'
        '{"messages": [{"role": "user", "content": "b"}]}\n'
    )
    assert len(dv.load_jsonl(p)) == 2


def test_load_jsonl_reports_the_offending_line(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text('{"messages": []}\n{not json}\n')
    with pytest.raises((dv.MetricError, json.JSONDecodeError)) as excinfo:
        dv.load_jsonl(p)
    assert "2" in str(excinfo.value), "a parse error must name the line number"


def test_summarise_reports_n_and_bounds():
    pytest.importorskip("numpy")
    s = dv.summarise([0.0, 1.0, 2.0, 3.0], [10, 10, 10, 10])
    assert s["n_samples"] == 4
    assert s["mean"] == pytest.approx(1.5)
    assert s["median"] == pytest.approx(1.5)
    assert s["min"] == 0.0 and s["max"] == 3.0


def test_summarise_refuses_an_empty_sample():
    pytest.importorskip("numpy")
    with pytest.raises(dv.MetricError):
        dv.summarise([], [])
