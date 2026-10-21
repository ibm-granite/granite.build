"""Distributional metrics between a student and its teacher, over a corpus.

PORTED FROM two earlier exploratory scripts (a 337-line JSD script and a 234-line
entropy script), which are the same pipeline twice: render a
conversation, mask the assistant span, run a forward pass, reduce the logits. Merged into one
module because the duplicated halves had already drifted -- the JSD copy derives the assistant
span from `offset_mapping` with a token-count fallback while the entropy copy has only the
fallback, so on a tokenizer without offsets the two disagreed about which tokens they were
measuring. The reductions differ; nothing else does.

WHAT THIS MEASURES THAT A BENCHMARK CANNOT. `steps/bfcl-eval` scores tool-calling accuracy
against ground truth: it asks "is the student right?". It cannot ask "did the student move
toward the TEACHER?", because that is not a property of one model's outputs -- it is a
distance between two models' distributions on the same inputs. For a distillation recipe that
distance is the direct read on whether distillation worked at all, and it is available with no
labelled data. Hence this module rather than another benchmark. The plan document asked for
the finding rather than a routing-around, and this is it: keep `bfcl-eval` for capability,
add divergence for transfer.

THE TOKENIZER IS NOT INCIDENTAL HERE. An earlier version loaded it with
`AutoTokenizer.from_pretrained(model1_path)`, which on a Granite
directory silently substitutes GPT2Tokenizer's plain ByteLevel pre_tokenizer for the trained
one. Both models then receive the SAME mis-segmented ids, so nothing crashes and the numbers
look plausible -- they are simply measured on text in a segmentation neither model was trained
on. Loading goes through `fast_tokenizer.load` instead, and the two models' tokenizer
identities are CHECKED rather than warned about (the original printed a warning on a vocab-size
mismatch and carried on computing a number that cannot mean anything).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from gb_steps_post_training.distillation import fast_tokenizer, tokenizer_identity

METRICS = ("jsd", "kld", "rkld", "entropy")

# Which metrics need two models. `entropy` is a property of one distribution, so it takes the
# student alone -- and it is the one metric here that can be run on a checkpoint before any
# teacher exists.
PAIRWISE = ("jsd", "kld", "rkld")


class MetricError(Exception):
    """Message is the operator-facing explanation."""


def load_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise MetricError(f"{path} does not exist. Pass distill-corpus-prep's corpus "
                          "(train.jsonl or eval.jsonl).")
    out = []
    with path.open() as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError as exc:
                raise MetricError(f"{path}:{i} is not valid JSON: {exc}") from exc
    if not out:
        raise MetricError(f"{path} is empty.")
    return out


def assistant_span(messages: list[dict], tok, max_length: int) -> dict | None:
    """input_ids/attention_mask plus a boolean mask over the FINAL assistant span.

    The span is found by rendering twice -- prompt-only and full -- and taking everything the
    full rendering adds. Deliberately NOT `return_assistant_tokens_mask`: that marks every
    assistant turn (granite-4.2's template wraps each one in `{% generation %}`), whereas this
    measures the completion, matching what a `last_message_only` run trained on. The two
    differ by 46% of assistant tokens on real multi-turn data, confirmed by direct measurement.

    Returns None when there is no span to measure. `truncated` in the returned dict flags a
    span that EXISTS but is cut short, which is a different and much sneakier problem -- see
    below.

    WHY TRUNCATION IS REPORTED AND NOT JUST APPLIED. Measured directly: at
    --max-length 2048 against a corpus rendering to ~3956 tokens, not one record of sixteen
    was skipped. The prompts fit easily, so every span was non-empty and every record was
    measured -- on a PREFIX of its completion, with the tail silently discarded. A
    skipped-record count cannot see that: it reports zero, and the run looks fully covered
    while half of what it claimed to measure was never evaluated. So the full rendered length
    is computed before truncation and the shortfall is carried out of here for the caller to
    account for.

    Do not read that as "2048 does not skip". Both damage modes are live on the same corpus
    at the same setting, and which one you get depends on which records you draw: the
    preflight measured 32 records at prompt min/median/max 1178/1572/3706 and
    full min/median/max 1720/2466/4082, so --max-length 2048 skips 11 of them (prompt alone
    over budget) AND truncates 17 (prompt fits, answer does not). A separate run of sixteen
    records was simply all in the second group. The point is that the two are independent,
    and only one of them used to be counted.
    """
    import torch

    if not messages or messages[-1].get("role") != "assistant":
        return None
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # NO truncation here -- the untruncated length is the thing that has to be known before
    # it is thrown away. The cut is applied by slicing below.
    enc = tok(full, truncation=False, padding=False, return_tensors="pt",
              add_special_tokens=False, return_offsets_mapping=True)
    full_len = int(enc["input_ids"].shape[1])
    keep_len = min(full_len, max_length)
    input_ids = enc["input_ids"][:, :keep_len]
    attention_mask = enc["attention_mask"][:, :keep_len]
    offsets = enc.get("offset_mapping")

    if offsets is not None:
        # Character offsets are exact; the token-count fallback below is not, because
        # re-tokenizing the prompt alone can merge differently at the boundary. An earlier
        # entropy-only version had only the fallback, which is the drift this port removes.
        prompt_chars = len(prompt)
        offs = offsets[0].tolist()
        start = next((i for i, (s, _) in enumerate(offs) if s >= prompt_chars), len(offs))
    else:
        start = len(tok(prompt, truncation=False, padding=False,
                        add_special_tokens=False)["input_ids"])

    if start >= keep_len:
        # The prompt alone fills the budget: there is no completion left to measure at all.
        return None
    mask = torch.zeros(keep_len, dtype=torch.bool)
    mask[start:] = True
    return {"input_ids": input_ids, "attention_mask": attention_mask, "assistant_mask": mask,
            "full_length": full_len, "truncated": full_len > max_length,
            "dropped_tokens": max(0, full_len - max_length)}


def collate(items: list[dict], pad_token_id: int) -> dict:
    import torch
    import torch.nn.functional as F

    max_len = max(it["input_ids"].shape[1] for it in items)
    ids, att, asst = [], [], []
    for it in items:
        pad = max_len - it["input_ids"].shape[1]
        ids.append(F.pad(it["input_ids"], (0, pad), value=pad_token_id))
        att.append(F.pad(it["attention_mask"], (0, pad), value=0))
        asst.append(F.pad(it["assistant_mask"].unsqueeze(0), (0, pad), value=False).squeeze(0))
    return {"input_ids": torch.cat(ids, 0), "attention_mask": torch.cat(att, 0),
            "assistant_mask": torch.stack(asst, 0)}


def reduce_logits(logits_a, logits_b, metric: str):
    """Per-token metric in nats. logits_b is None for `entropy`.

    Kept verbatim from the earlier version's arithmetic (including the 1e-10 floor inside the
    JSD mixture log) so a ported number is comparable to a recorded one.
    """
    import torch
    import torch.nn.functional as F

    if metric == "entropy":
        log_p = torch.log_softmax(logits_a, dim=-1)
        return -(torch.softmax(logits_a, dim=-1) * log_p).sum(dim=-1)

    log_p = F.log_softmax(logits_a, dim=-1)
    log_q = F.log_softmax(logits_b, dim=-1)
    if metric == "jsd":
        p, q = F.softmax(logits_a, dim=-1), F.softmax(logits_b, dim=-1)
        log_m = torch.log(0.5 * (p + q) + 1e-10)
        div = 0.5 * ((p * (log_p - log_m)).sum(-1) + (q * (log_q - log_m)).sum(-1))
    elif metric == "kld":
        div = F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(dim=-1)
    elif metric == "rkld":
        div = (torch.exp(log_q) * (log_q - log_p)).sum(dim=-1)
    else:
        raise MetricError(f"unknown metric {metric!r}; expected one of {METRICS}")
    return torch.clamp(div, min=0.0)


# Strings the compatibility probe encodes when the content hashes differ. Chosen to hit the
# places two Granite tokenizers actually diverge rather than to look thorough: ChatML control
# tokens (the retag's whole purpose -- these are single ids after it and six ids before),
# leading/among whitespace and digits (where granite-4.1's Sequence[Split(regex), ByteLevel]
# pre_tokenizer splits and 4.2's plain ByteLevel does not), and a tool-call payload, which is
# what this corpus is mostly made of.
PROBE_STRINGS = (
    "<|im_start|>assistant\n",
    "<|im_end|>",
    "  leading and    interior   whitespace",
    "1234567890 42 007",
    '{"name": "get_weather", "arguments": {"city": "Zurich"}}',
    "The quick brown fox jumps over the lazy dog.",
)


def check_tokenizers(model_a, model_b, *, allow_mismatch: bool = False) -> dict:
    """Refuse a pair whose tokenizers would assign DIFFERENT ids to the same text.

    A hard failure where an earlier version printed a warning about vocab sizes and carried
    on. A divergence between distributions over different vocabularies is
    not a large number, it is a meaningless one: index i denotes a different token to each
    model, so the arithmetic succeeds and measures nothing. That is worth a refusal because
    the run costs GPU hours and looks successful.

    WHAT IS COMPARED, AND WHY NOT tokenizer_identity.read(). The obvious move is to reuse the
    shared identity that distill-gold-train compares, and it is WRONG here -- measured, not
    reasoned: on the smoke pairing read() returns "ibm-granite/granite-4.2-3b" for the
    retagged student (the retag records a name) and "sha256:883975314d587437" for the teacher
    (an HF-cache directory has no identity file, so read() falls back to a hash). Those are
    never equal, so this guard would have refused every correctly retagged pairing. read() is
    built to compare ONE LINEAGE -- a corpus manifest against the student it was built from,
    where both sides are name-form -- and teacher-vs-student is not that comparison. Note the
    failure mode: distill-gold-train's original guard was INERT (it read a file nothing wrote,
    so it never fired), and this one would have been its mirror image, firing always. Both
    shapes report a tokenizer verdict that owes nothing to the tokenizers.

    So the primary test is content-hash equality of tokenizer.json, which is exactly the
    invariant the retag establishes -- it copies the teacher's tokenizer.json byte-for-byte
    (retag_student.py:382) -- and exactly what id-compatibility means.

    A hash mismatch does not go straight to a refusal, because byte-identical is stricter than
    id-compatible: a re-serialized or differently-ordered tokenizer.json can encode
    identically. So the fallback is BEHAVIOURAL rather than another piece of metadata -- both
    tokenizers encode PROBE_STRINGS and the ids are compared. That is the property the metric
    needs, tested directly. Names are used for the message only, so an operator reads which
    models disagreed rather than which digests did.
    """
    from pathlib import Path

    model_a = Path(model_a)
    name_a = tokenizer_identity.derive_name(model_a)
    hash_a = tokenizer_identity.hash_tokenizer(model_a)
    if hash_a is None:
        raise MetricError(f"{model_a} has no tokenizer.json, so it is not a model directory.")
    if model_b is None:
        return {"tokenizer": name_a, "tokenizer_sha256": hash_a, "compared": False}

    model_b = Path(model_b)
    name_b = tokenizer_identity.derive_name(model_b)
    hash_b = tokenizer_identity.hash_tokenizer(model_b)
    if hash_b is None:
        raise MetricError(f"{model_b} has no tokenizer.json, so it is not a model directory.")

    result = {"tokenizer": name_a, "tokenizer_sha256": hash_a, "compared": True,
              "student_tokenizer": name_a, "teacher_tokenizer": name_b}
    if hash_a == hash_b:
        result["agreement"] = "identical_tokenizer_json"
        return result

    disagreements = probe_disagreements(model_a, model_b)
    if not disagreements:
        # Different bytes, same ids on every probe. Recorded rather than silent: it means the
        # retag invariant does not hold for this pair and something re-serialized a tokenizer.
        result["agreement"] = "ids_agree_despite_differing_tokenizer_json"
        result["note"] = (f"{name_a} ({hash_a}) and {name_b} ({hash_b}) are not byte-identical "
                          "but agree on every probe string.")
        return result

    first = disagreements[0]
    message = (
        f"{name_a} and {name_b} do not agree on token ids, so a divergence between them is "
        f"not comparable -- the same index means a different token to each model.\n"
        f"  {name_a}: {hash_a}\n  {name_b}: {hash_b}\n"
        f"  {len(disagreements)} of {len(PROBE_STRINGS)} probe strings disagree; first:\n"
        f"    text : {first['text']!r}\n"
        f"    {name_a[:24]:>24} -> {first['ids_a']}\n"
        f"    {name_b[:24]:>24} -> {first['ids_b']}\n"
        f"  Pass distill-tokenizer-align's retagged_student as the student. If you have a "
        f"reason to want the number anyway, --allow-tokenizer-mismatch."
    )
    if not allow_mismatch:
        raise MetricError(message)
    result["agreement"] = "MISMATCH_ALLOWED"
    result["note"] = message
    return result


def probe_disagreements(model_a, model_b) -> list[dict]:
    """PROBE_STRINGS on which the two tokenizers produce different ids.

    Loaded through fast_tokenizer, which is the point: AutoTokenizer would impose GPT2's plain
    ByteLevel pre_tokenizer on a Granite directory (its tokenizer_config.json declares
    tokenizer_class "GPT2Tokenizer"), which is the very difference this probe exists to detect
    -- so a probe built on AutoTokenizer would erase the disagreement it is looking for and
    report agreement on genuinely incompatible tokenizers.
    """
    tok_a, _ = fast_tokenizer.load(model_a, require_chat_template=False)
    tok_b, _ = fast_tokenizer.load(model_b, require_chat_template=False)
    out = []
    for text in PROBE_STRINGS:
        ids_a = tok_a(text, add_special_tokens=False)["input_ids"]
        ids_b = tok_b(text, add_special_tokens=False)["input_ids"]
        if ids_a != ids_b:
            out.append({"text": text, "ids_a": ids_a, "ids_b": ids_b})
    return out


def summarise(per_sample: list[float], counts: list[int]) -> dict:
    import numpy as np

    if not per_sample:
        raise MetricError(
            "no sample produced a measurable assistant span. Either every conversation was "
            "truncated away by --max-length, or none ends on an assistant turn.")
    arr = np.asarray(per_sample, dtype=float)
    return {
        "n_samples": len(per_sample),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "total_assistant_tokens": int(sum(counts)),
        "mean_tokens_per_sample": float(sum(counts) / len(counts)),
    }
