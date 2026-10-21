"""The label-masking contract, DERIVED from a tokenizer rather than hardcoded in every config.

WHY THIS IS ONE STEP'S OUTPUT AND NOT A CONFIG FIELD.

Twenty-two gold configs each carry `response_template: "<|im_start|>assistant\\n"`. That value is
not a preference; it is a FACT about the chat template installed on the student, and the configs
restate it by hand. Restating a derived fact is how it goes stale silently: pair Granite 4 with a
different family -- Qwen 3.5, say -- and every one of those configs is wrong, the collator's scan
matches nothing, `CustomDataCollatorForChatML` (utils.py:465-500) leaves every label at
`ignore_index`, and the run trains on NOTHING while producing a loss curve, checkpoints and a
student. There is no post-condition on that path (unlike sft.py:896 and
custom_gold_trainer.py:1785, which raise).

So `distill-tokenizer-align` -- the step that INSTALLS the chat template -- also emits
`masking.json` describing what the template implies for masking, and the configs point at it.
One step owns it; nothing is scattered.

HOW IT IS DERIVED, AND WHY THE OBVIOUS METHOD IS WRONG.

The obvious derivation is to subtract: render `[user]` with and without `add_generation_prompt`
and take the difference. For THIS template that yields `<|im_start|>assistant\\n<think>\\n`
(chat_template.jinja:189-194), which is a plausible-looking answer and a live trap:

  * At TRAINING time the same template emits `<|im_start|>assistant\\n` (line 98) and then
    whatever the assistant content renders to -- `<think>\\n...\\n</think>\\n` for a record that
    carries reasoning, or an injected `<think></think>` for one that does not (lines 90/113/119).
  * So the generation-prompt string occurs in the render of SOME records and not others. A scan
    for it would mask a subset of assistant turns and silently skip the rest -- worse than
    matching nothing, because the loss curve looks fine.

The generation prompt is what you feed at INFERENCE. The response template is what precedes
assistant content at TRAINING time. They are allowed to differ and here they do.

The derivation used instead asks the template itself, in token space:

  1. Render a probe SET whose conversations exercise the template's branches (reasoning content
     present / absent, single and multi turn) under BOTH `enable_thinking` settings, with
     `return_assistant_tokens_mask=True`. The `{% generation %}` markers are ground truth for
     where assistant content starts.
  2. For every assistant turn in every probe, take the ids PRECEDING its first masked token.
  3. The longest common suffix of those id sequences bounds the answer -- common across branches
     is what "always precedes assistant content" means. It is then TRIMMED through the previous
     turn's eos and past any leading whitespace, because otherwise it encodes the probes' own
     conversation shape into the marker, and re-validated against the markers so a trim that
     over-generalised is refused rather than accepted.

Working in token space means the answer cannot be wrong about tokenization. But the collator
encodes the template TEXT standalone (utils.py:349), so the text is decoded back and re-encoded,
and a difference between in-context and standalone ids is a REFUSAL: that difference is the
silent trap in its purest form.

THREE REFUSALS, each naming a real failure:

  * No generation markers (an all-zero mask). Then there is no ground truth here, and GOLD would
    fail anyway at sft.py:909. Says to apply
    patches/granite-chat-template-generation-markers/.
  * A probe set that exercised only ONE branch. Then the common suffix is untested and can be too
    long -- the over-long template is precisely the "matches some records" trap. Refused rather
    than reported, because an over-long template that passes review is the worst outcome here.
  * eos disagreement between config.json and the tokenizer. The collator ends each span at
    `tokenizer.eos_token_id` (utils.py:487) while the model's generation is bounded by
    config.json's. If they differ, one assistant span runs to the end of the sequence and
    swallows the following user turn into the labels.

WHAT IS DELIBERATELY NOT HERE. No `instruction_template`. It is inert -- `instruction_token_ids`
is assigned at utils.py:354 and read nowhere, since the masking loop bounds spans with EOS -- and
a companion check asserts that inertness continues to hold. Emitting
a derived value for a field nothing reads would invite someone to start reading it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MASKING_NAME = "masking.json"
CONTRACT_VERSION = 1

# The probe set. Its job is to make the template take DIFFERENT branches, so the common suffix in
# step 3 is a real intersection and not one branch's whole prefix. `_assert_branch_coverage`
# enforces that it did; the probes are not trusted to be sufficient just because they look varied.
PROBES: tuple[tuple[str, list[dict]], ...] = (
    ("plain single turn", [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ]),
    ("reasoning content, explicit think tags", [
        {"role": "user", "content": "What is 17*3?"},
        {"role": "assistant", "content": "<think>\n17*3 = 51.\n</think>\n51."},
    ]),
    ("two assistant turns, mixed", [
        {"role": "user", "content": "Name a colour."},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "Now justify it."},
        {"role": "assistant", "content": "<think>\nBlue is calm.\n</think>\nIt is calm."},
    ]),
)


class MaskingError(RuntimeError):
    """A masking contract that cannot be derived, or that would mask the wrong tokens."""


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _common_suffix(seqs: list[list[int]]) -> list[int]:
    """Longest list that is a suffix of every input. [] if they share none."""
    if not seqs:
        return []
    out: list[int] = []
    for k in range(1, min(len(s) for s in seqs) + 1):
        tail = seqs[0][-k]
        if all(s[-k] == tail for s in seqs):
            out.insert(0, tail)
        else:
            break
    return out


def _spans_from_mask(mask: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of 1 in an assistant-tokens mask, as [start, end) pairs."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def spans_from_scan(ids: list[int], resp: list[int], eos_id: int | None) -> list[tuple[int, int]]:
    """Reimplements utils.py:475-497 EXACTLY, including the `i = end_idx` skip and the
    EOS-inclusive `end_idx = j + 1`.

    Reimplemented rather than imported on purpose: importing the collator would drag torch and a
    tokenizer instance into a function whose whole value is being a cheap, exact statement of what
    the collator does. If the collator's loop ever changes, a companion check compares the two
    on real batches and this drifts loudly rather than quietly. The duplication is the point of
    comparison.
    """
    spans: list[tuple[int, int]] = []
    n = len(resp)
    if n == 0:
        return spans
    i = 0
    while i <= len(ids) - n:
        if ids[i:i + n] == resp:
            start = i + n
            end = len(ids)
            if eos_id is not None:
                for j in range(start, len(ids)):
                    if ids[j] == eos_id:
                        end = j + 1
                        break
            spans.append((start, end))
            i = end
        else:
            i += 1
    return spans


# The template branches on `enable_thinking`, and the collator has ONE response_template for the
# whole run -- so a contract that held only under one setting would be a contract that silently
# breaks when a caller flips it. Both are rendered and both must agree.
#
# This is not hypothetical: a companion check reproduces custom_gold_trainer.py's own
# asymmetry, where prompt_text and full_text are rendered with enable_thinking=False while the call
# that produces `input_ids` is rendered without it. One run therefore uses both settings.
#
# A template that does not define the variable simply ignores it, so this costs nothing on
# families that have no thinking mode.
THINKING_VARIANTS = (True, False)


def _render(tok, messages: list[dict], *, enable_thinking: bool = True) -> tuple[list[int], list[int]]:
    """(ids, assistant_mask) for one probe. `tokenize=True` with the mask requested, because the
    mask is only produced on the tokenizing path."""
    out = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_assistant_tokens_mask=True,
        enable_thinking=enable_thinking,
    )
    ids = list(out["input_ids"])
    mask = list(out.get("assistant_masks") or [])
    return ids, mask


def _assert_branch_coverage(preceding: list[list[int]], suffix: list[int]) -> None:
    """The common suffix must be a PROPER suffix of at least one preceding sequence.

    Otherwise every probe agreed on its whole prefix, which means the probe set exercised one
    branch and the derived template is as long as that branch's prefix. That over-long template is
    exactly the failure this module exists to prevent: it matches the records that took that
    branch and silently skips the others. A probe set that cannot show disagreement has not tested
    anything, so this refuses rather than reports.
    """
    if not preceding:
        raise MaskingError("no assistant turns were found in any probe -- nothing to derive from")
    if all(len(p) == len(suffix) for p in preceding):
        raise MaskingError(
            f"every probe's assistant turn was preceded by the SAME {len(suffix)} ids, so the "
            "probe set exercised only one branch of the chat template and the derived response "
            "template is untested. It may be longer than what precedes assistant content in "
            "general, which would mask some records and silently skip the rest. Add a probe that "
            "renders differently (reasoning content present vs absent is the usual axis).")


def derive(tokenizer_dir: Path, *, config_dir: Path | None = None) -> dict:
    """The masking contract implied by the chat template installed at `tokenizer_dir`.

    `config_dir` defaults to `tokenizer_dir` and is where config.json's eos is read from; they are
    separable only so a tokenizer-only overlay can be interrogated against a model elsewhere.
    """
    from transformers import AutoTokenizer  # imported late: argparse and --help cost nothing

    tokenizer_dir = Path(tokenizer_dir)
    config_dir = Path(config_dir or tokenizer_dir)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    if not getattr(tok, "chat_template", None):
        raise MaskingError(
            f"{tokenizer_dir} has no chat template. Masking is a property OF the template, so "
            "there is nothing to derive. run-align.sh's --chat-template installs one.")

    tok_eos = tok.eos_token_id
    cfg_path = config_dir / "config.json"
    cfg_eos = None
    if cfg_path.is_file():
        cfg_eos = json.loads(cfg_path.read_text()).get("eos_token_id")
    # A list is legal in config.json (several stop ids). The collator can only use one, so a list
    # is only compatible if the tokenizer's id is among them.
    if cfg_eos is not None:
        ok = tok_eos in cfg_eos if isinstance(cfg_eos, list) else cfg_eos == tok_eos
        if not ok:
            raise MaskingError(
                f"eos disagreement: tokenizer says {tok_eos} "
                f"({tok.convert_ids_to_tokens([tok_eos])[0]!r}) and config.json says {cfg_eos}. "
                "The collator ends every assistant span at the TOKENIZER's eos (utils.py:487). "
                "If the render never emits that id, each span runs to the end of the sequence and "
                "swallows the following user turn into the labels.")

    preceding: list[list[int]] = []
    per_probe: list[dict] = []
    cases = [(f"{name} [thinking={t}]", messages, t)
             for name, messages in PROBES for t in THINKING_VARIANTS]
    for name, messages, thinking in cases:
        ids, mask = _render(tok, messages, enable_thinking=thinking)
        if not mask or not any(mask):
            raise MaskingError(
                f"probe {name!r}: the chat template produced an all-zero assistant mask, so it "
                "carries no {% generation %} markers and there is no ground truth for where "
                "assistant content begins. GOLD fails on this template anyway (sft.py:909). "
                "Apply patches/granite-chat-template-generation-markers/.")
        spans = _spans_from_mask(mask)
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        if len(spans) != n_assistant:
            raise MaskingError(
                f"probe {name!r}: the mask marks {len(spans)} span(s) for {n_assistant} assistant "
                "turn(s). The generation markers do not correspond one-to-one with assistant "
                "messages, so a per-turn derivation would be guessing.")
        for start, _end in spans:
            preceding.append(ids[:start])
        per_probe.append({"probe": name, "tokens": len(ids), "mask_spans": [list(s) for s in spans]})

    maximal = _common_suffix(preceding)
    if not maximal:
        raise MaskingError(
            "the assistant turns across the probe set share NO common preceding token, so no "
            "single response template can mark them all. This template needs positional masking "
            "(`last_message_only`), which is a different policy and must be chosen deliberately.")
    _assert_branch_coverage(preceding, maximal)

    # TRIM the maximal suffix, and this is not tidiness. The longest common suffix is
    # OVER-SPECIFIC: measured directly on this template, every assistant turn -- including one
    # that opens the conversation, because the template emits a default system turn first -- is
    # preceded by `<|im_end|>\n`, so the intersection reaches back through the PREVIOUS turn's
    # terminator. Direct measurement derived `<|im_end|>\n<|im_start|>assistant\n` for exactly that
    # reason. Such a marker is correct on every conversation shaped like the probes and wrong on
    # the first one that is not -- and "wrong" here means the collator finds nothing and every
    # label stays at ignore_index, silently (utils.py:465-500 has no post-condition).
    #
    # Two trims, each with a reason, applied in this order:
    #   1. Cut through the LAST eos in the suffix. Whatever precedes the previous turn's
    #      terminator cannot be part of what OPENS an assistant turn. eos comes from the
    #      tokenizer, so this is family-agnostic.
    #   2. Drop leading tokens that decode to nothing but whitespace. Inter-turn whitespace is
    #      separator, not marker: `<|im_end|>\n` is emitted as one unit by the turn that ENDS.
    #
    # Both trims can only GENERALISE the marker -- a shorter needle matches at least as many
    # positions -- so the risk they carry is over-matching, not under-matching. That risk is then
    # closed by re-validating against the generation markers below: a marker that matched an extra
    # position would produce an extra span and be refused. Neither trim is a guess about where the
    # boundary is; the markers remain the ground truth and the trims only propose candidates.
    cut = max((i for i, t in enumerate(maximal) if t == tok_eos), default=-1)
    trimmed = maximal[cut + 1:]
    while trimmed and not tok.decode(trimmed[:1]).strip():
        trimmed = trimmed[1:]
    if not trimmed:
        raise MaskingError(
            f"every token that always precedes assistant content in this template "
            f"({tok.decode(maximal)!r}) is the previous turn's terminator or whitespace, so there "
            "is no marker that opens an assistant turn. This template needs positional masking "
            "(`last_message_only`), which is a different policy and must be chosen deliberately.")

    # The collator encodes the template TEXT and scans for THOSE ids (utils.py:349). A candidate
    # that re-tokenizes differently standalone would match nothing at all -- silently -- so the
    # round trip is a precondition, not a sanity check.
    renders = [(name, _render(tok, messages, enable_thinking=t)) for name, messages, t in cases]
    text = tok.decode(trimmed)
    standalone = tok.encode(text, add_special_tokens=False)
    if standalone != trimmed:
        raise MaskingError(
            f"the derived template {text!r} tokenizes differently standalone than it does in "
            f"context: standalone {standalone} vs in-context {trimmed}. The collator encodes the "
            "template text standalone (utils.py:349) and scans for THOSE ids, so it would find "
            "nothing and every label would stay at ignore_index -- silently.")
    for name, (ids, mask) in renders:
        got = [st for st, _ in spans_from_scan(ids, trimmed, tok_eos)]
        want = [st for st, _ in _spans_from_mask(mask)]
        if got != want:
            raise MaskingError(
                f"probe {name!r}: scanning for the derived template {text!r} starts assistant "
                f"spans at {got} but the template's own generation markers say {want}. The two "
                "mechanisms disagree about which tokens are the assistant's, so masking cannot be "
                f"trusted either way. (Longest common suffix was {tok.decode(maximal)!r}, trimmed "
                f"to {text!r} at the previous turn's terminator.)")

    resp_in_context = trimmed

    # Now the whole point: does a scan with these ids reproduce the generation markers' spans?
    agreement: list[dict] = []
    for (name, (ids, mask)) in renders:
        scan = spans_from_scan(ids, resp_in_context, tok_eos)
        want = _spans_from_mask(mask)
        if [st for st, _ in scan] != [st for st, _ in want]:
            # Unreachable via the loop above, which selected on this very condition. Kept because
            # a future edit that moves the selection must not silently lose the assertion.
            raise MaskingError(
                f"probe {name!r}: scanning for {text!r} starts assistant spans at "
                f"{[st for st, _ in scan]} but the template's own generation markers say "
                f"{[st for st, _ in want]}.")
        agreement.append({
            "probe": name,
            "spans": len(scan),
            # Signed, and its SIGN is informative. Measured directly on granite-4.x: -1,
            # because the generation block closes AFTER the turn's trailing newline while the
            # collator's scan stops at the eos it includes. A positive delta would be the opposite
            # convention. Recorded per span rather than tolerated, so a reader sees 1 and not 40.
            "boundary_delta_tokens": [e1 - e2 for (_, e1), (_, e2) in zip(scan, want)],
            "labelled_tokens": sum(e - st for st, e in scan),
        })

    return {
        "contract_version": CONTRACT_VERSION,
        "response_template": text,
        "response_token_ids": resp_in_context,
        "boundary": {
            "policy": "eos_inclusive",
            "eos_token_id": tok_eos,
            "eos_token": tok.convert_ids_to_tokens([tok_eos])[0] if tok_eos is not None else None,
            "config_eos_token_id": cfg_eos,
            "note": ("utils.py:485-489 ends each span at the first eos at or after the span "
                     "start and INCLUDES it in the labels."),
        },
        "instruction_template": None,
        "instruction_template_note": (
            "not derived: instruction_token_ids is assigned at utils.py:354 and read nowhere, and "
            "a companion config check asserts that stays true."),
        "source": {
            "tokenizer_dir": str(tokenizer_dir),
            "tokenizer_json_sha256": _sha256(tokenizer_dir / "tokenizer.json"),
            "chat_template_sha256": (_sha256(tokenizer_dir / "chat_template.jinja")
                                     or hashlib.sha256(tok.chat_template.encode()).hexdigest()),
            "derivation": ("longest common suffix of the ids preceding each generation-marked "
                           "assistant span, over a probe set asserted to exercise more than one "
                           "branch, trimmed through the previous turn's eos and leading "
                           "whitespace, then re-validated against those same markers"),
            "maximal_common_suffix": tok.decode(maximal),
        },
        "probes": per_probe,
        "agreement": agreement,
    }


def check_config(doc: dict, cfg: dict) -> list[str]:
    """Problems with a gold config measured against a derived contract. Empty means sound.

    This is the preflight a launcher runs. It compares TEXT, because text is what the config
    carries and what the collator encodes -- comparing ids would pass a config whose text differs
    in a way this tokenizer happens to encode identically, and that config would be wrong for the
    next tokenizer.
    """
    problems: list[str] = []
    want = doc["response_template"]
    if "response_template" not in cfg:
        problems.append(
            f"response_template ABSENT -- CustomGOLDConfig's stale default would apply. "
            f"masking.json derives {want!r} from the installed chat template.")
    elif cfg["response_template"] != want:
        problems.append(
            f"response_template is {cfg['response_template']!r} but the chat template installed "
            f"on this student implies {want!r}. The config was not derived from the template it "
            "will train against.")
    if cfg.get("last_message_only") is True:
        problems.append(
            "last_message_only: true bypasses the response_template scan entirely "
            "(utils.py:470) for a positional split. A different masking policy; masking.json "
            "describes the scanning one.")
    return problems


def _self_test() -> int:
    """Exercises the pure functions against hand-built sequences. The derivation itself needs a
    real tokenizer and is covered by a companion smoke test.

    Each case states the answer it must produce, so a case cannot pass by tripping a different
    assertion than the one it was written for.
    """
    bad = 0

    def eq(name, got, want):
        nonlocal bad
        if got != want:
            print(f"  FAIL {name}: got {got!r}, want {want!r}")
            bad = 1

    eq("common suffix, plain", _common_suffix([[1, 2, 3], [9, 2, 3], [2, 3]]), [2, 3])
    eq("common suffix, none", _common_suffix([[1, 2], [3, 4]]), [])
    eq("common suffix, whole shortest", _common_suffix([[7, 8], [1, 7, 8]]), [7, 8])
    eq("common suffix, single seq is itself", _common_suffix([[4, 5, 6]]), [4, 5, 6])
    eq("common suffix, empty input", _common_suffix([]), [])

    eq("mask spans, two runs", _spans_from_mask([0, 1, 1, 0, 0, 1, 0]), [(1, 3), (5, 6)])
    eq("mask spans, run to end", _spans_from_mask([0, 1, 1]), [(1, 3)])
    eq("mask spans, all zero", _spans_from_mask([0, 0]), [])

    # The scan: two turns bounded by eos=99, template = [50, 51].
    ids = [10, 50, 51, 1, 2, 99, 11, 50, 51, 3, 99]
    eq("scan, two turns", spans_from_scan(ids, [50, 51], 99), [(3, 6), (9, 11)])
    # No eos: the span runs to the end, which is the collator's behaviour and not a bug.
    eq("scan, no eos", spans_from_scan([50, 51, 1, 2], [50, 51], None), [(2, 4)])
    # The trap: a template that does not occur yields no spans at all -- silently, in the
    # collator. Here it is an assertion.
    eq("scan, template absent", spans_from_scan(ids, [77, 78], 99), [])
    eq("scan, empty template", spans_from_scan(ids, [], 99), [])
    # `i = end_idx` means a template occurring INSIDE an assistant span is not re-matched.
    eq("scan, template inside a span is skipped",
       spans_from_scan([50, 51, 50, 51, 7, 99], [50, 51], 99), [(2, 6)])

    # Branch coverage: identical prefixes must refuse, a proper suffix must pass.
    try:
        _assert_branch_coverage([[1, 2, 3], [1, 2, 3]], [1, 2, 3])
        print("  FAIL branch coverage: identical prefixes were accepted")
        bad = 1
    except MaskingError as e:
        if "one branch" not in str(e):
            print(f"  FAIL branch coverage: wrong message {e}")
            bad = 1
    try:
        _assert_branch_coverage([[9, 1, 2, 3], [1, 2, 3]], [1, 2, 3])
    except MaskingError as e:
        print(f"  FAIL branch coverage: a proper suffix was refused: {e}")
        bad = 1
    try:
        _assert_branch_coverage([], [])
        print("  FAIL branch coverage: an empty probe set was accepted")
        bad = 1
    except MaskingError:
        pass

    doc = {"response_template": "<|im_start|>assistant\n"}
    eq("config check, matching", check_config(doc, {"response_template": "<|im_start|>assistant\n"}), [])
    for name, cfg, needle in (
        ("absent", {"lmbda": 0.3}, "ABSENT"),
        ("stale", {"response_template": "<|start_of_role|>assistant<|end_of_role|>"}, "implies"),
        ("literal backslash-n", {"response_template": "<|im_start|>assistant\\n"}, "implies"),
        ("bypass", {"response_template": "<|im_start|>assistant\n", "last_message_only": True}, "bypasses"),
    ):
        got = check_config(doc, cfg)
        if not any(needle in g for g in got):
            print(f"  FAIL config check {name}: expected {needle!r}, got {got}")
            bad = 1

    if not bad:
        print("  OK   masking self-test: suffix, mask spans, the collator's scan reimplementation, "
              "branch coverage and the config preflight")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("emit", "check-config", "self-test"))
    ap.add_argument("--tokenizer", type=Path, help="directory holding the installed chat template")
    ap.add_argument("--config-dir", type=Path, default=None,
                    help="where config.json is, if not --tokenizer")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"emit: where to write {MASKING_NAME} (default --tokenizer's parent)")
    ap.add_argument("--masking", type=Path, default=None, help=f"check-config: path to {MASKING_NAME}")
    ap.add_argument("--gold-config", type=Path, default=None, help="check-config: the yaml to audit")
    args = ap.parse_args(argv)

    if args.action == "self-test":
        return _self_test()

    if args.action == "emit":
        if args.tokenizer is None:
            ap.error("emit needs --tokenizer")
        try:
            doc = derive(args.tokenizer, config_dir=args.config_dir)
        except MaskingError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        # BESIDE the tokenizer, not beside its directory. `--tokenizer` names a model DIRECTORY on
        # every caller (run-align.sh passes $RETAGGED), so `.parent` here put masking.json one level
        # up -- outside the dir that align_state declares it in. Caught the only
        # way it could be caught: the step emitted the contract, printed a path, and then REFUSED to
        # write its marker because the output it promised was not where it promised to put it.
        tokenizer = Path(args.tokenizer)
        dest = Path(args.out) if args.out else (
            tokenizer / MASKING_NAME if tokenizer.is_dir() else tokenizer.parent / MASKING_NAME)
        if dest.is_dir():
            dest = dest / MASKING_NAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        tmp.replace(dest)
        print(f"  response_template : {doc['response_template']!r}")
        print(f"  token ids         : {doc['response_token_ids']}")
        print(f"  eos               : {doc['boundary']['eos_token_id']} "
              f"({doc['boundary']['eos_token']!r})")
        for a in doc["agreement"]:
            print(f"  agrees on {a['spans']} span(s), {a['labelled_tokens']:>4} labelled tokens, "
                  f"boundary delta {a['boundary_delta_tokens']} -- {a['probe']}")
        print(f"  wrote {dest}")
        return 0

    # check-config
    if args.masking is None or args.gold_config is None:
        ap.error("check-config needs --masking and --gold-config")
    import yaml
    doc = json.loads(Path(args.masking).read_text())
    cfg = yaml.safe_load(Path(args.gold_config).read_text())
    if not isinstance(cfg, dict):
        print(f"ERROR: {args.gold_config} is not a mapping", file=sys.stderr)
        return 1
    problems = check_config(doc, cfg)
    if problems:
        for p in problems:
            print(f"  FAIL {Path(args.gold_config).name}: {p}", file=sys.stderr)
        return 1
    print(f"  OK   {Path(args.gold_config).name}: response_template matches the contract derived "
          f"from {doc['source']['tokenizer_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
