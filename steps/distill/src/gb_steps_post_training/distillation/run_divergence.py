"""Entrypoint: measure divergence and/or entropy of a student against a teacher on a corpus.

Invoked as `python -m gb_steps_post_training.distillation.run_divergence`, the same convention
as build_overlay/retag_student (see steps/distill-tokenizer-align/src/run-align.sh).

REPLACES two earlier exploratory scripts (a JSD script and an entropy script) and their shell
wrappers. The scripts themselves were already argparse-driven and path-clean; the hardcoding
lived in the wrappers, so this port is mostly consolidation plus the tokenizer fixes recorded
in divergence.py.

ONE PROCESS COMPUTES ALL REQUESTED METRICS. The earlier scripts ran jsd and entropy as separate
invocations, which means loading a 30B teacher and re-running every forward pass per metric.
Every metric here is a reduction over the same two logit tensors, so `--metrics jsd,kld,entropy`
costs one pass, not three.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gb_steps_post_training.distillation import divergence as dv
from gb_steps_post_training.distillation import fast_tokenizer


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student-model", required=True,
                    help="Model whose distribution is measured. For a distillation run this "
                         "is the checkpoint under test (distill-hf-export's export), or "
                         "distill-tokenizer-align's retagged_student for a t=0 baseline.")
    ap.add_argument("--teacher-model", default=None,
                    help="Reference model. Required for jsd/kld/rkld; omit for entropy alone.")
    ap.add_argument("--corpus", required=True,
                    help="JSONL of {messages: [...]} records -- distill-corpus-prep's output. "
                         "Use its eval split, not the split the student trained on.")
    ap.add_argument("--out", required=True, help="Directory for metrics.json and per_sample.jsonl")
    ap.add_argument("--metrics", default="jsd,entropy",
                    help=f"Comma-separated subset of {','.join(dv.METRICS)} (default jsd,entropy)")
    ap.add_argument("--max-samples", type=int, default=256,
                    help="Cap on corpus records measured (default 256). A divergence mean is "
                         "stable well before the corpus is exhausted and this runs a 30B "
                         "forward pass per batch.")
    ap.add_argument("--max-length", type=int, default=4096,
                    help="Truncation length. Should match the training run's max_length, or "
                         "the metric is measured on a different distribution than was trained.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seeds the corpus subsample, so two runs measure the same records.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--allow-tokenizer-mismatch", action="store_true",
                    help="Compute anyway when the two tokenizers disagree on token ids. The "
                         "resulting number is not comparable across models; see "
                         "divergence.check_tokenizers.")
    ap.add_argument("--max-incomplete-fraction", type=float, default=0.25,
                    help="Refuse the run when more than this fraction of sampled records was "
                         "measured incompletely (default 0.25). Two things count as "
                         "incomplete, and they are different failures with the same cause -- "
                         "a --max-length below what the corpus needs. SKIPPED: the prompt "
                         "alone fills the budget, so no assistant tokens survive and the "
                         "record contributes nothing (confirmed directly: --max-length 1024 "
                         "against a corpus averaging 2347 tokens skipped 100%). TRUNCATED: "
                         "the span survives but its tail is cut, so the record contributes a "
                         "PREFIX of its completion (confirmed directly: --max-length 2048 "
                         "skipped nothing at all and quietly measured partial answers). "
                         "Neither skew is random -- both hit the longest conversations and "
                         "only those -- so the reported mean is a length-biased estimate "
                         "presented as the divergence. Set to 1.0 to measure a deliberately "
                         "short slice anyway.")
    ap.add_argument("--attn-implementation", default="eager",
                    help="eager is the default deliberately: this needs full logits and does "
                         "no generation, so flash-attention buys nothing here and is one more "
                         "thing that can fail to build.")
    return ap.parse_args(argv)


def resolve_metrics(raw: str) -> list[str]:
    wanted = [m.strip() for m in raw.split(",") if m.strip()]
    if not wanted:
        raise dv.MetricError("--metrics is empty")
    unknown = [m for m in wanted if m not in dv.METRICS]
    if unknown:
        raise dv.MetricError(f"unknown metric(s) {unknown}; expected a subset of {dv.METRICS}")
    # Deduplicate but keep order, so --metrics jsd,jsd is not two passes.
    return list(dict.fromkeys(wanted))


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        metrics = resolve_metrics(args.metrics)
        needs_teacher = [m for m in metrics if m in dv.PAIRWISE]
        if needs_teacher and not args.teacher_model:
            raise dv.MetricError(
                f"--teacher-model is required for {needs_teacher} (a divergence needs two "
                f"distributions). Use --metrics entropy to measure the student alone.")

        # The tokenizer check runs BEFORE any weights load. A 30B teacher takes minutes to
        # load and this refusal costs nothing, so ordering it first is the difference between
        # a fast error and a slow one.
        ident = dv.check_tokenizers(args.student_model, args.teacher_model,
                                    allow_mismatch=args.allow_tokenizer_mismatch)
        print(f"[tokenizer] {json.dumps(ident)}", flush=True)

        records = dv.load_jsonl(args.corpus)
        result = run(args, metrics, records, ident)
    except dv.MetricError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except fast_tokenizer.TokenizerLoadError as exc:
        print(f"ERROR: tokenizer: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result["metrics"], indent=2) + "\n")
    with (out / "per_sample.jsonl").open("w") as fh:
        for row in result["per_sample"]:
            fh.write(json.dumps(row) + "\n")

    print("\n=== divergence metrics ===")
    for name, summary in result["metrics"]["metrics"].items():
        print(f"  {name:>8}: mean {summary['mean']:.6f}  median {summary['median']:.6f}  "
              f"std {summary['std']:.6f}  (n={summary['n_samples']})")
    counts = result["metrics"]["counts"]
    if counts["skipped_no_assistant_span"]:
        print(f"  NOTE: {counts['skipped_no_assistant_span']} record(s) had no measurable "
              f"assistant span and were skipped.")
    if counts["truncated_at_max_length"]:
        print(f"  NOTE: {counts['truncated_at_max_length']} record(s) were TRUNCATED at "
              f"--max-length {result['metrics']['config']['max_length']}: "
              f"{counts['truncated_tokens_dropped']} completion token(s) were not scored. "
              f"The means above cover the measured prefixes only.")
    print(f"\nwrote {out}/metrics.json and {out}/per_sample.jsonl")
    return 0


def rendered_length(tok, messages) -> int:
    """Token length of a fully rendered conversation.

    NOT len(tok.apply_chat_template(..., tokenize=True)). Under transformers 5.x that returns
    a BatchEncoding, so len() counts its KEYS: it reported "median rendered length 2 tokens"
    for a corpus whose records are ~3956 tokens, confirmed by direct measurement. That is worse than having
    no diagnostic, because it tells an operator whose --max-length is four times too small
    that length is not the problem. Render to text, then tokenize.
    """
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def run(args, metrics: list[str], records: list[dict], ident: dict) -> dict:
    import random

    import torch
    from transformers import AutoModelForCausalLM

    tok, template_source = fast_tokenizer.load(args.student_model, require_chat_template=True)
    print(f"[tokenizer] chat template from {template_source}", flush=True)

    if len(records) > args.max_samples:
        # Seeded sample rather than head -n: the corpus is often sorted by source or length,
        # so a prefix measures one slice of it and the mean moves when max_samples changes.
        records = random.Random(args.seed).sample(records, args.max_samples)

    encoded, skipped, truncated, dropped = [], 0, 0, 0
    for rec in records:
        item = dv.assistant_span(rec.get("messages") or [], tok, args.max_length)
        if item is None:
            skipped += 1
            continue
        if item["truncated"]:
            truncated += 1
            dropped += item["dropped_tokens"]
        encoded.append(item)

    # NEITHER OF THESE IS A NEUTRAL LOSS OF SAMPLE SIZE, and that is the whole reason this
    # block exists instead of a log line. Both are caused by --max-length being too small for
    # the corpus, and both select on length:
    #   skipped   -- the prompt alone fills the budget, so nothing of the completion is left.
    #   truncated -- a span was measured, but only its head. The record is counted in
    #                n_samples and contributes a mean over a PREFIX of the answer.
    # Truncation is the more dangerous of the two precisely because it looks like success:
    # a direct measurement at --max-length 2048 skipped ZERO records of sixteen and reported a clean
    # jsd over conversations whose second halves were never seen. Long conversations are the
    # ones that get cut, so the surviving measurement is a length-biased estimate of the
    # divergence, reported as the divergence. So a high incomplete rate is a refusal with the
    # arithmetic shown.
    incomplete = skipped + truncated
    fraction = incomplete / len(records) if records else 1.0
    if not encoded or fraction > args.max_incomplete_fraction:
        lengths = sorted(rendered_length(tok, r["messages"])
                         for r in records[:64] if r.get("messages"))
        median = lengths[len(lengths) // 2] if lengths else 0
        longest = lengths[-1] if lengths else 0
        lengths_note = (f"  --max-length is {args.max_length}; a sample of this corpus renders "
                        f"to a median of {median} and a maximum of {longest} tokens.\n")
        if not encoded:
            # Distinct from the biased-subsample case and it needs a different fix: there is
            # no slice to measure, so --max-incomplete-fraction cannot rescue it.
            raise dv.MetricError(
                f"none of {len(records)} sampled records yielded a measurable assistant span, "
                f"so there is nothing to average.\n" + lengths_note +
                f"  A record is skipped when the prompt alone fills --max-length, leaving no "
                f"assistant tokens. Raise --max-length above {median} (ideally to the training "
                f"run's value), or confirm the corpus ends each conversation on an assistant "
                f"message -- distill-corpus-prep guarantees that, a raw corpus may not. "
                f"--max-incomplete-fraction does not apply here: it selects among records that "
                f"were measured at all, and none were.")
        raise dv.MetricError(
            f"{incomplete} of {len(records)} sampled records ({fraction:.0%}) were measured "
            f"incompletely, above --max-incomplete-fraction "
            f"{args.max_incomplete_fraction:.0%}:\n"
            f"    {skipped} skipped   -- the prompt filled --max-length, no assistant tokens "
            f"left to score.\n"
            f"    {truncated} truncated -- a span was scored, but {dropped} token(s) of "
            f"completion were cut off the ends.\n"
            + lengths_note +
            f"  Both only happen to conversations longer than --max-length, so what would be "
            f"reported is a length-biased estimate: the {len(encoded)} measured records are "
            f"the corpus's shorter ones, scored over the parts of their answers that fit.\n"
            f"  Raise --max-length (match the training run's), or pass "
            f"--max-incomplete-fraction 1.0 to measure the short slice deliberately.")

    dtype = getattr(torch, args.dtype)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def load_model(path):
        print(f"[model] loading {path}", flush=True)
        m = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, attn_implementation=args.attn_implementation)
        return m.to(args.device).eval()

    student = load_model(args.student_model)
    teacher = load_model(args.teacher_model) if any(m in dv.PAIRWISE for m in metrics) else None

    # A vocab-size difference survives an id-agreeing tokenizer pair (a resized embedding
    # table), and torch would broadcast or raise deep in the reduction. Checked here, where
    # the message can say which two numbers disagree.
    if teacher is not None:
        vs, vt = student.config.vocab_size, teacher.config.vocab_size
        if vs != vt:
            raise dv.MetricError(
                f"student vocab_size {vs} != teacher vocab_size {vt}. The tokenizers agree on "
                f"ids but the models' output spaces do not, so the distributions are not over "
                f"the same support.")

    sums = {m: [] for m in metrics}
    tok_counts = []
    per_sample = []
    with torch.no_grad():
        for start in range(0, len(encoded), args.batch_size):
            batch = dv.collate(encoded[start:start + args.batch_size], pad_id)
            ids = batch["input_ids"].to(args.device)
            att = batch["attention_mask"].to(args.device)
            asst = batch["assistant_mask"].to(args.device)

            s_logits = student(input_ids=ids, attention_mask=att).logits
            t_logits = teacher(input_ids=ids, attention_mask=att).logits if teacher else None

            # Shift so position i predicts token i+1, and drop the mask's first slot to match.
            # An earlier version measured logits against the mask unshifted, i.e. compared the
            # distribution that predicts token i against the mask for token i.
            s_pred = s_logits[:, :-1, :]
            t_pred = t_logits[:, :-1, :] if t_logits is not None else None
            keep = asst[:, 1:] & att[:, 1:].bool()

            for name in metrics:
                per_tok = dv.reduce_logits(s_pred, t_pred if name in dv.PAIRWISE else None, name)
                for row in range(ids.shape[0]):
                    sel = per_tok[row][keep[row]]
                    if sel.numel() == 0:
                        continue
                    sums[name].append(float(sel.mean()))

            for row in range(ids.shape[0]):
                n = int(keep[row].sum())
                if n == 0:
                    continue
                tok_counts.append(n)
                per_sample.append({"index": start + row, "assistant_tokens": n,
                                   **{m: sums[m][-1] for m in metrics if sums[m]}})

    summaries = {m: dv.summarise(sums[m], tok_counts) for m in metrics}
    return {
        "metrics": {
            "tokenizer": ident,
            "config": {"student_model": args.student_model, "teacher_model": args.teacher_model,
                       "corpus": args.corpus, "max_length": args.max_length,
                       "max_samples": args.max_samples, "batch_size": args.batch_size,
                       "seed": args.seed, "dtype": args.dtype,
                       "max_incomplete_fraction": args.max_incomplete_fraction,
                       "chat_template_source": template_source},
            # measured == truncated + fully-measured. A consumer that reads only `measured`
            # and `n_samples` cannot tell a complete run from a truncated one, which is
            # exactly the confusion a prior run walked into, so both are recorded.
            "counts": {"corpus_records": len(records), "measured": len(encoded),
                       "skipped_no_assistant_span": skipped,
                       "truncated_at_max_length": truncated,
                       "truncated_tokens_dropped": dropped,
                       "incomplete_fraction": round(incomplete / len(records), 4)
                       if records else None},
            "metrics": summaries,
        },
        "per_sample": per_sample,
    }


if __name__ == "__main__":
    raise SystemExit(main())
