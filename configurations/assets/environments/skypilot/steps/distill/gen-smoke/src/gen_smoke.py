#!/usr/bin/env python3
"""Does the model still write text? Nothing upstream asks that.

df8512e0's collapse was invisible to every metric the pipeline computed during training,
and the transfer evals that could see it ran at the end. What it was NOT invisible to is
a handful of greedy completions: the distilled model emits "// (true)" fifty times and
stops. Measured degeneracy was 20.8% on Java and 30.9% on Shell against a 0.4% / 6.7%
baseline. That is one GPU-minute of evidence, and it arrived only after ~4 GPU-h of eval
had been spent confirming a result already visible in the first sample.

One run for the whole ladder rather than one per rung: the model is 350m, so loading
several of them serially on one GPU is still minutes.

Usage: gen_smoke.py <out.json> <threshold> <new_tokens> <gate_final_rung> <step:path>...

gate_final_rung is "true" or "false". When true, the run FAILS only if the LAST rung is
degenerate, because that is the rung a downstream consumer would otherwise pick up. An
early rung above threshold is a finding to read in the table, not a reason to fail a
build whose later checkpoints may be fine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Raw completion prompts, deliberately: MultiPL-E is where the collapse showed worst, and
# it applies no chat template. A prompt set that went through the template would test a
# different thing from the benchmark that caught this.
PROMPTS = [
    "def has_close_elements(numbers, threshold):\n",
    'def sum_to_n(n: int):\n    """Sum numbers from 1 to n."""\n',
    "public class Solution {\n    public static int count(int[] a) {\n",
    "#!/bin/bash\n# Print the first argument twice.\n",
    "fn factorial(n: u64) -> u64 {\n",
    "The capital of France is",
    "Q: What is 17 + 25?\nA:",
    "Write a haiku about autumn.\n",
]


def adjacent_rate(lines: list[str]) -> float:
    """Fraction of adjacent line pairs that are identical.

    Reported, not gated on. This is the shape of the detector whose numbers the df8512e0
    post-mortem records (baseline 0.4% Java / 2.1% Rust / 6.7% Shell against the
    collapsed 20.8% / 18.2% / 30.9%), so it is kept for continuity with those figures --
    but it is NOT a safe gate on short samples. A correct 7-line Java function scores
    0.167 on it, because two dedented closing braces are legitimately identical and on
    six pairs one collision is 17%. Gating on that would fail healthy checkpoints.
    """
    if len(lines) < 2:
        return 0.0
    repeats = sum(1 for a, b in zip(lines, lines[1:]) if a == b)
    return repeats / (len(lines) - 1)


def looped_fraction(lines: list[str], min_run: int = 3) -> float:
    """Fraction of lines inside a run of >= min_run identical lines.

    THIS is what gates, because it is what the failure actually looks like: the
    collapsed model wrote "// (true)" about fifty times in a row. Real code repeats a
    line twice in passing; it does not repeat one three times running. On the two
    samples in the post-mortem this separates cleanly -- 1.00 for the collapsed Java
    output, 0.00 for the correct one -- where the adjacent-pair rate puts them 0.17
    apart.

    Lines are compared with indentation intact (rstrip only), so "    }" and "}" are
    different lines, which is the other half of why the brace case does not trigger it.
    """
    if not lines:
        return 0.0
    looped, i = 0, 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        run = j - i + 1
        if run >= min_run:
            looped += run
        i = j + 1
    return looped / len(lines)


def measure(text: str) -> tuple[float, float]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return looped_fraction(lines), adjacent_rate(lines)


def parse_rungs(args: list[str]) -> list[tuple[str, str]]:
    rungs = []
    for a in args:
        step, sep, path = a.partition(":")
        if not sep or not step.isdigit() or not path:
            raise SystemExit(f"FATAL: rung {a!r} is not <step>:<path>")
        rungs.append((step, path))
    if not rungs:
        raise SystemExit("FATAL: no rungs given")
    return rungs


def summarise(step: str, path: str, looped, adjacent, samples, threshold) -> dict:
    mean_looped = sum(looped) / len(looped)
    return {
        "step": int(step),
        "path": path,
        "mean_looped_fraction": round(mean_looped, 4),
        "worst_looped_fraction": round(max(looped), 4),
        "mean_adjacent_rate": round(sum(adjacent) / len(adjacent), 4),
        "degenerate": mean_looped > threshold,
        "samples": samples,
    }


def final_rung_failure(rows: list[dict], threshold: float) -> str | None:
    """The message to fail with, or None. Only the last rung gates. The earlier ones are
    the curve."""
    last = rows[-1]
    if not last["degenerate"]:
        return None
    return (
        f"GEN-SMOKE FATAL: the final checkpoint (step {last['step']}) has "
        f"{last['mean_looped_fraction']:.1%} of its generated lines inside a repeated "
        f"run, above the {threshold:.1%} threshold. Read repetition.json before "
        f"spending full-eval on it."
    )


def generate_rung(path: str, new_tokens: int):
    """Greedy completions of PROMPTS from one checkpoint. torch and transformers are
    imported here, not at module scope, so the detector above is testable without them.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    texts = []
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # GREEDY. Sampling would hide the failure: temperature is what lets a
            # narrowed distribution still look varied, and greedy decoding of a
            # collapsed model is a fixed-point search.
            gen = model.generate(
                **ids,
                max_new_tokens=new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        texts.append(
            tok.decode(gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
        )
    del model
    torch.cuda.empty_cache()
    return texts


def run(argv: list[str], generate=generate_rung) -> int:
    out = Path(argv[1])
    threshold = float(argv[2])
    new_tokens = int(argv[3])
    if argv[4] not in ("true", "false"):
        raise SystemExit(
            f"FATAL: gate_final_rung must be true or false, got {argv[4]!r}"
        )
    gate = argv[4] == "true"
    rungs = parse_rungs(argv[5:])

    rows = []
    for step, path in rungs:
        looped, adjacent, samples = [], [], []
        for prompt, text in zip(PROMPTS, generate(path, new_tokens)):
            one_looped, one_adjacent = measure(text)
            looped.append(one_looped)
            adjacent.append(one_adjacent)
            samples.append(
                {
                    "prompt": prompt,
                    "completion": text[:400],
                    "looped_fraction": round(one_looped, 4),
                }
            )
        row = summarise(step, path, looped, adjacent, samples, threshold)
        rows.append(row)
        print(
            f"GEN-SMOKE step={step} looped={row['mean_looped_fraction']:.4f} "
            f"worst_looped={row['worst_looped_fraction']:.4f} "
            f"adjacent={row['mean_adjacent_rate']:.4f} "
            f"threshold={threshold} "
            f"{'DEGENERATE' if row['degenerate'] else 'ok'}",
            flush=True,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"threshold": threshold, "rungs": rows}, indent=2))
    print(f"GB_ARTIFACT_ID:repetition_report GB_ARTIFACT_PATH:{out}", flush=True)

    print("GEN-SMOKE table (step, looped, worst, adjacent, verdict):", flush=True)
    for r in rows:
        print(
            f"  {r['step']:>6}  {r['mean_looped_fraction']:.4f}  "
            f"{r['worst_looped_fraction']:.4f}  "
            f"{r['mean_adjacent_rate']:.4f}  "
            f"{'DEGENERATE' if r['degenerate'] else 'ok'}",
            flush=True,
        )

    failure = final_rung_failure(rows, threshold)
    if failure and gate:
        print(failure, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv))
