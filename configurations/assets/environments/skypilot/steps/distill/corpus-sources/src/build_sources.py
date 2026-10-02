#!/usr/bin/env python3
"""Build ONE corpus file from several SFT splits, because three things are true at once
and no single one of them is optional:

  1. prep takes ONE path. prep_corpus.py:294 refuses a directory outright -- "Pass the
     .jsonl itself, or an HF dataset id" -- so there is no glob and no list form to hand
     it three files with.

  2. THE ROWS ARE SPELLED `conversations`, NOT `messages`. prep reads
     record.get("messages") (prep_corpus.py:367) and returns `no_messages` otherwise.
     Handed these files unchanged it would drop all 5,081,504 rows and raise PrepError.
     This rename is the single most load-bearing line in the step.

  3. The splits are wildly unequal -- general 84.86%, tools 13.98%, rag 1.17% -- and
     prep's max_examples stops after N KEPT rows IN INPUT ORDER. A plain concatenation
     plus max_examples would therefore yield a subset of pure `general` with no tools
     and no rag in it, silently. Sampling proportionally HERE, and shuffling, is what
     makes a subset representative rather than just short.

Reservoir sampling per split, seeded, so the selection is reproducible from the seed
alone and memory is bounded by the sample rather than the corpus. Two passes: one to
count, one to select, because the per-split quota cannot be computed until the totals
are known.

Usage: build_sources.py <out.jsonl> <target_rows> <seed> <source.jsonl>...
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def quotas(counts: list[int], target: int) -> list[int]:
    """Rows to take from each split. target 0 means "all rows"; anything larger than the
    corpus is the same."""
    total = sum(counts)
    take_all = target <= 0 or target >= total
    return [c if take_all else max(1, round(target * c / total)) for c in counts]


def sample_split(
    path: Path, quota: int, seed: int, tally: dict[str, int]
) -> list[dict]:
    """Reservoir-sample `quota` usable rows from one split, renaming as it goes.

    Per-split reservoirs keep the SHARE of each split fixed at its share of the corpus,
    which a single reservoir over the concatenation would only achieve in expectation.
    """
    rng = random.Random(f"{seed}:{path.name}")
    reservoir: list[dict] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                tally["bad"] += 1
                continue
            # THE RENAME. prep reads record["messages"] and drops everything else as
            # `no_messages`.
            if "messages" in rec:
                tally["already"] += 1
            elif "conversations" in rec:
                rec["messages"] = rec.pop("conversations")
                tally["renamed"] += 1
            else:
                tally["bad"] += 1
                continue
            if len(reservoir) < quota:
                reservoir.append(rec)
            else:
                j = rng.randint(0, i)
                if j < quota:
                    reservoir[j] = rec
    return reservoir


def main(argv: list[str]) -> int:
    out = Path(argv[1])
    target = int(argv[2])
    seed = int(argv[3])
    srcs = [Path(p) for p in argv[4:]]
    if not srcs:
        raise SystemExit("FATAL: no sources given")

    for p in srcs:
        if not p.is_file():
            raise SystemExit(f"FATAL: source is not a file: {p}")

    # ---- pass 1: count -------------------------------------------------
    counts = []
    for p in srcs:
        with p.open() as fh:
            counts.append(sum(1 for _ in fh))
        print(f"SOURCES count {p.name} {counts[-1]}", flush=True)
    total = sum(counts)
    print(f"SOURCES count TOTAL {total}", flush=True)
    if total == 0:
        raise SystemExit("FATAL: every source is empty")

    qs = quotas(counts, target)
    print(
        "SOURCES quota "
        + " ".join(
            f"{p.name}={q}({q/max(t,1):.4%})" for p, q, t in zip(srcs, qs, counts)
        ),
        flush=True,
    )

    # ---- pass 2: reservoir-sample each split ---------------------------
    tally = {"renamed": 0, "already": 0, "bad": 0}
    picked: list[dict] = []
    for p, q in zip(srcs, qs):
        reservoir = sample_split(p, q, seed, tally)
        print(f"SOURCES sampled {p.name} {len(reservoir)}", flush=True)
        picked.extend(reservoir)

    if tally["bad"]:
        print(f"SOURCES WARNING unusable rows skipped: {tally['bad']}", flush=True)
    # A corpus where NOTHING carried either spelling is a schema change, not a bad row:
    # fail rather than write an empty file prep will then refuse less informatively.
    if tally["renamed"] == 0 and tally["already"] == 0:
        raise SystemExit(
            "FATAL: no row carried `conversations` or `messages`. The source "
            "schema has changed; prep would drop every row as no_messages."
        )

    # Interleave the splits so an early truncation downstream still sees all three, and
    # so batches are mixed rather than blocked by domain.
    random.Random(seed).shuffle(picked)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for rec in picked:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"SOURCES renamed={tally['renamed']} already_messages={tally['already']} "
        f"bad={tally['bad']}",
        flush=True,
    )
    print(f"SOURCES wrote {len(picked)} rows -> {out}", flush=True)
    print(f"GB_ARTIFACT_ID:corpus_source GB_ARTIFACT_PATH:{out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
