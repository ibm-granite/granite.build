#!/usr/bin/env python3
"""Refuse a pinned corpus that this build does not describe.

The corpus is DEFINED by the retagged tokenizer and by the prep policies: the rows were
rendered through that chat template, the assistant masks were built with that tokenizer,
and 5,908 rows were dropped for exceeding THAT max_length. A pin taken from a run that
differed in any of those trains on a corpus this build does not describe, and every
metric downstream would still look normal.

So this turns that into a loud failure, and does it BEFORE the multi-node GPU allocation
is held. It reads one JSON file, so the cost of having it is seconds; the cost of not
having it is a run.

Usage:
  check_corpus_pin.py <corpus_dir> <teacher_model> <max_length> <think_policy>
                      <documents_policy> <eval_fraction> <tokenizer_dir> <out.json>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def pin_problems(
    corpus: Path,
    manifest: dict,
    *,
    teacher: str,
    max_length: int,
    think_policy: str,
    documents_policy: str,
    eval_fraction: float,
    tokenizer_dir: Path,
) -> tuple[list[str], str]:
    """Every mismatch between this build and the pinned manifest, and how strictly the
    tokenizer was compared. Every problem, not the first: whoever is re-pointing a sweep
    at a corpus wants the whole list in one read, not one per relaunch."""
    problems = []

    # The splits, before anything else: a pin whose files were cleaned up is the most
    # likely way this fires, and the cheapest to say.
    for name in ("train.jsonl", "eval.jsonl"):
        if not (corpus / name).is_file():
            problems.append(f"{name} is missing from {corpus}")

    # What the corpus was built AGAINST.
    want_identity = Path(teacher.rstrip("/")).name
    got_identity = manifest.get("tokenizer_identity")
    if got_identity != want_identity:
        problems.append(
            f"tokenizer_identity {got_identity!r} != this build's teacher "
            f"{want_identity!r}"
        )

    # What the corpus was built WITH. completion_boundary is not a parameter -- the
    # recipes hard-code last_message because the eval measures divergence over the final
    # assistant turn -- so it is compared against that literal rather than a parameter.
    policies = manifest.get("policies") or {}
    for key, want in (
        ("max_length", max_length),
        ("think_policy", think_policy),
        ("documents_policy", documents_policy),
        ("completion_boundary", "last_message"),
    ):
        got = policies.get(key)
        if got != want:
            problems.append(f"policies.{key} {got!r} != {want!r}")

    if float(manifest.get("eval_fraction", -1)) != eval_fraction:
        problems.append(
            f"eval_fraction {manifest.get('eval_fraction')!r} != {eval_fraction!r}"
        )

    # Strongest available check, when it is available: the manifest records the
    # tokenizer directory it used, so if that directory still exists we compare bytes
    # rather than trusting a name. It is allowed to be gone -- a pin should outlive the
    # build that produced it -- and the report says which mode ran, so a reader never
    # has to guess how strict this was.
    pinned_tok = Path(str(manifest.get("tokenizer_path") or ""))
    mode = "identity-only (pinned tokenizer_path no longer on disk)"
    if manifest.get("tokenizer_path") and pinned_tok.is_dir():
        mode = "byte-compared against the pinned tokenizer_path"
        for name in ("tokenizer.json", "chat_template.jinja"):
            here, there = tokenizer_dir / name, pinned_tok / name
            if not there.is_file():
                problems.append(f"{name} missing from {pinned_tok}")
            elif not here.is_file():
                problems.append(f"{name} missing from {tokenizer_dir}")
            else:
                hh = hashlib.sha256(here.read_bytes()).hexdigest()
                th = hashlib.sha256(there.read_bytes()).hexdigest()
                if hh != th:
                    problems.append(
                        f"{name} differs: this build {hh[:16]} vs pinned {th[:16]}"
                    )
    return problems, mode


def main(argv: list[str]) -> int:
    corpus = Path(argv[1])
    out = Path(argv[8])

    manifest_path = corpus / "corpus_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"FATAL: no corpus_manifest.json under {corpus}")
    manifest = json.loads(manifest_path.read_text())

    problems, mode = pin_problems(
        corpus,
        manifest,
        teacher=argv[2],
        max_length=int(argv[3]),
        think_policy=argv[4],
        documents_policy=argv[5],
        eval_fraction=float(argv[6]),
        tokenizer_dir=Path(argv[7]),
    )
    if problems:
        print(f"CORPUS-PIN REJECTED {corpus}", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        raise SystemExit(
            f"FATAL [corpus-pin-check]: {len(problems)} mismatch(es) "
            f"between this build and the pinned corpus manifest"
        )

    report = {
        "corpus_dir": str(corpus),
        "accepted": True,
        "tokenizer_check": mode,
        "tokenizer_identity": manifest.get("tokenizer_identity"),
        "policies": manifest.get("policies") or {},
        "eval_fraction": manifest.get("eval_fraction"),
        "seed": manifest.get("seed"),
        "counts": manifest.get("counts"),
        "splits": manifest.get("splits"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"CORPUS-PIN ACCEPTED {corpus} ({mode})", flush=True)
    print(f"GB_ARTIFACT_ID:pin_check GB_ARTIFACT_PATH:{out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
