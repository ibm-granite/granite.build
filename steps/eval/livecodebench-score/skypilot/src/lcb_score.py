#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score LiveCodeBench completions produced by ``eval/livecodebench-generate``.

Run from inside the LiveCodeBench checkout (its prompt files load by relative
path). It:

1. reads the completions directory (``completions.json`` + ``generation_meta.json``);
2. refuses when the completions were generated for a different release or date
   window than this step is asked to score, or for a different problem set than
   the dataset holds for that window. LiveCodeBench would otherwise fail with a
   bare ``AssertionError: N != M``, or, worse, score mismatched problems;
3. copies ``completions.json`` into the output directory, because LiveCodeBench
   writes its result files next to its input;
4. runs LiveCodeBench's own scorer, ``lcb_runner.runner.custom_evaluator``,
   unmodified, with the dataset read from the local copy (``lcb_dataset``);
5. writes ``summary.json`` (pass@1 overall and by difficulty) and prints it.

Generated code is executed by LiveCodeBench's checker in subprocesses with
timeouts and resource limits. That guard is NOT a security sandbox; isolation
comes from running this step on its own, with no credentials in its
environment (the step refuses to start with ``HF_TOKEN`` set).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import lcb_dataset

COMPLETIONS = "completions.json"
META = "generation_meta.json"
SUMMARY = "summary.json"
#: The scenario this step scores; the others need different inputs.
SCENARIO = "codegeneration"


def question_ids_sha256(ids: List[str]) -> str:
    """Hash a problem set independently of order, the same way generate does."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def load_completions(
    completions_dir: Path,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read and sanity-check the generate step's output directory."""
    comp_path = completions_dir / COMPLETIONS
    meta_path = completions_dir / META
    for p in (comp_path, meta_path):
        if not p.is_file():
            raise ValueError(
                f"{p} not found; is this the output of eval/livecodebench-generate?"
            )
    completions = json.loads(comp_path.read_text())
    meta = json.loads(meta_path.read_text())
    if not isinstance(completions, list) or not completions:
        raise ValueError(f"{comp_path} must be a non-empty JSON list")
    for i, c in enumerate(completions):
        if (
            not isinstance(c, dict)
            or "question_id" not in c
            or not isinstance(c.get("code_list"), list)
        ):
            raise ValueError(
                f"{comp_path} entry {i} needs question_id and a code_list list"
            )
    return completions, meta


def check_matches(
    meta: Dict[str, Any],
    completions: List[Dict[str, Any]],
    release_version: str,
    start_date: Optional[str],
    end_date: Optional[str],
    benchmark_ids: List[str],
) -> None:
    """Refuse completions that were generated for something else.

    :raises ValueError: naming every mismatch found.
    """
    problems = []
    for key, want in (
        ("release_version", release_version),
        ("start_date", start_date),
        ("end_date", end_date),
    ):
        got = meta.get(key) or None
        if (want or None) != got:
            problems.append(
                f"{key}: completions were generated for {got!r}, this step scores {want or None!r}"
            )
    comp_ids = [c["question_id"] for c in completions]
    if len(set(comp_ids)) != len(comp_ids):
        problems.append("completions.json lists a question_id more than once")
    if question_ids_sha256(comp_ids) != question_ids_sha256(benchmark_ids):
        missing = sorted(set(benchmark_ids) - set(comp_ids))
        extra = sorted(set(comp_ids) - set(benchmark_ids))
        problems.append(
            f"problem set differs from the dataset's window: {len(comp_ids)} completions vs "
            f"{len(benchmark_ids)} problems; missing {missing[:5]}{'...' if len(missing) > 5 else ''}, "
            f"extra {extra[:5]}{'...' if len(extra) > 5 else ''}"
        )
    if problems:
        raise ValueError(
            "completions do not match this scoring run:\n  " + "\n  ".join(problems)
        )


def summarize(
    eval_all: List[Dict[str, Any]], meta: Dict[str, Any], window: Dict[str, Any]
) -> Dict[str, Any]:
    """pass@1 overall and per difficulty, from LiveCodeBench's per-problem records."""
    by_diff: Dict[str, List[float]] = defaultdict(list)
    for r in eval_all:
        by_diff[str(r.get("difficulty", "unknown"))].append(float(r["pass@1"]))
    scores = [float(r["pass@1"]) for r in eval_all]
    return {
        "scenario": SCENARIO,
        "problems": len(scores),
        "pass@1": sum(scores) / len(scores) if scores else 0.0,
        "by_difficulty": {
            d: {"problems": len(v), "pass@1": sum(v) / len(v)}
            for d, v in sorted(by_diff.items())
        },
        **window,
        "generation": {
            k: meta.get(k)
            for k in (
                "model",
                "n",
                "temperature",
                "max_tokens",
                "enable_thinking",
                "lcb_commit",
            )
        },
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--completions-dir", required=True, help="the generate step's output directory"
    )
    ap.add_argument(
        "--dataset-dir",
        required=True,
        help="local copy of the livecodebench/code_generation_lite repo",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="where the result files are written (created)",
    )
    ap.add_argument("--release-version", required=True)
    ap.add_argument("--start-date", default="")
    ap.add_argument("--end-date", default="")
    ap.add_argument(
        "--timeout",
        type=int,
        default=6,
        help="per-test-case seconds (LiveCodeBench's default)",
    )
    ap.add_argument(
        "--num-process-evaluate",
        type=int,
        default=0,
        help="scorer processes; 0 = LiveCodeBench's default",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    start = args.start_date or None
    end = args.end_date or None
    window = {
        "release_version": args.release_version,
        "start_date": start,
        "end_date": end,
    }

    lcb_dataset.install(args.dataset_dir)
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

    completions, meta = load_completions(Path(args.completions_dir))
    benchmark = load_code_generation_dataset(
        args.release_version, start_date=start, end_date=end
    )
    if not benchmark:
        raise ValueError(
            f"no problems in {args.release_version} for window {start}..{end}"
        )
    check_matches(
        meta,
        completions,
        args.release_version,
        start,
        end,
        [p.question_id for p in benchmark],
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    staged = out / COMPLETIONS
    shutil.copyfile(Path(args.completions_dir) / COMPLETIONS, staged)
    shutil.copyfile(Path(args.completions_dir) / META, out / META)

    evaluator_argv = [
        "custom_evaluator",
        "--custom_output_file",
        str(staged),
        "--scenario",
        SCENARIO,
        "--release_version",
        args.release_version,
        "--timeout",
        str(args.timeout),
    ]
    if start:
        evaluator_argv += ["--start_date", start]
    if end:
        evaluator_argv += ["--end_date", end]
    if args.num_process_evaluate > 0:
        evaluator_argv += ["--num_process_evaluate", str(args.num_process_evaluate)]

    saved_argv = sys.argv
    sys.argv = evaluator_argv
    try:
        runpy.run_module("lcb_runner.runner.custom_evaluator", run_name="__main__")
    finally:
        sys.argv = saved_argv

    stem = staged.stem
    eval_all_path = out / f"{stem}_{SCENARIO}_output_eval_all.json"
    if not eval_all_path.is_file():
        raise RuntimeError(f"LiveCodeBench did not write {eval_all_path.name}")
    summary = summarize(json.loads(eval_all_path.read_text()), meta, window)
    (out / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n")
    print("eval/livecodebench-score: " + json.dumps(summary))
    return 0


if (
    __name__ == "__main__"
):  # the guard matters: LiveCodeBench's scorer uses multiprocessing
    try:
        sys.exit(main())
    except ValueError as e:
        print(f"eval/livecodebench-score: ERROR {e}", file=sys.stderr)
        sys.exit(2)
