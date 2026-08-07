#!/usr/bin/env python3
"""Exemplar eval entrypoint for the custom-image SkyPilot step.

Reads its parameters from CLI arguments (templated into the step.yaml `run` block
from config.eval_config), performs a placeholder "evaluation", and writes a single
results file to a fixed, well-known path (``<output-dir>/results.json``).

Unlike a training step, this workload does NOT print the Granite.build
``LLMB_ARTIFACT_ID`` line: its output is a single file at a path the step already
knows, so the step.yaml `run` block registers it. That keeps this script a plain
evaluator with no dependency on the Granite.build artifact convention.

Replace the body of ``evaluate`` with a real evaluation; the argument contract and
the fixed ``results.json`` output path are what the step depends on.
"""

import argparse
import json
import os

# The step.yaml `run` block registers this exact filename as the `results`
# output, so the name is part of the step contract — keep it in sync there.
RESULTS_FILENAME = "results.json"


def parse_args() -> argparse.Namespace:
    """Parse the eval parameters passed by the step's run block.

    :returns: Parsed arguments (model_path, tasks, output_dir, batch_size).
    """
    parser = argparse.ArgumentParser(description="Exemplar eval step")
    parser.add_argument(
        "--model-path", required=True, help="model path or id to evaluate"
    )
    parser.add_argument(
        "--tasks", default="", help="comma-separated benchmark/task names"
    )
    parser.add_argument(
        "--output-dir", required=True, help="dir the results file is written into"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> str:
    """Run the (placeholder) evaluation and write the results file.

    :param args: Parsed parameters.
    :returns: Absolute path to the written results file.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.abspath(os.path.join(args.output_dir, RESULTS_FILENAME))
    tasks = [t for t in args.tasks.split(",") if t] or ["placeholder"]
    # Real evaluation would go here. We emit placeholder per-task metrics so the
    # output is a well-formed, inspectable single artifact.
    results = {
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "metrics": {task: {"accuracy": 0.0} for task in tasks},
    }
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    return results_path


def main() -> None:
    """Entrypoint: evaluate and write results. The step.yaml registers the output."""
    args = parse_args()
    print(
        f"eval: starting on model={args.model_path} tasks={args.tasks or '(default)'}",
        flush=True,
    )
    results_path = evaluate(args)
    print(f"eval: wrote results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
