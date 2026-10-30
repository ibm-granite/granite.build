"""``granite-evals`` command line.

    granite-evals run swebench-verified --model /path/to/hf_model --output-dir out --limit 5
    granite-evals list [--suite granite42]
    granite-evals spend [ledger.jsonl]

``run`` is what every granite.build ``granite-*`` step calls. It serves the model
with vLLM (unless ``--base-url`` points at an existing server), runs one
benchmark and writes ``<output-dir>/results.json``. ``--phase generate`` stops
before scoring and writes ``generation.json``; ``--phase score`` on the same
output dir serves nothing and grades it (see ``registry``). The step, not this CLI,
prints the ``GB_ARTIFACT_ID`` marker, so the artifact contract stays visible in
the step template.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path

from granite_evals import meter, registry, suites
from granite_evals.registry import PHASES, RunConfig
from granite_evals.results import read_generation, write_results
from granite_evals.serving import ServerConfig, VLLMServer

def _kv(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--option expects key=value, got {pair!r}")
        out[key] = value
    return out


def _add_run(sub) -> None:
    p = sub.add_parser("run", help="run one benchmark")
    p.add_argument("benchmark")
    p.add_argument("--model", required=True, help="HF model dir or hub id")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--served-model-name", default="", help="default: basename of --model")
    p.add_argument("--base-url", default="", help="existing OpenAI-compatible endpoint; skips vLLM startup")
    p.add_argument("--limit", type=int, default=None, help="smoke mode: first N examples only")
    p.add_argument("--repeats", type=int, default=None, help="override pass@1[avg-of-k] repeats")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="", help="hub id or local path overriding the benchmark's pinned dataset")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--option", action="append", default=[], metavar="K=V", help="benchmark-specific option")
    p.add_argument("--phase", choices=PHASES, default="all",
                   help="generate: serve the model and save its outputs; score: grade them without a model (CPU)")
    g = p.add_argument_group("vLLM")
    g.add_argument("--tensor-parallel-size", type=int, default=1)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    g.add_argument("--max-model-len", type=int, default=None)
    g.add_argument("--tool-call-parser", default="auto")
    g.add_argument("--reasoning-parser", default="auto")
    g.add_argument("--vllm-arg", action="append", default=[], help="extra `vllm serve` argument (repeatable)")


def cmd_run(args) -> int:
    served = args.served_model_name or Path(args.model.rstrip("/")).name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = RunConfig(
        phase=args.phase,
        model=args.model,
        output_dir=args.output_dir,
        base_url=args.base_url,
        served_model_name=served,
        limit=args.limit,
        repeats=args.repeats,
        workers=args.workers,
        seed=args.seed,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        options=_kv(args.option),
    )
    benchmark = registry.get(args.benchmark)(config)
    generation = None
    if args.phase != "all":
        if not benchmark.splittable:
            raise SystemExit(f"{benchmark.id} scores inside generation; run it with --phase all")
        if args.phase == "score":
            if benchmark.score_needs_server():
                raise SystemExit(f"{benchmark.id}: these options score with the served model; run --phase all")
            generation = read_generation(benchmark)
            served = args.served_model_name or generation["served_model_name"]
            config.served_model_name = served
    started = time.time()

    if args.base_url or not benchmark.serves():
        server = contextlib.nullcontext()
        base_url = args.base_url
    else:
        server = VLLMServer(
            ServerConfig(
                model=args.model,
                served_model_name=served,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                tool_call_parser=args.tool_call_parser,
                reasoning_parser=args.reasoning_parser,
                extra_args=args.vllm_arg,
            ),
            log_path=args.output_dir / "vllm.log",
        )
        base_url = server.base_url

    tags = {"benchmark": benchmark.id, "job": os.environ.get("LSB_JOBID", ""), "run": str(args.output_dir)}
    with server, meter.Meters(config.options, tags) as meters:
        outcome = benchmark.run(base_url, served)
    if (spend := meters.summary()) is not None:
        outcome["api_spend"] = spend
        print(f"granite-evals: paid API spend ${spend['usd']:.4f} over {spend['calls']} calls")
    path = write_results(benchmark, outcome, started=started, served_model_name=served, generation=generation)
    if args.phase == "generate":
        print(f"granite-evals: {benchmark.id} generated {json.loads(path.read_text())['n']}; score with --phase score")
        print(f"granite-evals: generation {path.resolve()}")
        return 0
    print(f"granite-evals: {benchmark.id} = {benchmark_value(path)} ({benchmark.metric})")
    if outcome.get("incomplete"):
        print("granite-evals: INCOMPLETE: some items failed and were left out; rerun to retry them")
    print(f"granite-evals: results {path.resolve()}")
    return 0


def benchmark_value(path: Path) -> str:
    return f"{json.loads(path.read_text())['value']:.4f}"


def cmd_list(args) -> int:
    implemented = registry.all_benchmarks()
    if args.suite:
        for b in suites.load(args.suite)["benchmarks"]:
            mark = "x" if b["id"] in implemented else " "
            print(f"[{mark}] {b['id']:<26} {b['group']:<16} {b['name']} / {b['metric']}")
    else:
        for bid, cls in implemented.items():
            print(f"{bid:<26} {cls.metric}  (extra: {cls.extra or '-'})")
    return 0


def cmd_spend(args) -> int:
    print(json.dumps(meter.report(args.ledger), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="granite-evals")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)
    p = sub.add_parser("list", help="list implemented benchmarks, or a suite's bring-up status")
    p.add_argument("--suite", choices=suites.names())
    p = sub.add_parser("spend", help="paid API spend recorded in a ledger")
    p.add_argument("ledger", type=Path, nargs="?", default=Path(os.environ.get("GRANITE_EVALS_SPEND_LEDGER", "spend.jsonl")))
    args = parser.parse_args(argv)
    return {"run": cmd_run, "list": cmd_list, "spend": cmd_spend}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
