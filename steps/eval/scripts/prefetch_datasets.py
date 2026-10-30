# Copyright 2026 Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download the Hugging Face datasets the eval benchmarks read into an HF cache.

Run it once on a node with internet access (a laptop or login node), pointing
--hf-home at the directory the environment mounts into the eval containers as
the HF cache. Jobs on air-gapped compute nodes then read the pinned snapshots
from the cache (HF_HUB_OFFLINE=1).

Every entry is pinned to the commit the benchmark module pins
(<family>/src/*.py); `--check` verifies that against the source.
Datasets that are not on the Hub (GitHub files, PyPI packages, task data) are
pinned by commit/sha256 and baked into the image by the owning family's extras.sh instead.

    python scripts/prefetch_datasets.py --hf-home /path/to/hf-cache [--benchmark gpqa ...]

Gated datasets (gpqa, hle, ...) need HF_TOKEN with access granted.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# benchmark id -> (HF dataset repo, pinned revision, family)
DATASETS: dict[str, tuple[str, str, str]] = {
    "hmmt-feb25": (
        "MathArena/hmmt_feb_2025",
        "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
        "nemoskills",
    ),
    "gpqa": (
        "Idavidrein/gpqa",
        "83022cefff930aea54f654c0b282e74b9eeda5c6",
        "nemoskills",
    ),
    "mmlu-pro": (
        "TIGER-Lab/MMLU-Pro",
        "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "nemoskills",
    ),
    "livecodebench-v6": (
        "livecodebench/code_generation_lite",
        "c52cd175916e995019dcd848d1054b419d2e70b5",
        "nemoskills",
    ),
    "scicode": (
        "SciCode1/SciCode",
        "4510f6a6aa27c43fad7b43da2c59602a86e88480",
        "nemoskills",
    ),
    "hle": ("cais/hle", "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29", "nemoskills"),
    "omniscience": (
        "ArtificialAnalysis/AA-Omniscience-Public",
        "e4883edbb9f5ccf2b2a8fdc6fb65e01a58e99849",
        "nemoskills",
    ),
    "critpt": (
        "CritPt-Benchmark/CritPt",
        "9b9fc8498596ec08ab5437a72f4aa18beef2b876",
        "nemoskills",
    ),
    "aa-lcr": (
        "ArtificialAnalysis/AA-LCR",
        "9a77ef56b717057ade24ceab4d273712a0b4f19e",
        "nemoskills",
    ),
    "wmt24pp": (
        "google/wmt24pp",
        "fd7405c06494bc66a57b25f55d217a72f96e60dc",
        "nemoskills",
    ),
    "mmlu-prox-lite": (
        "li-lab/MMLU-ProX-Lite",
        "e82aafb9460529687d3c7e51b401d8dd1dd309dd",
        "lmeval",
    ),
    "terminal-bench-2.1": (
        "harborframework/terminal-bench-2.1",
        "3e235dff6880252a587fa479c09bcb1e16edf2eb",
        "tbench",
    ),
    "mcpatlas": (
        "ScaleAI/MCP-Atlas",
        "8c563b55d7c967755f474299848049834d624617",
        "judged",
    ),
    "profbench": (
        "nvidia/ProfBench",
        "f5c471cf2f277c264621e4b5c0cad5bb466f9260",
        "judged",
    ),
    "gdpval": ("openai/gdpval", "11e7900cdcac61bc4daf59e65feb238acda98fbf", "judged"),
    "fortress": (
        "ScaleAI/fortress_public",
        "0c096becbc75bb12065c8059a53960c7f0d4d35c",
        "judged",
    ),
    "swebench-verified": (
        "SWE-bench/SWE-bench_Verified",
        "78f471bf655a3137b2e8a75af1501690ec009ec3",
        "swebench",
    ),
    "swebench-multilingual": (
        "SWE-bench/SWE-bench_Multilingual",
        "846e647b9f33c0b51b739d005d13d85493c9af09",
        "swebench",
    ),
    "swebench-pro": (
        "ScaleAI/SWE-bench_Pro",
        "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50",
        "swebench",
    ),
}

# The shared core lives in src/core/; each family's modules in <family>/src/.
ROOT = Path(__file__).resolve().parent.parent


def check_pins() -> list[str]:
    """Every (repo, revision) above must appear verbatim in the family's source."""
    problems = []
    for bench, (repo, rev, fam) in DATASETS.items():
        text = "\n".join(
            p.read_text()
            for p in (ROOT / fam / "src").rglob("*.py")
            if "tests" not in p.parts
        )
        text += "\n".join(
            p.read_text()
            for p in (ROOT / "src" / "core").rglob("*.py")
            if "tests" not in p.parts
        )
        if repo not in text or rev not in text:
            problems.append(f"{bench}: {repo}@{rev} not found in {fam}/src or src/core")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--hf-home",
        type=Path,
        help="HF cache root (the directory mounted as HF_HOME in the containers)",
    )
    ap.add_argument(
        "--benchmark",
        action="append",
        help="only this benchmark id (repeatable); default: all",
    )
    ap.add_argument("--list", action="store_true", help="print the datasets and exit")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the pins against the vendored source and exit",
    )
    args = ap.parse_args(argv)

    if args.check:
        problems = check_pins()
        print(
            "\n".join(problems)
            if problems
            else f"OK: {len(DATASETS)} pins match the source"
        )
        return 1 if problems else 0
    chosen = args.benchmark or sorted(DATASETS)
    unknown = [b for b in chosen if b not in DATASETS]
    if unknown:
        ap.error(f"unknown benchmark(s) {unknown}; known: {sorted(DATASETS)}")
    if args.list:
        for b in chosen:
            repo, rev, fam = DATASETS[b]
            print(f"{b:24s} {fam:11s} {repo}@{rev[:12]}")
        return 0
    if not args.hf_home:
        ap.error("--hf-home is required")

    # Set before importing huggingface_hub: it reads HF_HOME at import time.
    os.environ["HF_HOME"] = str(args.hf_home)
    from huggingface_hub import snapshot_download

    args.hf_home.mkdir(parents=True, exist_ok=True)
    failed = []
    for b in chosen:
        repo, rev, _ = DATASETS[b]
        try:
            path = snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                revision=rev,
                token=os.environ.get("HF_TOKEN"),
            )
            print(f"ok    {b:24s} {repo}@{rev[:12]} -> {path}")
        except Exception as e:  # noqa: BLE001 - report every dataset, then fail once
            failed.append(b)
            reason = re.sub(r"\s+", " ", str(e))[:160]
            print(f"FAIL  {b:24s} {repo}@{rev[:12]}: {reason}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
