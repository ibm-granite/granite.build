# SPDX-License-Identifier: Apache-2.0
"""Download the Hugging Face datasets the eval benchmarks read into an HF cache.

Run it once on a node with internet access (a laptop or login node), pointing
--hf-home at the directory the environment mounts into the eval containers as
the HF cache. Jobs on air-gapped compute nodes then read the pinned data from
the cache (hf_offline: HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1).

Two caches, because benchmarks read the Hub two ways. Every entry is fetched
as a hub snapshot ($HF_HOME/hub), which serves snapshot_download and
hf_hub_download offline. A benchmark that reads through datasets.load_dataset
(directly, via granite_evals.data.load_split, or inside lm-eval and ns
prepare.py) needs more: offline, datasets 5 loads a hub id only from its
prepared cache ($HF_HOME/datasets), which a snapshot does not build. For those
entries (LOAD_CONFIGS) the prefetch also runs load_dataset for each config the
benchmark requests, at the pinned revision. A load without a revision (lm-eval
task yamls, e.g. the lm-eval bridge's gpqa_diamond_zeroshot) resolves offline
to that same prepared copy, so the gpqa entry serves it too. A non-default
selection option (languages=, ...) needs its configs warmed with --config.

Every entry is pinned to the commit the benchmark module pins
(<family>/src/*.py); `--check` verifies that against the source.
Datasets that are not on the Hub (GitHub files, PyPI packages, task data) are
pinned by commit/sha256 and baked into the image by the owning family's extras.sh instead.

    python scripts/prefetch_datasets.py --hf-home /path/to/hf-cache [--benchmark gpqa ...]
    python scripts/prefetch_datasets.py --hf-home ... --benchmark mmlu-prox-lite --config ko

Run it with a Python whose datasets matches the images' lock (steps/eval/src/
core/uv.lock: datasets 5.0.1, huggingface-hub 1.33.0): the jobs read the
prepared cache back with that version.

Gated datasets (gpqa, hle, officeqa, ...) need HF_TOKEN with access granted.
Entries that carry an allow_patterns list fetch only those files, matching what
the family's adapter downloads at run start (a warm cache must hold at least
that much, or an air-gapped job cannot complete the same fetch).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# benchmark id -> (HF dataset repo, pinned revision, family[, allow_patterns]);
# the optional allow_patterns restricts the fetch, as described in the docstring.
DATASETS: dict[str, tuple[str, str, str] | tuple[str, str, str, tuple[str, ...]]] = {
    "hmmt-feb25": (
        "MathArena/hmmt_feb_2025",
        "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
        "nemoskills",
    ),
    "aime26": (
        "MathArena/aime_2026",
        "d61f486805306e6e6b5913504c07763a99e1b112",
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
    "human-eval": (
        "openai/openai_humaneval",
        "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "nemoskills",
    ),
    "mbpp": (
        "google-research-datasets/mbpp",
        "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
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
    "browsecomp-plus": (
        "Tevatron/browsecomp-plus",
        "144cff8e35b5eaef7e526346aa60774a9deb941f",
        "nemoskills",
    ),
    "wmt24pp": (
        "google/wmt24pp",
        "fd7405c06494bc66a57b25f55d217a72f96e60dc",
        "nemoskills",
    ),
    "flores200": (
        "openlanguagedata/flores_plus",
        "1147e2dc59f37fbb5c22fd8478770de02cd399f9",
        "nemoskills",
    ),
    "mmlu-prox-lite": (
        "li-lab/MMLU-ProX-Lite",
        "e82aafb9460529687d3c7e51b401d8dd1dd309dd",
        "lmeval",
    ),
    "include-44": (
        "CohereLabs/include-base-44",
        "d2e1f6015f67a43c02a9a68db98e2298e2d6a660",
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
    # Both officeqa benchmarks read the same snapshot (the two question CSVs,
    # the parsed Treasury Bulletin corpus and the license files — exactly what
    # the adapter's ALLOW_PATTERNS fetches), so either entry warms the other.
    # The ~4 GB of PDFs and the LFS zips are deliberately excluded. Gated
    # (CC-BY-SA-4.0 click-through): HF_TOKEN needs the terms accepted.
    "officeqa-pro": (
        "databricks/OfficeQA",
        "763a8366abf2a3605c381d53586d844dc60fa756",
        "officeqa",
        (
            "officeqa_pro.csv",
            "officeqa_full.csv",
            "treasury_bulletins_parsed/jsons/*.json",
            "LICENSE-APACHE",
            "LICENSE-CC-BY-SA",
            "NOTICE",
        ),
    ),
    "officeqa-full": (
        "databricks/OfficeQA",
        "763a8366abf2a3605c381d53586d844dc60fa756",
        "officeqa",
        (
            "officeqa_pro.csv",
            "officeqa_full.csv",
            "treasury_bulletins_parsed/jsons/*.json",
            "LICENSE-APACHE",
            "LICENSE-CC-BY-SA",
            "NOTICE",
        ),
    ),
}

# benchmark id -> the configs its reader passes to datasets.load_dataset (None:
# the default config), each prepared at the entry's pinned revision (see the
# docstring). The defaults each benchmark requests with no selection option;
# where they are defined:
#   ns prepare.py reads (nemo_skills_*.py classes without a prepare_data):
#     NeMo-Skills' nemo_skills/dataset/<name>/prepare.py, with the class's
#     ns_prepare_args (gpqa: --split diamond -> gpqa_diamond; livecodebench:
#     release_v6; wmt24pp: en-<WMT_LANGUAGES>)
#   load_split reads: human-eval/mbpp (nemo_skills_codegen.py, MBPP_CONFIG),
#     browsecomp-plus (tools_agentic.py), flores200 (eng_Latn plus
#     flores_config() of FLORES_LANGUAGES, nemo_skills_extended.py), profbench
#     and gdpval (judge_general.py), the swebench sets (dataset_config(); Pro's
#     subset=default)
#   lm-eval reads: mmlu-prox-lite (chat_if.py OFFICIAL_LANGUAGES), include-44
#     (include.py LANGUAGES, all 44)
# Not listed (snapshot only): officeqa (snapshot_download), mcpatlas and
# fortress (hf_hub_download of one pinned file), terminal-bench-2.1
# (snapshot_download into a local_dir, which offline never reads the cache
# for: tbench cannot run under hf_offline).
LOAD_CONFIGS: dict[str, tuple[str | None, ...]] = {
    "hmmt-feb25": (None,),
    "aime26": (None,),
    "gpqa": ("gpqa_diamond",),
    "mmlu-pro": (None,),
    "livecodebench-v6": ("release_v6",),
    "scicode": (None,),
    "human-eval": (None,),
    "mbpp": ("sanitized",),
    "hle": (None,),
    "omniscience": (None,),
    "critpt": (None,),
    "aa-lcr": (None,),
    "browsecomp-plus": (None,),
    "wmt24pp": ("en-de_DE", "en-es_MX", "en-fr_FR", "en-it_IT", "en-ja_JP"),
    "flores200": (
        "eng_Latn",
        "arb_Arab",
        "ces_Latn",
        "deu_Latn",
        "spa_Latn",
        "fra_Latn",
        "ita_Latn",
        "jpn_Jpan",
        "kor_Hang",
        "nld_Latn",
        "por_Latn",
        "cmn_Hans",
    ),
    "mmlu-prox-lite": ("en", "de", "es", "fr", "it", "ja"),
    "include-44": (
        "Albanian",
        "Arabic",
        "Armenian",
        "Azerbaijani",
        "Basque",
        "Belarusian",
        "Bengali",
        "Bulgarian",
        "Chinese",
        "Croatian",
        "Dutch",
        "Estonian",
        "Finnish",
        "French",
        "Georgian",
        "German",
        "Greek",
        "Hebrew",
        "Hindi",
        "Hungarian",
        "Indonesian",
        "Italian",
        "Japanese",
        "Kazakh",
        "Korean",
        "Lithuanian",
        "Malay",
        "Malayalam",
        "Nepali",
        "North Macedonian",
        "Persian",
        "Polish",
        "Portuguese",
        "Russian",
        "Serbian",
        "Spanish",
        "Tagalog",
        "Tamil",
        "Telugu",
        "Turkish",
        "Ukrainian",
        "Urdu",
        "Uzbek",
        "Vietnamese",
    ),
    "profbench": (None,),
    "gdpval": (None,),
    "swebench-verified": (None,),
    "swebench-multilingual": (None,),
    "swebench-pro": ("default",),
}

# The shared core lives in src/core/; each family's modules in <family>/src/.
ROOT = Path(__file__).resolve().parent.parent


def check_pins() -> list[str]:
    """Every (repo, revision) above must appear verbatim in the family's source."""
    problems = []
    for bench, entry in DATASETS.items():
        repo, rev, fam = entry[0], entry[1], entry[2]
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
    problems += [
        f"{bench}: LOAD_CONFIGS entry has no DATASETS pin"
        for bench in LOAD_CONFIGS
        if bench not in DATASETS
    ]
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
    ap.add_argument(
        "--config",
        action="append",
        default=[],
        help="also prepare this load_dataset config (repeatable; needs exactly one "
        "--benchmark): a non-default selection, e.g. mmlu-prox-lite languages=ko",
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
    if args.config and len(chosen) != 1:
        ap.error("--config needs exactly one --benchmark")
    if args.list:
        for b in chosen:
            repo, rev, fam = DATASETS[b][:3]
            n = len(LOAD_CONFIGS.get(b, ()))
            how = f"snapshot + {n} load_dataset config(s)" if n else "snapshot"
            print(f"{b:24s} {fam:11s} {repo}@{rev[:12]}  {how}")
        return 0
    if not args.hf_home:
        ap.error("--hf-home is required")

    # Set before importing huggingface_hub / datasets: they read HF_HOME at
    # import time.
    os.environ["HF_HOME"] = str(args.hf_home)
    from huggingface_hub import snapshot_download

    args.hf_home.mkdir(parents=True, exist_ok=True)
    failed = []
    for b in chosen:
        entry = DATASETS[b]
        repo, rev = entry[0], entry[1]
        patterns = entry[3] if len(entry) > 3 else None
        try:
            path = snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                revision=rev,
                token=os.environ.get("HF_TOKEN"),
                allow_patterns=patterns,
            )
            print(f"ok    {b:24s} {repo}@{rev[:12]} -> {path}")
            configs = [*LOAD_CONFIGS.get(b, ()), *args.config]
            for config in dict.fromkeys(configs):
                warm_load_dataset(repo, rev, config)
                print(f"ok    {b:24s} {repo}@{rev[:12]} load_dataset config={config}")
        except Exception as e:  # noqa: BLE001 - report every dataset, then fail once
            failed.append(b)
            reason = re.sub(r"\s+", " ", str(e))[:160]
            print(f"FAIL  {b:24s} {repo}@{rev[:12]}: {reason}", file=sys.stderr)
    return 1 if failed else 0


def warm_load_dataset(repo: str, rev: str, config: str | None) -> None:
    """Build the datasets prepared cache for one config at the pinned
    revision (every split), as an offline load_dataset needs it."""
    import datasets

    datasets.load_dataset(
        repo, name=config, revision=rev, token=os.environ.get("HF_TOKEN")
    )


if __name__ == "__main__":
    sys.exit(main())
