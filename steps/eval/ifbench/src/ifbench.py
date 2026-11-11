"""IFBench (AllenAI) through NeMo-Skills; the `ifbench` benchmark family."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from granite_evals.benchmarks.nemo_skills import NemoSkillsBenchmark
from granite_evals.registry import register

IFBENCH_DATA_COMMIT = "fcd289db21d43aaa96c6d9291d32561cd6e19305"
"""allenai/IFBench main when the pinned ns commit was made (2026-09-11): the
last change to data/IFBench_test.jsonl (2026-08-30). ns's prepare.py reads the
file from ``refs/heads/main``; this is that read, pinned."""
IFBENCH_TEST_URL = "https://raw.githubusercontent.com/allenai/IFBench/refs/heads/main/data/IFBench_test.jsonl"
IFBENCH_TEST_SHA256 = "d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a"
IFBENCH_DIR = Path("/opt/benchmarks/IFBench")  # where ns's evaluator runs run_eval (docker/extras/ifbench.sh)
IFBENCH_NLTK_DATA = "/usr/local/share/nltk_data"  # the pinned NLTK data (docker/extras/ifbench.sh)


@register
class IFBench(NemoSkillsBenchmark):
    """IFBench (AllenAI: 300 prompts with out-of-distribution verifiable output
    constraints) as NeMo-Skills' ``ifbench`` benchmark: ns's generic/default
    prompt (the prompt is the user turn), ns's ifbench evaluator (IFBench's own
    ``run_eval`` strict and loose verifiers, at the IFBench commit ns's image
    pins, see docker/extras/ifbench.sh) and ns's ``if`` metrics. The value is
    the prompt-level loose accuracy (a prompt counts if all its instructions
    are followed by the response or one of IFBench's loose variants of it:
    first/last line or markdown ``*`` removed), pass@1 on one generation (mean
    over k with ``--repeats k``). ``details.metrics`` adds, from the same ns
    aggregation, the prompt-level strict accuracy (the response itself must
    follow every instruction) and the instruction-level loose and strict
    accuracies (the fraction of all instructions followed).

    Splits like every NeMo-Skills benchmark: ``--phase generate`` writes the
    responses, ``--phase score`` runs the verifiers on them (CPU only).

    No gold mode: IFBench has no reference responses."""

    id = "ifbench"
    metric = "pass@1 prompt loose / strict accuracy"
    default_repeats = 5
    """Nemo-Gym's ifbench/config.yaml: num_repeats: 5."""
    extra = "ifbench"
    harness_packages = ("nemo-skills", "litellm", "spacy", "nltk", "syllapy", "emoji", "langdetect")
    ns_benchmark = "ifbench"
    ns_metric = "prompt_loose_accuracy"
    ns_report_metrics = ("prompt_strict_accuracy", "instruction_loose_accuracy", "instruction_strict_accuracy")
    dataset = f"https://github.com/allenai/IFBench/blob/{IFBENCH_DATA_COMMIT}/data/IFBench_test.jsonl"
    dataset_revision = IFBENCH_DATA_COMMIT
    # ns prepare's test.jsonl from the pinned data .
    prepared_sha256 = "4dcc770a51d3d56d26c3b84410a734582587ce95a7cb7a5a0e58575f483308e3"
    pinned_urls = {
        IFBENCH_TEST_URL: (
            f"https://raw.githubusercontent.com/allenai/IFBench/{IFBENCH_DATA_COMMIT}/data/IFBench_test.jsonl",
            IFBENCH_TEST_SHA256,
        )
    }

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        if self.gold:
            raise SystemExit(f"{self.id}: IFBench has no reference responses, so there is no answers=gold mode")
        # IFBench's verifiers call nltk.download() when they are built, and nltk
        # searches ~/nltk_data before the image's pinned copy: a download there (or
        # data already in the job's home) would replace it. NLTK_DATA goes first on
        # nltk's search path, in the run_eval subprocess too. Only scoring uses it.
        if self.scoring:
            paths = [p for p in os.environ.get("NLTK_DATA", "").split(os.pathsep) if p]
            os.environ["NLTK_DATA"] = os.pathsep.join([IFBENCH_NLTK_DATA, *(p for p in paths if p != IFBENCH_NLTK_DATA)])
        return super().run(base_url, served_model_name)
