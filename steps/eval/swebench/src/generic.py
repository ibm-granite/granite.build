# SPDX-License-Identifier: Apache-2.0

"""swebench family: the generic bridge to any per-instance-spec swebench set.

One registered id (``swebench``) for every swebench-format dataset that carries
each instance's ``image``, ``eval_script`` and ``log_parser`` -- the
SWE-bench_Verified / SWE-bench_Multilingual shape, which is also what a team
building its own set produces. The recipe names the data: ``dataset=`` is
required (with ``dataset_revision=`` for reproducibility -- an unpinned set
floats with the Hub). Everything else is the Verified/Multilingual pipeline as
is: mini-swe-agent generates, the dataset's own eval script and log parser
grade, both phases split, instances resume. SWE-bench Pro keeps its own
curated id (its verifier and images are a different protocol).

``instances=`` (a regex on ``instance_id``) subsets the set, like on the
curated ids. The class-level pin is the harness wheel (the grading); the
recipe's ``dataset``/``dataset_revision`` pin the data.
"""

from granite_evals.benchmarks.swebench import SWEBench
from granite_evals.registry import register

SWEBENCH_VERSION = "5.0.2"


@register
class SWEBenchSet(SWEBench):
    id = "swebench"

    dataset = f"https://pypi.org/project/swebench/{SWEBENCH_VERSION}/"
    dataset_revision = (
        "sha256:b7f0416a1e686eca22c2f749b5f816685a202835032f6683080e2b53545bbb62"
    )

    def __init__(self, config):
        if not config.dataset:
            raise SystemExit(
                "swebench: dataset= is required (a set carrying each instance's image, "
                "eval_script and log_parser), with dataset_revision= for reproducibility; "
                "the curated ids pin the official sets"
            )
        super().__init__(config)
