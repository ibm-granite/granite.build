# SPDX-License-Identifier: Apache-2.0

"""nemoskills family: the generic bridge to any benchmark in the pinned NeMo-Skills.

One registered id (``nemoskills``) for every benchmark the pinned ns checkout
ships: ``benchmark=<ns name>`` (required, e.g. ``benchmark=math-500`` -- the
directory names under ns's ``nemo_skills/dataset``) runs it with **ns's own
defaults** -- its prepare, its prompts, its metrics -- where the curated
classes freeze Granite-specific protocol choices on top of the same machinery.
The curated ids stay as they are: they are the validated protocols.

Data: with ``dataset=`` a local path (a ``<dir>/<split>.jsonl`` or a single
jsonl file in ns's row format), the run reads it directly -- that is how a
recipe pins the data, and a hub id is refused (redirecting ns's fetch is a
curated-class feature). Without it, ns's own prepare fetches the benchmark's
data unpinned; the run's provenance records exactly what was read, but a rerun
floats with the source. ``metric=`` overrides the headline metric key (the
class default, ns's ``symbolic_correct``); everything else -- ``split=``, the
sampling options, the judge options, the two phases, resume -- is the base
class's own surface. See ``nemo_skills.py`` for the protocol.
"""

from granite_evals.benchmarks.nemo_skills import NS_COMMIT, NS_REPO, NemoSkillsBenchmark
from granite_evals.registry import register


@register
class Nemoskills(NemoSkillsBenchmark):
    id = "nemoskills"
    metric = "ns's own headline metric for the benchmark"

    # The pin every benchmark must carry: the ns checkout (the benchmark
    # configs, prompts and evaluators); the data is ns's own fetch or the
    # recipe's dataset= path.
    dataset = NS_REPO
    dataset_revision = NS_COMMIT

    def __init__(self, config):
        bench = config.options.get("benchmark", "").strip()
        if not bench:
            raise SystemExit(
                "nemoskills: benchmark= is required (an ns benchmark name, e.g. "
                "benchmark=math-500); the curated ids are the validated protocols"
            )
        super().__init__(config)
        self.ns_benchmark = bench
        if "metric" in config.options:
            self.ns_metric = config.options["metric"]
        from pathlib import Path

        if config.dataset and not Path(config.dataset).exists():
            raise SystemExit(
                "nemoskills: dataset= must be a local path in ns's row format (a dir with "
                "<split>.jsonl or a single jsonl file); hub ids are a curated-class feature"
            )
