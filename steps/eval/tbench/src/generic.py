# SPDX-License-Identifier: Apache-2.0

"""tbench family: the generic bridge to any Terminal-Bench-format dataset.

One registered id (``terminal-bench``) for every Terminal-Bench dataset on the
Hub (or a local directory of one): ``dataset=`` is required (with
``dataset_revision=`` for reproducibility). The loader reads both layouts the
publishers ship -- the harbor registry one (``registry.json`` + ``tasks/``,
like the terminal-bench-2.1 mirror the curated id pins) and the registry-less
one (one task directory per task at the root, like the terminal-bench-2.0
mirror). The curated id's manifest digest check applies only to its own
pinned revision; a bridged dataset is checked by its tasks running (harbor
verifies each task's own content), so pin the revision.

Everything else is the curated id's surface as is: ``tasks=`` / ``exclude=``
subset, ``agent=`` (terminus-2 or oracle), the per-task sandboxes, and the
resume behavior. See ``tbench.py`` for the protocol.
"""

from granite_evals.benchmarks.tbench import TerminalBench21
from granite_evals.registry import register

HARBOR_VERSION = "0.23.0"


@register
class TerminalBenchSet(TerminalBench21):
    id = "terminal-bench"

    dataset = f"https://pypi.org/project/harbor/{HARBOR_VERSION}/"
    dataset_revision = (
        "sha256:8747400dbb2a5e2298e1338e17e88eba38433c0433fd700f34d1a9021bba5c37"
    )

    def __init__(self, config):
        if not config.dataset:
            raise SystemExit(
                "terminal-bench: dataset= is required (an HF mirror of a Terminal-Bench "
                "release, or a local directory of one), with dataset_revision= for "
                "reproducibility; the curated terminal-bench-2.1 pins the official 2.1 mirror"
            )
        super().__init__(config)
