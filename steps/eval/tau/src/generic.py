# SPDX-License-Identifier: Apache-2.0

"""tau family: the generic bridge to any domain combination of the pinned data.

One registered id (``tau3``) for every combination of the τ³-release data the
image bakes: ``domains=airline,retail`` (required, comma-separated) runs those
domains exactly as the curated single-domain ids run theirs, and the headline
is the same aggregate shape -- the unweighted mean of the harness's pass^1 over
the chosen domains (what ``tau3-bench`` computes over its official three).
The curated ids stay as they are: ``tau3-bench`` is the official aggregate,
the single-domain ids the official per-domain numbers. A domain the pinned
data does not ship fails in the harness's own task lookup at run start.

Everything else is the base class's surface as is: ``task_split=`` (default
``base``) and ``tasks=`` (a task-name pattern) select within each domain, the
judge / user-simulator options (``judge_model=self`` for key-less smoke runs),
the two phases, and the resume behavior. See ``tau.py`` for the protocol.
"""

from granite_evals.benchmarks.tau import Tau3
from granite_evals.registry import register


@register
class Tau3Domains(Tau3):
    id = "tau3"
    metric = "pass@1 (mean over the chosen domains)"

    def __init__(self, config):
        spec = config.options.get("domains", "")
        domains = tuple(dict.fromkeys(d.strip() for d in spec.split(",") if d.strip()))
        if not domains:
            raise SystemExit(
                "tau3: domains= is required, e.g. domains=airline,retail "
                "(the curated single-domain ids are the official per-domain numbers)"
            )
        super().__init__(config)
        self.domains = domains
