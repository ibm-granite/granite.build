# SPDX-License-Identifier: Apache-2.0

"""mcpmark family: the generic bridge to the services the image can run.

One registered id (``mcpmark``) for the services of the pinned MCPMark
checkout that this image's pipeline sets up -- today the curated three,
``filesystem``, ``postgres`` and ``playwright`` (``N_TASKS`` in
``mcpmark.py``): ``service=<name>`` (required) runs that service's tasks
exactly as the curated ``mcpmark-<service>`` ids run theirs, and
``task_suite=`` (the curated ids' own option) selects the standard or easy
suite. The checkout ships more services (github, notion, insforge, supabase,
playwright_webarena), but they need accounts or backends the image does not
provide, and the pipeline has no setup or pinned task count for them: the
bridge refuses them up front, naming the ones it runs. The curated ids stay
as they are.

Everything else is the base class's surface as is: the agent options
(``compaction_token``, ``reasoning_effort``, ``timeout``), the failure policy,
and the resume behavior. See ``mcpmark.py`` for the protocol.
"""

from granite_evals.benchmarks.mcpmark import N_TASKS, Mcpmark
from granite_evals.registry import register

SERVICES = tuple(sorted({service for service, _ in N_TASKS}))


@register
class McpmarkService(Mcpmark):
    id = "mcpmark"
    metric = "pass@1"

    def __init__(self, config):
        service = config.options.get("service", "").strip()
        if not service:
            raise SystemExit(
                "mcpmark: service= is required, e.g. service=filesystem "
                "(the curated mcpmark-<service> ids are the same services)"
            )
        if service not in SERVICES:
            raise SystemExit(
                f"mcpmark: service={service} is not one this image runs; the bridge runs "
                f"{', '.join(SERVICES)} (the pinned checkout's other services need accounts or "
                "backends the image does not provide)"
            )
        super().__init__(config)
        self.service = service
