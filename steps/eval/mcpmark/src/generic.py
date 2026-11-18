# SPDX-License-Identifier: Apache-2.0

"""mcpmark family: the generic bridge to any service in the pinned MCPMark.

One registered id (``mcpmark``) for every service the pinned MCPMark checkout
ships: ``service=filesystem`` (required) runs that service's tasks exactly as
the curated ``mcpmark-<service>`` ids run theirs, and ``task_suite=`` (the
curated ids' own option) selects the standard or easy suite. The curated ids
stay as they are; a service the checkout does not ship fails fast when the run
reads its task directory.

Everything else is the base class's surface as is: the agent options
(``compaction_token``, ``reasoning_effort``, ``timeout``), the failure policy,
and the resume behavior. See ``mcpmark.py`` for the protocol.
"""

from granite_evals.benchmarks.mcpmark import Mcpmark
from granite_evals.registry import register


@register
class McpmarkService(Mcpmark):
    id = "mcpmark"

    def __init__(self, config):
        service = config.options.get("service", "").strip()
        if not service:
            raise SystemExit(
                "mcpmark: service= is required, e.g. service=filesystem "
                "(the curated mcpmark-<service> ids are the same services)"
            )
        super().__init__(config)
        self.service = service
