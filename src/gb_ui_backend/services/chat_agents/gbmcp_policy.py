"""Backend-agnostic gbmcp tool policy — the actual security boundary for chat.

Every gbmcp tool falls into exactly one of three buckets:
  - ALLOWED_GBMCP_TOOLS: executes directly, no gate at all.
  - CONFIRMABLE_GBMCP_TOOLS: exposed to the model with its real gbmcp schema,
    but the tool call only *proposes* the action (see
    tool_registry.py's build_confirmable_gbmcp_tools()) — actual execution
    happens later, outside the model loop, only if the user clicks Approve
    on the resulting confirmation card. See tool_loop_backend.py's
    ToolLoopBackend.confirm_action().
  - DISALLOWED_GBMCP_TOOLS: never even described to the model.

Every ChatAgentBackend implementation must consult these three sets rather
than deriving its own — so this boundary doesn't get weaker just because a
future backend has a different (or no) native permission model.

secret_delete is disallowed outright because it's the only one of the five
secret_* tools that actually executes a mutation directly — secret_get/
secret_create/secret_update never touch a secret value at all; they return
a gbcli shell command (with a <secret-value> placeholder for create/update)
for the user to run themselves in their own terminal, so the value never
transits the model or this server. secret_delete has no such deferral and
no confirmation surface to point at, so unlike build_start/gbserver_stop
below, a confirm-then-execute gate wasn't judged to be worth adding for it.

build_start/build_cancel/gbserver_stop were allowed directly for a period
during development, then walked back after a security review: the chat
agent's browser-awareness feature embeds a URL query param (an "id" value)
verbatim into the model's context with no sanitization, and with these
three tools unrestricted and ungated, a crafted link could inject text that
gets acted on. build_cancel has a real confirmation surface already built
for it — the build's own detail page, which has the actual Cancel button
(see ui_actions.py's build_detail route) — so it's handled through
suggest_navigation, not this module's confirmation mechanism.
build_start/gbserver_stop have no equivalent page anywhere in this
frontend, so instead of declining them outright, they're confirmable:
available to the model, but gated behind an explicit user Approve/Decline
before gbmcp is ever actually called.

gbserver_start mutates the same way gbserver_stop does (spawns a real
gbserver process — see gbmcp's lifecycle.py) and is gated for the same
reason, even though in the common standalone deployment (chat routes
mounted into the very gbserver process gbmcp would be starting) it's
usually a no-op: gbmcp's own docstring notes its tools stay reachable even
when gbserver is down, so in a deployment where chat is served independently
of the gbserver instance gbmcp controls, gbserver_start has a real,
unconfirmed effect if left auto-approved.
"""

from __future__ import annotations

# All 18 tools gbmcp exposes (per src/gbmcp/README.md).
ALL_GBMCP_TOOLS: list[str] = [
    "gbserver_status",
    "gbserver_start",
    "gbserver_stop",
    "build_list",
    "build_status",
    "build_describe",
    "build_log",
    "build_job_log",
    "build_start",
    "build_cancel",
    "secret_list",
    "secret_get",
    "secret_create",
    "secret_update",
    "secret_delete",
    "info_health",
    "info_version",
    "info_gb_version",
]

# Never even described to the model.
DISALLOWED_GBMCP_TOOLS: list[str] = ["secret_delete", "build_cancel"]

# Available, but only via propose-then-confirm — see module docstring.
CONFIRMABLE_GBMCP_TOOLS: list[str] = ["build_start", "gbserver_start", "gbserver_stop"]

# Auto-approved without a permission prompt — everything except the tools above.
ALLOWED_GBMCP_TOOLS: list[str] = [
    t
    for t in ALL_GBMCP_TOOLS
    if t not in DISALLOWED_GBMCP_TOOLS and t not in CONFIRMABLE_GBMCP_TOOLS
]

# Curated, manually-maintained list of gbmcp tools known to mutate real state
# directly (as opposed to secret_get/create/update, which only ever return a
# shell command for the user to run themselves — see above). Exists purely as
# a regression guard (test_gbmcp_policy.py) so a future edit here — or a
# future gbmcp tool added straight to ALLOWED_GBMCP_TOOLS without updating
# this list — fails loudly instead of silently landing a mutating tool in the
# auto-approved bucket.
KNOWN_MUTATING_GBMCP_TOOLS: list[str] = [
    "build_start",
    "build_cancel",
    "gbserver_start",
    "gbserver_stop",
    "secret_delete",
]
