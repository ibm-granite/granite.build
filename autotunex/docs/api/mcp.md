# MCP server

AutoTuneX can expose its assistant tool registry over the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP), so an external MCP client
can call the same tools the in-app chat assistant uses. This surface is **opt-in** and off
by default.

See [chat.md](chat.md) for the same tool registry exposed over HTTP, and
[authentication.md](authentication.md) for the credential model.

## Enabling it

The MCP server is mounted at `/mcp` **only** when `AUTOTUNEX_ENABLE_MCP` is set (the
`enable_mcp` setting, default off). When the flag is unset the mount is a no-op and the
server never even imports its MCP dependency, so the base install carries no cost for a
feature most deployments do not use. Enabling it requires the `[mcp]` install extra
(`fastmcp`).

The mount lives at exactly `/mcp` — at the service **root**, outside the `/api/v1`
resource prefix.

## Authentication

MCP requests authenticate with the **`X-API-Key`** header, resolved to the same scoped
principal the REST API uses; every tool call is therefore owner-scoped exactly as the
equivalent REST endpoint would be. With the `api_key` provider enabled, a request that
presents no valid key fails the tool call with an authentication error.

For an external client to have a credential it can present, the `api_key` provider must be
enabled (in `auth_providers`). MCP has no bearer or session-cookie transport of its own — it
threads only the `X-API-Key` header — so mounting MCP without that provider leaves no middle
ground, and the server logs a startup warning. Under standalone/disabled auth every tool call
*succeeds* as the standalone system owner, with no credential check at all; under an `oidc`-
or `session`-only deployment every tool call *fails* with a missing-credentials error, because
the key is the only credential MCP can carry.

## Exposed tools

The server registers every tool in the shared registry — the exact same handlers the
in-app chat assistant calls, so a tool's behavior and its ownership scoping are defined in
one place and consumed identically by both surfaces:

`list_jobs`, `get_job`, `get_job_trials`, `get_job_results`, `get_trial_logs`,
`list_configs`, `get_config`, `get_config_template`, `create_config`, `list_datasets`,
`get_dataset`, `get_supported_dataset_types`, `get_user_info`, `get_user_metadata`, and
`start_tuning_job`.

## See also

- [chat.md](chat.md) — the same tools driven by the in-app assistant
- [authentication.md](authentication.md) — API keys and the credential model
- [overview.md](overview.md) — base URL and shared conventions
