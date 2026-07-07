# Command-line interfaces

Granite.Build ships three CLIs:

| CLI | Who runs it | What it does |
|-----|-------------|--------------|
| [`gb`](gb-cli-reference.md) | Build authors / users | A thin client over gbserver's REST API (`/api/v1`) — submit and monitor builds, manage artifacts, spaces, secrets, steps, and templates. |
| [`gbserver`](gbserver-cli-reference.md) | Operators | The server side — run the REST API, the build/PR watchers, the build runner, the all-in-one standalone server, and admin tasks. |
| [`gbtest`](gbtest-cli-reference.md) | Build authors / CI | Run a `buildtest.yaml` through the YAML-driven build-test harness (assertions over a build's targets, steps, and artifacts). |

## How they relate

```
gb  ──HTTPS──▶  gbserver REST API (/api/v1)  ──▶  BuildWatcher ──▶ BuildRunner ──▶ Environments
(client)                    (server)
```

`gb` never runs builds itself — it talks to a running `gbserver`. For a laptop-only workflow,
`gbserver standalone` runs the whole server stack in one process (see the
[getting-started guide](../getting-started.md)), and `gb` points at it.

## References

- [`gb` CLI reference](gb-cli-reference.md) — the client CLI.
- [`gbserver` CLI reference](gbserver-cli-reference.md) — the server / operator CLI.
- [`gbtest`](gbtest-cli-reference.md) — the build-test harness CLI.
