# REST API

> **Audience:** operators and integrators calling gbserver over HTTP. For the command-line clients see
> the [CLIs](../cli/README.md).

gbserver exposes a REST API under the base path **`/api/v1`** (FastAPI). Rather than duplicate an
endpoint reference here (which would drift from the code), **browse the live, interactive OpenAPI docs**
served by a running server — they are always current.

## Browse the API locally

Start a server (see the [`gbserver` CLI reference](../cli/gbserver-cli-reference.md#running-the-server)):

```bash
gbserver standalone            # all-in-one: REST API + BuildWatcher (defaults to 127.0.0.1:8080)
# or just the API:
gbserver rest-server --port 8080
```

Then open the interactive OpenAPI (Swagger) UI. **Each API group is mounted as its own sub-app, so it has
its own docs page** — the root `/docs` only covers the root app. The Swagger UI for each group (append
`/openapi.json` instead of `/docs` for the raw schema):

- [`http://localhost:8080/api/v1/auth/docs`](http://localhost:8080/api/v1/auth/docs) — OIDC login proxy
- [`http://localhost:8080/api/v1/builds/docs`](http://localhost:8080/api/v1/builds/docs) — builds
- [`http://localhost:8080/api/v1/artifacts/docs`](http://localhost:8080/api/v1/artifacts/docs) — artifacts
- [`http://localhost:8080/api/v1/spaces/docs`](http://localhost:8080/api/v1/spaces/docs) — spaces
- [`http://localhost:8080/api/v1/logs/docs`](http://localhost:8080/api/v1/logs/docs) — logs
- [`http://localhost:8080/api/v1/lineage/docs`](http://localhost:8080/api/v1/lineage/docs) — lineage
- [`http://localhost:8080/api/v1/secrets/docs`](http://localhost:8080/api/v1/secrets/docs) — secrets
- [`http://localhost:8080/api/v1/node-health/docs`](http://localhost:8080/api/v1/node-health/docs) — node health

The `/docs` and `/openapi.json` pages load **without authentication** in every mode, so they work even
against a secured server. (Build-event subscription is `include_router`'d on the **root** app rather than
mounted as a sub-app, so — unlike the groups above — it appears in the **root** docs at
[`http://localhost:8080/docs`](http://localhost:8080/docs), even though its URL lives under `/api/v1/builds`.)

## Authentication

Requests to the API itself (not the docs pages) are authenticated per `GBSERVER_AUTH_MODE`:

| Mode | How you authenticate |
|------|----------------------|
| `apikey` | Send the shared `GBSERVER_API_KEY` as a bearer token. Localhost/standalone allows unauthenticated calls from `127.0.0.1`. |
| `github` | GitHub Enterprise token (bearer). |
| `ibmid` | IBMid OIDC token (bearer), obtained via the `gb` login flow. |
| `multi` | Either a GitHub or an IBMid token (auto-detected). |

See [multi-provider-authentication.md](multi-provider-authentication.md) for the full auth configuration
and the `gb auth login` token-exchange flow.

## API groups

Each group is mounted under `/api/v1/<group>`. Browse its endpoints at `<group>/docs`; the source is the
authoritative definition.

| Group | Prefix | Interactive docs | Source |
|-------|--------|------------------|--------|
| Auth (OIDC login proxy) | `/api/v1/auth` | `/api/v1/auth/docs` | [`auth_routes.py`](../../src/gbserver/api/auth_routes.py) |
| Builds | `/api/v1/builds` | `/api/v1/builds/docs` | [`builds.py`](../../src/gbserver/api/builds.py) |
| Artifacts | `/api/v1/artifacts` | `/api/v1/artifacts/docs` | [`artifacts.py`](../../src/gbserver/api/artifacts.py) |
| Spaces | `/api/v1/spaces` | `/api/v1/spaces/docs` | [`spaces.py`](../../src/gbserver/api/spaces.py) |
| Logs | `/api/v1/logs` | `/api/v1/logs/docs` | [`logs.py`](../../src/gbserver/api/logs.py) |
| Lineage | `/api/v1/lineage` | `/api/v1/lineage/docs` | [`lineage.py`](../../src/gbserver/api/lineage.py) |
| Secrets | `/api/v1/secrets` | `/api/v1/secrets/docs` | [`secrets.py`](../../src/gbserver/api/secrets.py) |
| Node health | `/api/v1/node-health` | `/api/v1/node-health/docs` | [`node_health.py`](../../src/gbserver/api/node_health.py) |
| Build events | `/api/v1/builds/{id}/events/subscribe` | root `/docs` (on the root app) | [`event_subscribe.py`](../../src/gbserver/api/event_subscribe.py) |

## How it's assembled

[`root_api.py`](../../src/gbserver/api/root_api.py) creates the FastAPI app, adds the auth middleware, and
**mounts each group as a sub-app** under `/api/v1/<group>` (build-event subscription is included on the
root app). That mounting is why each group carries its own `/docs` and `/openapi.json`.

## See also

- [`gbserver` CLI reference](../cli/gbserver-cli-reference.md) — starting the server
- [`gb` CLI reference](../cli/gb-cli-reference.md) — the client that talks to this API
- [Multi-provider authentication](multi-provider-authentication.md)
