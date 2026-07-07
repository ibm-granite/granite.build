# Using the environment-variable secrets manager in LLM.build

> **Audience:** operators configuring a space's secret manager. This is one of the `secret_manager`
> backends — see the [secrets overview](README.md) for the other backends (`local`, `ibmcloud`, `hybrid`)
> and [Spaces and `space.yaml`](../spaces/README.md) for where `secret_manager` sits in the schema.

The environment-variable secrets manager (`type: env`) resolves a space's secrets from **environment
variables**. It is read-only and requires no external service, which makes it ideal for CI/CD pipelines
(inject secrets as env vars) and containerized deployments.

The implementation is [`EnvSpaceSecretManager`](../../src/gbserver/spacesecretmanager/envspacesecretmanager.py).

## Configuration

Set `secret_manager.type` to `env`. The only config field is an optional `prefix`:

```yaml
secret_manager:
  type: env
  config:
    prefix: "GBSERVER_SECRET_"    # optional; default "GBSERVER_SECRET_"
```

| Field | Default | Purpose |
|-------|---------|---------|
| `prefix` | `GBSERVER_SECRET_` | The prefix prepended to a secret's (normalized) name to form the environment variable it reads. |

## How secret names map to environment variables

A secret name is **normalized** and prefixed to form the environment variable name:

- uppercased, and
- `-` and `.` are replaced with `_`.

So with the default prefix, all of these resolve to `GBSERVER_SECRET_API_KEY`:

| Secret name | Environment variable |
|-------------|----------------------|
| `api_key` | `GBSERVER_SECRET_API_KEY` |
| `api-key` | `GBSERVER_SECRET_API_KEY` |
| `api.key` | `GBSERVER_SECRET_API_KEY` |

When enumerating a space's secrets, the manager returns every environment variable that starts with the
prefix, with the prefix stripped from the key (so `GBSERVER_SECRET_HF_TOKEN` is exposed as `HF_TOKEN`).

## Read-only

This backend is read-only: `create_secret()` raises `NotImplementedError`. Set the variables directly in
your shell, CI/CD secret store, or deployment configuration — e.g. for a secret named `hf_token`:

```bash
export GBSERVER_SECRET_HF_TOKEN=hf_xxx
```

## Example

```yaml
name: ci-space
secret_manager:
  type: env
  config:
    prefix: "GBSERVER_SECRET_"
```

```bash
# secrets injected by the CI runner before the build
export GBSERVER_SECRET_HF_TOKEN=hf_xxx
export GBSERVER_SECRET_AWS_ACCESS_KEY_ID=AKIA...
export GBSERVER_SECRET_AWS_SECRET_ACCESS_KEY=...
```

## See also

- [Secrets overview](README.md) — the other `secret_manager` backends
- [Spaces and `space.yaml`](../spaces/README.md) — where `secret_manager` sits in the schema
- [Local secrets manager](local-secrets-manager.md) · [IBM Cloud secrets manager](ibmcloud-secrets-manager.md)
- Combine backends with the `hybrid` type — see the [secrets overview](README.md#backends)
