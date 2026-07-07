# Secrets

> **Audience:** operators configuring how a build's credentials are resolved, and anyone debugging a
> missing-secret failure. Secrets are selected per space — see [Spaces and `space.yaml`](../spaces/README.md#secret_manager).

**Secrets** are the credentials a build needs at run time — HuggingFace tokens, cloud API keys, SSH
private keys, registry passwords, and so on. Throughout gbserver, assets reference secrets **by name**,
never by value: an `environment.yaml` names the secret (e.g. an SSH key or a registry pull secret), and
the secret's value is resolved at build time by the space's configured **secret manager**. This keeps
secret material out of git-tracked assets.

## How secrets are configured

A space selects a secret-manager backend in its `space.yaml`:

```yaml
secret_manager:
  type: local          # local | env | hybrid | ibmcloud
  config: { ... }      # backend-specific
```

See the [`secret_manager` field](../spaces/README.md#secret_manager) in the spaces overview for where
this sits in the `space.yaml` schema.

## Backends

| `type` | Backend | Docs |
|--------|---------|------|
| `local` | File-backed secrets on disk, with optional one-way sync from a remote manager. | [local-secrets-manager.md](local-secrets-manager.md) |
| `env` | Secrets read from environment variables (`GBSERVER_SECRET_<NAME>` by default). Ideal for CI/CD and containers. | [env-secrets-manager.md](env-secrets-manager.md) |
| `ibmcloud` | IBM Cloud Secrets Manager (read-only at build time). | [ibmcloud-secrets-manager.md](ibmcloud-secrets-manager.md) |
| `hybrid` | Chains multiple managers with fallback priority (e.g. `env` overrides on top of `local` defaults). | [`hybridspacesecretmanager.py`](../../src/gbserver/spacesecretmanager/hybridspacesecretmanager.py) |

All backends implement the
[`SpaceSecretManager`](../../src/gbserver/spacesecretmanager/spacesecretmanager.py) interface, which the
space instantiates from the `secret_manager.type` value.

## How secrets are consumed

Once resolved, a space's secrets are available to the build's steps and environments, referenced **by
name**. Examples from `environment.yaml`:

- **Kubernetes** — `config.k8s.secrets` names secrets to mount as env vars or image pull secrets
  (see [k8s.md](../environments/k8s.md)).
- **SkyPilot** — `cluster_ssh_configs`, `cloud_config`, and `aws_credentials` values are secret-name-or-
  literal, resolved by exact name against the space's secrets (see [skypilot.md](../environments/skypilot.md)).
- **LSF** — the SSH private key is named via `login_node_ssh_key` (see [lsf.md](../environments/lsf.md)).

Per-user secrets (when supported) are merged over space secrets by the space at fetch time.

## Security

Keep secret **names** — not values — in any git-tracked asset (`environment.yaml`, `build.yaml`). The
value lives only in the secret manager (a local file, an env var, or IBM Cloud). A missing secret fails
the step fast; see [troubleshooting](../help/troubleshooting.md).

## See also

- [Spaces and `space.yaml`](../spaces/README.md) — where `secret_manager` is configured
- [Environments](../environments/README.md) — how environments reference secrets by name
- [Asset stores](../asset-stores/README.md) — the store credentials resolved by name from a secret manager
