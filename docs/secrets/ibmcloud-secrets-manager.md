# Using the IBM Cloud secrets manager in LLM.build

> **Audience:** operators configuring a space's secret manager. This is one of the `secret_manager`
> backends — see the [secrets overview](README.md) for the other backends (`local`, `env`, `hybrid`)
> and [Spaces and `space.yaml`](../spaces/README.md) for where `secret_manager` sits in the schema.

The IBM Cloud secrets manager (`type: ibmcloud`) resolves a space's secrets from
[IBM Cloud Secrets Manager](https://cloud.ibm.com/apidocs/secrets-manager/secrets-manager-v2). At build
time it is **read-only**: gbserver fetches the secrets the space is entitled to and makes them available
to steps. It is the backend used by the hosted (cloud) deployments; for local/standalone use, see the
[local secrets manager](local-secrets-manager.md).

The implementation is [`IbmcloudSpaceSecretManager`](../../src/gbserver/spacesecretmanager/ibmcloudspacesecretmanager.py).

## Prerequisites

The IBM Cloud SDK is an optional dependency:

```bash
pip install 'gbserver[ibmcloud]'
```

## Configuration

Set `secret_manager.type` to `ibmcloud` in the space's `space.yaml`. The `config` block takes two
optional fields; each falls back to an environment variable when omitted:

```yaml
secret_manager:
  type: ibmcloud
  config:
    service_url: https://<instance-id>.us-east.secrets-manager.appdomain.cloud
    # service_apikey: <IAM API key>    # prefer the env var below over inlining a key
```

| Field | Env var fallback | Purpose |
|-------|------------------|---------|
| `service_url` | `IBM_CLOUD_SECRETS_MANAGER_SERVICE_URL` | The Secrets Manager instance endpoint. Required (via field or env). |
| `service_apikey` | `IBM_CLOUD_API_KEY` | IAM API key used to authenticate. Required (via field or env). |

Authentication uses an IBM Cloud IAM API key (`IAMAuthenticator`). Prefer supplying the API key through
`IBM_CLOUD_API_KEY` rather than committing it into `space.yaml`.

## How secrets map to a space

Secrets in IBM Cloud Secrets Manager are organized into **secret groups**. gbserver decides which
groups a space may read by matching, so a single Secrets Manager instance can serve many spaces:

- **Group-to-space matching.** Each secret group's *description* holds one or more regex lines; a group
  is included for the space when one of those lines matches the space URI
  (`get_secret_groups()` searches each description line against the space's `uri`).
- **The public group.** A group named `gbspace-public` is always included (when present) and is loaded
  **first**, so space-specific groups loaded afterward override any same-named public secret.
- **Name prefixes are stripped.** A secret stored as `<group-name>-<NAME>` is exposed to the build as
  `<NAME>` (the `<group-name>-` prefix is removed). So `myspace-HF_TOKEN` becomes `HF_TOKEN`.
- **Base64 payloads.** Payloads carrying the `encode:base64` label are base64-decoded on read; the
  backend treats stored payloads as base64-encoded by default.

If no secret group matches the space URI, the space simply gets no secrets (a warning is logged).

## Resilience

The client enables retries (up to 10 attempts, 90s interval) and retries on transient `5xx` responses
as well as `403`s, which the IBM Secrets Manager has been observed to return intermittently.

## Administration

Creating and updating secrets is **not** done through this build-time backend
(`create_secret` raises `NotImplementedError`). Administrative CRUD on secret groups and secrets is
handled separately by `IbmcloudSpaceSecretManagerAdmin` in the same module — used by tooling that
provisions groups/secrets, not by the build runner.

## Relationship with the local secrets manager

The [local secrets manager](local-secrets-manager.md) can **sync from** an IBM Cloud instance: with
`do_remote_sync: true` and a `remote_sync_config` of `type: ibmcloud`, it bootstraps a local secrets
file from IBM Cloud on first run, then serves subsequent reads locally. Use that when you want IBM Cloud
as the source of truth but local file access at run time.

## See also

- [Secrets overview](README.md) — the other `secret_manager` backends
- [Spaces and `space.yaml`](../spaces/README.md) — where `secret_manager` sits in the schema
- [Local secrets manager](local-secrets-manager.md) — file-backed secrets, and remote sync from IBM Cloud
