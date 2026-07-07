# Using local secrets manager in LLM.build

> **Audience:** operators configuring a space's secret manager. This is one of the `secret_manager`
> backends — see the [secrets overview](README.md) for the other backends (`env`, `hybrid`, `ibmcloud`)
> and [Spaces and `space.yaml`](../spaces/README.md) for where `secret_manager` sits in the schema.

This document describes the specifications required to use the Local Secrets Manager as an alternative to the IBM Cloud Secrets Manager.

It explains supported configuration options, remote synchronization behavior, and the expected format of locally stored secrets.

The Local Secrets Manager allows secrets to be stored and accessed from a local file system, with optional one-way synchronization from a remote secrets manager (currently IBM Cloud Secrets Manager).

## Secret Manager Configuration

The configuration is provided in the **`space.yaml`** of the space that you are using (see the
[spaces overview](../spaces/README.md) for the full `space.yaml` schema). The secret manager
configuration is defined under the `secret_manager` section. To use the Local Secrets Manager, set the
type to `local`.

> **Two different files.** The **`space.yaml`** carries the `secret_manager` config shown below; the
> **secrets file** it points at via `secrets_dir` holds the actual secret values in the `spaces:` →
> `secrets:` layout described under [Local Secrets File Structure](#local-secrets-file-structure).
> They are not the same file.

### Basic Configuration (Local Only)

If remote synchronization is not required, users can provide only the local configuration. In this mode, secrets are read exclusively from the local secrets file.

Example:

```yaml
secret_manager:
  type: local
  config:
    secrets_dir: /path/to/secrets/file
```

`secrets_dir` may point to:

* A directory, in which case gbserver will look for the secrets file within it, or

* A direct path to the secrets file itself.

* File can be json, yaml or .env

### Remote Synchronization (Optional)

Remote synchronization allows secrets to be initially or periodically synced from a remote secrets manager into the local store. The only supported remote is IBM Cloud — see [ibmcloud-secrets-manager.md](ibmcloud-secrets-manager.md).

#### Enabling Remote Sync

To enable remote synchronization, set `do_remote_sync` to `true` and provide a `remote_sync_config`.

Example:

```yaml
secret_manager:
  type: local
  config:
    secrets_dir: /path/to/secrets/file
    do_remote_sync: true
    remote_sync_config:
      type: ibmcloud
      config:
        service_url: https://<instance-id>.us-east.secrets-manager.appdomain.cloud
```

## Configuration Fields

##### `do_remote_sync`
Enables remote synchronization when set to `true`.

---

##### `remote_sync_config.type`
Specifies the remote secrets provider.

**Currently supported values:**
- `ibmcloud`

---

##### `remote_sync_config.config.service_url`
The service endpoint URL for IBM Cloud Secrets Manager.

---

> **Note**  
> Assertion checks are enforced to ensure that when `do_remote_sync` is enabled, a valid `remote_sync_config` is provided.

### First-Time Local Sync Behavior

If the local secrets file does not exist and `do_remote_sync` is enabled:

* The system identifies this as a first-time local sync.

* Secrets are fetched from the remote secrets manager.

* A local secrets file is generated automatically at the specified path.

This allows bootstrapping of local secrets without manual file creation.

### Local Secrets File Structure

This is the structure of the **secrets file** at `secrets_dir` (not the `space.yaml`). Secrets are
organized by spaces, each containing one or more secrets.

YAML example:

```yaml
spaces:
  public:
    secrets:
      LAKEHOUSE_TOKEN_STAGING:
        payload: <base64 encoded>
        labels:
          - encode:base64
        secret_group: gbspace-public
      LAKEHOUSE_TOKEN_PROD:
        payload: <base64 encoded>
        labels:
          - encode:base64
        secret_group: gbspace-public
```

JSON example:

```json
{
  "spaces": {
    "public": {
      "secrets": {
        "LAKEHOUSE_TOKEN_STAGING": {
          "payload": "<base64 encoded>",
          "labels": [
            "encode:base64"
          ],
          "secret_group": "gbspace-public"
        },
        "LAKEHOUSE_TOKEN_PROD": {
          "payload": "<base64 encoded>",
          "labels": [
            "encode:base64"
          ],
          "secret_group": "gbspace-public"
        }
      }
    }
  }
}
```

### Secret Attributes

Each secret supports the following fields:

| Field          | Description |
|----------------|-------------|
| `payload`      | The secret value (base64 encoded). |
| `labels`       | Metadata labels (e.g., encode:base64). |
| `secret_group` | Secret group name emulating remote secrets manager semnatics. |


## Summary

- The Local Secrets Manager enables local, file-based secret storage.
- Remote synchronization from IBM Cloud Secrets Manager is optional.
- Users may specify a directory or a direct file path for secrets.
- First-time synchronization automatically populates the local secrets file when enabled.
- Secrets are structured in a consistent, space-based hierarchy.
