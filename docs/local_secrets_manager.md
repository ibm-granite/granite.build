# Using local secrets manager in LLM.build

This document describes the specifications required to use the Local Secrets Manager as an alternative to the IBM Cloud Secrets Manager.

It explains supported configuration options, remote synchronization behavior, and the expected format of locally stored secrets.

The Local Secrets Manager allows secrets to be stored and accessed from a local file system, with optional one-way synchronization from a remote secrets manager (currently IBM Cloud Secrets Manager).

## Secret Manager Configuration

The configuration should be provided in the `space.yaml` of the space that you are using. The secret manager configuration is defined under the `secret_manager` section. To use the Local Secrets Manager, set the type to `local`.

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

Remote synchronization allows secrets to be initially or periodically synced from a remote secrets manager into the local store.

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
        service_url: https://3a634e1e-2591-4a19-baa8-34cf8f12defb.us-east.secrets-manager.appdomain.cloud
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

Secrets are organized by spaces, each containing one or more secrets.

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
