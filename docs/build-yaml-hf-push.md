# HuggingFace Output Push Configuration (`store_push`)

The `store_push` block on an output artifact in `build.yaml` controls how a completed
artifact is pushed to a HuggingFace Hub repository.  It is evaluated per-output and takes
**precedence** over any equivalent settings in `environment.yaml`.

---

## Where it lives

```yaml
llm.build:
  targets:
    <target-name>:
      outputs:
        <output-name>:
          uri: hf:///datasets/<org>/<repo>
          store_push:           # <-- this section
            mode: "hfstore"
            config:
              hf:
                private: false
                resource_group_name: "public"
```

---

## Fields

### `mode`

| Value | Description |
|-------|-------------|
| `"hfstore"` | Push the output artifact to HuggingFace Hub. This is the only supported mode. |

If `store_push` is absent the environment-level push configuration from `environment.yaml`
is used instead (see [environment-yaml-config.md](environment-yaml-config.md)).

---

### `config.hf`

All fields are optional.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `private` | bool | `true` | Whether the HuggingFace repository should be **private**. Set to `false` to create or update a public repository. |
| `resource_group_name` | string | — | Name of the HuggingFace Enterprise resource group to assign the repository to. Overrides the build-level `space_name` fallback (see priority table below). |

The HuggingFace **repo type** (`model`, `dataset`, `space`) is not configurable here — it is
derived automatically from the `uri` scheme (e.g. `hf:///datasets/…` → `dataset`).

---

## Resource group name resolution

`resource_group_name` is resolved with the following priority (highest → lowest):

| Priority | Source | Notes |
|----------|--------|-------|
| 1 (highest) | `store_push.config.hf.resource_group_name` in `build.yaml` | Per-output override |
| 2 (lowest) | Build `space_name` (derived automatically from the g.b space) | Converted to a HF resource group name by prepending `gbspace-` |

If neither source provides a value, no resource group is set on the push.

---

## Examples

### Public dataset with explicit resource group

```yaml
outputs:
  download_file:
    uri: hf:///datasets/my-org/my-dataset-{{ binding.path | short_hash }}
    store_push:
      mode: "hfstore"
      config:
        hf:
          private: false
          resource_group_name: "public"
```

### Private model (default behaviour — `store_push` not required)

Omitting `store_push` is sufficient when the environment's asset store already
configures `hfstore` push.  To be explicit:

```yaml
outputs:
  model_checkpoint:
    uri: hf:///my-org/my-model
    store_push:
      mode: "hfstore"
      config:
        hf:
          private: true
```

### Use a specific resource group without changing visibility

```yaml
outputs:
  tuned_model:
    uri: hf:///my-org/tuned-model-{{ binding.path | short_hash }}
    store_push:
      mode: "hfstore"
      config:
        hf:
          resource_group_name: "research-team"
```

---

## Relationship with `environment.yaml`

The environment asset store may also declare a `push` block under `assetstores`:

```yaml
assetstores:
  - store_uri: hf://my-org
    push:
      - mode: hfstore
        config:
          hf:
            private: true
            resource_group_name: "default-group"
```

Fields in `build.yaml`'s `store_push` **override** the corresponding fields from the
environment-level push config.  Any field not set in `build.yaml` falls back to the
environment value.

---

## Related

- [environment-yaml-config.md](environment-yaml-config.md) — environment-level asset store push configuration
- `src/gbserver/types/buildconfig.py` — `BuildTargetOutputPushConfig`, `BuildTargetOutputConfig`
- `src/gbserver/environment/k8s.py` — `pushasset_hfstore()` — resolution logic
- `src/gbserver/builtins/steps/hfpush/` — the built-in step that performs the HF push
