# HuggingFace URIs and Output Push Configuration

This document covers two related things:

1. The **`hf://` URI scheme** (`HfURI`) used throughout `build.yaml` to identify
   HuggingFace models, datasets, spaces, and buckets.
2. The **`store_push` block** on a target output, which controls how a completed
   artifact is pushed to the HuggingFace Hub.

> **TL;DR:** You almost never need `store_push`. An `hf://` URI on the output is
> enough — the environment's asset store and the active g.b space supply the
> defaults (private repo, space-derived resource group). Only reach for
> `store_push` when you need a per-output override.

---

## HuggingFace URI format

`HfURI` (`src/gbcommon/uri/hf.py`) parses URIs in the following shape:

```
hf://[<host>/][<type>/]<owner>/<repo>[/<revision>[/<path_in_repo>]]
```

| Segment | Default | Notes |
|---------|---------|-------|
| `host` | `huggingface.co` | Omit (double `//`) to use the default; supply a host to target an Enterprise or custom hub. |
| `type` | `models` (implicit) | One of `models`, `datasets`, `spaces`, `buckets`. The `models/` segment may be omitted. |
| `owner` | — | Required. HF organization or user. |
| `repo` | — | Required. Repository name. |
| `revision` | `main` | Branch, tag, or commit SHA. |
| `path_in_repo` | `""` (repo root) | Subpath within the repo. If present, `revision` must be explicit (otherwise the parser cannot tell revision from path). |

### Examples

```yaml
# Models — "models/" segment is optional
hf:///mistralai/Mistral-7B-Instruct-v0.3                       # implicit MODEL, default host
hf:///models/mistralai/Mistral-7B-Instruct-v0.3                # explicit MODEL
hf://huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3  # explicit host
hf://ibm.com/models/mistralai/Mistral-7B-Instruct-v0.3         # custom host
hf:///ibm-granite/granite-3.0-8b-instruct/v1.0                 # explicit revision
hf:///ibm-granite/granite-3.0-8b-instruct/main/config.json     # revision + path_in_repo

# Datasets
hf://huggingface.co/datasets/wikitext/wikitext-103-v1
hf:///datasets/org/my-dataset
hf:///datasets/org/my-dataset/v2/data/train.csv                # revision + path_in_repo

# Spaces
hf://huggingface.co/spaces/huggingface/diffusers-gallery

# Buckets
hf://huggingface.co/buckets/org/test-bucket1
```

### Jinja templating in URIs

URIs in `build.yaml` can use Jinja expressions against the build context:

```yaml
outputs:
  download_file:
    uri: hf://huggingface.co/datasets/my-org/run-{{ binding.path | short_hash }}
```

`binding`, `run_metadata`, and the space variables are available. The rendered URI
is then parsed as `HfURI`.

---

## Minimal `build.yaml` — no `store_push` needed

For most builds an `hf://` URI on the output is all you need:

```yaml
llm.build:
  targets:
    publish_dataset:
      outputs:
        out:
          uri: hf://huggingface.co/datasets/my-org/my-dataset-{{ binding.path | short_hash }}
      steps:
        - step_uri: space://steps/download
          config: { ... }
```

With the above, the framework will:
- Derive the HF repo **type** (`dataset`) from the URI.
- Use `private: true` by default (inherited from the environment or the built-in default).
- Attach a resource group derived from the active g.b space (`gbspace-<space>`), so pushes into Enterprise-gated namespaces like `ibm-research` work out of the box.

**Use `store_push` only when you need to override one of these defaults for a single output.**

---

## The optional `store_push` block

```yaml
llm.build:
  targets:
    <target-name>:
      outputs:
        <output-name>:
          uri: hf://huggingface.co/datasets/<org>/<repo>
          store_push:                # <-- optional, omit when defaults suffice
            mode: "hfstore"
            config:
              hf:
                private: false
                resource_group_id: "abc123..."        # or use resource_group_name
                resource_group_name: "gbspace-public"
```

`store_push` is evaluated per-output and **takes precedence** over any equivalent
settings in `environment.yaml` (see [Relationship with `environment.yaml`](#relationship-with-environmentyaml)).

### Fields

#### `mode`

| Value | Description |
|-------|-------------|
| `"hfstore"` | Push the output artifact to HuggingFace Hub. This is the only supported mode. |

If `store_push` is absent the environment-level push configuration from `environment.yaml`
is used instead (see [environment-yaml-config.md](../operators/environment-yaml-config.md)).

#### `config.hf`

All fields are optional.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `private` | bool | `true` | Whether the HuggingFace repository should be **private**. Set to `false` to create or update a public repository. |
| `resource_group_id` | string | — | Pre-resolved HF Enterprise resource group id. When provided, no HF API lookup is performed — the id is used as-is. |
| `resource_group_name` | string | — | Resource group name. Resolved to an id via the HF API at push time. |

The HuggingFace **repo type** (`model`, `dataset`, `space`) is not configurable here — it
is derived automatically from the `uri` scheme (e.g. `hf:///datasets/…` → `dataset`).

---

## Resource group resolution

The effective resource-group id is determined with the following priority
(highest → lowest). Only one source needs to be set; when multiple are set they
must agree (the resolver raises `ValueError` on mismatch).

| Priority | Source | Notes |
|----------|--------|-------|
| 1 | `store_push.config.hf.resource_group_id` (build.yaml) | Per-output pre-resolved id. No HF API call. |
| 2 | `store_push.config.hf.resource_group_name` (build.yaml) | Per-output name. Resolved via HF API. |
| 3 | `environment.yaml` → `assetstores[].push[].config.hf.resource_group_id` / `resource_group_name` | Environment-level fallback. |
| 4 | Build `space_name` (automatic) | Populated at runtime from the g.b space; converted to resource group name by prepending `gbspace-`, then resolved via HF API. This is the default that makes `store_push` unnecessary in most cases. |

If none of the above yield a value, no resource group is attached to the push.

> **Note**: `space_name` is **not** a field you set in `build.yaml` — it is
> populated at runtime from the g.b space the build belongs to. It appears here
> only because it contributes to the final resource-group resolution.

The resolution itself is implemented in
[`HfURI.resolve_resource_group_id`](../src/gbcommon/uri/hf.py) and called from
both the K8s and LSF push paths before the step is dispatched.

---

## Override examples (when `store_push` *is* needed)

### Make a single output public

```yaml
outputs:
  download_file:
    uri: hf://huggingface.co/datasets/my-org/my-dataset-{{ binding.path | short_hash }}
    store_push:
      mode: "hfstore"
      config:
        hf:
          private: false
```

### Pin a specific resource group name (ignore the space default)

```yaml
outputs:
  tuned_model:
    uri: hf://huggingface.co/my-org/tuned-model-{{ binding.path | short_hash }}
    store_push:
      mode: "hfstore"
      config:
        hf:
          resource_group_name: "research-team"
```

### Pin a pre-resolved resource group id

Use when you've already looked up the id out-of-band and want to skip the HF API
call at push time:

```yaml
outputs:
  tuned_model:
    uri: hf://huggingface.co/my-org/tuned-model-{{ binding.path | short_hash }}
    store_push:
      mode: "hfstore"
      config:
        hf:
          resource_group_id: "5f8a...2c4"
```

---

## Relationship with `environment.yaml`

The environment asset store may also declare a `push` block under `assetstores`:

```yaml
assetstores:
  - store_uri: hf://huggingface.co/my-org
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

- [environment-yaml-config.md](../operators/environment-yaml-config.md) — environment-level asset store push configuration
- `src/gbcommon/uri/hf.py` — `HfURI` URI parser and `resolve_resource_group_id`
- `src/gbserver/asset/hfstore.py` — `Hfstore.build_hfpush_step_config` — builds the step config dict
- `src/gbserver/types/buildconfig.py` — `BuildTargetOutputPushConfig`, `BuildTargetOutputConfig`
- `src/gbserver/environment/k8s.py` — `K8s.pushasset_hfstore` — K8s push path
- `src/gbserver/environment/lsf.py` — `Lsf.pushasset_hfstore` — LSF push path
- `src/gbserver/builtins/steps/hfpush/` — the built-in step that performs the HF push
