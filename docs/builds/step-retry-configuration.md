# Step Retry Configuration

Retries allow a step that fails due to a transient error (e.g. a pod eviction, node
preemption, or insufficient resources) to be automatically re-launched without failing the
build. This document describes the three-level configuration system that controls whether
retries are enabled for a given step execution.

## Configuration levels

There are three places where retry behaviour can be configured, evaluated from highest to
lowest priority:

| Level | Location | Field | Scope |
|---|---|---|---|
| 1 (highest) | build.yaml step section | `retry_enabled` | Single step in one build run |
| 2 | step.yaml `config:` section | `retry_enabled_default` | All runs using that step type |
| 3 (lowest) | Server environment variable | `GBSERVER_ENABLE_STEP_RETRY` | All steps on the server |

### Global gate: `GBSERVER_ENABLE_STEP_RETRY`

This environment variable is evaluated first. If it is `false`, **all retries are disabled
immediately** — none of the step-level or build-level configuration is consulted.

- Default: `true`
- Set to `false` to disable retries server-wide (e.g. to debug failures without automatic
  re-launches).

When the gate is `true`, the priority chains below run.

### Step-type default: `retry_enabled_default` (step.yaml)

Each step type can declare its own retry default in the `config:` section of its `step.yaml`.

```yaml
# step.yaml
config:
  retry_enabled_default: true         # or false
  retry_transparently_default: true  # or false
```

Built-in step defaults:

| Step | `retry_enabled_default` | `retry_transparently_default` |
|---|---|---|
| `gbstep` (generic training/eval) | `false` | `true` (framework default) |
| `s3push` | `true` | `true` |
| `s3pull` | `true` | `true` |
| `lhpush` | `true` | `true` |
| `lhpull` | `true` | `true` |
| `hfpull` | `true` | `true` |
| `cosrclone` | `true` | `true` |

If `retry_enabled_default` is not set in the step.yaml, retry defaults to `false` when
`GBSERVER_ENABLE_STEP_RETRY=true`.

### Per-run override: `retry_enabled` and `retry_transparently` (build.yaml)

A build can override either setting for any individual step:

```yaml
# build.yaml
targets:
  my-target:
    steps:
      - step_uri: space://steps/my-step
        retry_enabled: true          # overrides retry_enabled_default from step.yaml
        retry_transparently: false  # overrides retry_transparently_default from step.yaml
```

These take precedence over the corresponding `*_default` values in the step.yaml.

### `retry_transparently` / `retry_transparently_default`

When a step is retried, it may emit `NEWARTIFACT_IN_ENVIRONMENT_EVENT` events again for
artifacts that were already recorded in the first attempt. Setting `retry_transparently: true`
causes duplicate artifact events (matched by path basename) to be silently filtered out,
preventing double-counting.

Priority chain (highest to lowest):

1. build.yaml `retry_transparently`
2. step.yaml `retry_transparently_default` (framework default: `true`)

---

## Priority chain summary

```
GBSERVER_ENABLE_STEP_RETRY == false
    └─► retries disabled (all steps)

GBSERVER_ENABLE_STEP_RETRY == true
    retry_enabled:
    ├─► build.yaml retry_enabled is set  →  use that value
    ├─► step.yaml retry_enabled_default is set  →  use that value
    └─► neither is set  →  false (disabled)

    retry_transparently:
    ├─► build.yaml retry_transparently is set  →  use that value
    └─► step.yaml retry_transparently_default  →  use that value (default: true)
```

---

## Examples

### Example 1 — default server behaviour, generic training step

The server uses the default (`GBSERVER_ENABLE_STEP_RETRY=true`). The build.yaml references a
`gbstep`-based step and does not set `retry_enabled`.

```yaml
# build.yaml
targets:
  fine-tune:
    steps:
      - step_uri: space://steps/my-training-step
```

`gbstep` has `retry_enabled_default: false`, so retries are **disabled**. A pod eviction will
fail the build.

---

### Example 2 — opt a training step into retries

The same setup, but the build explicitly enables retry for this run:

```yaml
# build.yaml
targets:
  fine-tune:
    steps:
      - step_uri: space://steps/my-training-step
        retry_enabled: true
```

Retries are **enabled** for this step, regardless of the `gbstep` default.

---

### Example 3 — builtin upload step, default behaviour

```yaml
# build.yaml
targets:
  upload:
    steps:
      - step_uri: space://steps/s3push
        config:
          s3push_config:
            local_path: /output
            s3_uri: s3://my-bucket/results
```

`s3push` has `retry_enabled_default: true` and `retry_transparently_default: true`. With the
server default (`GBSERVER_ENABLE_STEP_RETRY=true`), retries are **enabled** and duplicate
artifact events from re-runs are filtered automatically.

---

### Example 4 — disable retry for a specific upload step

```yaml
# build.yaml
targets:
  upload:
    steps:
      - step_uri: space://steps/s3push
        retry_enabled: false
        config:
          s3push_config:
            local_path: /output
            s3_uri: s3://my-bucket/results
```

Retries are **disabled** for this upload step even though `s3push` defaults to `true`.

---

### Example 5 — enable retry but disable transparent dedup

A step is retried but produces non-idempotent artifact events that should all be forwarded:

```yaml
# build.yaml
targets:
  upload:
    steps:
      - step_uri: space://steps/my-step
        retry_enabled: true
        retry_transparently: false
```

Retries are enabled, but duplicate `NEWARTIFACT_IN_ENVIRONMENT_EVENT` events are **not**
filtered — all artifact events from every attempt are forwarded downstream.

---

### Example 6 — custom step.yaml opting in by default

A custom step type that should always retry with transparent dedup:

```yaml
# my-eval-step/step.yaml
name: my-eval-step
version: 1.0.0
config:
  retry_enabled_default: true
  retry_transparently_default: true
```

Any build that references this step will have retries **enabled** by default with dedup
active. Either setting can still be overridden per-run in the build.yaml.

---

### Example 7 — server-wide disable for debugging

Set `GBSERVER_ENABLE_STEP_RETRY=false` in the server environment. Every step in every build
will run without retry, regardless of `retry_enabled_default` or `retry_enabled` values. Useful
for debugging transient failures without automatic re-launches obscuring the root cause.
