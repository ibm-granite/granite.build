# Monitoring and artifact events

How a step captures its outputs. When a step's workload finishes writing an output, it does
**not** call an API — it prints a line to stdout, and the step's **monitor** turns that line
into a `NEWARTIFACT_IN_ENVIRONMENT_EVENT` that registers the artifact against an output
declared in `build.yaml`. This page explains how to author that behaviour.

> **Audience:** anyone writing a custom step that produces output artifacts.

## The mental model

Every step runs under an environment (Bash, Docker, K8s, LSF, SkyPilot, RunPod). The
environment launches the workload and, alongside it, runs a **monitor** that tails the
workload's stdout/stderr. Each log line is matched against the step's `event_configs` rules;
a matching line becomes a `BuildEvent`:

- `NEWARTIFACT_IN_ENVIRONMENT_EVENT` — registers an output artifact.
- `MESSAGE_EVENT` — surfaces an informational line in the build UI.
- `WORKLOAD_STATUS_EVENT` — reports a progress/status change.

The `binding_id` on an artifact event must match an **output name** in the target's
`build.yaml`. That is the whole contract: workload prints a marker → monitor matches it →
the named output is bound to a value.

## Where monitors live in `step.yaml`

Monitors and their rules are declared per environment type under `environment_configs`:

```yaml
environment_configs:
  Bash:                              # or Docker, K8s, Lsf, Skypilot, Runpod
    launchers:
      command:
        type: nohup
        monitors: [log_monitor]      # monitors run concurrently with this launcher
    monitors:
      log_monitor:
        type: log_monitor            # maps to monitor_log_monitor() on the env class
        config:
          event_configs: [ ... ]     # the log-line parsing rules
```

The full `event_configs` field schema (`event_type`, `line_regex`, `is_json`,
`event_fields` with `field_regex` / `field_value_template` / `is_data` / `is_json`) is
documented once in the
[`environment.yaml` reference](../environments/README.md#event_configs--log-line-parsing-rules).
This page covers the artifact-producing patterns built on top of it.

## The `LLMB_ARTIFACT_*` marker convention

Rather than write a bespoke `line_regex` per step, most steps standardise on a marker the
workload prints, so a single event rule works across environments. There are two forms,
distinguished by whether the output value is a **path** or an opaque **value**:

| Marker the workload prints | Produces a binding | Use for outputs stored via |
|----------------------------|--------------------|----------------------------|
| `LLMB_ARTIFACT_ID:<output> LLMB_ARTIFACT_PATH:<path>` | `{"path": "<path>"}` | `file://`, `env://`, and other filesystem-backed stores |
| `LLMB_ARTIFACT_ID:<output> LLMB_ARTIFACT_STATE:<value>` | `{"state": "<value>"}` | `mem://` |

`<output>` must match an output name declared in `build.yaml`.

### Path outputs (`file://`, `env://`)

The workload prints, for an output named `results`:

```
LLMB_ARTIFACT_ID:results LLMB_ARTIFACT_PATH:/workspace/output
```

and the step's monitor carries this rule:

```yaml
- event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
  line_regex: "^LLMB_ARTIFACT_ID:.* LLMB_ARTIFACT_PATH:.*"
  is_json: false
  event_fields:
    - field_name: binding_id
      field_regex: "(?<=LLMB_ARTIFACT_ID:)[^ ]+"
    - field_name: path
      field_regex: "(?<=LLMB_ARTIFACT_PATH:).*"
      is_data: true
    - field_name: binding
      field_value_template: '{ "path": "{{ fields.data.path }}" }'
      is_json: true
```

The asset store reads `binding["path"]` and copies from that filesystem location.

### Value outputs (`mem://`)

The workload prints, for an output named `server_url`:

```
LLMB_ARTIFACT_ID:server_url LLMB_ARTIFACT_STATE:http://host:8000
```

and the rule is identical except it emits `state` instead of `path`:

```yaml
- event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
  line_regex: "^LLMB_ARTIFACT_ID:.* LLMB_ARTIFACT_STATE:.*"
  is_json: false
  event_fields:
    - field_name: binding_id
      field_regex: "(?<=LLMB_ARTIFACT_ID:)[^ ]+"
    - field_name: state
      field_regex: "(?<=LLMB_ARTIFACT_STATE:).*"
      is_data: true
    - field_name: binding
      field_value_template: '{ "state": "{{ fields.data.state }}" }'
      is_json: true
```

## Path vs state — the key distinction

The `binding` key you emit determines how the value reaches the consuming step:

- **`path`** is treated as a filesystem location. The asset store copies from it, and path
  normalisation may be applied.
- **`state`** is passed to the consumer **verbatim** — no copying, no path normalisation.
  This is exactly what `mem://` needs: it hands a producer's value straight through the
  build's shared memory, so a value like a service URL (`http://host:8000`) or a cluster
  name survives intact.

A consumer reads a `state` binding via a template:

```yaml
{{ bindings.<input_name>.binding.state }}
```

For the full picture of how `mem://` (and `env://`) stores are resolved, see
[Asset stores](../asset-stores/README.md).

## Worked example: a `mem://` output, producer to consumer

Producer target declares a `mem://` output (mem:// transfers nothing, so no `type`):

```yaml
outputs:
  server_url:
    uri: "mem://server_url"
```

Its step emits the marker and carries the `LLMB_ARTIFACT_STATE` rule shown above. A
downstream target then consumes it:

```yaml
inputs:
  rm_url:
    binding: start_server.server_url     # <producer_target>.<output_name>
steps:
  - step_uri: space://steps/train
    config:
      bash_config:
        command: "python train.py --reward-url {{ bindings.rm_url.binding.state }}"
```

Two reference step definitions show both variants in real use:

- [`bash/steps/command/step.yaml`](../../configurations/assets/environments/bash/steps/command/step.yaml) — carries the `LLMB_ARTIFACT_PATH` and `LLMB_ARTIFACT_STATE` rules side by side.
- [`skypilot/.../rm-server/step.yaml`](../../configurations/assets/environments/skypilot/lsf/ibm-bluevela/steps/rm-server/step.yaml) — a long-lived service that scrapes its own startup log to publish its URL as a `mem://` `state` binding.

## Per-environment monitor behaviour

Which monitor tails logs — and whether it does so live or in a batch after the job
finishes — is a property of the **environment**, not the step. The `event_configs` schema
is shared; the monitor `type` differs. See each environment page for its monitor types:
[bash](../environments/bash.md), [docker](../environments/docker.md),
[k8s](../environments/k8s.md), [lsf](../environments/lsf.md),
[skypilot](../environments/skypilot.md), [runpod](../environments/runpod.md).

One exception worth calling out: **RunPod's `pod_status_monitor` does not apply
`event_configs`** — it tracks pod status only and does not stream the container's logs. A
RunPod step cannot register artifacts by printing `LLMB_ARTIFACT_*` markers; instead push
its outputs to an asset store (e.g. an `s3push` step) that the orchestrator can read. See
[runpod.md](../environments/runpod.md).

### Gotcha: one line, one rule

The engine's `get_events_from_log_line`
([architecture/environment-classes.md](../architecture/environment-classes.md))
reassigns `log_line` to the matched substring after the first matching rule. In practice:

- Anchor artifact rules (`^LLMB_ARTIFACT_ID:...`) so a diagnostic line that merely *echoes*
  the command doesn't also match.
- If you need two events from related information, emit them from **distinct log lines**
  (as the `rm-server` step does with its `Starting FastAPI server on ...` and
  `GB_CLUSTER_NAME: ...` lines) rather than two rules against the same line.

## See also

- [Steps overview](README.md) — step.yaml structure and built-in steps
- [`environment.yaml` reference — `event_configs`](../environments/README.md#event_configs--log-line-parsing-rules) — the full field schema
- [Asset stores](../asset-stores/README.md) — how `mem://`, `env://`, `file://` and other URI schemes are resolved
- [`build.yaml` reference](../builds/build-yaml-reference.md) — declaring target inputs and outputs
