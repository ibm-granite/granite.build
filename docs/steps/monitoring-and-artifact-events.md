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

Monitors and their rules are declared per environment type under `environment_configs`. A
monitor entry is either a **reference** to a shared monitor in the library (preferred) or an
**inline** definition:

```yaml
environment_configs:
  Bash:                              # or Docker, K8s, Lsf, Skypilot, Runpod
    launchers:
      command:
        type: nohup
        monitors: [log_monitor]      # monitors run concurrently with this launcher
    monitors:
      log_monitor:
        ref: space://monitors/bash   # reference the shared bash monitor (recommended)
```

The inline form spells the whole monitor out in the step instead:

```yaml
    monitors:
      log_monitor:
        type: log_monitor            # maps to monitor_log_monitor() on the env class
        config:
          event_configs: [ ... ]     # the log-line parsing rules
```

Most built-in steps use the reference form so the artifact-marker rules live in one place; see
[Referencing a shared monitor](#referencing-a-shared-monitor-the-monitor-library) below.

The full `event_configs` field schema (`event_type`, `line_regex`, `is_json`,
`event_fields` with `field_regex` / `field_value_template` / `is_data` / `is_json`) is
documented once in the
[`environment.yaml` reference](../environments/README.md#event_configs--log-line-parsing-rules).
This page covers the artifact-producing patterns built on top of it.

## Referencing a shared monitor (the monitor library)

Rather than copy the same `event_configs` into every step, monitors are defined once in the
shipped **monitor library** and referenced by name. A library monitor lives at
`src/gbserver/builtins/monitors/<name>/monitor.yaml` and a step points at it with a `ref`:

```yaml
    monitors:
      skypilot_monitor:
        ref: space://monitors/skypilot   # resolves to builtins/monitors/skypilot/monitor.yaml
```

`space://monitors/<name>` resolves to the library directory whose `monitor.yaml` holds the
monitor; a same-named directory under a configuration's own `monitors/` tree overrides the
built-in. The shipped monitors — `bash`, `docker`, `skypilot`, `lsf` — all carry the standard
`LLMB_ARTIFACT_*` artifact rules (both the `PATH` and `STATE` variants below), so a step that
just references one gets artifact capture for free.

### Overriding a referenced monitor

A `ref` entry may add a `config:` overlay and, optionally, a `type:` override. The overlay is
**deep-merged** over the referenced config (the step's value wins on any conflicting key), so a
step can tune a single field without restating the rest:

```yaml
    monitors:
      skypilot_monitor:
        ref: space://monitors/skypilot
        config:
          # A long-running eval polls less often; base default is 300s.
          poll_interval_seconds: "{{ config.poll_interval_seconds | default(900) }}"
```

To **add** event rules without discarding the inherited ones, use the reserved
`extra_event_configs` list — its entries are appended to the referenced monitor's
`event_configs` (a plain `event_configs:` key in the overlay would replace them instead):

```yaml
    monitors:
      skypilot_monitor:
        ref: space://monitors/skypilot
        config:
          extra_event_configs:
            - event_type: MESSAGE_EVENT
              line_regex: "^Step .* complete"
              event_fields:
                - {field_name: msg, field_regex: ".*"}
```

If the referenced monitor sets a `type:`, an overriding `type:` on the referring entry must
**match it** — a monitor may only reference another of the **same type**. A library monitor can
itself `ref` a parent (of the same type), forming a chain that is merged base-first; cycles are
rejected.

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
  line_regex: "LLMB_ARTIFACT_ID:.* LLMB_ARTIFACT_PATH:.*"
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

The asset store reads `binding["path"]` and copies from that filesystem location. (Whether the
regex should be `^`-anchored depends on the environment — see
[Anchoring is per-environment](#anchoring-is-per-environment) below.)

### Value outputs (`mem://`)

The workload prints, for an output named `server_url`:

```
LLMB_ARTIFACT_ID:server_url LLMB_ARTIFACT_STATE:http://host:8000
```

and the rule is identical except it emits `state` instead of `path`:

```yaml
- event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
  line_regex: "LLMB_ARTIFACT_ID:.* LLMB_ARTIFACT_STATE:.*"
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

**Which environments can produce a `mem://` output.** Consuming a `mem://` input works everywhere — the
transport is host-side and environment-agnostic. Producing one additionally needs the step's monitor to
carry this STATE rule. The shipped `space://monitors/bash`, `space://monitors/skypilot`,
`space://monitors/docker`, and `space://monitors/lsf` monitors already include it (referencing
one gives it for free); the k8s / runpod monitors do not yet. To add it to a step whose monitor lacks it, append the rule via
`extra_event_configs` (see [Overriding a referenced monitor](#overriding-a-referenced-monitor)) or inline
it. (RunPod's `pod_status_monitor` applies no `event_configs` at all — see below — so it cannot register
`mem://` outputs by marker.)

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
      command_config:
        command: "python train.py --reward-url {{ bindings.rm_url.binding.state }}"
```

Reference definitions that show both variants in real use:

- [`builtins/monitors/bash/monitor.yaml`](../../src/gbserver/builtins/monitors/bash/monitor.yaml) — the shared bash monitor, carrying the `LLMB_ARTIFACT_PATH` and `LLMB_ARTIFACT_STATE` rules side by side. The [`builtins/steps/bash/command/step.yaml`](../../src/gbserver/builtins/steps/bash/command/step.yaml) step references it with `ref: space://monitors/bash`.
- [`skypilot/.../rm-server/step.yaml`](../../configurations/assets/environments/skypilot/lsf/ibm-bluevela/steps/rm-server/step.yaml) — a long-lived service that keeps its monitor **inline** (its startup-log scraping is bespoke) and publishes its URL as a `mem://` `state` binding.

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

### Anchoring is per-environment

Whether an artifact rule should be `^`-anchored is a property of **how that environment
delivers logs**, not a universal best practice — the shipped monitors differ deliberately:

| Monitor | Anchored? | Why |
|---------|-----------|-----|
| `bash` (`space://monitors/bash`) | **Yes** (`^LLMB_ARTIFACT_ID:...`) | The bash launcher echoes the command back (`command step start: <cmd>`); anchoring stops the rule from matching that echo. Steps print the real marker at column 0, so the anchor still matches them. |
| `skypilot` (`space://monitors/skypilot`) | **No** | SkyPilot's *retrieved* job logs prefix stdout lines (e.g. `(worker1, rank=0) …`), so the marker is not at column 0 and a `^` anchor would never match. |
| `docker` (`space://monitors/docker`) | **No** | Docker streams raw stdout so either works; kept unanchored to match SkyPilot. |
| `lsf` (`space://monitors/lsf`) | **No** | LSF tails the job's raw stdout log (no worker prefix), so either works; kept unanchored for forward compatibility with future/wrapped LSF log sources. |

Anchoring `space://monitors/skypilot` "for consistency" once caused a real regression — the
markers stopped matching and the job registered **zero** artifacts. If you author a new monitor,
anchor only when the environment injects a line you must avoid matching.

> **Exception — the step-metadata marker is anchored on every monitor.** Unlike the artifact
> rules above, `LLMB_STEP_METADATA_KEY/VALUE` surfaces in lineage as *authoritative build
> provenance* (e.g. a byoc step's resolved `commit_hash`), so it must not be injectable by
> arbitrary mid-stream output from cloned-repo code. Bash, Docker, and LSF anchor it with a
> plain `^`; SkyPilot anchors it too but permits only its own `(name, pid=N) ` log prefix before the
> marker (`^(\([^)]*\)\s+)?LLMB_STEP_METADATA_KEY:...`), so a marker embedded after other text
> on the line is ignored. Do **not** un-anchor these to match the artifact rules — the trade-off
> is deliberate. (Caveat: because SkyPilot prefixes every line, a *deliberate* clean-line echo of
> just the marker is still indistinguishable from the scaffold's own emission; anchoring closes
> the incidental/embedded case, which is the realistic one.)
>
> **ANSI is handled once, centrally — not per monitor.** SkyPilot colourises its line prefix
> (`\x1b[36m(name, pid=N)\x1b[0m`) in retrieved logs. Rather than teach each monitor's regex to
> skip escape codes (which would also be needed for the `^` anchor to match), `get_events_from_log_line`
> strips all ANSI escape sequences from every log line before any rule runs. So monitor regexes —
> in **all** environments — only ever see clean text, and captured values never absorb a stray
> escape (e.g. a reset folded onto a commit hash). Keep ANSI handling there, not in `line_regex`.

### Gotcha: one line, one rule

The engine's `get_events_from_log_line`
([architecture/environment-classes.md](../architecture/environment-classes.md))
reassigns `log_line` to the matched substring after the first matching rule. In practice:

- If you need two events from related information, emit them from **distinct log lines**
  (as the `rm-server` step does with its `Starting FastAPI server on ...` and
  `GB_CLUSTER_NAME: ...` lines) rather than two rules against the same line.

## See also

- [Steps overview](README.md) — step.yaml structure and built-in steps
- [`environment.yaml` reference — `event_configs`](../environments/README.md#event_configs--log-line-parsing-rules) — the full field schema
- [Asset stores](../asset-stores/README.md) — how `mem://`, `env://`, `file://` and other URI schemes are resolved
- [`build.yaml` reference](../builds/build-yaml-reference.md) — declaring target inputs and outputs
