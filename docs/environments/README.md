# Environments

> **Audience:** operators configuring `environment.yaml`, and step authors who need to know
> which compute backend a step runs on. For how `space://steps/<name>` URIs route to an impl,
> see [step-resolution.md](step-resolution.md).

An **environment** is the compute backend a build target runs on. Each target in a `build.yaml`
names an `environment_uri` (e.g. `space://environments/bash`); that points at
an environment asset whose `environment.yaml` declares the environment **type**, its credentials and
behaviour, and the asset stores reachable from it. The `space://` URI is resolved through the active
space's `base_uris` — see [Spaces and `space.yaml`](../spaces/README.md).

This page covers the framework common to **all** environment types — the `environment.yaml` schema,
asset stores, the shared `step.yaml` launcher/monitor structure, the `event_configs` log-parsing
schema, and the common `config:` fields. Each environment type then has its own page covering only
what is unique to it.

## Compute-endpoint map

Every compute endpoint gbserver can run a step on, and the page that documents it:

| Compute endpoint | `type:` | Reached via | Page |
|------------------|---------|-------------|------|
| Local OS process | `Bash` | direct | [bash.md](bash.md) |
| Local container (Docker / Podman) | `Docker` | direct | [docker.md](docker.md) |
| Kubernetes / OpenShift | `K8s` | direct (Helm + AppWrapper) | [k8s.md](k8s.md) |
| IBM LSF cluster | `Lsf` | direct (`bsub` over SSH) | [lsf.md](lsf.md) |
| RunPod GPU pods | `Runpod` | direct (RunPod API) | [runpod.md](runpod.md) |
| **SkyPilot** (multi-cloud) | `Skypilot` | per-step `sky.launch()` | [skypilot.md](skypilot.md) |
| &nbsp;&nbsp;↳ SLURM | `Skypilot` | `default_cloud: slurm` | [skypilot-slurm.md](skypilot-slurm.md) |
| &nbsp;&nbsp;↳ LSF | `Skypilot` | `default_cloud: lsf` | [skypilot-lsf.md](skypilot-lsf.md) |
| &nbsp;&nbsp;↳ Kubernetes | `Skypilot` | `default_cloud: kubernetes` | [skypilot-kubernetes.md](skypilot-kubernetes.md) |
| &nbsp;&nbsp;↳ AWS | `Skypilot` | `default_cloud: aws` | [skypilot-aws.md](skypilot-aws.md) |

Some target compute is reachable **two ways**. LSF can be driven natively ([lsf.md](lsf.md), gbserver
submits `bsub` itself) or through SkyPilot ([skypilot-lsf.md](skypilot-lsf.md), SkyPilot's LSF
provisioner submits the job). Kubernetes likewise runs natively via Helm/AppWrapper ([k8s.md](k8s.md))
or through SkyPilot ([skypilot-kubernetes.md](skypilot-kubernetes.md)). Pick the native type when you
want gbserver's first-class lifecycle (AppWrapper retries, RabbitMQ event streaming for K8s; SSH
workspace management for LSF); pick SkyPilot when you want one environment definition that can target
several clouds with a uniform launcher.

SkyPilot can in principle target additional clouds (GCP, Azure, Lambda, …); only the four above are
documented in depth here.

## How environments work

The base class is [`Environment`](../../src/gbserver/environment/environment.py); each type is a
subclass in [`src/gbserver/environment/`](../../src/gbserver/environment/) (`bash.py`, `docker.py`,
`k8s.py`, `lsf.py`, `runpod.py`, `skypilot.py`). Types are discovered dynamically: the filename,
capitalized, is the class name and the `type:` value (`bash.py` → `Bash`). See
[architecture/environment-classes.md](../architecture/environment-classes.md) for the internals.

A step runs through a small lifecycle the environment implements as suffixed methods —
`setup_<suffix>()`, `launch_<suffix>()`, `monitor_<suffix>()`, `cleanup_<suffix>()`. The step's
`step.yaml` `environment_configs` section (below) selects which launcher and monitors to use per
environment type.

## `environment.yaml` — common top-level structure

```yaml
name: <string>          # Human-readable name (informational).
type: <string>          # Environment class to instantiate: Bash, Docker, K8s, Lsf, Runpod, Skypilot.
subtype: <string>       # Optional: distinguishes envs of the same type (see below).
config:                 # Environment-type-specific config — see the per-type page.
  ...
assetstores:            # Asset stores accessible from this environment (see below).
  - store_uri: <uri>
    pull:
      - mode: <mode>
        config: {}
    push:
      - mode: <mode>
        config: {}
```

`name`, `type`, `subtype`, and `assetstores` are the fields the shared schema
([`EnvironmentConfig`](../../src/gbserver/types/environmentconfig.py)) defines; everything under
`config:` is a free-form dict interpreted by the environment class. The per-type pages document each
type's `config:` block.

### Sharing step implementations across environments

Steps are resolved relative to the active environment (see
[step-resolution.md](step-resolution.md)). A **single** `step.yaml` at a shared ancestor of
several environments' directories — e.g. `environments/skypilot/steps/digit/step.yaml` — is
discovered by every environment beneath it (`skypilot/kubernetes`, `skypilot/slurm`, …) via the
resolver's ancestor-walk. An env's own `steps/<name>/` still overrides. No configuration needed.

### Restricting a shared step by sub-type (`subtype` / `subtypes`)

Because environments that differ only in compute endpoint often share one `type` (all skypilot
endpoints are class `Skypilot`, differing only by `config.default_cloud`), the class alone can't
say *which* of them may use a shared step. A **sub-type** provides that discriminator:

- An `environment.yaml` may declare an optional `subtype` (any free-form string — no predefined
  set; the skypilot endpoints use `kubernetes`/`slurm`/`aws`/`lsf` by convention).
- A step's `environment_configs.<Class>` may declare an optional `subtypes` list. **Empty (the
  default) = universal** — the step matches any environment of that class (this is how builtins
  and general shared steps keep resolving everywhere). A **non-empty** list restricts the step to
  environments whose `subtype` is one of the listed values (exact string match); an environment
  with no `subtype` does not match such a step.

Example — one shared `digit` usable only by the kubernetes and slurm endpoints:

```yaml
# environments/skypilot/steps/digit/step.yaml
environment_configs:
  Skypilot:
    subtypes: [kubernetes, slurm]   # aws / lsf endpoints are excluded
    ...
```

The sub-type filter is applied by both resolver tiers (ancestor-walk and env-class-match), so the
restriction holds regardless of which tier finds the file.

### Asset stores

`assetstores` map a store URI to the **pull** (input) and **push** (output) behaviour for this
environment. Each `pull`/`push` entry has a `mode` and an optional `config`. The input key is `pull`
(it pairs with `push` and matches the `pullasset_*` handler); the former name `load` is still accepted
as a **deprecated alias** (parsing it logs a warning recommending `pull`). Modes are implemented by
`pullasset_*` / `pushasset_*` methods on the environment class, which may queue a built-in step; the
exact set is per-type — see each page.

The env-local (`env://`) and in-memory (`mem://`) stores are the exceptions: each is registered
implicitly for **every** environment (their push/pull transfer nothing — env:// is a shared-filesystem
no-op, mem:// passes a value through the build's shared memory — handled by the base class), so `env://`
and `mem://` inputs and outputs work without an `assetstores` entry here. Declare one only to override
its default `load`/`push` mode. The `file:` store is **not** implicit — an environment that supports it
declares `space://assetstores/file/` (the builtin File store) here with the modes it implements (e.g.
`bash` = load+push; `docker` = push only, since it has no `pullasset_filestore`).

For the full list of modes and the store types themselves (URI schemes, secrets, configuration), see
[Asset stores](../asset-stores/README.md#load-and-push-modes).

## `step.yaml` — `environment_configs` (common structure)

`environment_configs` declares, per environment type, which launchers and monitors run a step. The
shape is the same for every type; the available launcher/monitor `type:` values differ and are
documented on each per-type page.

```yaml
environment_configs:
  K8s:                          # or Bash, Docker, Lsf, Runpod, Skypilot. Case-insensitive match.
    default_launcher: <name>    # Optional: launcher used when the step names none.
    launchers:
      <launcher_name>:
        type: <suffix>          # Maps to launch_<suffix>() on the environment class.
        monitors:               # Monitor names (from monitors:) run concurrently with this launcher.
          - <monitor_name>
        config: { ... }         # Launcher-specific kwargs.
    monitors:
      <monitor_name>:
        type: <suffix>          # Maps to monitor_<suffix>() on the environment class.
        config:
          event_configs: [ ... ]  # Log-line parsing rules (see below).
```

A monitor entry may instead **reference** a shared monitor from the library rather than inline
`type`/`config`:

```yaml
    monitors:
      <monitor_name>:
        ref: space://monitors/<name>   # e.g. bash, docker, skypilot
        config: { ... }                # Optional overlay deep-merged over the referenced monitor.
```

The referenced monitor lives at `src/gbserver/builtins/monitors/<name>/monitor.yaml` and already
carries the standard `LLMB_ARTIFACT_*` rules. See
[Referencing a shared monitor](../steps/monitoring-and-artifact-events.md#referencing-a-shared-monitor-the-monitor-library)
for the overlay rules (`extra_event_configs`, same-type constraint).

## `event_configs` — log-line parsing rules

`event_configs` live under a monitor's `config` and turn matching log lines into `BuildEvent`s
(artifact registration, status, messages). The schema is shared across the environment types whose
monitors tail logs (Bash, Docker, K8s, Lsf, Skypilot).

```yaml
event_configs:
  - event_type: <BuildEventType>   # NEWARTIFACT_IN_ENVIRONMENT_EVENT | MESSAGE_EVENT |
                                   # WORKLOAD_STATUS_EVENT | VALIDATION_DATA_EVENT | ARTIFACT_PUSHED_EVENT
    line_regex: "<regex>"          # Matched against each log line; the matched portion feeds field extraction.
    is_json: false                 # If true, parse the matched portion as JSON into event_data["data"].
    event_fields:
      - field_name: <name>         # Key in the event payload.
        field_regex: "<regex>"     # Extract value via regex (full match). Mutually exclusive with field_value_template.
        field_value_template: "..." # Jinja2 template. Context: {{ fields.<name> }}, {{ fields.data.<key> }}.
        is_json: false             # Parse the extracted value as JSON before storing.
        is_data: false             # Store under event_data["data"] instead of the top-level payload.
```

### Event type conventions

| `event_type` | Typical trigger | Common fields |
|--------------|-----------------|---------------|
| `NEWARTIFACT_IN_ENVIRONMENT_EVENT` | Workload writes an output | `binding_id` (matches an output name in `build.yaml`), `binding` (JSON with `"path"` for filesystem outputs, or `"state"` for `mem://` outputs) |
| `MESSAGE_EVENT` | Informational line for the build UI | `msg` |
| `WORKLOAD_STATUS_EVENT` | Progress update | `status` |
| `VALIDATION_DATA_EVENT` | Structured metrics | `data` |
| `ARTIFACT_PUSHED_EVENT` | Upload step confirms a push | `uri`, `binding_id` |

### Example: artifact detection from a log line

```
# Log line emitted by the workload:
Final checkpoint saved in /gpfs/workspace/output/checkpoint-final

# Matching rule:
- event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
  line_regex: "Final\\scheckpoint\\ssaved\\sin\\s.*"
  is_json: false
  event_fields:
    - field_name: binding_id
      field_value_template: final_checkpoint   # Static value; matches an output name in build.yaml
    - field_name: path
      field_regex: "/.*"
      is_data: true                            # Stored in data dict for the binding template
    - field_name: binding
      field_value_template: '{ "path": "{{ fields.data.path }}" }'
      is_json: true
```

Most steps standardize on the `LLMB_ARTIFACT_ID:<id> LLMB_ARTIFACT_PATH:<path>` convention so a single
rule works across environments:

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

For a `mem://` output the workload emits `LLMB_ARTIFACT_STATE:<value>` instead, and the rule
builds a `"state"` binding (passed to the consumer verbatim, no path normalisation). See
[Monitoring and artifact events](../steps/monitoring-and-artifact-events.md) for the step
author's guide to both variants.

## `step.yaml` `config:` — common fields read by environments

The `config:` section of `step.yaml` (and per-step `config` overrides in `build.yaml`) carries fields
environments read at launch time. These are common; type-specific blocks (`k8s.*`, `lsf.*`,
`docker.*`, launcher `resources`, …) are documented on each page.

```yaml
config:
  retry_enabled_default: false      # Whether retry is enabled for this step type by default.
                                    # Overridable per-run in build.yaml. Default: false.
  retry_transparently_default: true # Deduplicate NEWARTIFACT events across retries. Default: true.

  compute_config:                   # Generic resource hints. K8s/Lsf/Docker/Runpod translate these
    num_nodes: 1                    # into backend resource specs. (SkyPilot ignores compute_config —
    num_gpus_per_node: 0            # it reads resources from the launcher; see skypilot.md.)
    num_cpus_per_node: 0
    total_memory_per_node: ""

  workload:                         # Used by Lsf (and bash-style steps) to derive workspace/log paths.
    path: ""
    args: ""
    workspace_dir: ""
    output_dir: ""
```

## See also

- [Setup guides](setup/) — provisioning the backends (SkyPilot Kubernetes/SLURM, RunPod) and build-time secret scripts
- [build.yaml reference](../builds/build-yaml-reference.md)
- [Steps](../steps/README.md) — built-in steps and step.yaml structure
- [architecture/environment-classes.md](../architecture/environment-classes.md) — the `Environment` base class internals
- [Troubleshooting](../help/troubleshooting.md)
