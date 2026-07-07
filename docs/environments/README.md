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
config:                 # Environment-type-specific config — see the per-type page.
  ...
assetstores:            # Asset stores accessible from this environment (see below).
  - store_uri: <uri>
    load:
      - mode: <mode>
        config: {}
    push:
      - mode: <mode>
        config: {}
```

`name`, `type`, and `assetstores` are the only fields the shared schema
([`EnvironmentConfig`](../../src/gbserver/types/environmentconfig.py)) defines; everything under
`config:` is a free-form dict interpreted by the environment class. The per-type pages document each
type's `config:` block.

### Asset stores

`assetstores` map a store URI to the **load** (input) and **push** (output) behaviour for this
environment. Each `load`/`push` entry has a `mode` (e.g. `hf_pull`/`hf_push`, `cos_rclone`, `env_local`,
`default`) and an optional `config`. Modes are implemented by `pullasset_*` / `pushasset_*` methods on the
environment class, which may queue a built-in step (e.g. `hf_pull` injects an
[hfpull](../../src/gbserver/builtins/) step); the exact set is per-type — see each page.

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
| `NEWARTIFACT_IN_ENVIRONMENT_EVENT` | Workload writes an output | `binding_id` (matches an output name in `build.yaml`), `binding` (JSON with `"path"`) |
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
