---
name: create-build
description: Use a new Granite.build build.yaml for any compute workload — model training (SFT/LoRA/EPT), inference/serving, data generation/processing, evaluation, or arbitrary compute. Targets, steps, inputs/outputs (artifacts), compute resources, and how they resolve within a space. Use when asked to create, write, or scaffold a new build, build.yaml, or pipeline.
argument-hint: "[build-name]"
allowed-tools: Bash(ls *) Bash(test *) Bash(cat *) Bash(grep *) Bash(find *)
---

# Author a Granite.build build

This skill makes a workload (train a model, run inference, generate/process data, evaluate, or any script) run on Granite.build. You produce **two kinds of artifacts**:

1. A **step** — the unit of execution. A step is a directory under the space's assets containing a `step.yaml` (how to launch it) and a `bash_scripts/<step-name>/` folder with the script(s) it runs. Steps live in the space, are reusable, and are referenced by URI.
2. A **build.yaml** — the run definition. It references steps by URI, wires inputs/outputs, and passes per-run settings. The build.yaml is *not* a script and does not live under assets; keep it with the user's project (e.g. a `lora-granite/build.yaml` folder).

Granite.build is **workload-agnostic**: training, inference, data-gen, and evaluation are all "a step that runs a command." There is no special "training build" shape. What changes between workloads is the script inside the step and the settings in build.yaml — not the structure.

## Operating assumptions (read first)

These reflect how this environment actually behaves. Follow them unless the user says otherwise.

- **Default to standalone.** Assume `GB_ENVIRONMENT=STANDALONE`, a local `gbserver`, the **bash** compute backend, and SQLite. No cloud creds, no Kubernetes. Everything below targets that backend.
- **A server must be running before you create or run anything.** Building, validating, and running all talk to a live `gbserver`. Before your first `gb` command, confirm one is up; if not, start one. See "Ensure the server is running."
- **Compose with targets and steps as the workload needs.** A workload is one or more **targets**, each containing one or more **steps**. Steps within a target run sequentially and share a filesystem; targets can be wired together with `binding` (a downstream target's input bound to an upstream target's output) and run on the standalone bash backend. Use sequential steps in one target for phases that share files (e.g. generate data → train in the same workspace); use separate bound targets when phases have distinct inputs/outputs or compute and you want them as separate lineage-tracked units. A single target with one step is still the simplest shape — reach for more structure only when the workload calls for it.
- **The server picks up new/edited steps dynamically.** After you add or change a step under the space's assets, you do **not** need to restart gbserver. The next build resolves the new step. (Restart only if a change genuinely isn't being seen and you've ruled out everything else.)
- **Steps, step definitions, and scripts live under the space's assets.** Only the build.yaml lives with the user's project. Never inline a training script into build.yaml; put it in the step's `bash_scripts/` dir.
- **Each step's script sets up its own runtime (venv) internally.** Don't rely on a pre-activated environment. If the workload needs Python deps (torch/trl/peft/…), the script creates/activates its own venv (idempotently) before importing them — see "The step script owns its environment." Never install heavy deps into the gbserver venv.

## Ensure the server is running

Before any `gb` call. The environment sets a per-job `GBSERVER_PORT` (defaults to 8080); check for a server on THAT port. Match by process (works from any call, the same way `run-gbserver` launches and detects it):

```
pgrep -f "gbserver standalone --port ${GBSERVER_PORT:-8080}" >/dev/null && echo "server up" || echo "server DOWN"
```

If it's down, start it with the **`run-gbserver`** skill (it clones/sets up the repo and launches the standalone server on `${GBSERVER_PORT:-8080}` with a space). Do not author against a dead server — you can't discover real step/env names and every command errors.

Throughout, run `gb` with the venv active and the env set — including `GBSERVER_HOST` pointed at the per-job port (it must match the port `run-gbserver` used; the client otherwise defaults to `http://localhost:8080` and can't reach a server on a non-default port). Run `gb` in a Bash call **with `dangerouslyDisableSandbox: true`** so it shares the server's namespace:

```
source <repo>/.venv/bin/activate
export GB_ENVIRONMENT=STANDALONE
export GBSERVER_HOST="http://127.0.0.1:${GBSERVER_PORT:-8080}"
```

## Discover what already exists (don't invent URIs)

`gb step list` is **not supported in standalone mode** — it errors. Discover by reading the space on disk instead:

- Find the space: `gb space list` (works in standalone) gives space names; the space dir is the one passed to `--space-dir` (commonly `configurations/spaces/local`).
- The space's `space.yaml` has a `base_uris:` chain (e.g. `file://../../assets`) — that's where steps/environments/assetstores resolve from.
- **Environments:** `ls <assets>/environments/` → typically `bash docker runpod skypilot ...`. Default to `bash`.
- **Steps:** steps usable on the bash backend live under `<assets>/environments/bash/steps/<name>/`. A step is referenced as `space://steps/<name>` even though it lives in an env-keyed subdir — the resolver maps it.
- **On-disk reference steps** live under `<assets>/environments/bash/steps/` (`hello`, `command`, `inference`, `inference-lora`, `lora-finetune`). `hello`/`command` are **minimal** steps (launcher + a `MESSAGE_EVENT` log monitor, no artifact capture) — copy them for shape. `inference`, `inference-lora`, and `lora-finetune` are the reference for a step that **captures outputs as artifacts** — they carry the `NEWARTIFACT_IN_ENVIRONMENT_EVENT` monitor. Copy `inference` (or `lora-finetune`) when your step must emit artifacts.

When you reference `space://steps/<name>` or `space://environments/<name>`, the name must be a real directory you found above.

## CRITICAL: bash steps need a `nohup` launcher — the generic `gbstep` does NOT work on bash

The builtin `gbstep` step defines launchers only for `k8s` and `Lsf`. On the **bash** backend it falls back to a `helm` launcher and dies with `KeyError: 'helm'`. **Do not reference `space://steps/gbstep` on the bash backend.** Instead, author your own step whose `step.yaml` declares a `Bash` launcher of `type: nohup`. This is the single most common mistake — avoid it.

## Authoring a bash step (the core skill)

A step is a directory under `<assets>/environments/bash/steps/<step-name>/`:

```
<assets>/environments/bash/steps/<step-name>/
├── step.yaml
└── bash_scripts/
    └── <step-name>/          # MUST match the step name
        └── command.sh        # default entrypoint, OR run.py via script_path
```

### `step.yaml` — the launch contract

Model it on this. `type: nohup` is mandatory for bash. The monitor's `NEWARTIFACT_IN_ENVIRONMENT_EVENT` block is what captures your outputs — keep it. **On-disk reference:** this artifact-capturing shape matches the `inference`, `inference-lora`, and `lora-finetune` steps (which carry this monitor); `hello`/`command` are minimal and have only the `MESSAGE_EVENT` block, so copy `inference` (or `lora-finetune`) when your step emits artifacts.

```yaml
name: <step-name>            # must equal the directory name
version: v1
type: custom
config: {}
environment_configs:
  Bash:
    launchers:
      <step-name>:
        type: nohup
        monitors:
          - log_monitor
        config:
          script_path: run.py   # optional; omit to default to command.sh
    monitors:
      log_monitor:
        type: log_monitor
        config:
          event_configs:
            # Captures outputs: the script prints a line of the form
            #   LLMB_ARTIFACT_ID:<id> LLMB_ARTIFACT_PATH:<abs-path>
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
            - event_type: MESSAGE_EVENT
              line_regex: ".*"
              is_json: false
              event_fields:
                - field_name: msg
                  field_regex: ".*"
```

- Omit `config.script_path` → the launcher runs `bash_scripts/<step-name>/command.sh`.
- Set `config.script_path: run.py` → it runs `bash_scripts/<step-name>/run.py` (must be executable).

### What the script receives at runtime (env vars)

gbserver wraps your script and injects:

- `LLMB_BASH_OUTPUT_DIR` — write outputs here. Each sub-directory you create becomes an artifact; or set `config.bash.single_output_artifact: true` in build.yaml to make the whole dir one artifact; or emit `LLMB_ARTIFACT_ID:<id> LLMB_ARTIFACT_PATH:<path>` explicitly (preferred — gives the artifact a stable id matching your `outputs:` name).
- `LLMB_BASH_INPUT_<NAME>` — for each input named `<name>` in build.yaml, the resolved **local path** to that artifact, with `<NAME>` uppercased. E.g. input `base_model` → `$LLMB_BASH_INPUT_BASE_MODEL`.
- Anything you put under `config.bash.env` in build.yaml is exported as an env var. **This is how per-run settings reach the script** (hyperparameters, the fact to teach, etc.). Read them with defaults in the script.
- `LLMB_BASH_ASSET_DIR`, `LLMB_BASH_LAUNCH_ID`, build/target ids — usually not needed.

### The step script owns its environment

The script must not assume torch/trl/etc. are importable. It creates and uses its own venv idempotently, inside the step, at run time. Two clean patterns — pick one:

**(a) `command.sh` creates the venv, then runs the python** (recommended):
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV="$HOME/.gb-<workload>-venv"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install torch transformers trl peft datasets accelerate
fi
exec "$VENV/bin/python" "$SCRIPT_DIR/run.py"
```

**(b) a single `run.py`** whose first action re-execs into a self-built venv if its deps are missing. Use (a) unless the user wants a one-file step.

Either way: the venv is created **inside the step at run time**, is idempotent (reused on later runs), and is separate from the gbserver venv. Pin to a small Granite model for CPU (`ibm-granite/granite-4.0-350m`); CPU training of a few dozen LoRA steps takes seconds-to-minutes.

## The build.yaml — targets and steps

Working build.yaml examples ship under `samples/standalone/` in the granite.build repo — read those for real, runnable references (single- and multi-target shapes).

Keep it with the user's project (not under assets). The example below uses a single target referencing one step — the simplest shape; add more steps to a target (sequential, shared filesystem) or more targets (wired by `binding`) as the workload needs. Settings go in `config.bash.env`. The base model can be an `hf:///owner/repo` input — the bash env's HF assetstore pulls it automatically (no separate pull step), surfacing it as `$LLMB_BASH_INPUT_BASE_MODEL`.

```yaml
granite.build:
  name: <build-name>
  version: 0.0.1
  targets:
    run:                                   # one target
      environment_uri: space://environments/bash
      inputs:
        base_model:
          uri: hf:///ibm-granite/granite-4.0-350m   # triple slash; host defaults to huggingface.co
      outputs:
        adapter:
          base_uri: file:workspace/adapter/  # produced artifacts use base_uri
      steps:
        - step_uri: space://steps/<step-name>
          config:
            bash:
              env:                           # per-run settings -> env vars in the script
                TRAIN_MAX_STEPS: "30"
                TRAIN_LR: "2e-4"
                LORA_R: "8"
            compute_config:
              num_nodes: 1
```

Notes:
- **Inputs** carry exactly one of `uri:` (fixed location: `hf:///…`, `file:…`) or `binding:` (wire to another target's output for multi-target workloads). Use `uri` for fixed external inputs; use `binding` to chain targets.
- **Produced outputs** use `base_uri:` (a push destination); a fixed input file uses `uri:`.
- Multiple steps in one target run sequentially and share the filesystem — use this for phases that share files. Use separate bound targets when phases have distinct inputs/outputs or compute.

## Validate, then run

```
gb build validate -f <path>/build.yaml      # schema + URI existence
gb build start    -f <path>/build.yaml       # prints a build id
gb build list                                 # are builds still running? lists builds + their statuses
gb build status   <build-id>                  # monitor a specific build's status
```

**If any `gb` command warns the CLI is out of date** (a client/server version mismatch), add **`--skip-version-check`** *after* the subcommand: `gb build start -f <path>/build.yaml --skip-version-check`. The check is advisory; skipping it is safe.

**Monitor with `gb build status`; use `gb build list` to see if a build is still running.** `gb build list` lists in-progress builds — **if your build is not in `gb build list`, it is done.** `gb build status <build-id>` is how you monitor a specific build as it progresses toward SUCCESS.

**Validation is necessary but not sufficient.** `gb build validate` passes even when a step would hit the `helm`-launcher bug. It does not prove the build runs. Always do a real `gb build start` and watch it reach SUCCESS (via `gb build list` / `gb build status`) with the expected steps actually executing.

## Debugging — where the REAL output is

`gb build log` and `gb build status` mostly show **status events**, not your script's stdout. The actual script output (prints, tracebacks, training loss) is on disk:

```
~/.granite.build/workdir/llm-build-<build-id>/target-<t>/target-run-*/step-<s>/step-run-*/launch-*/outputs/job.log
```

Find and read it:
```
find ~/.granite.build/workdir/llm-build-<build-id> -name job.log
```
This `job.log` is your primary debugging artifact. If a step "succeeded" but did nothing, or failed mysteriously, read `job.log` first. Reading a *prior* successful build's `job.log` and its copied `step.yaml`/`run.py` (under the same tree) is the fastest way to learn a working pattern.

## Authoring procedure (checklist)

1. Confirm a gbserver is running (start via `run-gbserver` if not). Set `GB_ENVIRONMENT=STANDALONE`, activate the repo venv.
2. Discover the space, its `bash` environment, and existing steps by reading `<assets>/`. Read an on-disk step as a template — `hello` for the minimal shape, `inference`/`lora-finetune` for the artifact-capturing shape.
3. Author the step under `<assets>/environments/bash/steps/<name>/`: `step.yaml` (Bash/`nohup` launcher + artifact monitor) and `bash_scripts/<name>/` (the script, which sets up its own venv and reads settings from env). Make scripts executable.
4. Write the build.yaml with the user's project: target(s) and step(s) matching the workload (one target/one step is the simplest), settings in `config.bash.env`, base model as an `hf:///` input, outputs via `base_uri`. (No restart needed — the server picks the step up.)
5. `gb build validate`, then **`gb build start`**. Monitor it with `gb build status <build-id>`; the build is done once it no longer appears in `gb build list`. Then read `job.log` to confirm the workload actually ran (not just that status flipped to SUCCESS).
6. If anything is unclear about a field/option/error, consult the **`gb-docs`** skill.

## When unsure

Invoke **`gb-docs`** for the authoritative schema/CLI/troubleshooting docs. If the docs and this skill disagree on a documented field, trust the docs — but note the docs are k8s/LSF-centric, and several documented conveniences (`config.workload.commands`, `gb.files_to_create`) are **k8s/LSF-only and do not apply to the bash backend**. For bash behavior, the source of truth is the on-disk steps (`hello`/`command` for a minimal launch; `inference`/`lora-finetune` for artifact capture) plus a working build's `job.log`/`step.yaml`.
