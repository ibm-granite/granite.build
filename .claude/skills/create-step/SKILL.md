---
name: create-step
description: Author a full, reusable Granite.build custom step — a step directory (step.yaml + bash_scripts/) referenced from a build.yaml by a file:// URI. Use this (instead of create-build's inline command+heredoc) when you have owned/multi-file code, a step you want to version and reuse across builds, or need per-environment launch behavior. Covers the step.yaml launch contract, the artifact monitor, the runtime env-var contract, and per-compute-environment differences.
argument-hint: "[step-name]"
allowed-tools: Bash(ls *) Bash(test *) Bash(cat *) Bash(grep *) Bash(find *) Bash(mkdir *) Bash(chmod *) mcp__gbmcp__gbserver_status mcp__gbmcp__gbserver_start mcp__gbmcp__build_start mcp__gbmcp__build_status mcp__gbmcp__build_log mcp__gbmcp__build_list mcp__gbmcp__build_describe mcp__gbmcp__build_job_log
---

# Author a Granite.build custom step

A **step** is the reusable unit of execution: a directory containing a `step.yaml` (how to launch it) and a `bash_scripts/<step-name>/` folder with the script(s) it runs. Use this skill when a step is the right tool; use **`create-build`** (inline `command` + heredoc) for a simple, self-contained, one-off workload.

**Use a step (this skill) when:**
- the workload is owned code you'll version and reuse across builds,
- it's more than fits comfortably in a heredoc (multiple files, real modules),
- you want a clean step/build separation, or per-environment launch behavior.

**Use `create-build` instead when:** it's a self-contained one-off — the `command` step + a heredoc is simpler and needs nothing on disk.

**Before authoring anything, check whether a step already ships for your workload** (`inference`, `lora-finetune`, `inference-lora`, …) — see `create-build`'s **`references/steps.md`**. If one fits, don't author a duplicate; just reference `space://steps/<name>`.

(Build actions go through the `mcp__gbmcp__*` tools, not the `gb` CLI; the tools are always attached — if a build call reports the backend is unreachable, `gbserver_start()` brings it up. See **`run-gbserver`**.)

## The distinction: steps are referenced by a `file://` URI

Custom steps in this workflow live **in the user's project** (e.g. `<project>/steps/<step-name>/`), *not* under the space's assets. The build references them by an **absolute `file://` URI**:

```yaml
steps:
  - step_uri: file:///abs/path/to/<project>/steps/<step-name>
```

- **Use an absolute `file:///…` URI.** A *relative* `file:` URI resolves against the current working directory (not the build.yaml's location or the space), which is unstable — an absolute path is unambiguous. (`space://…` URIs resolve via the space's `base_uris` chain regardless of cwd; `file://` URIs resolve by path.)
- Packaged/shipped steps that live in the space are still referenced as `space://steps/<name>` — but for your own custom steps, `file://` keeps them with the project and out of the space.
- **Verify resolution** on your gbserver: if a `file://` step URI doesn't resolve, consult the **`gb-docs`** skill (`docs/environments/step-resolution.md`) for how step URIs resolve in your version.

## Step anatomy

```
<project>/steps/<step-name>/
├── step.yaml
└── bash_scripts/
    └── <step-name>/          # MUST match the step name
        └── command.sh        # default entrypoint, OR run.py via script_path
```

### `step.yaml` — the launch contract (bash environment)

`type: nohup` is mandatory for the bash backend. The `NEWARTIFACT_IN_ENVIRONMENT_EVENT` monitor is what captures your outputs — keep it if the step emits artifacts.

```yaml
name: <step-name>            # MUST equal the directory name
version: v1
type: custom
config: {}
environment_configs:
  Bash:                       # environment TYPE key (see "Compute-environment differences")
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
            # Capture outputs: the script prints a line beginning with
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
- Set `config.script_path: run.py` → it runs `bash_scripts/<step-name>/run.py`.

> **CRITICAL:** do **not** reference the builtin `gbstep` on the bash backend — it declares launchers only for `k8s`/`Lsf` and dies with `KeyError: 'helm'`. Author your own `Bash`/`nohup` launcher as above.

### The script owns its runtime

The launcher hands the script a **clean environment** — no `PATH`/`HOME`/`os.environ`. `command.sh` establishes PATH from the injected python dir and builds its own venv (idempotently) before importing torch/etc.:

```bash
#!/bin/bash
set -eu
export PATH="${LLMB_BASH_PYTHON_DIR:?}:/usr/local/bin:/usr/bin:/bin"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# cache venv OUTSIDE the per-run output dir so reruns are fast
OUT="${LLMB_BASH_OUTPUT_DIR:-}"
case "$OUT" in */workdir/*) VENV_BASE="${OUT%%/workdir/*}/.gb-venvs";; *) VENV_BASE="${TMPDIR:-/tmp}/.gb-venvs";; esac
mkdir -p "$VENV_BASE"; export HF_HOME="$VENV_BASE/hf-cache"
PY="$LLMB_BASH_PYTHON_DIR/python3"; [ -x "$PY" ] || PY=python3
VENV="$VENV_BASE/<step-name>"
[ -x "$VENV/bin/python" ] || { "$PY" -m venv "$VENV"; "$VENV/bin/pip" install --quiet --upgrade pip; }
"$VENV/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
exec "$VENV/bin/python" "$SCRIPT_DIR/run.py"
```

### Runtime env-var contract (what the script receives)

- `LLMB_BASH_OUTPUT_DIR` — write outputs here; register them by printing (at line start) `LLMB_ARTIFACT_ID:<id> LLMB_ARTIFACT_PATH:<abs-path>`, where `<id>` matches an `outputs.<id>` in the build.
- `LLMB_BASH_INPUT_<NAME>` — resolved local path of each target input `<name>` (uppercased). E.g. input `model` → `$LLMB_BASH_INPUT_MODEL`.
- `LLMB_BASH_PYTHON_DIR` — dir of a pinned Python ≥3.11 (lead PATH with it; there is no HOME).
- `config.bash.env` entries from the build.yaml — exported as env vars (per-run params).

## Compute-environment differences (steps are env-specific)

**There is no automatic translation of a step across environments.** A step is keyed by environment **type** under `environment_configs`, and the runtime *selects the block matching the active environment* — anything not authored for that env can't run there. The workload *script* (`run.py`) can be shared, but the launch/provisioning wrapper is authored per environment:

| Environment | `environment_configs` key | Launcher `type` | How Python/deps are provided | Inputs |
|---|---|---|---|---|
| bash (local) | `Bash` | `nohup` | script builds its own **venv** at runtime | auto-exported as `$LLMB_BASH_INPUT_*` |
| docker | `Docker` | `docker` | the container **image** (deps prebaked; no venv) | wire manually: `docker.env: LLMB_BASH_INPUT_X: {value: "{{ bindings.x.binding.path }}"}` |
| LSF | `Lsf` | `bsub` | a cluster **module / python env** | staged to the remote host |
| skypilot (aws/k8s/slurm/lsf) | `Skypilot` | `skypilot` | a **conda env in the cluster image** + a `setup:` block; needs cluster resources | via `run:`/`setup:` blocks |
| runpod | `Runpod` | `runpod` | the container **image** | via the image/command |

Consequences to keep in mind:
- **Each step available in a space is specific to a specific environment** — e.g. the on-disk `command`/`hello`/`inference` steps under `environments/bash/steps/` are the **bash** copies; `docker`/`lsf`/`skypilot` have their own copies. A step resolves only for an environment whose type it declares.
- The launcher `type` and monitor `type` are **not interchangeable** — `nohup` exists only on bash, `docker` only on docker, `bsub` only on LSF, `helm` only on k8s. Referencing a launcher the active env doesn't implement fails at launch (this is the `gbstep`→`helm` bug on bash).
- **Building a venv is a bash-ism.** On docker/skypilot/lsf the deps come from a prebuilt image/conda/module, and building a fresh venv would shadow (or omit) them — so those blocks run the script directly against the provided interpreter.
- To make one step run on multiple environments, author a block per env in the same `step.yaml`, sharing the script but supplying each env's launcher + deps source. For the bash-only standalone default, a single `Bash` block is enough.

## Reference it from the build, then run

```yaml
granite.build:
  name: <build-name>
  version: 0.0.1
  targets:
    run:
      environment_uri: space://environments/bash
      inputs:
        model: { uri: hf:///ibm-granite/granite-4.0-h-350m }
      outputs:
        result: { uri: file:outputs/run_{{ binding.path | short_hash }}/ }
      steps:
        - step_uri: file:///abs/path/to/<project>/steps/<step-name>   # absolute file:// URI
          config:
            bash: { env: { PROMPT: "..." } }
            compute_config: { num_nodes: 1 }
```

Submit the build.yaml **text** with `build_start(file_content=<yaml text>)`:

```
build_start(file_content=<the build.yaml as a string>)   # returns a build_id
build_status(build_id)                                    # monitor; done once it leaves build_list()
```

There is **no `build_validate` tool**; validation never proved a build runs anyway — always do a real `build_start` and watch it reach SUCCESS.

## Debugging — the real output

`build_log`/`build_status` show status events, not stdout. Fetch the script's real output with `build_job_log(build_id)` — the tail of the on-disk `job.log` (prints, tracebacks, the `LLMB_ARTIFACT_ID` line). If a step "succeeded" but did nothing, read it first.

## Checklist

1. `gbserver_status()`; if it isn't `ready`, `gbserver_start()` to bring the backend up (see **`run-gbserver`**).
2. Create `<project>/steps/<name>/step.yaml` (Bash/`nohup` launcher + artifact monitor) and `bash_scripts/<name>/` (script that owns its venv, reads inputs/params from env, emits the `LLMB_ARTIFACT_ID` line). Make scripts executable.
3. Reference it in the build.yaml by an **absolute `file:///…` `step_uri`**; declare inputs/outputs; per-run params in `config.bash.env`.
4. Submit with `build_start(file_content=<yaml text>)`. Monitor to SUCCESS; read `build_job_log(build_id)` to confirm it actually ran.
5. For any field/option/error or `file://`-resolution question, consult **`gb-docs`**.
