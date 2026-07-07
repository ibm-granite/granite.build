# Bash environment

> **Audience:** operators configuring a `Bash` environment, anyone authoring a `build.yaml` for it,
> or writing a custom bash step. For the common schema see [Environment overview](README.md); for step
> resolution see [step-resolution.md](step-resolution.md).

## Compute environment

The **bash** environment runs a step as a local OS process — no container image, no cluster. It is the
simplest backend and the one used by the standalone samples under
[`samples/standalone/`](../../samples/standalone/). The step's script runs under `nohup`; its
stdout/stderr are tailed by the monitor, which turns matching lines into build events.

The implementation is [`Bash`](../../src/gbserver/environment/bash.py).

## `environment.yaml`

The bash environment needs no type-specific config:

```yaml
name: bash
type: Bash
config: {}
assetstores:
  - store_uri: space://assetstores/hf/
    load:
      - mode: default
  - store_uri: space://assetstores/local/
    load:
      - mode: default
    push:
      - mode: default
```

## `step.yaml` — launcher and monitor types

| `type` | Method | Notes |
|--------|--------|-------|
| `nohup` (launcher) | `launch_nohup` | The only launcher. Runs `./bash_scripts/<step.name>/<job_sub>.sh` as a `nohup` process. |
| `log_monitor` | `monitor_log_monitor` | Tails the process stdout/stderr and applies `event_configs`. |

> The `bash_scripts/<name>/` subdirectory **must match the step's `name:`** — the launcher runs
> `./bash_scripts/<step.name>/<job_sub>.sh`.

## Execution flow

When a target whose `environment_uri` is `space://environments/bash` runs a step:

1. The step is resolved to a directory containing `step.yaml` (see [step-resolution.md](step-resolution.md)).
2. The bash environment selects the step's `Bash` launcher (type `nohup`) and prepares a working copy
   of the step's `bash_scripts/<step.name>/` directory.
3. gbserver renders the built-in **job-submission script** (`llmb_bash_jobsub.sh`), which exports a set
   of `LLMB_BASH_*` environment variables and then runs the step's `script_path` (e.g. `run.py`).
4. The script runs as a `nohup` process; its output is tailed by the monitor, which emits build events
   (including artifact registration).

The launcher is [`Bash.launch_nohup`](../../src/gbserver/environment/bash.py); the job template is
[`llmb_bash_jobsub.sh`](../../src/gbserver/builtins/steps/gbstep/bash_scripts/).

## How inputs reach your script

Declare inputs **on the target** in `build.yaml`. gbserver resolves each input (downloads an `hf://`
model, a `file:` dataset, …) and **automatically exports its local path** as
`LLMB_BASH_INPUT_<NAME>` — the uppercased input name, prefixed `LLMB_BASH_INPUT_`.

```yaml
# build.yaml
targets:
  inference:
    environment_uri: space://environments/bash
    inputs:
      model:                                   # <- target input named "model"
        uri: hf:///ibm-granite/granite-4.0-h-350m
    steps:
      - step_uri: space://steps/inference
```

```python
# run.py — the resolved local path arrives automatically:
model_path = os.environ["LLMB_BASH_INPUT_MODEL"]   # input "model" -> LLMB_BASH_INPUT_MODEL
```

This auto-export is done by the job template (the `bindings` loop): for every resolved input binding
it emits `export LLMB_BASH_INPUT_<NAME>="<local path>"`. **You never set these by hand** in the bash
environment — declaring the target input is enough. (The Docker environment, by contrast, requires you
to wire the binding into env manually; see [docker.md](docker.md).) An input that isn't bound simply
has no `LLMB_BASH_INPUT_<NAME>` variable — treat unset/empty as "not provided".

### Declaring the input contract (`step.yaml` I/O schema)

A step should declare which inputs it expects so the build is **validated up front**:

```yaml
# step.yaml
inputs:
  allow_unknown: false
  required:
    model:   { type: model, accept: [uri, binding] }
  optional:
    adapter: { type: model, accept: [uri, binding] }
outputs:
  allow_unknown: true        # the script may emit artifacts dynamically
  optional:
    generation: { type: fileset }
```

gbserver validates target inputs against this schema before running
(`Build.__validate_step_inputs_and_outputs` in [`build.py`](../../src/gbserver/build/build.py)): a
missing **required** input fails the build with a clear error, and `accept` controls whether a `uri:`
and/or a cross-target `binding:` is permitted. `type` values come from `StepIOTypeEnum` (`model`,
`dataset`, `fileset`, `bucket`); `accept` from `StepInputsAcceptEnum` (`uri`, `binding`).

## How configuration reaches your script

Tunable knobs that are **not** artifacts (prompts, hyperparameters, …) are passed as environment
variables via the step's `config.bash.env` block in `build.yaml`:

```yaml
steps:
  - step_uri: space://steps/inference
    config:
      bash:
        env:
          PROMPT: "what are the top five states in the us"
          MAX_NEW_TOKENS: "512"
```

```python
prompt = os.environ.get("PROMPT", "<default>")          # read in run.py
```

Precedence, lowest to highest (later wins):

1. space secrets and the `environment.yaml` `env` block,
2. the `step.yaml` launcher `env` (defaults baked into the step),
3. **`config.bash.env` from `build.yaml`** (per-build overrides).

So `build.yaml` is the single source of truth for a run. (Layer 3 is handled in `Bash.launch_nohup`;
it mirrors the Docker launcher's `config.docker.env`.) For this reason the example steps keep **no**
launcher `env` defaults — they default inside the script with `os.environ.get(KEY, default)` so
build.yaml always wins.

## How your script reports outputs

The script writes files under `$LLMB_BASH_OUTPUT_DIR` and registers an artifact by printing a line the
monitor recognizes:

```python
print(f"LLMB_ARTIFACT_ID:{artifact_id} LLMB_ARTIFACT_PATH:{output_dir}")
```

The monitor's `NEWARTIFACT_IN_ENVIRONMENT_EVENT` rule (in the step's `step.yaml`, see the
[event_configs schema](README.md#event_configs--log-line-parsing-rules)) parses `LLMB_ARTIFACT_ID:` and
`LLMB_ARTIFACT_PATH:` and binds the artifact. **The id must match an output name declared on the
target**, so the artifact is routed to that output's URI.

## Standalone caveats (multi-step pipelines)

When running under `gbserver standalone` / `gbserver build run` (the samples' mode):

- **Each step in a target gets its own isolated launch directory.** Two steps in the same target do
  **not** automatically share an output directory. To pass data between steps *within* a target, all
  steps share `$LLMB_BASH_TARGET_RUN_ID`, so a step can write to a dir keyed on it and a later step can
  read it back.

To pass an output from one target to another, use the normal **cross-target binding**
(`binding: <target>.<output>`) — these schedule correctly in standalone. The `lora-finetune` sample
does exactly this: its `inference` target binds its `adapter` input to the `finetune` target's
`adapter` output, and gbserver runs them in dependency order.

## See also

- [Environments overview](README.md) and the shared [event_configs schema](README.md#event_configs--log-line-parsing-rules)
- [Docker environment](docker.md) — the containerized analogue of bash
- [Steps overview](../steps/README.md) and per-step docs:
  [inference](../../configurations/assets/environments/bash/steps/inference/README.md),
  [lora-finetune](../../configurations/assets/environments/bash/steps/lora-finetune/README.md)
- [build.yaml reference](../builds/build-yaml-reference.md)
