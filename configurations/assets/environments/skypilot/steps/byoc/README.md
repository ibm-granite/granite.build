# byoc (SkyPilot)

Bring-Your-Own-Code step for SkyPilot clusters. It clones a **public git repo** during
`setup` (optionally running a `setup_command` afterwards, e.g. to install dependencies)
and runs a user-defined command during `run`, inside a **public container image** or on
the bare launcher node — no custom image is built for this step.

> **Developing or testing this step?** See `steps/byoc/skypilot/README.md` in the
> granite.build repository for how the step is generated, tested, and published —
> including the environment variables the SkyPilot **aws** integration test requires
> (`AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, or `AWS_PROFILE`).

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/byoc` URI:

```yaml
steps:
  - step_uri: space://steps/byoc
```

## Config contract (`byoc_config`)

All fields live under the step's `config.byoc_config`.

### Required

| Field | Type | Purpose |
|---|---|---|
| `repo` | string | Public git repository URL cloned during `setup`. An empty value fails the step. |
| `command` | string | Bash command executed during `run`, from the step's working directory (the workdir root). `cd` into the cloned repo yourself if needed, e.g. `cd code && python main.py`. |

### Optional

| Field | Type | Purpose |
|---|---|---|
| `image` | string | Public container image the step runs in, e.g. `python:3.12-slim`. Rendered at runtime as SkyPilot `docker:<image>`. Defaults to `quay.io/fedora/fedora-minimal:42`; set to `""` to run on the bare launcher node instead (e.g. a cluster without Pyxis, which cannot run container images). |
| `ref` | string | Branch, tag, or commit checked out after cloning. Default: the repo's default branch. |
| `workdir` | string | Subdirectory (under the workdir root) the repo is cloned into. Default: `code`. |
| `setup_command` | string | Bash command run during `setup`, **after** the clone, from the workdir root. Use it for dependency installation, e.g. `cd code && pip install -r requirements.txt`. `set -eu` is in effect, so a failure fails the build. Empty (default) => skipped. |

> **`image` is a runtime choice, not a built image.** Unlike custom-image steps (which
> build a Dockerfile), `byoc` builds no image; `image` selects an existing public image
> and the code arrives at run time via `git clone`.

## Inputs and outputs

`byoc` declares no step-level I/O schema of its own, but the **target** can declare
**any number** of keyed `inputs:` and `outputs:` — there is nothing to change on the
step to consume more of either.

### Inputs (any number)

Declare each input on the target (a direct `uri:`, or a `binding:` to an upstream
target's output). The byoc `command` references each one by name via Jinja, which is
rendered into `config.byoc_config.command` **before** `run` (inputs are resolved by the
environment first, so they are concrete paths/values at launch time):

- filesystem-backed inputs (`env://`, `hf://`, `file://`, `lh://`, …):
  `{{ bindings.<name>.binding.path }}`
- `mem://` state inputs: `{{ bindings.<name>.binding.state }}`

Reference each input by the scheme it actually uses (`.path` for filesystem, `.state`
for `mem://`) — byoc accesses bindings explicitly, so there is no auto-exported
`GB_BYOC_INPUT_*` env var.

**File mounts** — the optional `src/` directory beside the template is mounted to
`./src` at the workdir root on the cluster (see [`src/helpers.sh`](src/helpers.sh)),
demonstrating the SkyPilot `file_mounts` input.

### Outputs (any number)

Declare each output on the target, then have your `command` register an artifact by
printing one marker line per artifact (captured by `skypilot_monitor`):

```
LLMB_ARTIFACT_ID:<output-id> LLMB_ARTIFACT_PATH:<abs-path>
```

- `<output-id>` must match an `outputs.<id>` declared on the target.
- Emit one line per artifact; **repeat** the same `<output-id>` on additional lines to
  register multiple artifacts under a single output.
- For `mem://` outputs, use `LLMB_ARTIFACT_STATE:<value>` instead of `LLMB_ARTIFACT_PATH`.

For the full target I/O schema and the `bindings.*` Jinja variables, see
`docs/builds/build-yaml-reference.md`; for the marker convention (PATH vs STATE,
anchoring, the shipped monitors), see `docs/steps/monitoring-and-artifact-events.md` in
the granite.build repository.

## Working directory and paths

Both `setup` and `run` start in the same **working directory** (the step's per-run
workdir), so the step never needs to know its absolute location. `repo` is cloned
into `<workdir>` (default `code/`) beneath it, and `src/` is mounted at `./src`.
Use relative paths from there; when you need an absolute path — e.g. for an
`LLMB_ARTIFACT_PATH` marker — derive it at run time with `$(pwd)`.

## Example build.yaml

This example declares **two byoc targets**: an upstream `pretrain` producer whose
`model` output the downstream `run` target binds to. `run` wires **two inputs** (one
direct `uri:`, one `binding:` to `pretrain`'s output) and **two outputs** (one of which
receives two artifacts via repeated markers). The `command` reads each input via
`{{ bindings.<name>.binding.path }}` and registers each artifact via a marker line:

```yaml
granite.build:
  name: byoc-example
  version: 0.0.1
  targets:
    pretrain:
      environment_uri: space://environments/skypilot/aws
      outputs:
        model:
          uri: hf://huggingface.co/my-org/byoc-pretrain-{{ run_metadata.targetsteprun_id | short_hash }}
      steps:
        - step_uri: space://steps/byoc
          config:
            compute_config: { num_nodes: 1, num_gpus_per_node: 1 }
            byoc_config:
              image: "python:3.12-slim"
              repo: "https://github.com/org/repo"
              command: >-
                cd code;
                python pretrain.py --out-dir "$(pwd)/out";
                echo "LLMB_ARTIFACT_ID:model LLMB_ARTIFACT_PATH:$(pwd)/out/model.ckpt"
    run:
      environment_uri: space://environments/skypilot/aws
      inputs:
        dataset:
          uri: hf:///datasets/org/my-dataset
        init_model:
          binding: pretrain.model   # <upstream-target>.<output-id>
      outputs:
        checkpoints:
          uri: hf://huggingface.co/my-org/byoc-out-{{ run_metadata.targetsteprun_id | short_hash }}
        report:
          uri: hf://huggingface.co/datasets/my-org/byoc-report-{{ run_metadata.targetsteprun_id | short_hash }}
      steps:
        - step_uri: space://steps/byoc
          config:
            compute_config: { num_nodes: 1, num_gpus_per_node: 1 }
            byoc_config:
              image: "python:3.12-slim"
              repo: "https://github.com/org/repo"
              ref: "main"
              workdir: "code"
              setup_command: "cd code && pip install -r requirements.txt"
              command: >-
                cd code;
                python main.py
                --dataset "{{ bindings.dataset.binding.path }}"
                --init-model "{{ bindings.init_model.binding.path }}"
                --out-dir "$(pwd)/out";
                echo "LLMB_ARTIFACT_ID:checkpoints LLMB_ARTIFACT_PATH:$(pwd)/out/epoch-1.ckpt";
                echo "LLMB_ARTIFACT_ID:checkpoints LLMB_ARTIFACT_PATH:$(pwd)/out/epoch-2.ckpt";
                echo "LLMB_ARTIFACT_ID:report LLMB_ARTIFACT_PATH:$(pwd)/out/report.json"
```

A single direct input and output are just the one-entry case of the above.

## Notes and limitations

- **Public repos only.** `setup` runs an unauthenticated `git clone`; private
  repositories (and the credential/secret wiring the LSF `custom_code_lsf` step
  provides) are out of scope for this exemplar.
- **No dependency caching.** Dependencies are whatever the chosen public `image`
  provides plus anything your `command`/repo installs at run time.
- **Variable inputs/outputs are a target-level feature, not a step limitation.** Declare
  any number of `inputs:`/`outputs:` on the target and wire them from the `command` as
  shown in [Inputs and outputs](#inputs-and-outputs) — no change to the step is required.
