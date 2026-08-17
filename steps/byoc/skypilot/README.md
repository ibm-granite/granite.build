# byoc (SkyPilot)

Bring-Your-Own-Code step for SkyPilot clusters. Runs in a **public container
image** (or on the bare launcher node), clones a public git repo during `setup`
(optionally running a `setup_command` afterwards, e.g. to install dependencies),
and runs a user-defined command during `run` — no custom image is built or
published for this step.

This is the SkyPilot counterpart of the LSF `custom_code_lsf` and Kubernetes
`custom_code` steps. It is *generated* from the sources in this directory by the
shared Makefile conventions — see the framework overview:
[steps/README.md](../../README.md).

## Referencing the step

`make space` renders a self-contained Space into `space/` (see `SPACE_DIR` in the
[framework overview](../../README.md)). Point the build's Space at that directory
and reference the step by the stable `space://steps/byoc` URI:

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
| `command` | string | Bash command executed during `run`, from inside the cloned repo directory. |

### Optional

| Field | Type | Purpose |
|---|---|---|
| `image` | string | Public container image the step runs in, e.g. `python:3.12-slim`. Rendered at runtime as SkyPilot `docker:<image>`. Defaults to `quay.io/fedora/fedora-minimal:42`; set to `""` to run on the bare launcher node instead (e.g. a cluster without Pyxis, which cannot run container images). |
| `ref` | string | Branch, tag, or commit checked out after cloning. Default: the repo's default branch. |
| `workdir` | string | Subdirectory (under `$GB_BUILD_WORKDIR`) the repo is cloned into. Default: `code`. |
| `setup_command` | string | Bash command run during `setup`, **after** the clone, from `$GB_BUILD_WORKDIR` (the workdir root). Use it for dependency installation, e.g. `cd code && pip install -r requirements.txt`. `set -eu` is in effect, so a failure fails the build. Empty (default) => skipped. |

> **`image` is a runtime choice, not a built image.** Unlike custom-image steps
> (e.g. [eval](../../eval/skypilot/README.md)), `byoc` builds no Dockerfile;
> `image` selects an existing public image and the code arrives at run time via
> `git clone`.

## Inputs and outputs

- **Inputs** — this step declares no target `inputs:` of its own; it brings its
  code by cloning `repo`. Bind target inputs in your `build.yaml` as usual if the
  cloned command needs them (they are resolved by the environment before `run`).
- **File mounts** — the optional `src/` directory beside the template is mounted
  to `$GB_BUILD_WORKDIR/src` on the cluster (see [`src/helpers.sh`](src/helpers.sh)),
  demonstrating the SkyPilot `file_mounts` input.
- **Outputs** — to register an artifact, have your `command` print a line that
  begins with the Granite.build marker (captured by `skypilot_monitor`):

  ```
  LLMB_ARTIFACT_ID:<output-id> LLMB_ARTIFACT_PATH:<abs-path>
  ```

  `<output-id>` must match an `outputs.<id>` declared on the target.

## Env vars the step provides to your commands

The SkyPilot launcher exports `$GB_BUILD_WORKDIR` into both `setup` and `run`: the
per-run workdir and the run script's initial CWD. `repo` is cloned to
`$GB_BUILD_WORKDIR/<workdir>` and `src/` is mounted at `$GB_BUILD_WORKDIR/src`.

## Generating and deploying the step

`byoc` has no `Dockerfile`, so the `image`/`publish-image` targets are no-ops;
only `make space` (render the Space + bundle `src/`), `make clean`, and
`make help` do anything here. For the full target list and variables, see the
shared [Makefile target conventions](../../README.md#makefile-target-conventions).

Then point the build's Space at the generated `space/` directory and reference
the step by `space://steps/byoc` (see above).

To promote the step into the repo's committed assets tree
(`configurations/assets/environments/skypilot/steps/byoc/`) and copy its slurm
build test into `test/steps/byoc/skypilot/` so it is runnable from VSCode against
the published step, run `make publish-step`. See
[Two test modes](../../README.md#two-test-modes) for how the same test runs both
against the locally rendered `space/` (Mode 1, `make test`) and against the
published step (Mode 2, under `test/steps/`).

## Example build.yaml

```yaml
granite.build:
  name: byoc-example
  version: 0.0.1
  targets:
    run:
      environment_uri: space://environments/skypilot/aws
      outputs:
        result:
          uri: lh://prod/myspace/models/shared/byoc-out-{{ run_metadata.targetsteprun_id | short_hash }}/1
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
              command: "python main.py --out $GB_BUILD_WORKDIR/result"
```

## Notes and limitations

- **Public repos only.** `setup` runs an unauthenticated `git clone`; private
  repositories (and the credential/secret wiring the LSF `custom_code_lsf` step
  provides) are out of scope for this exemplar.
- **No dependency caching.** Unlike `custom_code_lsf`'s hash-keyed conda cache,
  dependencies are whatever the chosen public `image` provides plus anything your
  `command`/repo installs at run time.
- **Single output artifact** in the exemplar; emit additional `LLMB_ARTIFACT_ID`
  lines and declare matching `outputs` to register more.
