# byoc (SkyPilot)

Bring-Your-Own-Code step for SkyPilot clusters. Runs in a **public container
image** (or on the bare launcher node), clones a public git repo during `setup`,
and runs a user-defined command during `run` — no custom image is built or
published for this step.

This is the SkyPilot counterpart of the LSF `custom_code_lsf` and Kubernetes
`custom_code` steps. It is *generated* from the sources in this directory by the
shared Makefile conventions — see the framework overview:
[steps/README.md](../../README.md).

## Referencing the step

`byoc` is a generated bundle referenced by an **absolute `file://` URI** to the
`step/` directory produced by `make step`:

```yaml
steps:
  - step_uri: file:///abs/path/to/steps/byoc/skypilot/step
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
| `image` | string | Public container image the step runs in, e.g. `python:3.12-slim`. Rendered at runtime as SkyPilot `docker:<image>`. Empty (default) runs on the bare launcher node. |
| `ref` | string | Branch, tag, or commit checked out after cloning. Default: the repo's default branch. |
| `workdir` | string | Subdirectory (under `$GB_BUILD_WORKDIR`) the repo is cloned into. Default: `code`. |

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

Exported by the SkyPilot launcher into both `setup` and `run`:

| Variable | Set when | Meaning |
|---|---|---|
| `$GB_BUILD_WORKDIR` | always | Per-run workdir; the run script's initial CWD. `repo` is cloned to `$GB_BUILD_WORKDIR/<workdir>`; `src/` is mounted at `$GB_BUILD_WORKDIR/src`. |
| `$GB_SHARED_WORKDIR` | env sets `shared_workdir` | Env-level shared dir mounted on every worker, for cross-step state. |
| `$GB_BUILD_ID`, `$GB_TARGETRUN_ID` | run metadata present | Build / target-run identifiers, for correlating logs and shared state. |
| `$GB_SKYPILOT_LAUNCH_ID`, `$GB_SKYPILOT_CLUSTER_NAME` | always | This launch's id and the SkyPilot cluster name. |

## Generating and deploying the step

`byoc` has no `Dockerfile`, so the `image`/`publish-image` targets are no-ops;
only `make step` (render + bundle `src/`), `make clean`, and `make help` do
anything here. For the full target list and variables, see the shared
[Makefile target conventions](../../README.md#makefile-target-conventions).

Then reference `step/` by absolute `file://` URI (see above).

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
        - step_uri: file:///abs/path/to/steps/byoc/skypilot/step
          config:
            compute_config: { num_nodes: 1, num_gpus_per_node: 1 }
            byoc_config:
              image: "python:3.12-slim"
              repo: "https://github.com/org/repo"
              ref: "main"
              workdir: "code"
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
