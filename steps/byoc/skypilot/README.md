# byoc (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `byoc` in a `build.yaml` (config contract, inputs/outputs, examples). This file
> covers how the step is *generated, tested, and published*.

Bring-Your-Own-Code step for SkyPilot clusters. Runs in a **public container image**
(or on the bare launcher node), clones a public git repo during `setup`, and runs a
user-defined command during `run` — no custom image is built or published for this step.

This is the SkyPilot counterpart of the LSF `custom_code_lsf` and Kubernetes
`custom_code` steps, and the public-image counterpart of the custom-image
[eval](../../eval/skypilot/README.md) step. It is *generated* from the sources in this
directory by the shared Makefile conventions — see the framework overview:
[steps/README.md](../../README.md).

## Generating and deploying the step

`byoc` has no `Dockerfile`, so the `image`/`publish-image` targets are no-ops;
only `make space` (render the Space + bundle `src/`), `make clean`, and
`make help` do anything here. For the full target list and variables, see the
shared [Makefile target conventions](../../README.md#makefile-target-conventions).

`make space` renders a self-contained Space into `space/` (see `SPACE_DIR` in the
framework overview). Point the build's Space at that directory to reference the step by
`space://steps/byoc`.

To promote the step into the repo's committed assets tree
(`configurations/assets/environments/skypilot/steps/byoc/`) and copy its slurm
build test into `test/steps/byoc/skypilot/` so it is runnable from VSCode against
the published step, run `make publish-step`. Publishing also copies
[USAGE.md](USAGE.md) to `README.md` beside the published `step.yaml`, so the released
step ships user-facing docs. See
[Two test modes](../../README.md#two-test-modes) for how the same test runs both
against the locally rendered `space/` (Mode 1, `make test`) and against the
published step (Mode 2, under `test/steps/`).
