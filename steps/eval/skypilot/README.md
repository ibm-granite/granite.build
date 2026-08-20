# eval (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `eval` in a `build.yaml` (config contract, inputs/outputs, examples). This file
> covers how the step is *built, tested, and published*.

Evaluation step for SkyPilot clusters. Its evaluation code is baked into a
**custom image** built from [`Dockerfile`](Dockerfile), published to a registry,
and referenced from the generated `step.yaml` via `image_id`. The `run` block
invokes the baked entrypoint ([`src/eval.sh`](src/eval.sh)) with parameters from
`config.eval_config`, then registers the single results file as the step's output.

This is a custom-image counterpart to the public-image
[byoc](../../byoc/skypilot/README.md) step. It is *generated* from the sources in
this directory by the shared Makefile conventions — see the framework overview:
[steps/README.md](../../README.md).

> **This is an exemplar, not a working evaluator.** The shipped
> [`src/eval.sh`](src/eval.sh) is a **placeholder shell script** — it writes a
> `results.json` recording its parameters but performs no real evaluation, so the
> image needs no Python or dependencies (just a minimal Fedora base). When you
> implement eval for real, replace the script body with a real harness and give
> the image a suitable runtime + dependencies; the flag contract and the fixed
> `results.json` output path are what the step depends on.

## Building, publishing, and deploying the step

Because a `Dockerfile` is present, this is an image step: `make all` runs
`image` → `publish-image` → `space`. For the full target list, variables, and
[registry credentials](../../README.md#registry-credentials), see the shared
[Makefile target conventions](../../README.md#makefile-target-conventions).

To promote the step into the repo's committed assets tree
(`configurations/assets/environments/skypilot/steps/eval/`) and copy its build
test into `test/steps/eval/skypilot/` so it is runnable from VSCode against
the published step, run `make publish-step`. Publishing also copies
[USAGE.md](USAGE.md) to `README.md` beside the published `step.yaml`, so the released
step ships user-facing docs. See
[Two test modes](../../README.md#two-test-modes) for how the same test runs both
against the locally rendered `space/` (Mode 1, `make test`) and against the
published step (Mode 2, under `test/steps/`).

With the local `Docker` launcher removed, there is **no longer a way to exercise
the built image locally**: `make test` runs the cluster-agnostic `eval.sh` unit
tests ([test/test_eval.py](test/test_eval.py)) plus the real-EC2 integration test
([test/aws/](test/aws/)), and the latter **skips unless AWS credentials are
present**. Running the image end to end now requires a reachable remote cluster
(the Skypilot launcher).

Eval-specific notes:

- `REGISTRY` ships as a **placeholder** (`quay.io/your-org`) so the offline
  targets work out of the box; replace it in the `Makefile`, or override per
  release, e.g. `make all REGISTRY=quay.io/myorg IMAGE_TAG=0.1.0`.
  `make publish-image` against the placeholder will fail auth — set a real
  registry first. `IMAGE_TAG` defaults to the git short SHA.
- At `make space` time the image reference
  `$(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)` is substituted into the Skypilot
  launcher's `image_id: "docker:${IMAGE_REF}"`.
- **Image is required at run time.** On a real remote cluster the image must be
  **published and reachable** — run `make publish-image` (after `podman login`)
  before submitting a build.
