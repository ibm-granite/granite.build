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
(`configurations/assets/environments/skypilot/steps/eval/`) and copy its Docker
build test into `test/steps/eval/skypilot/` so it is runnable from VSCode against
the published step, run `make publish-step`. Publishing also copies
[USAGE.md](USAGE.md) to `README.md` beside the published `step.yaml`, so the released
step ships user-facing docs. See
[Two test modes](../../README.md#two-test-modes) for how the same test runs both
against the locally rendered `space/` (Mode 1, `make test`) and against the
published step (Mode 2, under `test/steps/`).

Eval-specific notes:

- `REGISTRY` ships as a **placeholder** (`quay.io/your-org`) so the offline
  targets work out of the box; replace it in the `Makefile`, or override per
  release, e.g. `make all REGISTRY=quay.io/myorg IMAGE_TAG=0.1.0`.
  `make publish-image` against the placeholder will fail auth — set a real
  registry first. `IMAGE_TAG` defaults to the git short SHA.
- At `make space` time the image reference
  `$(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)` is substituted into **both** launcher
  blocks — the Skypilot `image_id: "docker:${IMAGE_REF}"` and the Docker
  launcher's `image` (see below).

### Running locally with no publish (the `Docker` launcher)

The step's `step-template.yaml` also carries a **`Docker`** environment config
(a `docker` launcher running the same image and `eval.sh`). This lets the image
be **built and exercised locally with no registry publish**: `make test` renders
the Space and builds the image locally (`make image`), then a docker build test
under `test/docker/` (the per-cluster test dir the `Docker` launcher expects) runs
it against the local Docker daemon. The
`docker` environment's `pull_policy` is `if-not-present`, so the just-built local
image is used as-is — no push, no pull, no container-capable cluster needed. Run
it (with the repo-root `.venv` active) via:

```sh
make -C steps/eval/skypilot test
```

> **Running the *committed* Mode-2 docker test — tag coupling.** The Mode-1 flow
> above always works because `make test` builds the image and renders the Space at
> the same commit, so their tags agree. The **committed** Mode-2 test
> (`test/steps/eval/skypilot/docker/`, created by `make publish-step`) instead runs
> against the **published** `step.yaml` under
> `configurations/assets/…/steps/eval/`, whose `image`/`image_id` bake in
> `IMAGE_TAG` (the git short SHA by default) **as of the commit `make publish-step` was
> last run at**. Because `pull_policy` is `if-not-present`, that test only resolves
> if a local image at that exact tag exists — otherwise it tries to pull the
> placeholder `quay.io/your-org` registry and fails. So before running it, rebuild
> **and** re-publish at your current commit so both sides agree:
>
> ```sh
> make -C steps/eval/skypilot image publish-step   # builds gb-step-eval:<HEAD-sha> and re-renders the assets to match
> ```
>
> (Then commit the regenerated `step.yaml`.) Pin a stable `IMAGE_TAG` — e.g. `make
> … image publish-step IMAGE_TAG=local` on both sides — if you would rather the
> committed assets not drift per commit.

- **Image is required at run time.** On a real remote cluster (the Skypilot
  launcher) the image must be **published and reachable** — run `make
  publish-image` (after `podman login`) before submitting such a build. The
  `Docker` launcher is the exception: it uses the **local** image, so `make
  image` (done for you by `make test`) is enough — no publish.
