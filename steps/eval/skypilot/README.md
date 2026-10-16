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

## Running the AWS test (real EC2)

`make -C steps/eval/skypilot test` runs the whole `test/` tree, including
`test/aws/test_skypilot_aws_eval.py`, which runs the eval step on a real EC2
node via SkyPilot. Like the [byoc AWS
test](../../byoc/skypilot/README.md#environment-variables-for-the-aws-test) it is
**extended-suite only** (`@extended_testing_only` + the `skypilot_integration`
marker) and **self-skips** unless AWS credentials are in the environment, so it
never provisions an instance by accident.

### Environment variables

The skip gate is the shared
`gbserver.environment.skypilot.aws_credentials_present()` predicate, which
requires **either**:

| Variable(s) | Meaning |
|---|---|
| `AWS_ACCESS_KEY_ID` **and** `AWS_SECRET_ACCESS_KEY` | An explicit access-key pair. |
| `AWS_PROFILE` | A named profile in `~/.aws/credentials` (e.g. `gb-skypilot`). |

If neither is set the test skips cleanly. Also required for the launch itself
(not part of the skip gate): `sky check aws` must report `AWS: enabled`, and a
region must be configured — via the AWS profile / `~/.aws/config`, or
`AWS_DEFAULT_REGION`. No `HF_TOKEN` is needed: the exemplar fixture uses `env://`
I/O and downloads nothing.

#### Where the build's AWS creds actually come from (`GB_AWS_*`)

The env vars above are only the **skip gate** — they decide whether the test runs
on this host. They are **not** how the build authenticates to AWS. The AWS
`environment.yaml`
(`configurations/assets/environments/skypilot/aws/environment.yaml`) resolves the
secret *names* `GB_AWS_ACCESS_KEY_ID` / `GB_AWS_SECRET_ACCESS_KEY` through the
space **secret manager** and materializes them into a non-default `gb-skypilot`
profile that SkyPilot then selects:

```yaml
aws_credentials:
  - profile: gb-skypilot
    aws_access_key_id: GB_AWS_ACCESS_KEY_ID
    aws_secret_access_key: GB_AWS_SECRET_ACCESS_KEY
cloud_config: { workspaces: { default: { aws: { profile: gb-skypilot } } } }
```

Selecting an explicit profile **disables the ambient `AWS_*` env-var provider**, so
provisioning uses the materialized profile, not your shell. Standalone: seed those
secret names (base64) into `~/.granite.build/space_secrets/`; shared: the
server-managed secret store supplies them. Full runbook:
[docs/environments/skypilot-aws.md](../../../docs/environments/skypilot-aws.md).

### The image must be published first (unlike byoc)

eval bakes its evaluator into a **custom image**, and the EC2 node *pulls* that
image by the `image_id` (`docker:${IMAGE_REF}`) frozen into the step at `make
space` time. The aws test only passes once the image is pushed to a
**public/pullable** registry — the committed default `quay.io/your-org` is a
placeholder that fails auth. Build + publish at a real registry, then run the
tests with that same `REGISTRY` so the rendered `image_id` points at the image you
pushed:

```sh
make -C steps/eval/skypilot image publish-image REGISTRY=quay.io/<you>   # build + push (after `podman login`)
make -C steps/eval/skypilot test REGISTRY=quay.io/<you>                  # renders image_id -> your ref, runs the aws test
```
