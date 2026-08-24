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

## Running the tests

`make test` runs the whole `test/` tree — both the `slurm/` and `aws/` build
tests — against the locally rendered `space/`. Both are **extended-suite only**
(`@extended_testing_only`) and carry the `skypilot_integration` marker, and each
self-skips unless its backend is reachable, so neither runs by accident:

```
make -C steps/byoc/skypilot test   # uses the repo-root .venv; runs `make space` first
```

- **slurm test** — needs the local Docker SLURM cluster + MinIO endpoint. Bring
  them up once with `make test-setup` (delegates to the repo-root `slurm-setup` /
  `minio-setup`).
- **aws test** — needs AWS credentials in the environment (below).

### Environment variables for the aws test

`test/aws/test_skypilot_aws_byoc.py` provisions a **real EC2 instance** via
SkyPilot, so it is gated on AWS credentials being present in the environment. The
gate is the shared `gbserver.environment.skypilot.aws_credentials_present()`
predicate (the same one gbserver uses to decide boto3 has something to try),
which requires **either**:

| Variable(s) | Meaning |
|---|---|
| `AWS_ACCESS_KEY_ID` **and** `AWS_SECRET_ACCESS_KEY` | An explicit access-key pair. |
| `AWS_PROFILE` | A named profile in `~/.aws/credentials` (e.g. `gb-skypilot`). |

If neither is set the test **skips cleanly** — no EC2 instance is ever
provisioned without credentials explicitly exported. Note this gate reads the
**environment only**: a bare `~/.aws/credentials` `[default]` is picked up only
because boto3/SkyPilot read it at launch, but it does *not* satisfy the skip gate
unless `AWS_PROFILE` (or the key pair) is exported.

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

Also required for the launch itself to succeed (not part of the skip gate):

- **`sky check aws` must report `AWS: enabled`** — SkyPilot uses boto3 to
  provision the instance.
- **A region** must be configured — via the AWS profile / `~/.aws/config`, or
  `AWS_DEFAULT_REGION`.

No `HF_TOKEN` is needed: the fixture is HF-free (`env://` I/O) and runs on the
bare node (`image: ""`).
