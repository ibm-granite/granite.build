# SkyPilot AWS build testing — plan & status

Resumption doc for the "standalone build tests for the SkyPilot AWS environment" work.
Branch: `feat/aws-config-and-build-testing`.

## Goal

Test the SkyPilot **AWS** (`subtype: aws`) compute environment. Split into two phases:

1. **Now — component test of the AWS wiring** (free, deterministic, no cloud). ✅ DONE.
2. **Later — real end-to-end integration test** against a real AWS EC2 instance,
   once AWS credentials are set up. ⏳ DEFERRED.

## Key decision log (why it's split this way)

- SkyPilot's AWS path provisions **real EC2 instances and SSHes into them** to run
  the workload. The EC2 provisioning happens *inside* the SkyPilot SDK; gbserver's
  own "AWS" code is thin (it sets `infra=aws` and hands a `sky.Task`/`sky.Resources`
  to the SDK).
- **No free tool can run a real workload through the AWS path.** LocalStack
  Community / moto mock `RunInstances` (no bootable SSH-able VM); LocalStack Pro
  isn't free and needs SkyPilot patching; the EC2 provisioning endpoint isn't even
  configurable through gbserver.
- An earlier attempt replaced the **whole `sky` SDK** with a functional local-executor
  fake (`FakeSky`) to force a green end-to-end build against local MinIO. **Rejected**:
  once you fake the entire SDK, the test exercises neither SkyPilot nor anything
  AWS-specific — it's a BuildRunner + command-step + S3-push test wearing an "aws"
  label. That scaffolding has been removed.
- **Chosen (Option A):** a focused **component test** that mocks `sky` at its call
  boundary and asserts gbserver translates an AWS config into the correct SkyPilot
  calls. This tests exactly the code gbserver *owns* for AWS, with no pretense of
  running a workload. High-fidelity AWS end-to-end is Phase 2 (real EC2).

## Phase 1 status: DONE ✅

**File:** `test/unit/environment/test_skypilot_aws.py` (plain unit test; runs in the
default/quick suite — no marker, no infra, mocks `sky` entirely).

Mirrors the established pattern in `test/unit/environment/test_skypilot.py`
(mock `sky`, assert on `sky.Resources.call_args.kwargs`). 6 tests, all passing:

- `test_default_cloud_aws_routes_infra_to_aws` — `default_cloud: aws` ⇒ `sky.Resources(infra="aws")`.
- `test_aws_instance_type_and_spot_flow_to_resources` — `instance_type`/`use_spot` reach `sky.Resources`.
- `test_aws_zone_without_cluster_folds_into_infra` — bare `zone` folds to `infra="aws/<zone>"`, `zone=None`.
- `test_explicit_resources_cloud_overrides_default` — step `resources.cloud` overrides env default.
- `test_inline_aws_credentials_are_parsed_and_forwarded` — `config.aws_credentials` ⇒ `skypilot_config.materialize(..., aws=[AwsCredentialProfile], ...)`.
- `test_no_inline_config_does_not_materialize` — no inline config ⇒ `materialize` not called.

Run: `.venv/bin/pytest test/unit/environment/test_skypilot_aws.py -v`  →  6 passed.

### Relevant source (for reference)

- `src/gbserver/environment/skypilot.py`
  - `_launch_skypilot_inner` (~603): builds `infra` (`override_res.get("infra") or cloud`),
    `sky.Resources(infra=, instance_type=, cpus=, memory=, use_spot=, zone=, image_id=, ...)` (~712).
  - `_get_cloud()` (~383): returns `config.default_cloud`.
  - `_ensure_inline_configs_materialized()` (~347): parses `aws_credentials`/`cloud_config`/`cluster_ssh_configs`
    and calls `skypilot_config.materialize(name, ssh, cloud_config, aws, secrets)`.
- `src/gbserver/types/environmentconfig.py` — `AwsCredentialProfile` (`profile`,
  `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`).

## Removed (the rejected whole-SDK approach)

- `test/libgbtest/skypilot_fake.py` (FakeSky local-executor double)
- `test/unit/libgbtest/` (its unit test)
- `test/integration/standalone/buildrunner/skypilot_aws/` (end-to-end test)
- `test-data/integration/standalone/buildrunner/skypilot_aws/` (build/space/MinIO fixtures)

## Loose ends when resuming Phase 1

- I was mid-verify that the new test collects under the quick/standalone marker
  expression (it's a plain unit test, so it should — just finish that check):
  `pytest -m "not secret_manager and not nats_server and not docker_required and not ibm and not nats and not extended" test/unit/environment/test_skypilot_aws.py`
- **MinIO may still be running** (from the abandoned end-to-end attempt). Stop it:
  `make minio-teardown`.
- **Stale docs to reconcile** — these describe the *rejected* end-to-end approach and
  should be updated or removed to avoid confusion:
  - `docs/superpowers/specs/2026-07-23-skypilot-aws-standalone-build-test-design.md`
  - `docs/superpowers/plans/2026-07-23-skypilot-aws-standalone-build-test.md`
- Nothing has been committed (per the manual-git workflow). New/changed file to
  stage: `test/unit/environment/test_skypilot_aws.py`.

## Phase 2 (later): real-EC2 end-to-end integration test

When AWS credentials are available:

- Model on the SkyPilot **Slurm** buildrunner tests
  (`test/integration/standalone/buildrunner/skypilot_slurm/` + the
  `AbstractYamlBuildRunnerTest` base) — a YAML-driven `build.yaml` + `buildtest.yaml`
  fixture against the `space://environments/skypilot/aws` environment.
- Provide AWS creds via the env's `config.aws_credentials` (secret-name-or-literal,
  materialized to `~/.aws/credentials`) and region via `config.cloud_config.aws`
  (or `AWS_DEFAULT_REGION`).
- Add a reachability/credentials **skip gate** (analogous to `_slurm_cluster_reachable()`
  in `test/integration/environment/test_skypilot_slurm_e2e.py`) — e.g. `sky check aws`
  or presence of creds — so it skips cleanly without credentials.
- Mark it `@extended_testing_only` + `pytest.mark.skypilot_integration`.
- Real S3 outputs (`s3://...`) exercise the S3 assetstore push for real; no MinIO
  needed once using real AWS.
- Cost/teardown: use the smallest instance, `idle_minutes_to_autostop` low, and
  ensure `sky down` in teardown (bounded, like the slurm e2e test).
