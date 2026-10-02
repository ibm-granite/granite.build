# bfcl-eval (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `bfcl-eval` in a `build.yaml` (config contract, inputs/outputs, examples). This file
> covers how the step is *built, tested, and published*.

Scores a model's tool calling with the Berkeley Function Calling Leaderboard harness. The
harness, vLLM, and this step's helper scripts are baked into a **custom image** built from
[`Dockerfile`](Dockerfile), published to a registry, and referenced from the generated
`step.yaml` via `image_id: "docker:${IMAGE_REF}"`. Structurally this is the same shape as
[eval](../../eval/skypilot/README.md); see [steps/README.md](../../README.md) for the
shared Makefile conventions.

## Why this step exists

`space://steps/bfcl-eval` already resolves on this branch — to a committed asset at
`configurations/assets/environments/skypilot/lsf/ibm-bluevela/steps/bfcl-eval/step.yaml`,
with **no authoring directory**, running
`docker:us.icr.io/cil15-shared-registry/bfcl-py311:0.02`. That image is IBM-internal and
cannot be pulled from outside, so the final target of
`recipes/granite4-gold-distillation/lsf/distill-pipeline-smoke` is unrunnable for anyone without ICR
access. This step is the public replacement: same config surface, image built from public
bases, harness version pinned in [`uv.lock`](uv.lock) rather than baked into an opaque tag.

Three differences from the legacy asset are deliberate, not oversights:

| Legacy asset | Here | Why |
|---|---|---|
| `type: EVAL` | `type: custom` | There is no `EVAL` member in `gbcommon.types.stepconfig.StepType`; the values are `data_processing`, `data_generation`, `training`, `tuning`, `custom`. `type` is a free string, so `EVAL` neither errors nor validates — it is simply not a type. Follows [distill-eval](../../distill-eval/skypilot/step-template.yaml), which reasoned this through first. `steps/eval` still says `EVAL`; that is left alone here rather than changed under cover of an unrelated step. |
| `outputs.optional.bfcl_results` | `outputs.required.bfcl_results` | The step exits 1 when the harness leaves no score file, so a successful run always has one. `optional` on an always-present output only lets a real failure complete green. |
| `config.workload.cwd` / `hf_home` | `config.bfcl_config.hf_home` | `config.workload` is parsed as `StepConfigWorkloadSection` (`path`/`args`/`workspace_dir`/`output_dir`/`python_env`). Pydantic **ignores** unknown keys there, so both of those settings have never had any effect. |

The monitor override is a fourth, smaller fix: the recipe passes `poll_interval_seconds`
and `log_retrieval_interval_seconds` to this step, and the legacy asset threaded only the
first into its monitor config — the other was silently ignored. Its progress regex
(`Starting BFCL|Running.*stage|Evaluation complete`) also matched **no line
`run-bfcl.sh` prints**, so periodic log retrieval produced no status events at all. Both are
now asserted against the real script by `test/test_step_template.py`.

## ⚠️ The image has never been built or pushed

Authored on a host with **no `/etc/subuid` entry and no registry credentials**, so:

- `make image` has never run to completion here,
- `make publish-image` has never run (and `make publish-step` refuses for an image step
  until the image exists on the registry — `PUBLISH_REQUIRE_IMAGE=true`),
- `test/lsf/` has never executed, and
- no build has ever run this step end to end.

Everything below the image line **is** verified: the unit tests pass, the template renders,
`bash -n` accepts the `run` block, and the artifact contract is asserted. Treat the first
`make image` as unproven work, not a formality.

`REGISTRY` ships as the placeholder `quay.io/your-org`, matching `steps/eval` — `common.mk`
makes it mandatory for image steps with no default, and `make publish-image` against the
placeholder will fail auth. Build and push at a real registry, then render with the same
value so `image_id` points at what you pushed:

```sh
make image publish-image REGISTRY=quay.io/<you>   # after `podman login`
make publish-step        REGISTRY=quay.io/<you>
```

### Building rootless without a `/etc/subuid` entry

If your build host has no subuid range for your account (the condition above), `podman
build` fails in apt's privilege-drop before it installs anything. Two things get past it,
both already applied or documented:

```sh
podman build --isolation=chroot --cgroup-manager=cgroupfs -t bfcl-eval .
```

and the Dockerfile's `-o APT::Sandbox::User=root` on both `apt-get` invocations, which is
commented there with the exact `setgroups()` failure. The flags are **not** baked into the
Makefile: they are a property of a broken build host, not of this step, and `common.mk`'s
`PODMAN_BUILD_ARGS` is the right place to pass them when needed.

### Base image constraints, if you change them

- **The base must be Debian/Ubuntu.** SkyPilot's docker runtime runs `apt-get` *inside* the
  container during setup; on a non-apt base the task dies with rc 127 before the step's
  `run` block is reached (documented in [eval's Dockerfile](../../eval/skypilot/Dockerfile)).
  `python:3.14-slim` is Debian, so this holds today — a switch to e.g. an Alpine or UBI base
  would break it.
- **Not NGC.** The comment at the top of the Dockerfile records a real build in which an
  `nvcr.io/nvidia/vllm` base had its own torch/triton/vllm uninstalled and replaced by the
  pip wheels this lockfile pulls anyway. The CUDA runtime comes from `nvidia-*-cu12` wheels;
  only a host driver is needed.
- **Two floating references remain**, both known and both a judgement call for whoever
  builds first: `FROM python:3.14-slim` is Docker Hub, which CI rate-limits (sibling steps
  mirror through `public.ecr.aws/docker/library` for exactly this reason), and
  `COPY --from=ghcr.io/astral-sh/uv:latest` pins uv to `latest`. Neither was changed here,
  because no build could be run to confirm the substitution — `ARG PYTHON_BASE_TAG` exists
  so the base can be overridden without editing the file.

## Running the tests

```sh
make unit-tests     # fast loop: no image build, nothing to install
make test           # the framework's own target -- see the caveat below
```

`make unit-tests` is **additive**, not an override. `common.mk` defines `test: space image`,
so the framework's `make test` on an image step builds the whole torch+vLLM image (and wants
a repo-root `.venv`) before running a single pure-Python test. That is correct for a release
gate and wrong for editing `src/`, so both exist.

### What is skipped without the harness, and why

`bfcl_eval` — the leaderboard harness — is a dependency of this step's **image**, not of
this repository. In a plain granite.build checkout it is absent, and three groups of tests
handle that explicitly rather than erroring at import:

| What | How it skips | Covers |
|---|---|---|
| `test_resolve_test_categories.py`, `test_sample_test_ids.py`, `test_shard_test_ids.py` | `collect_ignore` in [`test/conftest.py`](test/conftest.py) | Their modules under test import the harness at module scope. |
| Two tests in `test_tool_call_tag_repair.py` | `pytest.importorskip` | They patch the real `Granite4FCHandler`. The other eight tests exercise the repair logic directly and always run. |
| Three tests in `test_run_bfcl.py` | the `needs_harness` marker | `run-bfcl.sh` delegates id selection to helpers that import the harness, so those shell paths exit 1 without it. The other seven run against stub `bfcl`/`vllm` binaries. |

To run the full set, do it where the harness is present:

```sh
uv sync --locked && uv run pytest test
```

`pydantic` appears in `make unit-tests` because `test_step_template.py` asserts `type`
against the real `StepType` enum, as every ported step's suite does, and importing
`gbcommon` pulls it. The repo-root `pyproject.toml` puts `src` on pytest's `pythonpath`, so
`gbcommon` resolves from this subdirectory with nothing installed.

### The LSF build test

[`test/lsf/test_skypilot_lsf_bfcl_eval.py`](test/lsf/test_skypilot_lsf_bfcl_eval.py) runs a
real build on BlueVela, and is gated four ways, matching the sibling distill steps:
`pytest.mark.ibm`, `@extended_testing_only`, an explicit `GB_STEP_BLUEVELA_BUILD=1` opt-in,
and an SSH reachability probe. It will not fire in CI or from a routine `make test`. **It has
never been run** — it needs both a published image and a GPU reservation.

## Publishing

`make publish-step` promotes the step into the committed assets tree and copies the
per-cluster test and fixtures under `test/steps/` and `test-data/steps/`; it also copies
[USAGE.md](USAGE.md) to `README.md` beside the published `step.yaml`, so the released step
ships user-facing docs and this development file stays here. `make check-published` re-renders
and fails on drift — it substitutes the *already committed* image ref, so it is immune to
`IMAGE_TAG` churn. **Author under `steps/`; never hand-edit the published asset.**

Note that publishing does not remove the legacy asset at
`configurations/assets/environments/skypilot/lsf/ibm-bluevela/steps/bfcl-eval/`. Both would
then provide `space://steps/bfcl-eval` — resolution order decides which a recipe gets, so
retiring the legacy one is a follow-up that belongs with whoever owns that assets tree.

## Dependency notes

[`pyproject.toml`](pyproject.toml) carries the reasoning inline; two points worth finding
before you touch it:

- **`numpy>=2` and `faiss-cpu>=1.15.0` are overrides**, not preferences. `bfcl-eval` pins
  `numpy==1.26.4` in every release from 2025.6.8 to 2026.3.23 — traced to a blanket
  "pin what's installed" pass upstream, not a real numpy-2 incompatibility (its own usage is
  `np.percentile`/`np.asarray`/`np.array(dtype=...)`, none of which 2.0 touched). The
  `faiss-cpu` override is what makes Python 3.14 possible at all: `faiss-cpu==1.11.0` has no
  cp314 wheel.
- **`tree-sitter` is deliberately NOT overridden.** `bfcl_eval`'s js/java parsers call the
  pre-0.22 API, and overriding it raised `TypeError: __init__() takes exactly 1 argument
  (2 given)` in a real run. It stays at bfcl-eval's own `==0.21.3`, which has no cp314 wheel
  and so builds from source — that is what `build-essential` is for in the Dockerfile.
- **`clearml` and `wandb` are in the lockfile but unused by `src/`.** They are inherited from
  the tracking conventions of the repository this step came from. Left in place so
  `uv.lock` stays exactly the resolution that was tested; dropping them is a safe, separate
  change for whoever next rebuilds the image.
