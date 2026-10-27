# openinstruct-sft (SkyPilot): development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `openinstruct-sft` in a `build.yaml` (config contract, inputs and output,
> examples). This file covers how the step is built, tested and published, and why
> it is shaped the way it is.

A reusable SFT step: upstream Open-Instruct `finetune.py` (v0.3.0), single node,
multi-GPU, tracked with ClearML. It is a custom-image step: the `Dockerfile` here
bakes Open-Instruct and `src/` into the image. It is referenced as
`space://steps/train/openinstruct-sft` (the URI follows the nested
`steps/train/openinstruct-sft` authoring path). It has build tests on two clusters,
AWS and IBM BlueVela SLURM. The framework it follows is described in
[steps/README.md](../../../README.md).

## Layout

```
step-template.yaml          the step: config defaults, render-time validation, launcher blocks
Makefile                    STEP_NAME, REGISTRY, IMAGE_VERSION; includes steps/common.mk
Dockerfile                  the image: CUDA base, pinned Open-Instruct, clearml, src/
.gbignore                   files gbserver ships without Jinja-rendering (docs, src/)
src/                        baked into the image at /opt/openinstruct-sft/
  openinstruct_sft_guard.sh   on-node checks of the staged base_model/dataset paths
  openinstruct_sft_setup.sh   image sanity check: finetune.py present, training stack imports
  openinstruct_sft_run.sh     GPU count, ClearML on/off, accelerate launch, swap, markers
patches/finetune_local_model.py   build-time patch applied to upstream finetune.py
test/                       unit tests at the root; one dir per cluster build test
  test_dockerfile.py
  test_exemplar_build.py
  test_finetune_local_model_patch.py
  test_openinstruct_sft_guard_sh.py
  test_openinstruct_sft_run_sh.py
  test_openinstruct_sft_setup_sh.py
  test_skypilot_train_openinstruct_sft_render.py
  aws/test_skypilot_aws_train_openinstruct_sft.py                   real EC2 (A10G:1)
  slurm-bluevela/test_skypilot_slurm_bluevela_train_openinstruct_sft.py  real BlueVela (H200:1)
test-data/
  build.yaml                exemplar build a user copies (pasted into USAGE.md)
  parameters.yaml           parameters for the exemplar
  aws/build.yaml            aws build test fixture
  aws/buildtest.yaml        aws build test expectations
  slurm-bluevela/build.yaml       BlueVela build test fixture
  slurm-bluevela/buildtest.yaml   BlueVela build test expectations
USAGE.md                    user doc; published as the released step's README.md
README.md                   this file; not published
```

## Why the shell lives in src/

The template only computes values with Jinja (paths, the mixer list, the `args`
map) and passes them as argv. The scripts do the shell. So every line of shell is
`bash -n`-able and unit-testable, not only renderable.

Config-only checks are `raise_error()`s in the template's `validate` macro, not
script checks. The launcher prepends the inline `hf download` (aws) to `setup:`, so
a script there runs only after the node is up and the model is downloaded. A
`raise_error` fails the strict launcher fill at step creation instead, before
provisioning. The guard script keeps only what is knowable on the node: the staged
paths. Both `setup:` and `run:` call the guard and repeat the `validate` macro,
because a warm-cluster re-run renders `run:` without `setup:`. A render test pins
the two macro copies identical.

`.gbignore` lists `src/**` because gbserver runs a strict Jinja pass over the merged
step directory, and `openinstruct_sft_guard.sh` contains literal `{{` and `{%` (its
unrendered-binding detector). `**/*.md` is listed so the docs can show `{{ ... }}`.

## Image

`quay.io/granitebuild/gb-steps:train-openinstruct-sft-skypilot-oi-v0.3.0-4`. It holds:

| Part | Pin |
|---|---|
| Base | `nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04`, by digest. `devel` because DeepSpeed JIT-compiles its ops. |
| uv | `ghcr.io/astral-sh/uv:0.13.5`, by digest (copied in as a stage). |
| Open-Instruct | allenai/open-instruct tag v0.3.0, commit `26de99604002d6a7b25a2c5f2ee880e388a21dca` (Apache-2.0), at `/opt/open-instruct`. The build fetches exactly that commit and fails if `HEAD` differs. |
| Training stack | `uv sync --frozen` from Open-Instruct's own `uv.lock` into `/opt/open-instruct/.venv`. |
| clearml | `clearml==2.1.12`, installed into the same venv. |
| Security patches | On top of the frozen lock, in the same install RUN: `vllm` uninstalled (the fixed release needs torch 2.11; SFT does not use it); `pillow`, `nltk`, `litellm`, `PyJWT`, `Authlib`, `fastmcp`, `anyio`, `GitPython`, `certifi`, `wandb` upgraded to their first fixed releases (versions pinned in the Dockerfile and `test/test_dockerfile.py`). `litellm` and `wandb` therefore leave upstream's pins. The NVIDIA Nsight Compute profiler is purged from the CUDA base. |
| Step scripts | `src/` at `/opt/openinstruct-sft/`. |

Why `finetune.py`: it is deprecated upstream in favour of `olmo_core_finetune.py`,
which supports only OLMo and Qwen3 configs (no Granite). The commit pin makes the
deprecation an explicit decision.

Choices in the Dockerfile that look odd, and why:

- **`uv sync --frozen` keeps the default groups.** `open_instruct/utils.py` imports
  `beaker` unconditionally, and `beaker-py` is in the `dev` group at v0.3.0. Dropping
  the groups breaks `import open_instruct`.
- **`OPEN_INSTRUCT_COMMIT`, not `GIT_COMMIT`.** `finetune.py`'s `get_wandb_tags()`
  calls the GitHub search API when `GIT_COMMIT` is set. The step reads
  `OPEN_INSTRUCT_COMMIT` to print it at setup and to emit the
  `open_instruct_commit` metadata.
- **`chmod -R a+rX`.** Pyxis on SLURM runs the container as the job user, not
  root. The venv, uv's managed Python (`/opt/uv-python`, not root's home, since the
  venv's interpreter is a symlink into it) and the scripts are readable by any uid.
- **The venv is first on `PATH`.** That also provides the `hf` CLI the aws
  environment's inline `hf download` uses.
- **finetune.py is patched at build time** (`patches/finetune_local_model.py`): upstream pre-fetches `--model_name_or_path` with `snapshot_download()`, which rejects a local directory, so the pre-fetch is skipped when the path is a local dir (the step always passes the staged `base_model`). The build fails if the upstream line changes.

**Security trade-off.** The base is Ubuntu, not UBI or a minimal image. SkyPilot
bootstraps itself by running `apt-get` inside the container, so a UBI/RPM base fails
provisioning. The mitigations:

- every base is pinned by digest (`test_dockerfile.py` checks this);
- `apt-get upgrade` runs at build time;
- the container runs as **root**, like the repo's other step images. A non-root
  default `USER` does not work under SkyPilot's docker runtime: SkyPilot sets the
  container up as root and saves root's environment (`HOME=/root`) to
  `/etc/profile.d`, so the non-root user logs in with `HOME=/root` and provisioning
  fails. The VM exists for one job and is torn down after; Pyxis on SLURM still runs
  the container as the job user. Restoring a non-root user needs a SkyPilot fix
  (follow-up);
- files under `/opt` are readable by any uid but writable only by root;
- the image is scanned for vulnerabilities before `make publish-image` (below).

### Scanning

Scan every image before `make publish-image`:

1. Run the team's container scanner on the built image.
2. As a stop-gap, run `trivy image --severity CRITICAL,HIGH <ref>`.
3. Do not publish with a fixable CRITICAL in a package the Dockerfile controls (the
   apt layer, clearml, the security-patched venv packages listed under Image, the
   step scripts).
4. Record the CRITICAL and HIGH counts in the PR.

A fixable CRITICAL in a package that comes from upstream's lock is fixed by
upgrading that package on top of `uv sync --frozen` in the install RUN (or removing
it when SFT does not use it and the fix would move the training stack), then
rebuilding and rescanning. Result for `oi-v0.3.0-2` (trivy, CRITICAL+HIGH; `oi-v0.3.0-3` differs only by dropping the non-root user): 0
fixable CRITICAL; 18 CRITICAL without a fix, all in Ubuntu's `linux-libc-dev`
(kernel UAPI headers, needed by DeepSpeed's JIT build); 585 HIGH, 66 of them
fixable (57 Python packages, 8 jar, 1 Go binary), tracked as a follow-up.

### Bumping IMAGE_VERSION

The Makefile's `IMAGE_VERSION` is a stable version (`oi-v0.3.0-4`), not the
per-commit git SHA. The image is large, and BlueVela runs it from a `.sqsh` built
from this exact tag, so it must not churn on unrelated commits.

- Bump the suffix (`-1` to `-2`) when the Dockerfile or `src/` changes.
- Bump the prefix with the Open-Instruct pin (for example `oi-v0.4.0-1`).

After a bump, publish the image, rebuild the BlueVela `.sqsh`, update the `.sqsh`
path in `test-data/slurm-bluevela/build.yaml` and in `USAGE.md`, then re-run
`make publish-step`.

## Publishing the image

The image is `$(REGISTRY)/gb-steps:train-openinstruct-sft-skypilot-<IMAGE_VERSION>`:
every custom step pushes to one shared repository, `gb-steps`, and `steps/common.mk`
derives the tag from the step directory (`/` becomes `-`).

The granitebuild quay account is a user namespace, whose push robot cannot create
repositories. So `gb-steps` was pre-created once under `quay.io/granitebuild`, set
Public, and grants the robot `granitebuild+gbpush` Write. Every step pushes a new
tag into it, so no per-step quay setup is needed:

```bash
podman login -u 'granitebuild+gbpush' --password-stdin quay.io/granitebuild   # robot token on stdin
make -C steps/train/openinstruct-sft/skypilot image publish-image
skopeo inspect --raw docker://quay.io/granitebuild/gb-steps:train-openinstruct-sft-skypilot-oi-v0.3.0-4
```

Run the `skopeo inspect` with no credentials, to check the image pulls
anonymously.

The image is large (CUDA, torch, DeepSpeed and flash-attn wheels). A local
build needs well over 60 GB free in the podman machine.

`make publish-step` refuses a custom-image step until its `IMAGE_REF` exists on the
registry, so publish the image first. Run `make check-published` after any change
to the template, `src/`, `USAGE.md` or a build test.

## BlueVela image

On SLURM the step runs from a prebuilt squashfs, not a registry pull. Build it on a
BlueVela login node as `granitebuild`:

```bash
enroot import -o /proj/data-eng/llmb-read-write/images/openinstruct-sft-oi-v0.3.0-4.sqsh \
  "docker://quay.io#granitebuild/gb-steps:train-openinstruct-sft-skypilot-oi-v0.3.0-4"
```

Rebuild it for each `IMAGE_VERSION`, with the version in the file name. The BlueVela
fixture sets `openinstruct_sft_config.image_id` to that path. The step adds
`docker:`, and SkyPilot's SLURM provisioner hands a `.sqsh` to Pyxis as is.

## How ClearML is wired

Open-Instruct has no ClearML option, and `finetune.py` assumes wandb when tracking
is on. With `--with_tracking`, upstream `finetune.py`:

- calls `maybe_use_ai2_wandb_entity()`, which calls `wandb.login()`, when
  `--wandb_entity` is unset;
- calls `accelerator.get_tracker("wandb")`, which raises `ValueError` if no wandb
  tracker exists, and reads `wandb_tracker.run.url`.

So ClearML runs as accelerate's built-in ClearML tracker, next to a disabled wandb
tracker. When ClearML is on, `openinstruct_sft_run.sh` passes:

```
--with_tracking --report_to wandb clearml --wandb_entity none
```

and always exports `WANDB_MODE=disabled`. wandb becomes a no-op (`run.url` is
`None`), `--wandb_entity` skips the login, and accelerate's ClearML tracker gets all
scalars and hyperparameters. The project and task come from `CLEARML_PROJECT` and
`CLEARML_TASK`, which the script exports from `clearml_project`/`clearml_task_name`
(defaults `openinstruct-sft`/`finetune`). When ClearML is off, no tracking flags
are passed.

Verified 2026-10-23 by a CPU spike: accelerate with wandb disabled and
`CLEARML_OFFLINE_MODE=1` ran with no error, and the offline session recorded
`train/loss`.

## Tests

The unit tests at `test/`'s root need no cluster or image. They stub `accelerate`,
`nvidia-smi` and the venv Python, so they pin the exact argv the step emits.

| Test | Pins |
|---|---|
| `test_dockerfile.py` | every `FROM` digest-pinned; Open-Instruct pinned to a full commit and checked after fetch; `uv sync --frozen` and an exact `clearml==` pin; no `USER` and no `useradd` (runs as root); `OPEN_INSTRUCT_COMMIT` set and no `GIT_COMMIT`; every script the template calls exists in `src/` |
| `test_skypilot_train_openinstruct_sft_render.py` | the template rendered with gbserver's strict renderer: identity and I/O schema, the shared monitor, no GPU request, identical macro copies, the `docker:` prefix (including `.sqsh`), valid bash, every `raise_error` (data source, mixer and splits shape, `output`, `num_nodes`, `deepspeed_config`, `clearml_enabled`, step-owned and map `args`), and the guard and run argv (defaults, bound dataset, splits, `args` rendering, ClearML mode, output path) |
| `test_openinstruct_sft_guard_sh.py` | staged-path checks: empty or unrendered `base_model`, missing dir, no `config.json`, missing dataset, unknown option; valid paths pass silently |
| `test_openinstruct_sft_setup_sh.py` | the image check: passes on a good image, fails without `finetune.py` or when the training stack does not import |
| `test_openinstruct_sft_run_sh.py` | the `accelerate launch` argv and cwd, step-owned flags before build args, no-DeepSpeed, GPU count (SkyPilot, `nvidia-smi` fallback, none), ClearML off/on/auto and default names with `WANDB_MODE=disabled`, the staging swap and re-run, refusing a non-model `output_path`, no marker without a model, and the markers last and absolute |
| `test_exemplar_build.py` | every shipped build.yaml (exemplar and both fixtures) against the contract: step URI, `output` names a model output, exactly one data source, `base_model` bound, GPUs requested, no step-owned `args` (parsed from the template's `owned` list, 15 names), single node, the BlueVela GPU matching the environment's constraint, every exemplar parameter defined |

Run the unit tests only with `-m "not extended"`, and with AWS credentials unset.
`make test` and a plain `pytest test` do not deselect `extended`, so with AWS
credentials or the BlueVela key present they provision real clusters:

```bash
cd steps/train/openinstruct-sft/skypilot
env -u AWS_PROFILE -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
  GB_ENVIRONMENT=STANDALONE pytest test -m "not extended"
```

The two build tests are marked `extended` and train three steps on a real GPU. Both
need the image published first.

| | aws | slurm-bluevela |
|---|---|---|
| environment | `skypilot/aws` | `skypilot/slurm/ibm-bluevela` |
| image | quay.io ref, pulled at launch | prebuilt `.sqsh` on `/proj` |
| GPU | `A10G:1` | `H200:1` (the environment pins `h200`), `memory: "64"` |
| data | `dataset_mixer_list` (HF id) | bound `dataset` input |
| DeepSpeed | ZeRO-3 (default) | none (`deepspeed_config: ""`) |
| ClearML | `auto` (off without creds) | `true`, `CLEARML_OFFLINE_MODE=1` |
| `step_count` | 1 (hf:// staged inline) | 3 (two hfpull steps + the step) |

Both check the `model` artifact and the `open_instruct_commit` metadata. The
BlueVela `step_count` is marked to calibrate against the first green run.

### aws

Skips unless AWS credentials are in the environment. Credentials go in the
generated space's secrets, per `docs/environments/skypilot-aws.md`.

```bash
export AWS_PROFILE=gb-skypilot
make -C steps/train/openinstruct-sft/skypilot space
GB_ENVIRONMENT=STANDALONE make -C steps/train/openinstruct-sft/skypilot test TEST_DIR=test/aws
```

### slurm-bluevela

Marked `ibm`. Skips unless the BlueVela key for the `granitebuild` functional ID is
at `~/.ssh/ibm-bluevela.key`, and skips in SPS CI. The `.sqsh` must exist first (see
[BlueVela image](#bluevela-image)).

```bash
make -C steps/train/openinstruct-sft/skypilot space
GB_ENVIRONMENT=STANDALONE make -C steps/train/openinstruct-sft/skypilot test TEST_DIR=test/slurm-bluevela
```

## Follow-ups

- **More environments.** Build tests for K8s, SkyPilot-K8s and LSF. Issue #229
  stays open for them.
- **Multi-node.** The step is single-node only (`num_nodes` must be 1).
- **Legacy recipe compatibility.** The 8 `recipes/granite4-350m/{lsf,aws}/sft*`
  builds use the hand-authored `space://steps/openinstruct-sft` (the
  `skypilot/lsf/ibm-bluevela` and `skypilot/aws` assets, unchanged here). They
  depend on:
  - fork-only `finetune.py` flags: `--tokenized_data` (pre-tokenized data),
    `--padfree`, `--reduce_loss`, `--hf_dir`, `--chat_template <file>`;
  - FSDP with `fsdp_accelerate_mamba.yaml` for granite-4;
  - the `sft_config.*` keys and the `model`/`checkpoint`/`final` names.

  Carrying them over needs a pre-tokenized `tokenized_data` input (a fork image or
  an upstream patch), FSDP, a chat-template file option, and a translation layer
  for the old names. The two steps do not collide: `space://steps/openinstruct-sft`
  never resolves to this nested step, because step names resolve by longest prefix.
