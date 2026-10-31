# hf-peft-sft (SkyPilot): development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `hf-peft-sft` in a `build.yaml` (config contract, methods, inputs and output,
> examples). This file covers how the step is built, tested and published, and why
> it is shaped the way it is.

A reusable PEFT SFT step: LoRA, QLoRA or aLoRA adapters trained with TRL's
`SFTTrainer` and peft's `LoraConfig`, single node, one or more GPUs (DDP, FSDP or
DeepSpeed ZeRO-3), tracked with ClearML. It is a custom-image step: the
`Dockerfile` here bakes a hash-locked training stack and `src/` into the image. It
is referenced as `space://steps/train/hf-peft-sft` (the URI follows the nested
`steps/train/hf-peft-sft` authoring path). It has build tests on two clusters, AWS
and IBM BlueVela SLURM. The framework it follows is described in
[steps/README.md](../../../README.md); the shape follows
[openinstruct-sft](../../openinstruct-sft/skypilot/README.md).

The training entrypoint is our own `src/hf_peft_sft_train.py`, not the `trl sft`
CLI (it has no aLoRA) and not peft's example scripts (not a stable API).

## Layout

```
step-template.yaml          the step: config defaults, render-time validation, launcher blocks
Makefile                    STEP_NAME, REGISTRY, IMAGE_VERSION, `make lock`; includes steps/common.mk
Dockerfile                  the image: CUDA base, uv, the hashed lock, src/
requirements.in             direct pins (torch, peft, trl, transformers, accelerate, ...)
requirements.lock           full closure with hashes, compiled from requirements.in by `make lock`
.gbignore                   files gbserver ships without Jinja-rendering (docs, src/)
src/                        baked into the image at /opt/hf-peft-sft/
  hf_peft_sft_guard.sh        on-node checks of the staged base_model/dataset paths
  hf_peft_sft_setup.sh        image sanity check: train script present, training stack imports
  hf_peft_sft_run.sh          GPU count, accelerate config, ClearML on/off, launch, swap, markers
  hf_peft_sft_train.py        StepArgs + SFTConfig parsing, model/LoraConfig, SFTTrainer, save
  accelerate/                 fsdp.yaml, fsdp_qlora.yaml, deepspeed_z3.yaml (from peft's examples)
test/                       unit tests at the root; one dir per cluster build test
  test_docs.py
  test_dockerfile.py
  test_exemplar_build.py
  test_hf_peft_sft_guard_sh.py
  test_hf_peft_sft_run_sh.py
  test_hf_peft_sft_setup_sh.py
  test_hf_peft_sft_train.py
  test_hf_peft_sft_train_cpu.py
  test_skypilot_train_hf_peft_sft_render.py
  aws/test_skypilot_aws_train_hf_peft_sft.py                         real EC2 (A10G:1)
  slurm-bluevela/test_skypilot_slurm_bluevela_train_hf_peft_sft.py   real BlueVela (H200:2)
test-data/
  build.yaml                exemplar build a user copies (pasted into USAGE.md)
  parameters.yaml           parameters for the exemplar
  aws/build.yaml            aws build test fixture (qlora, alora)
  aws/buildtest.yaml        aws build test expectations
  slurm-bluevela/build.yaml       BlueVela build test fixture (lora/fsdp, qlora/deepspeed)
  slurm-bluevela/buildtest.yaml   BlueVela build test expectations
USAGE.md                    user doc; published as the released step's README.md
README.md                   this file; not published
```

The accelerate configs are adapted from peft's `examples/sft/configs/` (Apache-2.0;
each file says so). There is one DeepSpeed config: ZeRO-3 works for QLoRA as is,
because the 4-bit storage dtype is set to bf16 when sharding (see below).

## Why the shell lives in src/

The template only computes values with Jinja (paths, the `lora` map, the `args`
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

The run script backstops two render-time checks it can only see on the node: no
GPUs at all, and sharding on one GPU when the build asked for an `instance_type`
instead of a counted `accelerators` string.

`hf_peft_sft_train.py` imports only the standard library at module level. torch,
peft, trl, transformers and datasets are imported inside `main()`. So the pure
helpers (format detection, the aLoRA invocation check, `use_reentrant`, the model
and bitsandbytes kwargs, the `Step:` line, the `base_model_name_or_path` rewrite)
are unit-tested in the repo venv, which has none of those packages.

`.gbignore` lists `src/**` because gbserver runs a strict Jinja pass over the merged
step directory, and `hf_peft_sft_guard.sh` contains literal `{{` and `{%` (its
unrendered-binding detector). `**/*.md` is listed so the docs can show `{{ ... }}`.

A few rules worth knowing when changing the trainer:

- **Step-owned flags.** `StepArgs` fields and the `SFTConfig` fields the run script
  sets form the template's `owned` list (28 names), refused in `args`. The render
  test pins that every `StepArgs` field is in the list. A new `StepArgs` field must
  not share a name with an `SFTConfig` field: `HfArgumentParser` rejects the
  duplicate when the trainer starts.
- **`use_reentrant`.** With `gradient_checkpointing`, the trainer sets
  `gradient_checkpointing_kwargs={"use_reentrant": ...}`: `True` for
  `distributed: deepspeed` (TRL's `SFTTrainer` forces `True` for PEFT with ZeRO-3,
  and warns, so the step picks the value TRL would use), otherwise `True` only for
  `qlora` outside multi-process DDP (peft's examples).
- **QLoRA placement.** Under DDP the 4-bit base is pinned to the local GPU
  (`device_map={"": local_rank}`) and stored as `uint8`. Under FSDP or DeepSpeed the
  storage dtype is bf16 (sharding needs a float dtype) and the launcher places the
  shards.
- **aLoRA check.** The invocation is checked against the first 32 rows of the
  trainer's already-tokenized dataset (`trainer.train_dataset[i]["input_ids"]`), so it
  sees exactly what the chat template produced. None matching is an error; some
  missing is a warning.
- **Saving.** Under FSDP the state dict type is switched to `FULL_STATE_DICT` before
  `trainer.save_model`, so the adapter is saved whole. `SFTTrainer` saves the
  tokenizer with the adapter. The run script then reads the safetensors header and
  refuses to register a file with no tensors.
- **FSDP2, not FSDP1.** With FSDP1, peft 0.21.x (huggingface/peft#3876, a
  regression from peft #3490) saved a 40-byte adapter with no tensors: FSDP1 wraps
  each decoder layer, so module names read
  `...layers.0._fsdp_wrapped_module.self_attn.q_proj`, and peft strips that segment
  only at the start of a name. The full-state-dict gather was correct (a BlueVela
  probe saw 420 LoRA keys reach peft; it kept 0). FSDP2 (`fsdp_version: 2`) leaves
  the names clean, and a CPU check kept all 28 LoRA tensors of a tiny Llama under
  FSDP2, with gradient checkpointing too. Move back to FSDP1 only after a peft
  release carries the fix.

## Image

`quay.io/granitebuild/steps/train/hf-peft-sft/skypilot:peft-v0.21.2-3`. It holds:

| Part | Pin |
|---|---|
| Base | `nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04`, by digest. `devel` because DeepSpeed JIT-compiles its ops. |
| uv | `ghcr.io/astral-sh/uv:0.13.6`, by digest (copied in as a stage). |
| Python | 3.12, uv-managed, under `/opt/uv-python`; venv at `/opt/hf-peft-sft/.venv`. |
| Training stack | `torch==2.15.0`, `peft==0.21.2`, `trl==1.17.1`, `transformers==5.22.1`, `accelerate==1.15.0`, `bitsandbytes==0.50.2`, `deepspeed==0.19.9`, `datasets==5.1.0`, `clearml==2.1.12` (`requirements.in`), with every transitive dependency pinned and hashed in `requirements.lock` (linux x86_64, Python 3.12). |
| Not included | flash-attn (attention is `sdpa`), vllm, unsloth, wandb. `test_dockerfile.py` checks the lock has none of the first three. |
| apt | `apt-get upgrade`; `ca-certificates`, `curl`, `git`, `sudo` (SkyPilot); NVIDIA Nsight Compute purged (`cuda-nsight-compute-13-2`, `nsight-compute-2026.1.1`). |
| Step scripts | `src/` at `/opt/hf-peft-sft/`. |

The locally built image is 17.3 GB.

### The torch/CUDA pair

torch 2.12 and later ship CUDA 13 wheels by default, which need an NVIDIA driver
of 580 or newer. The newest CUDA 12.8 build is torch 2.11.0. So the driver on each
cluster decided the pair. Measured on 2026-10-30 with `nvidia-smi`:

| Cluster | GPU | Driver |
|---|---|---|
| IBM BlueVela | H200 | 590.48.01 |
| aws (SkyPilot's default image) | A10G | 580.159.04 |

Both are 580 or newer, so the image uses **torch 2.15.0** from PyPI, the latest
release. It bundles the CUDA 13.2 runtime (2.15.0 ships only cu132 and cu134
builds). The base is `cuda:13.2.1-devel`, the same CUDA minor, so the `nvcc`
DeepSpeed uses for its JIT ops matches torch's runtime. The base's
`NVIDIA_REQUIRE_CUDA` admits both the 580 and 590 driver branches.

Only the driver versions above were measured (2026-10-30). On either cluster the
image's CUDA 13.2 runtime may rely on CUDA minor-version compatibility. On aws it
works: the aws build test passed on the 580.159.04 driver (2026-10-30, QLoRA with
bitsandbytes 4-bit and aLoRA, 3 steps each). On BlueVela it works under Pyxis on the
590.48.01 driver (build 611571b7, 2026-10-31, LoRA/FSDP2 and QLoRA/DeepSpeed). If a
future driver breaks it, the fallback is torch 2.11.0+cu128 on
`cuda:12.8.1-devel-ubuntu22.04`.

### Choices in the Dockerfile that look odd, and why

- **A hashed lock with `--no-deps`.** Every package comes from `requirements.lock`
  with `--require-hashes --no-deps`, so the install cannot pull anything the lock
  does not name. One exception: DeepSpeed ships only an sdist, and uv's build
  isolation resolves its build dependencies (setuptools) at build time, outside the
  lock's hashes. The installed DeepSpeed itself is hash-checked.
- **`DS_BUILD_OPS=0`.** DeepSpeed is built without pre-compiling its ops; any op
  a run uses JIT-compiles on the node at first use.
  That is why the base is `devel` and why its CUDA minor matches torch's.
- **A CUDA 13.2 runtime on 580/590 drivers.** See [The torch/CUDA pair](#the-torchcuda-pair):
  it may rely on minor-version compatibility. Confirmed on aws (580) and BlueVela (590).
- **chmod in two layers.** Pyxis on SLURM runs the container as the job user, not
  root. The venv and uv's managed Python (`/opt/uv-python`, not root's home, since
  the venv's interpreter is a symlink into it) get `chmod -R a+rX` in the install
  layer itself. `src/` is copied after it and made readable with a `find -maxdepth 2`
  that skips `.venv`. A later recursive chmod over `/opt/hf-peft-sft` would copy the
  whole venv layer up again (`test_dockerfile.py` checks there is none).
- **The venv is first on `PATH`.** That also provides the `hf` CLI the aws
  environment's inline `hf download` uses.
- **No flash-attn.** It would need a wheel built for this exact torch/CUDA pair (or
  a long source build in the image), and adapter training runs fine on PyTorch's
  `sdpa`. The model loads with `attn_implementation="sdpa"`.

### Security trade-off

The base is Ubuntu, not UBI or a minimal image. SkyPilot bootstraps itself by
running `apt-get` inside the container, so a UBI/RPM base fails provisioning. The
mitigations:

- every base is pinned by digest (`test_dockerfile.py` checks this);
- every Python package is pinned and hash-checked (above);
- `apt-get upgrade` runs at build time;
- the container runs as **root**, like the repo's other step images. A non-root
  default `USER` does not work under SkyPilot's docker runtime: SkyPilot sets the
  container up as root and saves root's environment (`HOME=/root`) to
  `/etc/profile.d`, so the non-root user logs in with `HOME=/root` and provisioning
  fails. The VM exists for one job and is torn down after; Pyxis on SLURM still runs
  the container as the job user. Restoring a non-root user needs a SkyPilot fix
  (#566);
- files under `/opt` are readable by any uid but writable only by root;
- the image is scanned for vulnerabilities before `make publish-image` (below).

### Scanning

Scan every image before `make publish-image`:

1. Run the team's container scanner on the built image.
2. As a stop-gap, run `trivy image --severity CRITICAL,HIGH <ref>`.
3. Do not publish with a fixable CRITICAL in a package the Dockerfile controls (the
   apt layer, the locked Python packages, the step scripts).
4. Record the CRITICAL and HIGH counts in the PR.

A fixable CRITICAL in a locked Python package is fixed by raising its pin (in
`requirements.in` if it is a direct pin, else as a constraint there), running
`make lock`, bumping `IMAGE_VERSION`, then rebuilding and rescanning. Check quay's
own scan of the pushed tag too: its database differs from trivy's.

Results for `peft-v0.21.2-3`, the shipped image (manifest
`sha256:cdae82b13aabf628cb5a4ad2dd04bcecd9ee216cdaa2635a5e52400bfc74afd4`, scanned
2026-10-31; `-3` changes only `src/` over `-2`, and `-2` scanned the same):

- **trivy 0.76.1** (CRITICAL+HIGH): nothing fixable. 14 CRITICAL without a fix, all in
  Ubuntu's `linux-libc-dev` (kernel UAPI headers from the CUDA devel base, needed by
  DeepSpeed's JIT build); 372 HIGH, none with a fix.
- **quay (Clair)**: 0 Critical, 0 High; 357 Medium, 109 Low, 6 Negligible, none with a
  fix.

`peft-v0.21.2-1` had 5 more fixable HIGH, all in the `msgpack` 1.1.2, `urllib3` 2.7.0
and `setuptools` 70.3.0 vendored inside **pip** in uv's managed CPython (never run by
the step). `-2` removes pip and ensurepip from the managed CPython.

### Bumping IMAGE_VERSION

The Makefile's `IMAGE_VERSION` is a stable version (`peft-v0.21.2-3`), not the
per-commit git SHA. The image is large, and BlueVela caches a `.sqsh` per tag
(each new tag means a fresh cold import), so it must not churn on unrelated commits.

- Bump the suffix (`-1` to `-2`) when the Dockerfile, `requirements.lock` or `src/`
  changes.
- Bump the prefix with the peft pin (for example `peft-v0.22.0-1`).

After any edit to `requirements.in`, run `make lock` to recompile
`requirements.lock` (uv 0.13.6, linux x86_64, Python 3.12, with hashes) and commit
both files. `test_dockerfile.py` checks the direct pins agree between the two.

After a bump, publish the image, then re-run `make publish-step`. BlueVela
re-imports the new tag automatically on its next run (see
[BlueVela image cache](#bluevela-image-cache)).

## Publishing the image

The image is `$(REGISTRY)/steps/train/hf-peft-sft/skypilot:<IMAGE_VERSION>`:
`steps/common.mk` gives every custom step its own repository, named after the step
directory, and the tag is `IMAGE_VERSION`. The repository must be public (SkyPilot
pulls anonymously); see `steps/README.md`, "A custom image-based step's image
repository". With `QUAY_API_TOKEN` set, `publish-image` creates a missing quay.io
repository as public.

Log in first (`podman login quay.io/granitebuild`, or `REGISTRY_USER` /
`REGISTRY_PASSWORD` in the environment; see `steps/README.md`, "Registry
credentials"), then:

```bash
make -C steps/train/hf-peft-sft/skypilot image publish-image
sh steps/scripts/registry-tag-status.sh quay.io/granitebuild/steps/train/hf-peft-sft/skypilot:peft-v0.21.2-3
```

`publish-image` refuses a tag that already exists, so bump `IMAGE_VERSION` (see
above) before publishing a changed image. Run `registry-tag-status.sh` with no
credentials: `present` means the image pulls anonymously.

The image is large (CUDA devel, torch, DeepSpeed). A local build needs well over
60 GB free in the podman machine.

`make publish-step` refuses a custom-image step until its `IMAGE_REF` exists on the
registry, so publish the image first. Run `make check-published` after any change
to the template, `src/`, `USAGE.md` or a build test.

## BlueVela image cache

The `ibm-bluevela` SLURM environment's `slurm_image_cache` block (#560) caches the
step's registry image as `/proj/granite-build/g4os/enroot/<image>.sqsh`. A cold cache
is filled by a CPU import job before the first GPU job (gbserver waits up to 5 h,
queueing included); later launches start the container from the `.sqsh` in seconds.
Bumping `IMAGE_VERSION` therefore needs nothing on BlueVela: the next run imports
the new tag. To force a re-import of an unchanged tag, delete its `.sqsh` there.
See `docs/environments/skypilot-slurm.md`, "slurm_image_cache".

## How ClearML is wired

transformers has a built-in `ClearMLCallback`, so the step needs no tracker shim
and no wandb. When ClearML is on, `hf_peft_sft_run.sh` passes `--report_to clearml`
to the trainer and exports:

```
CLEARML_PROJECT=<clearml_project, default hf-peft-sft>
CLEARML_TASK=<clearml_task_name, default sft-<method>>
CLEARML_LOG_MODEL=${CLEARML_LOG_MODEL:-false}
```

The callback reads the project and task from those variables. Its own default for
`CLEARML_LOG_MODEL` is true (upload every save to the ClearML file server), which
would duplicate the Granite.Build artifact, so the step defaults it to false and
keeps a value set through `launcher_config.envs`. When ClearML is off, the trainer
gets `--report_to none`.

`auto` mode is decided in the run script, on the node: on iff `CLEARML_API_HOST`,
`CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` are all non-empty. The
template maps `clearml_enabled` (`auto`, `true`/`on`/`1`, `false`/`off`/`0`) to the
script's `--clearml-mode auto|on|off`.

The BlueVela build test runs ClearML with `CLEARML_OFFLINE_MODE=1`, so the callback
path runs on a real GPU without a server.

## Tests

The unit tests at `test/`'s root need no cluster, GPU or image. They stub
`accelerate`, `nvidia-smi` and the venv Python, so they pin the exact argv the step
emits.

| Test | Pins |
|---|---|
| `test_docs.py` | `USAGE.md` documents every config field and every step-owned flag, contains the exemplar build verbatim, and links only to anchors, `src/` or `https://`; this README links USAGE.md and records the torch/driver decision |
| `test_dockerfile.py` | every `FROM` digest-pinned; the direct pins in `requirements.in` and the lock agree with the spec; every lock entry `==`-pinned with hashes, installed with `--require-hashes`; no flash-attn, vllm or unsloth; no `USER` and no `useradd` (runs as root); the `/opt/hf-peft-sft` layout and `PATH`; no recursive chmod of the venv after `COPY src/`; Makefile identity; every script and accelerate config the template and run script call exists in `src/` |
| `test_skypilot_train_hf_peft_sft_render.py` | the template rendered with gbserver's strict renderer: identity and I/O schema, the shared monitor and `Step:` event, no GPU request, identical macro copies, the `owned` list covering every `StepArgs` field and the run script's flags, the `docker:` prefix (including `.sqsh`), valid bash, every `raise_error` (data source, split, enums, aLoRA invocation and packing, qlora + fp16, `output`, `num_nodes`, no GPU request, sharding on one GPU, `lora.*`, step-owned, map and empty-list `args`), and the guard and run argv (defaults, bound dataset, `lora` lists, aLoRA, `args` rendering, ClearML mode, output path and names) |
| `test_hf_peft_sft_guard_sh.py` | staged-path checks: empty or unrendered `base_model` or dataset, missing dir, no `config.json`, missing dataset, unknown option; valid paths (dataset dir or file) pass silently |
| `test_hf_peft_sft_setup_sh.py` | the image check: passes on a good image and imports the whole stack, fails without the train script or when the stack does not import, only warns without `nvidia-smi` |
| `test_hf_peft_sft_run_sh.py` | the `accelerate launch` argv: DDP single and multi GPU, the config file per method × distributed, sharding on one GPU refused, step-owned flags before build args, `--bf16`/`--fp16`, GPU count (SkyPilot, `nvidia-smi` fallback, none), the dataset cache in the workdir, bad enums and missing values as prefixed errors; ClearML off/on/auto, default names and `CLEARML_LOG_MODEL` default and override; the staging swap and re-run, a stale `.partial`, refusing a non-adapter, degenerate (`/`, `.`, `..`) or `base_model`-overlapping `output_path` (same, inside, containing, via a symlink, or at the staging dir), no marker without both adapter files, a training failure propagating, and the markers last and absolute |
| `test_hf_peft_sft_train.py` | the trainer's pure helpers: format detection, the aLoRA invocation check, `target_modules`, the `use_reentrant` rule, model and bitsandbytes kwargs, `LoraConfig` kwargs, the `Step:` line, the `base_model_name_or_path` rewrite, the `StepArgs` field list |
| `test_hf_peft_sft_train_cpu.py` | the real trainer on CPU (tiny random Llama, 2 steps): `lora` and `alora` save an adapter with the tokenizer and the rewritten base id, and a wrong aLoRA invocation fails before training. Skipped unless `peft` and `trl` import (see below) |
| `test_exemplar_build.py` | every shipped build.yaml (exemplar and both fixtures, per target) against the contract: step URI, `output` names a model output, exactly one data source, `base_model` bound, GPUs requested and enough for sharding, aLoRA iff an invocation string, no step-owned `args` (parsed from the template's `owned` list), single node, a portable `base_model_id`, the fixtures together cover qlora/ddp, alora/ddp, lora/fsdp and qlora/deepspeed, the BlueVela GPU matching the environment's constraint, every exemplar parameter defined |

Run the unit tests only with `-m "not extended"`, and with AWS credentials unset.
`make test` and a plain `pytest test` do not deselect `extended`, so with AWS
credentials or the BlueVela key present they provision real clusters:

```bash
cd steps/train/hf-peft-sft/skypilot
env -u AWS_PROFILE -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
  GB_ENVIRONMENT=STANDALONE PYTHONPATH=src pytest test -m "not extended"
```

### The CPU smoke test

`test_hf_peft_sft_train_cpu.py` runs the real training script, so it needs peft and
trl, which the repo venv does not have. It catches TRL and peft API drift before
any GPU time. Build a separate venv with the image's pins (CPU torch: PyPI's macOS
torch wheels are CPU-only; on Linux add `--extra-index-url
https://download.pytorch.org/whl/cpu` for torch):

```bash
S=steps/train/hf-peft-sft/skypilot
uvx --from uv==0.13.6 uv venv -q --python 3.12 /tmp/hfpeft-cpu-venv
grep -E "^(peft|trl|transformers|accelerate|datasets|torch)==" "$S/requirements.in" > /tmp/hfpeft-cpu.in
VIRTUAL_ENV=/tmp/hfpeft-cpu-venv uvx --from uv==0.13.6 uv pip install -r /tmp/hfpeft-cpu.in pytest
```

Then run it without the repo's conftest (`steps/conftest.py` loads the repo-wide
`test/conftest.py`, which needs pydantic):

```bash
cd "$S" && PYTHONPATH=src /tmp/hfpeft-cpu-venv/bin/python -m pytest -c /dev/null --noconftest test/test_hf_peft_sft_train_cpu.py -q
```

It downloads `trl-internal-testing/tiny-LlamaForCausalLM-3.2` from the HF Hub and
takes about 15 seconds. `qlora` needs CUDA, so it is covered only by the build tests.

### Build tests

The two build tests are marked `extended` and train three steps per target on real
GPUs. Both need the image published first, and both need `PYTEST_ADDOPTS=-s`: each
runs two targets (two launches in one pytest process), which otherwise fails on
click's captured file descriptors.

| | aws | slurm-bluevela |
|---|---|---|
| environment | `skypilot/aws` | `skypilot/slurm/ibm-bluevela` |
| image | quay.io ref, pulled at launch | registry ref, cached as a shared `.sqsh` (`slurm_image_cache`) |
| base model | `hf:///HuggingFaceTB/SmolLM2-135M-Instruct` | `hf:///HuggingFaceTB/SmolLM2-135M` with `args.chat_template_path: HuggingFaceTB/SmolLM2-135M-Instruct`. The Instruct repo also ships 1.7 GB of onnx exports, and BlueVela's hfpull (1 GiB cpu-partition cgroup, whole-repo parallel download) is OOM-killed on it; the base repo is 272 MB with the same vocab |
| target 1 | `qlora`, `ddp`, `A10G:1` | `lora`, `fsdp`, `H200:2`, `memory: "128"` |
| target 2 | `alora`, `ddp`, `A10G:1`, invocation `<\|im_start\|>assistant` | `qlora`, `deepspeed`, `H200:2`, `memory: "128"` |
| data | `dataset_name: trl-lib/Capybara` (both) | target 1: bound `dataset` (`hf:///datasets/trl-lib/Capybara`); target 2: `dataset_name` |
| training | `train[:64]`, `max_steps: 3`, `max_length: 1024` | same |
| ClearML | `auto` (off without creds) | `true`, `CLEARML_OFFLINE_MODE=1` |
| `step_count` | 1 per target (hf:// staged inline) | 3 and 2 (one hfpull step per hf:// input + the step), calibrated on build 611571b7 |

Both check each target's `adapter` artifact and the `method` and `peft_version`
metadata.

| Result | aws | slurm-bluevela |
|---|---|---|
| status | **passed** 2026-10-30, 7m21s for both targets (`peft-v0.21.2-1`) | **passed** 2026-10-31, 14m32s for both targets, build 611571b7 (`peft-v0.21.2-3`, cached `.sqsh`) |
| loss, steps 1→3 | qlora and alora each logged `Step: 1/3`…`3/3` | lora/fsdp 2.15 → 4.26 → 2.27; qlora/deepspeed `Step: 1/3`…`3/3` |
| adapter | `adapter` at `/root/sky_workdir/adapter_output` per target | lora/fsdp: 420 tensors, 19.6 MB (read on the node); both registered |
| markers | `method`, `peft_version` 0.21.2, `trl_version` 1.17.1 | same |

Earlier BlueVela runs, and what each one fixed:

- `-1`: the SmolLM2-135M-Instruct hfpull was OOM-killed three times in the cpu
  partition's 1 GiB cgroup (the repo ships 1.7 GB of onnx exports). The fixture now
  stages the base SmolLM2-135M.
- `-1`: on one node the lora/fsdp container saw 1 of its 2 GPUs, and NCCL failed with
  "Multiple Ranks are using the same GPU". The run script now refuses an undercount.
- `-2`: lora/fsdp "passed" with a 40-byte, 0-tensor adapter (FSDP1 + peft#3876). The
  run script now refuses an adapter with no weights, and FSDP is FSDP2.

#### aws

Skips unless AWS credentials are in the environment. Credentials go in the
generated space's secrets, per `docs/environments/skypilot-aws.md`.

```bash
export AWS_PROFILE=gb-skypilot
make -C steps/train/hf-peft-sft/skypilot space
cd steps/train/hf-peft-sft/skypilot && PYTEST_ADDOPTS=-s GB_ENVIRONMENT=STANDALONE PYTHONPATH=src \
  python -m pytest test/aws -k test_runner    # with the repo .venv active
```

`make test` would also rebuild the image (needs >60 GB free); the direct pytest call
reuses the published image.

#### slurm-bluevela

Marked `ibm`. Skips unless the BlueVela key for the `granitebuild` functional ID is
at `~/.ssh/ibm-bluevela.key`, and skips in SPS CI. A cold image cache adds an
import job before the GPU jobs (see [BlueVela image cache](#bluevela-image-cache)).

```bash
make -C steps/train/hf-peft-sft/skypilot space
cd steps/train/hf-peft-sft/skypilot && PYTEST_ADDOPTS=-s GB_ENVIRONMENT=STANDALONE PYTHONPATH=src \
  python -m pytest test/slurm-bluevela -k test_runner    # with the repo .venv active
```

## Follow-ups

- **Merged-model output.** Optionally merge a `lora` adapter into the base and
  register the merged model (not possible for aLoRA; lossy for a 4-bit base).
- **More environments.** Build tests for K8s, SkyPilot-K8s and LSF.
- **Multi-node.** The step is single-node only (`num_nodes` must be 1).
- **flash-attn in the image.** Needs a wheel for the torch/CUDA pair.
- **Non-root image (#566).** Blocked on a SkyPilot fork fix; see Security trade-off.
