# fms-hf-tuning (SkyPilot): development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `fms-hf-tuning` in a `build.yaml` (config contract, data paths, methods, examples).
> This file covers how the step is built, tested and published, and why it is
> shaped the way it is.

A reusable fine-tuning step around
[fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning)
3.2.2 (ibm-granite/granite.build#231): full fine-tuning, LoRA or aLoRA, single node,
DDP or FSDP, tracked with ClearML. It is a custom-image step: the `Dockerfile` bakes
a hash-locked fms-hf-tuning stack and `src/` into the image. It is referenced as
`space://steps/train/fms-hf-tuning`. It has build tests on AWS and IBM BlueVela
SLURM. The framework is described in [steps/README.md](../../../README.md); the shape
follows [hf-peft-sft](../../hf-peft-sft/skypilot/README.md) (PR #596).

**Why a separate step from hf-peft-sft.** fms-hf-tuning 3.2.2 caps
`transformers<=4.55.4`, `torch<2.9`, `trl` and `peft` at 0.19.x and `datasets` at 4.x;
hf-peft-sft's image is on transformers 5.22, torch 2.15, trl 1.17 and peft 0.21. No
image can hold both. fms-hf-tuning is also its own trainer (`tuning.sft_trainer`,
its own parser, data pipeline and trackers), so this step wraps it rather than
reimplementing it. The scaffolding (run and guard scripts, step-creation
validation, the `Step:` monitor rule, the hashed-lock Dockerfile, the test layout)
comes from #596.

## Layout

```
step-template.yaml          the step: config defaults, step-creation validation, launcher blocks
Makefile                    STEP_NAME, REGISTRY, IMAGE_VERSION, `make lock`; includes steps/common.mk
Dockerfile, .dockerignore   the image: CUDA 12.8 base, uv, the hashed lock, the patch, src/
patches/                    image build only (not published)
  lightglue_no_remote_code.py   CVE-2026-5241 backport on transformers 4.55.4
requirements.in             direct pins (fms-hf-tuning, the flash-attn wheel, clearml)
requirements.lock           full closure with hashes, compiled by `make lock`
.gbignore                   files gbserver ships without Jinja-rendering (docs, src/)
src/                        baked into the image at /opt/fms-hf-tuning/
  fms_hf_tuning_guard.sh      on-node checks of the staged base_model/dataset paths
  fms_hf_tuning_setup.sh      image check: scripts present, the stack (flash-attn too) imports
  fms_hf_tuning_run.sh        GPUs, accelerate config, output paths, ClearML, job, launch, check, swap, markers
  fms_hf_tuning_job.py        builds + strictly checks the job JSON; checks the saved output
  fms_hf_tuning_entry.py      upstream sft_trainer.main() plus a rank-0 `Step:` callback
  accelerate/                 single_gpu.yaml, ddp.yaml (step-authored); fsdp.yaml (upstream's defaults)
test/                       unit tests at the root; one dir per cluster build test
test-data/                  the exemplar (build.yaml + parameters.yaml) and per-cluster fixtures
USAGE.md                    user docs, published as the step's README.md
```

## How a run works

1. **Step creation.** The template's `validate` macro (repeated in `setup:` and
   `run:`; a render test pins the copies identical) checks the config with
   `raise_error()`, so a bad config fails before any node is provisioned.
2. **setup:** `fms_hf_tuning_guard.sh` checks the staged paths;
   `fms_hf_tuning_setup.sh` imports the whole stack.
3. **run:** the guard again (a warm-cluster re-run renders `run:` without `setup:`),
   then `fms_hf_tuning_run.sh`. It receives `args` and `data_config` as base64 JSON
   (`| json_dumps | b64encode`), so free text (chat templates, `{{ }}`, quotes)
   never passes through shell quoting.
4. **The job JSON.** `fms_hf_tuning_job.py build` merges the step defaults, the
   user's `args` and the step-owned keys, then checks the result against
   fms-hf-tuning's own parser (below) and writes `fms_hf_tuning_job.json`.
5. **Training.** `accelerate launch --config_file <step config> fms_hf_tuning_entry.py`
   with `SFT_TRAINER_CONFIG_JSON_PATH` pointing at the job and
   `TERMINATION_LOG_FILE` at `fms_hf_tuning_error.log`. On failure the run script
   prints that log (or says none was written) and accelerate's exit code.
   `accelerate launch` turns every child failure into exit 1, so upstream's
   1 (input error) / 203 (internal error) codes don't reach the script; the
   termination log is what tells them apart.
6. **Register.** `fms_hf_tuning_job.py check-output` refuses missing files or 0
   tensors (from the safetensors headers, no torch needed) and rewrites an
   adapter's `base_model_name_or_path`. A malformed index or adapter_config JSON,
   or a failed write, ends as an `fms-hf-tuning: ERROR` message. The run script then swaps
   `<out>.partial` into place and prints the markers, `GB_ARTIFACT_ID` last.

### The strict check

fms-hf-tuning reads the job with `HfArgumentParser.parse_dict(...,
allow_extra_keys=True)`. That ignores a misspelled key, and it does no type
conversion: a YAML `1e-5` (a string under PyYAML) or `packing: maybe` passes
through. `coerce_args` runs every value through its argparse action's `type` and
`choices` exactly as the command line would, refuses unknown keys and lists for
single-value arguments, and then `parse_dict(..., allow_extra_keys=False)` runs on
the dataclass keys. It keeps the first action per dest: HfArgumentParser follows a
default-True bool (such as `use_flash_attn`) with an untyped `--no_<field>`
store_false twin, so keeping the typed one is what converts `"false"` and refuses
`"maybe"`. `EXTRA_ARGS` (`peft_method`, `quantization_method`,
`exp_metadata`, `alora_invocation_string`) are arguments sft_trainer adds outside
its dataclasses; they are type-checked only. `test_fms_hf_tuning_job_upstream.py`
pins all of this against the real parser, including every `args` map the step
ships.

### The progress line

Upstream's `main()` passes `additional_callbacks=None` to `train()`, with no hook.
`fms_hf_tuning_entry.py` replaces `tuning.sft_trainer.train` with a wrapper that
appends a `TrainerCallback` printing `Step: <n>/<max> loss=<x>` on world process
zero, then calls `main()`, which looks `train` up at call time. This relies on
`train()`'s `additional_callbacks` keyword in the pinned 3.2.2;
`test_upstream_train_still_takes_additional_callbacks` fails if a bump drops it.
fms-hf-tuning logs once per epoch by default, so the step defaults
`logging_strategy: steps` and `logging_steps: 10` (overridable in `args`).

### FSDP1 and adapters

`fsdp` uses fms-hf-tuning's own `fixtures/accelerate_fsdp_defaults.yaml` (FSDP1,
FULL_SHARD, FULL_STATE_DICT), the config its launcher uses for every multi-GPU run.
hf-peft-sft moved to FSDP2 because peft 0.21 saves an empty adapter from
FSDP1-wrapped models (huggingface/peft#3876, a regression from peft #3490). This
image has peft 0.19.1, before that regression. `check-output` refuses a 0-tensor
adapter either way, and the BlueVela `lora-fsdp` build test is the proof. If it
ever saves 0 tensors, either switch `fsdp.yaml` to FSDP2 for adapters or refuse
`lora`/`alora` with `fsdp`.

## Image

`nvcr.io/nvidia/cuda:12.8.2-base-ubuntu24.04` (digest-pinned), uv 0.13.6 (digest-
pinned), Python 3.12 under `/opt/uv-python`, and a venv at
`/opt/fms-hf-tuning/.venv` installed from `requirements.lock` with
`--require-hashes --no-deps`. torch 2.8 is the newest fms-hf-tuning 3.2.2 allows; it
bundles the CUDA 12.8 runtime, so the base is `base`, not `devel` (nothing compiles
at runtime). Its `NVIDIA_REQUIRE_CUDA` admits the aws 580 and BlueVela 590 drivers.

flash-attn comes from the project's GitHub release wheel
(`flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`), hashed
in the lock: PyPI has only the sdist, which takes hours to compile. The `abiTRUE`
build matches torch 2.8's `_GLIBCXX_USE_CXX11_ABI` (True). fms-hf-tuning turns
`use_flash_attn` on by default.

The image is 8.5 GB locally (4.7 GB compressed on quay; measured on `fms-v3.2.2-2`).
`.dockerignore` keeps local `__pycache__` out of `COPY src/` (the first test build
shipped it).

**Apple Silicon.** Under podman's qemu x86 emulation, `import datasets` (pyarrow)
hangs and `import flash_attn` segfaults; the same versions import natively on arm64
and the wheel's `.so` is in the image. So don't run `fms_hf_tuning_setup.sh` in the
image on a Mac: check the layout instead (versions via `importlib.metadata`), and
let the build tests run the real import on the GPU node.

### Security trade-off

Three documented exceptions to the image rules:

1. **Ubuntu base.** SkyPilot bootstraps itself by running `apt-get` inside the
   container, so a UBI/RPM base fails provisioning.
2. **Root.** SkyPilot's docker runtime saves root's environment (`HOME=/root`) to
   `/etc/profile.d`, so a non-root default `USER` fails provisioning (#566). Pyxis
   on SLURM remaps the job to root inside the container (`--container-remap-root`);
   on the host it is still the submitting user, with that user's file access.
3. **Below latest.** fms-hf-tuning 3.2.2 caps torch 2.8 (latest 2.15),
   transformers 4.55.4 (latest 5.22), trl 0.19.1, peft 0.19.1 and datasets 4.8.5. Only
   upstream can lift those caps; overriding them would run fms-hf-tuning on versions
   it does not support. Record which scan findings come from them.

Mitigations: digest pins, a hashed lock, `apt-get upgrade`, pip and ensurepip
removed from the managed CPython, no bytecode (`UV_COMPILE_BYTECODE=0`, #616),
files under `/opt` readable by any uid but writable only by root, and a scan before
`make publish-image`.

### Scanning

Scan every image before `make publish-image`, as for hf-peft-sft:

1. Run the team's container scanner on the built image.
2. As a stop-gap, `trivy image --severity CRITICAL,HIGH --ignore-unfixed <ref>`.
3. Don't publish with a fixable CRITICAL in a package the Dockerfile controls that
   upstream's caps allow raising.
4. Record the CRITICAL and HIGH counts here and in the PR, separating findings in
   packages held back by fms-hf-tuning's caps.

Results for `fms-v3.2.2-1` (manifest
`sha256:b683b78bf5a4e0d2b6af71af3f143953f25c3b84a9e47439e574cb7175e6b293`, scanned
2026-11-06):

- **trivy 0.76.2** (CRITICAL+HIGH): 0 CRITICAL; 7 HIGH, all in Python packages and
  none in the Ubuntu layer. 5 have a fix, and every one is held back by an upstream
  cap, so none is fixable in this image:

  | package | installed | fixed in | CVEs | held back by |
  |---|---|---|---|---|
  | transformers | 4.55.4 | 5.3.0 / 5.5.0 / 5.10.0 | CVE-2026-4372, CVE-2026-5241 (patched from `-3`, below), CVE-2026-9856 | fms-hf-tuning `transformers<=4.55.4` |
  | torch | 2.8.0 | 2.10.0 | CVE-2026-24747 | fms-hf-tuning `torch<2.9` |
  | fsspec | 2026.2.0 | 2026.6.0 | CVE-2026-104851 | datasets 4.8.5 `fsspec<=2026.2.0` (fms-hf-tuning `datasets<5`) |

  The other 2 have no fix yet: transformers 4.55.4 (CVE-2026-80047) and flash_attn
  2.8.3 (CVE-2026-31253).
- **quay (Clair)**: 1 Critical, 20 High (114 Medium, 29 Low, 2 Negligible). Clair
  counts each advisory ID (GHSA and PYSEC aliases separately), so it lists more than
  trivy for the same three packages, and it marks all of them fixable. The Critical
  and High are only in the same three capped packages:
  - **transformers 4.55.4**: 13 High, and the Critical (PYSEC-2026-2290, fixed after
    5.2.0);
  - **torch 2.8.0**: 5 High (fixed in 2.9.0 / 2.10.0);
  - **fsspec 2026.2.0**: 2 High (fixed in 2026.6.0).

  None of these can be upgraded away in this image, for the reasons in the table
  above. The Critical is the same advisory trivy rates High, CVE-2026-5241: from
  `-3` it is **patched in place** (below).

`fms-v3.2.2-2` (manifest
`sha256:3b4acd1122961016323663ee280245ca3723f5e8c7f4bd4ee019e242f84c0fbe`, published
2026-11-06) changes only `src/` (run.sh: the scratch-dir overlap refusal and the
accelerate exit-code message), on the same lock and base. trivy 0.76.2 gives the same
result as `-1`: 0 CRITICAL, 7 HIGH, the same packages and CVEs.

**The CVE-2026-5241 backport (from `fms-v3.2.2-3`).** This is the quay Critical
(GHSA-fgcw-684q-jj6r, PYSEC-2026-2290). transformers' LightGlueConfig takes
`trust_remote_code` from the model's own `config.json` and passes it to a nested
`AutoConfig.from_pretrained` for a keypoint detector that is not a built-in type. So
loading an attacker-controlled `base_model` runs its Python, even though
fms-hf-tuning never sets `trust_remote_code`; a local check against 4.55.4 confirmed
the payload runs.
- The fixed transformers releases are 5.x, which fms-hf-tuning's cap excludes.
- So the Dockerfile applies `patches/lightglue_no_remote_code.py`, a backport of
  upstream's fix (huggingface/transformers 676559d5, "[LightGlue] Remove remote code
  execution"), after the hashed install.
- After the patch the keypoint detector config always comes from `CONFIG_MAPPING`
  (an unknown type raises `KeyError`), and the detector model never gets
  `trust_remote_code`. LightGlue is a keypoint matcher, so nothing the step trains
  uses it.
- Like upstream's fix, it edits three files: `configuration_lightglue.py` and
  `modeling_lightglue.py` (the code transformers runs), and, from `-4`,
  `modular_lightglue.py`. That is the source the other two are generated from;
  transformers never imports it, but the wheel ships it with the same two
  pass-throughs.
- The patch fails the build if its target lines change.
- `test/test_lightglue_patch.py` runs the exploit against a copy of the real 4.55.4:
  the payload runs unpatched and not patched. It also checks that no LightGlue
  source file in the patched copy keeps a pass-through.
- Scanners match on the package version, so trivy and quay still report
  CVE-2026-5241 against transformers 4.55.4. That finding is patched, not open.

`fms-v3.2.2-3` (manifest
`sha256:fc696a40d415559b3752981b169ac1dbf8f01fc192f2e8dd44f55f7e3e04d414`, published
2026-11-06) adds only that patch layer, on the same lock, base and `src/` as `-2`.
- The patched lines were checked inside the built image: neither `trust_remote_code`
  pass-through remains.
- trivy 0.76.2 reports the same 0 CRITICAL / 7 HIGH, because it matches on the
  version.
- quay rates `-3` 1 Critical / 20 High (114 Medium, 29 Low, 2 Negligible), the same as
  `-1` and `-2`. The Critical is PYSEC-2026-2290: a version-matched false positive,
  recorded below.
- `-1` and `-2` carried the unpatched code. Both tags were **deleted from quay** on
  2026-11-06.
- `-3` left `modular_lightglue.py` unpatched. transformers never imports that file,
  so `-3` is not exploitable through it, but `-4` patches it too. The `-3` tag was
  **deleted from quay** on 2026-11-10; `-4` is the only published tag.

`fms-v3.2.2-4` (manifest
`sha256:2d4e803ece50d4f6bb62570035eb7715857cbb66280e086af111e3b9e48165cd`, published
2026-11-10) is on the same lock and base as `-3`. Two changes:
- the backport also patches `modular_lightglue.py`, as upstream's fix does;
- a lint-only change to `src/fms_hf_tuning_job.py` (a mypy `cast` and a wrapped
  line), with the same behaviour.

It was checked inside the built image:
- no LightGlue source file keeps a `trust_remote_code` pass-through;
- there are no `.pyc` files under `/opt/fms-hf-tuning`;
- it has fms-hf-tuning 3.2.2 and transformers 4.55.4;
- every `src/` file was byte-identical to the source at publish time (later
  comment-only edits to `src/fms_hf_tuning_run.sh` are not in the image; see
  [Bumping IMAGE_VERSION](#bumping-image_version)).

Scan results:
- trivy 0.76.2: 0 CRITICAL / 7 HIGH, the same packages and CVEs as `-3`.
- quay: 1 Critical / 20 High (89 Medium, 29 Low, 2 Negligible). That is the same as
  a rescan of `-3` on the same day; Clair's database has changed since
  2026-11-06, which is why Medium is lower than the 114 above.

The build tests ran on `-3` (results below); `-4` trains the same way.

#### Reviewed false positive: PYSEC-2026-2290 / CVE-2026-5241 in `fms-v3.2.2-3` and `-4`

quay's Critical on `fms-v3.2.2-3` and `fms-v3.2.2-4` is a reviewed false positive.
Keep this record until fms-hf-tuning allows transformers 5
(foundation-model-stack/fms-hf-tuning#687) and the image moves to a fixed release.

| | |
|---|---|
| Finding | PYSEC-2026-2290, alias CVE-2026-5241 / GHSA-fgcw-684q-jj6r. Critical in quay (Clair), High in trivy |
| Package | transformers 4.55.4 (installed from the hashed lock; fms-hf-tuning 3.2.2 caps `transformers<=4.55.4`) |
| Why it is reported | Clair and trivy match on the installed package's name and version. The patch edits source files in that package and leaves its version metadata at 4.55.4, so every advisory for 4.55.4 is still listed |
| Vulnerable code | `models/lightglue/configuration_lightglue.py`: the nested `AutoConfig.from_pretrained(..., trust_remote_code=self.trust_remote_code)` for a keypoint detector type outside `CONFIG_MAPPING`; `models/lightglue/modeling_lightglue.py`: `from_config(..., trust_remote_code=config.trust_remote_code)`. `models/lightglue/modular_lightglue.py` holds both lines too; it is never imported (transformers' lazy loader skips `modular_*` files) |
| Remediation | `patches/lightglue_no_remote_code.py`, a backport of huggingface/transformers 676559d5 ("[LightGlue] Remove remote code execution", #45122), applied by the Dockerfile right after the hashed install. It patches the two runtime files and, from `-4`, `modular_lightglue.py`, as upstream does. It fails the image build if any target is not found exactly once |
| Evidence | (1) Inside the published images, no `trust_remote_code` pass-through remains in the two runtime files (`-3`), or in any LightGlue source file (`-4`). (2) `test/test_lightglue_patch.py` loads a crafted LightGlue repo the way fms-hf-tuning loads `base_model` (`AutoModelForCausalLM.from_pretrained`, no `trust_remote_code`), against a copy of the real 4.55.4: the repo's Python runs unpatched, and does not run patched |
| Scope | Only this advisory. The other transformers, torch and fsspec findings are real and unpatched, held back by fms-hf-tuning's caps (the tables above) |
| Re-check on | any change to `requirements.lock`, the base image, or the patch. A transformers bump that moves the target lines fails the build. A move to transformers 5.x removes the patch and this record |
| Reviewed | 2026-11-06, for `fms-v3.2.2-3` (`sha256:fc696a40d415559b3752981b169ac1dbf8f01fc192f2e8dd44f55f7e3e04d414`); 2026-11-10, for `fms-v3.2.2-4` (`sha256:2d4e803ece50d4f6bb62570035eb7715857cbb66280e086af111e3b9e48165cd`) |

### Bumping IMAGE_VERSION

`IMAGE_VERSION` is stable (`fms-v3.2.2-4`), not the git SHA: BlueVela caches a
`.sqsh` per tag. Bump the suffix when the Dockerfile, `requirements.lock` or `src/`
changes; the prefix with the fms-hf-tuning pin (for example `fms-v3.3.0-1`).
A change to comments only needs no bump: the published image keeps the old comment
text and the step runs the same (hf-peft-sft's rule, #626). The review fixes after
`-4` changed only comments in `src/fms_hf_tuning_run.sh` and the Dockerfile, so the
image stays `-4`. A local `make image` then no longer matches the published tag,
and under podman `make test` warns until the next bump; `TEST_PUBLISHED=true` skips
that comparison. After
editing `requirements.in`, run `make lock` and commit both files. On an fms-hf-tuning
bump, re-run the CPU venv tests first: they catch parser and `train()` drift.

## Publishing the image

The image is `quay.io/granitebuild/steps/train/fms-hf-tuning/skypilot:<IMAGE_VERSION>`
(public). Log in (`podman login quay.io/granitebuild`, or `REGISTRY_USER` /
`REGISTRY_PASSWORD`; set `QUAY_API_TOKEN` so the first push creates the repository
public), then:

```bash
make -C steps/train/fms-hf-tuning/skypilot image publish-image
sh steps/scripts/registry-tag-status.sh quay.io/granitebuild/steps/train/fms-hf-tuning/skypilot:fms-v3.2.2-4
```

`present` means it pulls anonymously. `make publish-step` refuses until the image
exists. Run `make check-published` after any change to the template, `src/`,
`USAGE.md` or a build test.

## BlueVela image cache

The `bluevela` environment's `slurm_image_cache` (#560) imports the registry image
once into `/proj/granite-build/g4os/enroot/<image>.sqsh` in a CPU job; later runs
start from it. A new `IMAGE_VERSION` is imported on its first run.

## How ClearML is wired

fms-hf-tuning has its own ClearML tracker (`trackers: [clearml]`, with
`clearml_project` / `clearml_task`), which adds transformers' `ClearMLCallback` and
exports the run URI. When ClearML is on, the job gets those three keys and
`report_to` stays `none`, so the callback is not added twice. The run script
exports `CLEARML_LOG_MODEL=${CLEARML_LOG_MODEL:-false}` (the callback would
otherwise upload every save). `auto` is decided on the node: on iff
`CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` are all set.
The image has clearml 2.1.12, not the 2.0.0 upstream's `[clearml]` extra pins; the
BlueVela build test runs the tracker with `CLEARML_OFFLINE_MODE=1`. If 2.1.12 breaks
it, pin 2.0.0 in `requirements.in`.

## Tests

The unit tests at `test/`'s root need no cluster, GPU or image:

| Test | Pins |
|---|---|
| `test_skypilot_train_fms_hf_tuning_render.py` | the template under gbserver's strict renderer: identity, I/O schema, defaults, the monitor rule, identical macro copies, `owned` == the job script's `OWNED`, the `docker:` prefix, every `raise_error`, every owned key refused, and the guard/run argv (base64 round trip, ClearML mode, data paths) |
| `test_fms_hf_tuning_job.py` | the job script without fms-hf-tuning: value conversion and refusals on an argparse stand-in, the merge (owned keys, defaults, data_config, ClearML), output checks (adapter, single-file and sharded model, 0 tensors), the base-id rewrite, the CLI |
| `test_fms_hf_tuning_job_upstream.py` | the same against the real parser: `EXTRA_ARGS` and `OWNED` match upstream, DiGiT-style and aLoRA configs pass, typos/bad values fail, every shipped `args` map passes (skipped without `tuning`) |
| `test_fms_hf_tuning_entry.py` | `step_line`, the `train` wrapper, rank-0 printing, upstream's `additional_callbacks`, and a 2-step CPU run (full + LoRA) through the wrapper (the last three need `tuning`) |
| `test_fms_hf_tuning_guard_sh.py`, `test_fms_hf_tuning_setup_sh.py` | staged-path checks; the image check (files present, stack imported, `nvidia-smi` optional) |
| `test_accelerate_configs.py`, `test_fms_hf_tuning_run_sh.py` | the three configs (accelerate parses them: needs `accelerate.commands`); the run script with stubs: launch argv per mode and GPU count, #604, the job, ClearML, output safety, the swap, failures and markers |
| `test_lightglue_patch.py` | the CVE-2026-5241 backport: every target replaced, a missing or duplicated target refused, applied after the hashed install; and (CPU venv) a crafted LightGlue repo runs its payload against a copy of real transformers 4.55.4 unpatched, and not once patched |
| `test_dockerfile.py`, `test_docs.py`, `test_exemplar_build.py` | the image recipe; USAGE.md vs the template; every shipped build.yaml vs the contract, and every shipped target rendered through the template's `setup:`/`run:` blocks under gbserver's strict renderer (after the inner fill of `{% raw %}` and bindings, which is checked too) |

Run them with `-m "not extended"` and AWS credentials unset:

```bash
cd steps/train/fms-hf-tuning/skypilot
env -u AWS_PROFILE -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
  GB_ENVIRONMENT=STANDALONE PYTHONPATH=src ../../../../.venv/bin/python -m pytest test -m "not extended"
```

### The CPU venv

The upstream-dependent tests need fms-hf-tuning, which the repo venv does not have:

```bash
uvx --from uv==0.13.6 uv venv -q --python 3.12 /tmp/fms-cpu-venv
VIRTUAL_ENV=/tmp/fms-cpu-venv uvx --from uv==0.13.6 uv pip install fms-hf-tuning==3.2.2 clearml==2.1.12 pytest
PYTHONPATH=src /tmp/fms-cpu-venv/bin/python -m pytest -c /dev/null --noconftest \
  test/test_fms_hf_tuning_job.py test/test_fms_hf_tuning_job_upstream.py \
  test/test_fms_hf_tuning_entry.py test/test_accelerate_configs.py test/test_lightglue_patch.py
```

(`--noconftest`: `steps/conftest.py` loads the repo-wide conftest, which needs
pydantic.) The entry test downloads `trl-internal-testing/tiny-LlamaForCausalLM-3.2`.

### Build tests

Both are `extended`, train 3 steps per target on real GPUs, need the image
published, and need `PYTEST_ADDOPTS=-s` (two launches in one pytest process).

| Cluster | Targets (peft_method / distributed) | Proves |
|---|---|---|
| `test/aws` (A10G:1) | `full` (none / ddp), `alora` (alora / ddp) | image pull, inline `hf download`, full fine-tuning with flash-attn on Ampere, aLoRA, the chat handler on HF-id data |
| `test/slurm-bluevela` (H200:2, `ibm`) | `lora-fsdp` (lora / fsdp, bound `dataset`), `full-fsdp` (none / fsdp, `data_config`) | Pyxis + `.sqsh`, FSDP1 saving a non-empty adapter, `data_config` reading a bound input, ClearML offline |

**BlueVela result** (build `718527f5`, `fms-v3.2.2-3`, 2026-11-09): passed in 25m52s.
- The image cache imported the image into a `.sqsh` in 450 s on the first run.
- `lora-fsdp`: loss 3.30 → 1.65 over 3 steps. The adapter has 422 tensors (210 `lora_A`
  + 210 `lora_B`, plus the resized `lm_head`/`embed_tokens`), with `r=8`, and the base
  id is rewritten. So peft 0.19.1 is not affected by peft#3876 under FSDP1.
- `full-fsdp` (from `data_config`): loss 3.29 → 1.65; model registered.
- Both reported `Step:` progress as workload status. Metadata: `peft_method`,
  `distributed=fsdp`, `fms_hf_tuning_version=3.2.2`, `transformers_version=4.55.4`.
- flash-attn and pyarrow import on the GPU node; the Mac cannot check that under qemu.

**aws result** (`fms-v3.2.2-3`, 2026-11-09): passed in 6m45s, with `full` (none / ddp)
and `alora` (alora / ddp), each on one A10G (g5.xlarge).
- Both trained 3 steps (loss to about 1.62) with flash-attn on Ampere and registered
  `model`. Metadata: `distributed=ddp`, `fms_hf_tuning_version=3.2.2`,
  `transformers_version=4.55.4`.
- SkyPilot hit `InsufficientInstanceCapacity` for g5.xlarge in one zone and failed
  over to another by itself.
- No EC2 instances were left running.

The first attempt (build `00e72069`) failed on a bad node: `srun ... Communication
connection failure` on p1-r14-n3 while provisioning `lora-fsdp`. gbserver treats that
as non-transient and does not retry, and it left the half-provisioned SkyPilot cluster
UP (removed with `sky down`). If a BlueVela run fails that way, rerun it, and run
`sky status` afterwards.

```bash
# aws (costs money)
AWS_PROFILE=gb-skypilot PYTEST_ADDOPTS=-s make -C steps/train/fms-hf-tuning/skypilot test TESTS=test/aws
# BlueVela (needs ~/.ssh/ibm-bluevela.key and the BV_SSH_PRIVATE_KEY space secret)
env -u AWS_PROFILE PYTEST_ADDOPTS=-s make -C steps/train/fms-hf-tuning/skypilot test TESTS=test/slurm-bluevela
```

## Follow-ups

QLoRA and the fms-acceleration plugins; native K8s (moving the Helm
`space://steps/tuning` into the framework); SkyPilot-K8s and LSF; multi-node; a
checkpoints output; `lora_post_process_for_vllm`; DeepSpeed; prompt tuning (`pt`).
