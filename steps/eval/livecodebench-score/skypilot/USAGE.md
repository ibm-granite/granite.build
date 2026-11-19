# eval/livecodebench-score (SkyPilot)

Scores [LiveCodeBench](https://livecodebench.github.io/) ([code](https://github.com/LiveCodeBench/LiveCodeBench)) code
generation completions, as written by `eval/livecodebench-generate`, and registers
the result files as the step's `results` artifact. CPU only.

It runs LiveCodeBench's own scorer, unchanged, after checking that the completions
were generated for the same release, date window and problem set. The step runs in
a public container image (`python:3.12-slim` by default) and installs a pinned
LiveCodeBench commit there at setup time, with its Python packages pinned to that
commit's lock file.

## Scoring runs model-generated code

LiveCodeBench executes every completion against the problem's tests. Its checker
runs each one in a subprocess with time and memory limits, and disables some
`os`, `shutil` and `subprocess` functions first. **LiveCodeBench's own documentation
says that guard is not a security sandbox.**

This step adds the following:

- **It runs separately** from the model, in its own job, with no GPU.
- **It runs in a sandbox.** With `sandbox: true` (the default) the scorer, and so
  every completion, runs under a [Landlock](https://docs.kernel.org/userspace-api/landlock.html)
  ruleset (`src/lcb_sandbox.py`). Landlock is Linux's unprivileged access control:
  the scorer restricts itself before it starts, every process it starts inherits the
  restrictions, and none can lift them. It needs no root and no user namespaces, so
  it works in unprivileged containers too.
  - **No other process's environment.** Every process that started the scorer (the
    cluster's job launcher, the step's shell) runs as the job's user and still holds
    the job's original environment, credentials included, in `/proc/<pid>/environ`.
    A sandboxed process can't trace a process outside its sandbox, and reading that
    file counts as tracing, so none of them can be read, whatever wraps the step.
  - **Only the step's own files.** The scorer can read the system directories
    (`/usr`, `/lib`, `/etc` and the like), its Python, the LiveCodeBench checkout,
    `src/` and the two inputs, and write only the output directory, its own `HOME`
    and HF cache under the working directory, and its own `TMPDIR`. The job user's
    real home (keys, `~/.netrc`, tokens), the rest of the shared `/tmp` and other
    builds' files can't be opened.
  - **No TCP.** Binding and connecting TCP sockets are denied.
  - **No signals to, or abstract unix sockets of, processes outside the sandbox**
    (Landlock ABI 6 or later).

  It needs Linux with Landlock enabled (`landlock` in `/sys/kernel/security/lsm`),
  ABI 4 or later: Linux 6.7, or a distribution kernel with it backported (RHEL 9.7's
  5.14 kernels report ABI 6). Where the kernel can't apply it, the step **fails**
  rather than run the code without it. `sandbox: false` runs without it, with a
  warning in the log; use it only for completions you trust.
- **Its environment is an allowlist.** It declares no secrets and sets no `envs`, and
  `run` starts the scorer under `env -i` and an allowlist: `PATH`, an empty `HOME`,
  a private `TMPDIR`, the locale if the job set one, `PYTHONPATH`,
  `HF_HOME`, the two offline switches, the venv's `VIRTUAL_ENV`, the dynamic loader's
  and module system's variables (`LD_LIBRARY_PATH` and similar), and the thread-count
  variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `SLURM_CPUS_PER_TASK`,
  `SLURM_CPUS_ON_NODE`). If the job environment carried `HF_TOKEN` or
  `HUGGING_FACE_HUB_TOKEN`, the log says so. Its inputs arrive through separate
  download steps (hfpull) before it starts.
- **Its `HOME` is empty.** Generated code that reads `~/.ssh`,
  `~/.cache/huggingface` or a dotfile finds nothing.
- **It runs offline.** `HF_HUB_OFFLINE` and `HF_DATASETS_OFFLINE` are set.

**What the sandbox doesn't cover.** UDP and pathname unix sockets aren't restricted,
so generated code could still send a DNS query or a UDP packet. It can read the
system directories, which may hold site configuration, and other processes'
`/proc/<pid>/cmdline` (not their environments). It runs as the job's user, so a
kernel bug in Landlock itself would hand it that user's access. Before scoring a
model you don't trust, also run this step as a user with no credentials of its own
on disk.

### Known advisories in the scorer's packages

The step pins its scorer's packages, and moves them past LiveCodeBench's own lock for
security fixes (see the comment in `src/requirements-score.txt`), and torch
(`torch_package`) is pinned past its advisories. Two can't move:

- **`datasets` 3.5.0** (CVE-2026-66007, fixed only in 5.0.1). LiveCodeBench ships its
  benchmark as a loading-script dataset, and `datasets` 4.0 removed support for loading
  scripts, so the scorer can't load the benchmark on any fixed release. The advisory is
  a path traversal through `file_name` metadata in folder-based builders. This step
  reads only the pinned LiveCodeBench dataset, offline, through its script, and never
  saves a dataset to disk or pushes one to the Hub.
- **`fsspec` 2024.12.0** (GHSA-27vj-qcqg-25rc, fixed in 2026.6.0), capped at
  `<=2024.12.0` by that `datasets` release. The advisory is code execution through a
  Jinja2 template in a `ReferenceFileSystem` (`reference://`) reference. This step
  never opens a reference filesystem: the dataset is read from local files.

Removing the scorer's use of `datasets`, by reading the dataset's JSONL directly, would
clear both.

## Referencing the step

```yaml
steps:
  - step_uri: space://steps/eval/livecodebench-score
```

## Inputs and outputs

### Inputs

| Input | Required | Type | Typical source |
|---|---|---|---|
| `completions` | yes | dataset | A `binding` to the `completions` output of an `eval/livecodebench-generate` target, or a `uri` to where that output was pushed (`hf:///datasets/<org>/<name>`) |
| `dataset` | yes | dataset | `uri: hf:///datasets/livecodebench/code_generation_lite`: the benchmark's problems and hidden tests. The SkyPilot launcher downloads it before setup. |

The step reads them as `{{ bindings.completions.binding.path }}` and
`{{ bindings.dataset.binding.path }}`. A target that doesn't bind both fails build
validation before anything is queued.

**The dataset is large:** `release_v6` uses six JSONL files, 4.2 GB in all. The date
window only narrows what is scored; the whole release is still downloaded.

### Outputs

| Output | Type | What it is |
|---|---|---|
| `results` | `dataset` | The `output_path` directory: `summary.json`, plus LiveCodeBench's own `completions_codegeneration_output_eval.json` (pass@1 and per-problem detail), `…_eval_all.json` (one record per problem, with `graded_list` and `metadata` for each sample) and `…_output.json`, and a copy of the scored `completions.json` and `generation_meta.json`. |

`summary.json` has pass@1 overall and by difficulty, the problem count, the window,
and the generation settings recorded by the generate step.

## Config contract (`lcb_score_config`)

| Field | Type | Required | Purpose |
|---|---|---|---|
| `release_version` | string | **required** | LiveCodeBench release, e.g. `release_v6`. Must equal the generate step's. |
| `start_date`, `end_date` | string | optional | Date window (`YYYY-MM-DD`, inclusive) on the problems' contest dates. Must equal the generate step's. Empty means no bound on that side. |
| `timeout` | int | optional (default `6`) | Seconds per test case (LiveCodeBench's `--timeout`). |
| `num_process_evaluate` | int | optional (default `0`) | Scorer processes. `0` keeps LiveCodeBench's default; set it to the CPUs the build requests. |
| `output_path` | string | optional (default `output`) | Where the result files are written. Relative paths resolve in the step's working directory. |
| `image` | string | optional (default `python:3.12-slim`) | Public container image, rendered as SkyPilot `docker:<image>`. Debian/Ubuntu-based, run as root, with `python3` and `pip` (setup installs `git` with `apt-get` if missing). `""` runs on the bare launcher node: use that on a SLURM cluster without the Pyxis plugin. |
| `install` | bool | optional (default `true`) | Install LiveCodeBench and its packages into a venv at setup, on a Python 3.12 that uv downloads into the workdir (not the node's own `python3`, which only runs pip to install uv). Set `false` only when `image` already provides them and a checkout at `lcb_checkout`. |
| `lcb_commit` | string | optional | LiveCodeBench commit to install (it isn't on PyPI). Its packages are pinned in the step's bundled `src/requirements-score.txt` to that commit's lock; a different commit needs that file regenerated. |
| `lcb_repo` | string | optional | Where to fetch LiveCodeBench from. |
| `lcb_checkout` | string | optional (default `LiveCodeBench`) | Where the checkout goes. LiveCodeBench reads files by relative path, so the scorer runs from inside it. |
| `torch_package`, `torch_index_url` | string | optional (default `torch==2.15.0`, CPU index) | LiveCodeBench imports torch even to score. Installed first, from the CPU index by default. |
| `sandbox` | bool | optional (default `true`) | Run the scorer under a Landlock sandbox (see above). The step fails if the kernel can't apply it. `false` runs without it, with a warning: only for completions you trust. |
| `pip_index_url` | string | optional | Package index for everything else. |
| `uv_version` | string | optional (default `0.13.6`) | uv release setup installs from `pip_index_url` and builds the venv with. Must not be empty. |

## Example build.yaml

Generate and score one contest day (6 problems) of `release_v6`. The generate
target is described in `eval/livecodebench-generate`'s own README; its `completions`
output feeds this step's `completions` input by binding.

```yaml
granite.build:
  name: livecodebench-example
  version: 0.0.1
  targets:
    generate:
      environment_uri: space://environments/skypilot/aws
      inputs:
        model:
          uri: hf:///models/ibm-granite/granite-4.2-3b
        dataset:
          uri: hf:///datasets/livecodebench/code_generation_lite
      outputs:
        completions:
          uri: "hf://huggingface.co/datasets/<your-org>/lcb-completions_{{ binding.path | short_hash }}"
          type: dataset
      steps:
        - step_uri: space://steps/eval/livecodebench-generate
          config:
            launcher_config:
              resources: {accelerators: "A10G:1", cpus: 8, memory: 32}
            lcb_generate_config:
              release_version: release_v6
              start_date: "2025-04-05"
              end_date: "2025-04-05"
              max_tokens: 8192
              enable_thinking: "false"
    score:
      environment_uri: space://environments/skypilot/aws
      inputs:
        completions:
          binding: generate.completions
        dataset:
          uri: hf:///datasets/livecodebench/code_generation_lite
      outputs:
        results:
          uri: "hf://huggingface.co/datasets/<your-org>/lcb-results_{{ binding.path | short_hash }}"
          type: dataset
      steps:
        - step_uri: space://steps/eval/livecodebench-score
          config:
            launcher_config:
              resources: {cpus: 4, memory: 8}
            lcb_score_config:
              release_version: release_v6
              start_date: "2025-04-05"
              end_date: "2025-04-05"
              num_process_evaluate: 4
```

## Notes and limitations

- **Memory.** The step builds the dataset's local cache with one row per write,
  because single rows reach 93 MB. Scoring a small window then needs well under
  1 GB. Scoring all of `release_v6` decodes every problem's tests, which needs
  several GB; size `memory` for it.
- **Only the `codegeneration` scenario** is scored. LiveCodeBench's other scenarios
  (self-repair, test output prediction, code execution) need different inputs.
- **Scores can vary slightly between machines.** LiveCodeBench's tests are timed,
  so a slower node can turn a pass into a timeout. Its authors report a spread of
  under half a point; if it is larger, lower `num_process_evaluate` or raise
  `timeout`.
- **The dataset is loaded with `trust_remote_code`:** it ships a loading script.
