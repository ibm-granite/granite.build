# Contributing to AutoTune

Thank you for your interest in contributing to AutoTune (the `fm-tune` repository).
This guide will help you [get started](#getting-started) with developing and
contributing to the project.

## Contribution Pathways

There are several ways to contribute to AutoTune:

### 1. Contributing to This Repository

- New tuning algorithms (PEFT methods, online/offline RL variants)
- New training drivers or distributed-training backends
- New search algorithms for Ray Tune integration
- Bug fixes, performance improvements, refactoring
- Documentation, examples, configuration recipes
- Test coverage and CI improvements

**Process:** see the [Pull Request Process](#pull-request-process) section below.

### 2. Reward Functions and Configuration Recipes

- Reward functions for online RL (see `autotune/rewards/`)
- New or tuned YAML configuration recipes (see `autotune/configs/`)

These can live in your own repository if they target a specific deployment, or be
contributed back if they have general utility.

### 3. Bug Reports and Feature Requests

Open an issue at
[github.com/ibm-granite/granite.build/issues](https://github.com/ibm-granite/granite.build/issues)
with a minimal reproduction (config snippet, command line, error trace). For
feature requests, describe the use case before proposing an implementation —
this helps avoid wasted work if the feature overlaps with something already in
flight.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. To report unacceptable
behavior, contact one of the maintainers listed in
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md#enforcement).

To report a security vulnerability, follow [`SECURITY.md`](SECURITY.md) — do not open a public issue.

## Getting Started

### Prerequisites

- **Python 3.10+** (3.12 recommended)
- **NVIDIA GPU(s) with a CUDA 13 driver (>= 580)** for GPU training: torch
  2.11's PyPI wheels are built for CUDA 13.0
- **Linux** for training (flash-attn is Linux-only); macOS works for editing,
  config validation, and unit tests that don't require GPU
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) recommended,
  or `pip` + `conda`/`mamba`

### Installation with `uv` (recommended)

```bash
# Fork the granite.build monorepo on GitHub, then clone your fork:
git clone git@github.com:<your-username>/granite.build.git
cd granite.build/autotunex/src/fm-tune

# Create and activate a virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Editable install with full extras (verl, vLLM, flash-attn) and dev tooling
uv pip install -e ".[full,dev]"

# Install pre-commit hooks
pre-commit install
```

There is no prebuilt flash-attn wheel for torch 2.11 + CUDA 13, so the `full`
extra **compiles flash-attn and the Mamba kernels (`mamba-ssm`, `causal-conv1d`)
from source**, configured by `[tool.uv.extra-build-dependencies]` /
`[tool.uv.extra-build-variables]` in `pyproject.toml`. Run it on a GPU node or
any host with nvcc 13 (~15 minutes cold; cached by uv afterwards). On macOS or a
CPU-only box, install `".[core,dev]"` instead. See
[Installation in `README.md`](README.md#installation) for details.

### Installation with `pip` and `conda`

```bash
conda create -n autotune python=3.12 -y
conda activate autotune
# Plain pip ignores the [tool.uv] build settings: pre-build flash-attn and the
# Mamba kernels with --no-build-isolation first, exactly as in
# README.md "With pip and conda", then:
pip install -e ".[full,dev]"
pre-commit install
```

### Verify Installation

```bash
# Lint passes
ruff check .

# Unit tests (skip slow + GPU markers; runs in <1 minute on CPU)
pytest -m "not slow and not gpu"
```

For a smoke test of the training pipeline, see the **Quick Start** section in
[`README.md`](README.md).

## Repository Layout

| Path | Contents |
|------|----------|
| `main.py` | CLI entry point. Parses args, starts Ray, runs `AutotuneOptimizer`. |
| `autotune/optimizer.py` | `AutotuneOptimizer` — orchestrates HPO via Ray Tune. |
| `autotune/config.py` | `AutotuneConfig` — loads YAML configs. |
| `autotune/pipeline.py` | `AutotunePipeline` — validates `tuning_algo` × `rl_algo` combos. |
| `autotune/constants.py` | Algorithm registries, PEFT type mappings. |
| `autotune/utils.py` | Data loading, tokenization, memory model, FSDP/DS strategy estimators. |
| `autotune/trainers/` | Training drivers (single/multi-GPU, HF/TRL/verl backends). |
| `autotune/configs/` | YAML config templates. |
| `autotune/rewards/` | Reward functions for online RL. |
| `autotune/tools/` | Dataset builders (GSM8K, factuality) and parquet/JSON conversion helpers. |
| `autotune/callbacks/` | Ray Tune callbacks, buffered logging service. |
| `autotune/lsf/` | Optional multi-node Ray launcher for LSF/HPC clusters (`--scheduler lsf`, blaunch). |
| `autotune/slurm/` | Optional multi-node Ray launcher for Slurm clusters (`--scheduler slurm`, srun); reuses the scheduler-agnostic pieces of `autotune/lsf/`. |
| `tests/` | Pytest suite. |
| `docs/` | Reference documentation: GPU sizing, MPS/MLX, dataset formats. |

For deeper architectural context — driver selection, the verl integration,
memory levers, and known gotchas — see [`CLAUDE.md`](CLAUDE.md) and the files
under [`docs/`](docs/).

## Coding Standards

### Type Annotations

Type annotations are **required** on public functions and on driver entry points
(`train_driver_*_gpu`). They're encouraged elsewhere when they add clarity.

```python
def fit(
    self,
    train_dataset: "Dataset",
    eval_dataset: "Dataset" | None = None,
) -> "tune.ResultGrid": ...
```

### Docstrings

Use Google-style docstrings on public functions, classes, and any non-trivial
helper. Keep them factual and short — readers are skimming, not reading.

```python
def estimate_fsdp_strategy(
    model_name_or_path: str,
    max_seq_length: int,
    per_device_batch_size: int,
    num_gpus: int,
    peft_config: dict | None = None,
    gpu_memory_gb: int = 75,
) -> str:
    """Estimate the fastest FSDP sharding strategy that fits in GPU memory.

    Tries strategies from least sharding (fastest) to most sharding:
    no_shard (DDP) → shard_grad_op → full_shard.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        max_seq_length: Maximum sequence length for training.
        per_device_batch_size: Batch size per GPU.
        num_gpus: Number of A100 80GB GPUs allocated.
        peft_config: Optional dict with PEFT params (e.g. {"r": 128}). If None,
            assumes full fine-tuning.
        gpu_memory_gb: Per-GPU memory in GB (default 75 to leave 5 GB headroom).

    Returns:
        One of "no_shard", "shard_grad_op", "full_shard".

    Raises:
        ValueError: If the model doesn't fit even with full_shard.
    """
```

Skip docstrings on trivial private helpers — well-named code beats redundant prose.

### Code Style

- **Ruff** for linting and formatting (`E`, `F`, `W`, `I` rule sets).
- Line length: 120.
- Quotes: double (`ruff format` enforces).
- Match existing patterns in the file you're editing — when adding to a driver,
  follow the conventions already in that driver.
- **Don't add abstractions or features beyond what the task requires.** Bug fixes
  shouldn't include surrounding cleanup; one-shot operations don't need a helper.

### Formatting and Linting

```bash
# Lint
ruff check .

# Auto-fix what can be auto-fixed
ruff check --fix .

# Format
ruff format .
```

Pre-commit runs `ruff check --fix` and `ruff format` on staged files before each
commit (see `.pre-commit-config.yaml`). Bypass with `git commit --no-verify`
only for genuine intermediate commits.

## Development Workflow

### Branching

Branch off `main` in your fork. Common naming patterns:

- `iss<n>` — fixing or implementing issue #n
- `feat-<short-name>` — new feature
- `fix-<short-name>` — bug fix
- `docs-<short-name>` — documentation only

### Commit Messages

Follow [Angular commit format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit):

```
<type>: <subject>

<body>

<footer>
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`.

**Example:**

```
feat: add hybrid_shard support to FSDP driver

Wires hybrid_shard into estimate_fsdp_strategy and the multi-GPU TRL
driver. Shards within node, replicates across nodes — lower all-gather
overhead than full_shard for multi-node runs.

Closes #186
```

Keep the subject under 70 characters. The body explains *why*, not *what* — the
diff already shows what.

### Developer Certificate of Origin (DCO)

This project uses the [Developer Certificate of Origin](https://developercertificate.org/)
to certify that contributors have the right to submit their work under the
project's Apache-2.0 license. By signing off on a commit you agree to the terms
of the DCO.

**Sign off every commit** with `-s` (or `--signoff`):

```bash
git commit -s -m "feat: your commit message"
```

This appends a `Signed-off-by` trailer using your `user.name` and `user.email`
from git config. Use your real name and a reachable email.

To retroactively sign existing commits on a branch:

```bash
git rebase --signoff <base>
git push --force-with-lease
```

### AI Coding Assistants

AI-assisted development is welcome. **You are responsible for reviewing and
understanding every change before submitting** — the assistant is a tool, not a
co-author with judgment.

When AI tooling materially shaped a commit, add an `Assisted-by:` trailer:

```text
Assisted-by: Claude Code
Assisted-by: GitHub Copilot
```

Use the tool's common name. One trailer per tool. This is for transparency, not
attribution.

### Pull Request Process

1. **Open or pick up an issue** describing what you're changing. For non-trivial
   work, discuss the approach in the issue before writing code.
2. **Branch off `main`** in your fork.
3. **Make your changes** following the coding standards above.
4. **Add or update tests.** New drivers, new search algorithms, and new
   utilities under `autotune/utils.py` should ship with unit tests. If a fix
   addresses a bug, add a regression test that fails before the fix.
5. **Update documentation** as needed — `README.md`, `CLAUDE.md`, `docs/`, or
   the relevant config YAML's inline comments.
6. **Run lint and tests locally**:
   ```bash
   ruff check .
   pytest -m "not slow and not gpu"
   ```
7. **Push and open a PR** against `main`. Reference the issue in the PR body.
8. **Respond to review.** Squash or fix-up commits are fine; rebases are
   preferred over merge commits when resolving conflicts with `main`.

## Testing

### Quick Reference

```bash
# Fast tests only (no model downloads, no GPU; ~30 s)
pytest -m "not slow and not gpu"

# Tests that download HF model weights (>50 MB)
pytest -m slow

# Tests that need a CUDA GPU
pytest -m gpu

# Single test file
pytest tests/test_blds_fidelity_schedule.py

# Single test function
pytest tests/test_config.py::TestLoadFromYaml::test_load_yaml

# Stop on first failure with verbose output
pytest -x -v
```

### Test Markers

Defined in `pyproject.toml` under `[tool.pytest.ini_options]`:

| Marker | Meaning |
|--------|---------|
| `slow` | Downloads or loads HF model weights (>50 MB). Skip in fast iterations. |
| `gpu`  | Requires CUDA. Skipped on CPU-only machines. |

Use markers when adding tests:

```python
import pytest


@pytest.mark.slow
def test_tokenization_with_real_model(): ...


@pytest.mark.gpu
def test_fsdp_init(): ...
```

### Writing Tests

- Tests live in `tests/` and follow the `test_*.py` pattern.
- Use the fixtures under `tests/fixtures/` rather than reinventing them.
- Don't write tests that require launching a Ray cluster unless absolutely
  necessary — they're slow, flaky in CI, and hard to debug. Prefer testing the
  pure-Python pieces (config loading, search-space construction, the memory
  model, dataset utilities) directly.
- Tests that exercise GPU code paths must be marked `gpu`. Tests that pull
  weights from HuggingFace must be marked `slow`.

### Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:

- **Ruff** with `--fix` (linting + import sorting)
- **Ruff-format** (formatting)

Install once per clone:

```bash
pre-commit install
```

Run manually across the whole repo:

```bash
pre-commit run --all-files
```

> **Note:** `pre-commit run --all-files` re-runs the hooks on every tracked
> file. On a fresh clone or after dependency upgrades this can take a minute.
> Don't cancel mid-run.

## Common Issues & Troubleshooting

| Problem | Fix |
|---------|-----|
| `flash-attn` install fails | macOS isn't supported. On Linux it is compiled from source (no prebuilt wheel exists for torch 2.11 + CUDA 13): run `uv pip install -e ".[full]"` on a host with nvcc 13 (`/usr/local/cuda` or `$CUDA_HOME`), which builds it against the env's torch via `[tool.uv.extra-build-dependencies]`. With plain pip, install torch first and then `pip install --no-build-isolation flash-attn==2.8.3` (see README). `undefined symbol` on import means it was built against a different torch. |
| `ImportError: verl` | Install the `full` extra: `pip install -e ".[full]"`. The `core` extra omits verl. |
| `CUDA-capable device(s) is/are busy or unavailable` in a verl (PPO/GRPO/DAPO) job | vLLM runs in its own processes next to the FSDP workers, which `exclusive_process` compute mode forbids. On LSF request `-gpu "num=N:mode=shared:j_exclusive=yes"` (`j_exclusive=yes` keeps other jobs off the GPUs, which `mode=shared` alone does not). |
| RL run trains but logs nothing after startup (no `[AutoTune] verl step N` lines) | A vLLM dependency (`model_hosting_container_standards`) lowers the root logger to ERROR unless `SAGEMAKER_CONTAINER_LOG_LEVEL` is set; `driver_multi_verl.py` defaults it to `INFO`. If you set it yourself, use `INFO` or lower. |
| `Total available GPUs 0 is less than total desired GPUs N` | A previous trial's verl placement groups weren't released. See `autotune/trainers/_verl_cleanup.py` and "verl Worker Cleanup" in `CLAUDE.md`. |
| `Result(metrics=None)` from Ray Train | HPO trials must save at least one checkpoint per epoch. Use `save_strategy="epoch"`, `save_total_limit=1`. See `CLAUDE.md` "Key Gotchas". |
| vLLM OOM on long-context models | `max_model_len` defaults to the model's full position embeddings (often 128K). Set it to `max_prompt_length + max_response_length`. |
| `ConnectionRefusedError` on Ray init | A previous Ray cluster wasn't cleaned up. `ray stop --force`, then re-run. |
| Pre-commit hook fails | Run `pre-commit run --all-files` to see the specific failure. Fix the underlying issue rather than `--no-verify`-ing past it. |

If you hit something not on this list and figure out the fix, please open a PR
adding it here.

### Getting Help

- Search [existing issues](https://github.com/ibm-granite/granite.build/issues)
- Check [`CLAUDE.md`](CLAUDE.md) — the architectural reference, gotchas, and verl integration notes
- Check the inline comments in `autotune/configs/autotune.yaml` — config-level documentation
- Open a new issue with a minimal reproduction

## Additional Resources

### Documentation

- [`README.md`](README.md) — project overview, quick start, CLI reference
- [`CLAUDE.md`](CLAUDE.md) — architecture, gotchas, memory & resource reference, verl integration notes
- [`docs/RESOURCES.md`](docs/RESOURCES.md) — GPU sizing guide for 3B/8B/30B on 8× A100
- [`docs/MPS.md`](docs/MPS.md) — Apple Silicon (MPS) + MLX backend support
- [`docs/dataset-sft.md`](docs/dataset-sft.md), [`docs/dataset-offline-rl.md`](docs/dataset-offline-rl.md), [`docs/dataset-online-rl.md`](docs/dataset-online-rl.md) — dataset format references

### Project Links

- **Source:** [github.com/ibm-granite/granite.build](https://github.com/ibm-granite/granite.build) (the `autotunex/src/fm-tune` project)
- **Issues:** [github.com/ibm-granite/granite.build/issues](https://github.com/ibm-granite/granite.build/issues)

---

## Feedback Loop

Found a bug, workaround, or pattern while contributing?

- **Recurring issue?** → Add to [Common Issues](#common-issues--troubleshooting).
- **Architectural insight or new gotcha?** → Add to [`CLAUDE.md`](CLAUDE.md).
- **Config-level documentation?** → Add inline comments to the relevant YAML in `autotune/configs/`.

Help us improve this guide by opening a PR with your additions.

---

Thank you for contributing to AutoTune!
