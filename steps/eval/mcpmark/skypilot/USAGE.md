# mcpmark (SkyPilot)

Scores a checkpoint on one of the granite-evals **mcpmark** family's benchmarks:
MCPMark services, each run through the pinned
[eval-sys/mcpmark](https://github.com/eval-sys/mcpmark) harness (Apache-2.0, baked
into the image at a pinned commit). The served model is the tool-calling agent;
per task, the service's state manager sets up the initial state, the agent drives
it against the service's live MCP server, and the task's own `verify.py` scores the
final state. `benchmark` selects the service.

The code is the vendored granite-evals runtime (`steps/eval/src`) plus this
family's adapter (`steps/eval/mcpmark/src/mcpmark.py`), baked into this family's
image by `Dockerfile.gpu` one directory up (built by `steps/eval/Makefile`), which
also carries the pinned MCPMark checkout with its own venv and the service
runtimes (node/chromium for the npx MCP servers, a PGDG postgres 17 for
`mcpmark-postgres`, a pipx'd `postgres-mcp`). Nothing is built from this step at
run time.

```yaml
steps:
  - step_uri: space://steps/eval/mcpmark
    config:
      benchmark: mcpmark-filesystem
```

## Data and its licenses

- The harness and the task definitions are the pinned
  `eval-sys/mcpmark` commit (`cd45b7f…`, Apache-2.0), verified at image build
  time by the codeload tarball's sha256 and a digest of the extracted tree
  (`mcpmark/extras.sh`), and run from a writable per-run copy of that checkout
  under `output/mcpmark-checkout`.
- The task **state** payloads (the filesystem zips and the five postgres
  category backups) have an **unstated license** and are never vendored or
  baked: the harness downloads them at run time from `storage.mcpmark.ai`, and
  `results.json` records a digest of what it got (`details.state_digest`) so CDN
  drift is detectable across runs. Runs need network access.
- `mcpmark-playwright`'s four tasks drive the **live public web** with in-image
  chromium. Expect run-to-run variance; treat its score as a signal, not a
  stable number.

## Benchmarks in this family

| `benchmark` | Metric | Size |
|---|---|---|
| `mcpmark-filesystem` | pass@1 | 30 standard + 10 easy tasks |
| `mcpmark-postgres` | pass@1 | 21 standard + 10 easy tasks |
| `mcpmark-playwright` | pass@1 | 4 standard tasks (no easy suite) |

See `granite-evals list` for the current, authoritative set and sizes.

## Config

| Field | Purpose |
|---|---|
| `benchmark` | Required. One of the ids in the table above. |
| `image` | Required. The granite-evals `mcpmark` image, pinned by tag (`GRANITE_EVALS_IMAGE_MCPMARK`). |
| `model_path` | Required. Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | Name vLLM serves under. Empty = basename of `model_path`. |
| `output_dir` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | **Smoke knob:** score only the first N tasks by sorted id, so two smoke runs are comparable. Empty = the full set. |
| `repeats` | Must stay empty (1): one agent run per task; the module refuses otherwise (upstream's own `--k` is what varies samples, and it is pinned to 1). |
| `workers` | Must stay `1`: upstream's pipeline runs one service's tasks serially. Run the three services as separate targets for parallelism. |
| `dataset` / `dataset_revision` | Override with a LOCAL PATH to an existing MCPMark checkout (trusted as-is: no commit marker or task-count checks; recorded as such in `results.json`). |
| `options` | Space-separated `key=value` benchmark options; see below. |
| `phase` | `all` only: scoring happens inside generation (the key exists because every granite step shares one contract). |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | vLLM. |
| `sandbox_cache` | Unused by this family (no enroot sandboxes; the MCP services run in the job container itself); kept for the shared config contract. |
| `hf_home` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. No HF token is needed.

### Options (`options`)

| Option | Default | Purpose |
|---|---|---|
| `task_suite` | `standard` | `easy` runs the smaller suite (10 tasks per service; playwright has none and refuses to run empty). |
| `timeout` | `3600` | Per-task wall-clock bound, seconds (a timed-out task is a scored 0 — upstream treats timeouts as failures, not infra). |
| `compaction_token` | (unset) | Upstream's context-compaction threshold; unset disables compaction. |
| `reasoning_effort` | (unset) | Passed through to the model provider; left unset for vLLM by default. |
| `max_failed_frac` | `0.05` | Infrastructure-failed tasks above this fraction fail the run (a retryable error or a task that never produced a result). |

## The postgres service

`mcpmark-postgres` starts a private postgres 17 cluster per job (`initdb` into
node-local scratch, `pg_ctl` on a node-locked port — two jobs on one node never
share a port) and stops it at the end. `initdb` refuses to run as root, so this
benchmark needs the job's usual non-root container user. A preflight runs the
pipx'd `postgres-mcp` server's `--help` before any task, so a broken install
fails the run loudly instead of mis-scoring every task.

## Resume

The harness's own results tree under `output/mcpmark/` is the resume store: a
task with a success or a non-retryable failure is skipped on re-run, a
retryable one (rate limit, state duplication, …) is retried, and a missing one
is re-run. The same tree survives job retries, so a re-dispatched target
continues where it stopped.

## Output

`granite_results` (dataset): the `results.json` file. Per-task records (the
agent trajectory and `meta.json` per task) stay in the harness's own tree under
`output/mcpmark/run/`; the writable checkout and the downloaded state payloads
stay under `output/` (`mcpmark-checkout/`, `test_environments/`) so nothing is
re-fetched on a resumed run.
