# data-designer (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `data-designer` in a `build.yaml`. This file covers how the step is *generated, tested,
> and published*.

A [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) runner for SkyPilot
clusters. Runs on the **bare launcher node** by default, installing `data-designer` from
PyPI with `uv` during `setup`. No custom image is built or published. It is modelled on
the [dpk](../../dpk/skypilot/README.md) step and keeps its shape: Jinja computes only
*values*, the shell and logic live in bundled `src/` files that are tested directly, the
required-config guards run in both phases, and the artifact marker is the last thing
printed.

## Design: dpk with three deltas

| | dpk | data-designer |
|---|---|---|
| **What the build names** | one DPK `transform` (module + pip extra derived) | one Data Designer `config` (inline) + the `model_providers` it calls |
| **Invocation** | `python -m dpk_<t>.runtime` from `dpk_run.sh` | Data Designer's Python API from `src/dd_run.py` |
| **External dependency** | none | an LLM endpoint with a credential delivered as a space secret, or a small model served on the node (`local_model`) |

**Why the Python API, not `data-designer create`.** The CLI reads providers from
`$DATA_DESIGNER_HOME/model_providers.yaml`, so a credential would have to be written to
a file on the node. Some endpoints need a custom header: RITS takes `RITS_API_KEY: <key>`,
and Data Designer's `extra_headers` are literal strings. `dd_run.py` builds the providers
in process, resolving `api_key_env` / `extra_headers_from_env` from the environment, so a
key never reaches disk or the persisted config. Each provider is validated with its
credentials replaced by placeholders first, so a malformed entry is reported by field
name without the key reaching the log (a pydantic error prints the whole entry; found in
review).

**How the config reaches the node.** The `config`, `model_providers`, `run_config` and
`local_model` travel together as base64 JSON in `$GB_DD_SPEC_B64`, which the launcher's `envs` render with the server's
`json_dumps | b64encode` filters. This avoids YAML and shell quoting, and the Data
Designer prompts inside are data at that point, never re-rendered. They *are* rendered
once before that, as part of the step config, which is why USAGE.md asks for
`{% raw %}` around prompts.

| File | Invoked from | Does |
|---|---|---|
| `src/dd_guard.sh` | `setup` and `run` | required fields, enums, counts (incl. `min_records` ≤ `num_records`), marker-safe `output`, unrendered `seed` binding, `local_model` fields, and refuses `local_model` with `dd_image` |
| `src/dd_setup.sh` | `setup` (bare-node only) | the dpk uv install: bootstrap `uv`, shared `UV_CACHE_DIR`, venv, `data-designer==<pin>` (plus `zstandard` with `local_model`) |
| `src/dd_ollama.sh` | `setup` and `run` (`local_model` only) | install: download Ollama, verify its published sha256, keep the CPU libraries, pull the model under a lock; run: start the server, refuse a taken port, run `dd_run.py`, stop the server |
| `src/dd_run.py` | `run` (always) | decode spec → resolve credentials → validate providers without secrets → seed-path override → `validate()` → check a previous run like Data Designer would (clear it, refuse, or resume) → `create()` → row-count check → optional export → metadata + artifact marker |

## Running the tests

The step's tests split the way dpk's do. What lives in this directory is what anyone can
run, and what `make publish-step` ships. Tests that need IBM infrastructure live in the
repo's IBM tree and never ship.

| Kind | Where | Runs in | Ships? |
|---|---|---|---|
| Unit tests | `test/*.py` here | `make test` (Mode 1) | No: only per-cluster subdirs are copied |
| Open build tests: no keys, no IBM network | `test/<cluster>/` + `test-data/<cluster>/` here | `make test`, then `test/steps/...` (Mode 2) | **Yes** |
| IBM-only tests: RITS, BlueVela | `test/integration/ibm/buildrunner/skypilot/...` | IBM suites (`pytest.mark.ibm`) | No |

The open build tests use `local_model` (Ollama on the node), so they need no key, no
secret and no model endpoint. `slurm-ollama` and `kube-ollama` exist; aws is next.

### Unit tests (here)

```
make -C steps/data/data-designer/skypilot test    # uses the repo-root .venv; runs `make space` first
```

Mode 1 only; no infrastructure and no model endpoint. Some of them need `data-designer`
or `zstandard`, which are deliberately **not** in the repo `.venv` (installing
data-designer there replaces the repo's pinned duckdb, fsspec, pyarrow and
opentelemetry). Under `make test` those skip, and the skip reason names the command to
run them. To run all of them, use a separate venv that has `data-designer==0.9.5`,
`zstandard`, `pytest` and `pyyaml`:

```
cd steps/data/data-designer/skypilot
PYTHONPATH=src <venv>/bin/python -m pytest -c /dev/null --noconftest \
  test/test_dd_step_render.py test/test_dd_guard_sh.py test/test_dd_run.py test/test_dd_ollama_sh.py
```

There are 110 tests in all:

- `test_dd_step_render.py` pins what the template computes:
  - the rendered blocks parse under `bash -n`;
  - the guard receives identical argv from both phases;
  - every runner argument arrives intact, quotes included;
  - `dd_image` switches between installing and the `docker:` image;
  - `local_model` on and off;
  - `$GB_DD_SPEC_B64` decodes back to the config, with prompts verbatim, and carries
    `run_config` and `local_model`.
- `test_dd_guard_sh.py` runs the guard on every refusal and on the passing cases,
  including `min_records` > `num_records`, an out-of-range port, and `local_model` with
  `dd_image`.
- `test_dd_run.py` covers the runner:
  - spec decoding;
  - credential resolution, including that no value appears in an error, and provider
    validation that never shows a secret;
  - the seed-path override;
  - `clear_previous_run` removing only Data Designer's files, and not following
    symlinks;
  - `count_resumable_records`.

  With `data-designer`, `TestRunEndToEnd` also runs the whole runner with a sampler-only
  config, so no model is called:
  - the marker is the last line, and the export is written;
  - `min_records` is checked before any work;
  - a finished run is extended;
  - `if_possible` restarts in place on a changed config, `buffer_size` or
    `preserve_dropped_columns`, with no timestamped sibling dir;
  - `always` refuses a mismatch, and explains a first run;
  - user files survive a restart;
  - a malformed provider is reported without its key.
- `test_dd_ollama_sh.py` runs the real `src/dd_ollama.sh` with `curl` and the Ollama
  binary stubbed, so nothing is downloaded:
  - the install keeps only the CPU libraries;
  - a checksum mismatch or a failed download leaves nothing in the cache;
  - a second install reuses the cache;
  - a taken port fails instead of borrowing another server;
  - `run` returns the wrapped command's exit status and always stops the server.

### Open build tests (here)

- **slurm-ollama**: the local Docker SLURM cluster (`make test-setup`, once). A 4-row
  sampler + `llm-text` generation against `granite4:350m` served on the node by
  `local_model`, with `model_providers` empty, `min_records: 4` and a jsonl export.
  - **Setup:** installs data-designer and Ollama (CPU libraries only, via the venv's
    `zstandard` because these nodes have no `zstd`) and pulls the model.
  - **Run:** starts the server on `127.0.0.1`, generates through it, and stops it.
  - **Checked:** the build expects `records_generated: 4` and `resumed_records: 0`.
  - **Timing:** green on 2026-10-21, 4m39s with a cold cache and 3m04s warm. No Ollama
    process was left on any node afterwards.

  Run with `GB_ENVIRONMENT=STANDALONE GBTEST_MODE=live make test TEST_DIR=test/slurm-ollama`.

- **kube-ollama**: the same fixture in a pod on `skypilot/kubernetes/local`, the local
  kind cluster (`make test-setup-kube`, once). It self-skips unless that cluster and its
  `gb-shared` volume are up.
  - **Pod image:** SkyPilot's default kube image has no `python3`, `pip` or `zstd` of
    its own. The install works anyway, through SkyPilot's runtime Python and the venv's
    `zstandard`, as on SLURM.
  - **Cache:** goes to the env's ReadWriteMany `gb-shared` volume at `/shared`, so a
    second run reuses Ollama and the model.
  - **Timing:** green on 2026-10-21 in 3m53s with a cold cache. The workdir-teardown pod
    (`-td-`) removed itself about 50s later, leaving no data-designer pods behind.

  Run with `GB_ENVIRONMENT=STANDALONE GBTEST_MODE=live make test TEST_DIR=test/kube-ollama`.

### IBM-only tests (`test/integration/ibm/buildrunner/skypilot/slurm_local/`)

These run against RITS (IBM-internal) on the local Docker SLURM cluster, and are marked
`pytest.mark.ibm`. Fixtures are under the matching `test-data/...` directory.

| Test | What it proves |
|---|---|
| `test_data_designer_runner_rits.py` | `src/dd_run.py` in-process against RITS: header auth, generation, export, marker. No cluster. Fastest check. |
| `test_data_designer_rits.py` | A 4-row build reaches SUCCESS: install, spec hand-off, space secret → header, live generation, marker. |
| `test_data_designer_rits_keep.py` | 6 rows into `/shared/data-designer-test/slurm-rits-keep`, which survives teardown. Reads them back: 3 row groups, every `question` non-empty, no key in any file; lineage metadata `records_generated: 6`, `resumed_records: 0`. |
| `test_data_designer_rits_resume.py` | Two builds: the first cancelled after ≥2 row groups, the second resubmitted. The second must keep those row groups byte-identical and generate only the rest. |

**Before running:**

1. **Cluster:** `make slurm-setup` from the repo root, once.
2. **Space:** `make -C steps/data/data-designer/skypilot space`. The fixtures resolve the
   unpublished step from that rendered Space, whose space name is `public` (the
   `common.mk` default since #513, the same name as `configurations/spaces/local`).
3. **Secret:** store the RITS key in that space, with `gb secret create RITS_API_KEY
   --value "$RITS_API_KEY" --space public` (needs a running gbserver). Without
   gbserver, the file must be the local secret manager's own YAML format under
   `~/.granite.build/space_secrets/public/`. A hand-written plain JSON file is
   silently ignored, and the build then fails with "declared secret 'RITS_API_KEY' ...
   was not found among the 0 available secret(s)".
4. **Skip gate:** export `RITS_API_KEY`. That only decides whether the tests run; the
   builds read the space secret.

```
GB_ENVIRONMENT=STANDALONE GBTEST_MODE=live \
  .venv/bin/python -m pytest -s test/integration/ibm/buildrunner/skypilot/slurm_local/test_data_designer_rits.py
```

`test_data_designer_runner_rits.py` imports nothing from gbserver, but needs
`data-designer`, which is deliberately not in the repo venv: installing it there
replaces the repo's pinned duckdb, fsspec, pyarrow and opentelemetry. Run it with a
separate venv: `<venv>/bin/python -m pytest -c /dev/null --noconftest <file>`.

**Cancellation and the resume test.** Cancelling a SkyPilot SLURM build doesn't stop its
task with the currently pinned SkyPilot fork
(https://github.com/ibm-granite/granite.build/issues/533). The fix is in
https://github.com/cmadam/skypilot/pull/5. Until that lands, the resume test also
`scancel`s the step's SLURM job after requesting cancellation. With
`GBTEST_DD_BUILD_CANCEL_ONLY=true` it relies on the build cancellation alone, which
passes only against a SkyPilot carrying the fix. The fix was verified that way on
2026-10-21: the task step was CANCELLED at the signal, and nothing was written after it.

Measured history: resume was first checked off-cluster against RITS (8 rows,
`buffer_size: 2`). These cases are now covered without a model by `TestRunEndToEnd`:

1. A `kill -9` after 2 of 4 row groups resumed with "2 of 4 row group(s) already complete
   … skipping them", and `batch_00000.parquet` was unchanged.
2. A re-run of the finished dataset was a no-op.
3. A changed prompt under `always` refused.
4. A changed prompt under `if_possible` showed that Data Designer starts fresh in a
   timestamped sibling directory, not in place. A later review found the same for a
   changed `preserve_dropped_columns`, and a hard failure for a changed `buffer_size`.
   `dd_run.resume_mismatch` now runs Data Designer's own checks before generating, and
   `clear_previous_run` clears just the previous run's files.

### Planned

- **More open build tests**: `aws-ollama` (a copy of `slurm-ollama` with another
  `environment_uri`, gated on AWS credentials), then keep-output and resume variants.
  These ship with the step.
- **BlueVela**: siblings under `slurm_bluevela/`, once the step is in the gb-test assets
  repo.

## Known gaps

- **No model-server option yet.** `distill/vllm-server` publishes its URL as a `mem://`
  binding but only runs on LSF. An optional `model_endpoint` input that overrides a
  provider's `endpoint` is the intended way in, once there is a non-LSF server to test it
  against.
- **Config size.** The spec travels in one environment variable, so a very large inline
  config (thousands of lines, or a long inline category list) could hit OS argument
  limits. A `config` file input would lift that.
- **Mistyped `output`** behaves as in dpk: the marker names an undeclared output, the
  server logs and ignores it, and the target goes green having registered nothing.
- **Personas.** Nemotron-Personas samplers download 0.3–4 GB per locale into
  `DATA_DESIGNER_HOME`, which is per-run here, so they would re-download each run.
  Nothing caches them yet.
