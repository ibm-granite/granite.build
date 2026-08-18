# DPK Tokenization + Validation — SkyPilot

Tokenizes Parquet documents into Arrow token files with the
[Data Prep Kit](https://github.com/data-prep-kit/data-prep-kit) (DPK)
`tokenization2arrow` transform, then validates that the output is internally
consistent.

The same `build.yaml` runs on **every SkyPilot endpoint** — SLURM, Kubernetes,
AWS, and LSF — because both targets use the builtin `command` step, which is
resolved by any environment of class `Skypilot`. Switch endpoints with one
parameter; nothing else changes.

DPK runs through its **pure-Python runtime**: no Ray cluster, no GPU, and no
private repository. DPK also ships a Ray runtime
(`dpk_tokenization2arrow.ray.runtime`) which is considerably faster on large
corpora, but it needs a Ray cluster — see [Scaling up](#scaling-up).

## How it works

| Target | Description |
|---|---|
| `tokenize` | Installs DPK from public PyPI and runs `dpk_tokenization2arrow.runtime` over the input Parquet files. Emits the output directory as the `tokens` artifact. |
| `validate` | Binds `tokenize.tokens` and runs [`scripts/validate_tokens.py`](scripts/validate_tokens.py) against it. Exits non-zero — failing the build — if the token stream and its metadata disagree. |

| Artifact | Direction | Type | Description |
|---|---|---|---|
| `tokens` | Output of `tokenize` | `dataset` | Directory of `.arrow` token files plus a `meta/` tree. |
| `report` | Output of `validate` | `fileset` | Directory containing `validation.json` — the check summary. |

`validate` depends on `tokenize` through the artifact **binding**
`tokens: {binding: tokenize.tokens}`, so it is dispatched only once the tokenize
output has been registered. There is no explicit ordering to declare.

### What the transform produces

For each input Parquet file, `tokenization2arrow` writes:

```
<name>.arrow                  # single `tokens: uint32` column — concatenated token IDs
meta/<name>.docs              # "<file>, documents: <n>, tokens: <n>"
meta/<name>.docs.ids          # one "<document_id>, <token_count>" line per document
metadata.json                 # job-level stats (num_tokens, result_files, ...)
```

Note this is a **flat token stream**, not one row per document — the per-document
boundaries live in the `meta/` sidecars. (DPK's other transform,
`dpk_tokenization`, writes per-document rows to Parquet instead.)

### What `validate` checks

Because the token stream and its metadata are written separately, the useful
check is that they **agree**:

1. `metadata.json` exists and reports at least one result file.
2. At least one `.arrow` file was produced.
3. Every `.arrow` file is readable Arrow IPC with a `tokens` column, and non-empty.
4. Both `meta/` sidecars exist for each `.arrow` file.
5. Per file: `sum(.docs.ids counts)` == `.arrow` row count == `.docs` summary total,
   and the `.docs` document count matches the number of `.docs.ids` lines.
6. Document IDs are unique within a file.
7. Build-wide token total matches `metadata.json`'s `num_tokens`.

This catches truncated, duplicated, or mis-ordered writes that a
"file exists and is non-empty" check would silently pass.

## Prerequisites

- **SkyPilot configured** for your target endpoint — `sky check <cloud>` passes.
- **Outbound PyPI access** on the worker (the step `pip install`s DPK at runtime).
- **No credentials needed** for the defaults: the sample input is committed to
  this repo and the default tokenizer is public.

For a local SLURM cluster on your own machine:

```bash
make g4os-skypilot-venv PYTHON=python3.13
source .venv/bin/activate
make slurm-setup          # Docker SLURM cluster (requires a running Docker/Podman)
```

## Running

```bash
gbserver standalone --space-dir configurations/spaces/local     # terminal 1

# terminal 2
source .venv/bin/activate
export GB_ENVIRONMENT=STANDALONE
gb build start -f configurations/assets/templates/DPK_Tokenize_Skypilot/build.yaml
gb build status <build-id>
gb build log <build-id>
```

`parameters.yaml` alongside `build.yaml` is picked up automatically.

### Switching environments

```bash
gb build start -f build.yaml --param ENVIRONMENT=skypilot/kubernetes
gb build start -f build.yaml --param ENVIRONMENT=skypilot/aws
gb build start -f build.yaml --param ENVIRONMENT=skypilot/lsf/ibm-bluevela
```

> **Native Kubernetes (`space://environments/k8s`) is not supported by this
> template.** The `K8s` environment class has no builtin `command` step — it
> ships only artifact-transfer steps (`hfpull`, `s3push`, `lhpull`, …), so a
> `command` step for it would need a new Helm chart and AppWrapper template.
> Use `skypilot/kubernetes` to run on a Kubernetes cluster.

## Configuration

Everything is parameterized in [`parameters.yaml`](parameters.yaml); override at
submit time with `--param KEY=VALUE`.

| Parameter | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `skypilot/slurm` | SkyPilot endpoint to run on. |
| `DPK_TRANSFORMS_VERSION` | `1.1.8` | `data-prep-toolkit-transforms` version from PyPI. |
| `TRANSFORMERS_VERSION` | `>=4.38.2` | HuggingFace `transformers` constraint. |
| `INPUT_LOCAL_DIR` | `samples/data/dpk-tokenization/input` | Local dir rsync'd to the worker via `file_mounts`. |
| `VALIDATE_SCRIPT` | `.../scripts/validate_tokens.py` | Validation script shipped to the worker. |
| `TOKENS_URI` | `env:///tmp/dpk-tokenize-output` | How `tokens` is handed from `tokenize` to `validate`. Keep `env://` on shared-filesystem endpoints; use `s3://…` on `skypilot/aws`. See [Cross-node handoff](#cross-node-handoff). |
| `TOKENIZER` | `hf-internal-testing/llama-tokenizer` | Any HF AutoTokenizer-compatible name or path. |
| `DOC_ID_COLUMN` | `document_id` | Input column holding unique document IDs. |
| `DOC_CONTENT_COLUMN` | `contents` | Input column holding document text. |
| `TEXT_LANG` | `en` | Language hint for text splitting. |
| `CHUNK_SIZE` | `0` | Tokenize in N-character chunks; `0` = whole document. Use `20000` for very long English documents. |
| `NUM_GPUS_PER_NODE` | `0` | CPU-only workload. |
| `TOTAL_MEMORY_PER_NODE` | `4Gi` | Raise for large inputs (DPK suggests 64Gi at scale). |
| `POLL_INTERVAL_SECONDS` | `30` | Completion-detection interval. The monitor default is 300s, which would dominate wall-clock here. |

### Cross-node handoff

`validate` **reads** the `tokens` directory that `tokenize` produced — so the two
targets must be able to see the same bytes. They are separate workloads, and on
`skypilot/aws` each provisions its **own EC2 instance with no shared filesystem**.
Because `env://` registers only a node-local path (it moves no bytes), an
`env://` handoff leaves `validate` on a different instance reading an absent
directory, and the build fails with `no .arrow files found`.

- **Shared-filesystem endpoints** (`skypilot/slurm`, `skypilot/lsf` — both define
  a `shared_workdir` of `/shared`): the default `TOKENS_URI=env://…` is fine
  *when the targets co-locate or write under the shared workdir*.
- **`skypilot/aws`** (and any endpoint without a shared FS): route `tokens`
  through S3 so `tokenize` pushes it and `validate` pulls it locally:

  ```bash
  gb build start -f build.yaml \
    --param ENVIRONMENT=skypilot/aws \
    --param TOKENS_URI=s3://<your-bucket>/dpk-tokenize/tokens
  ```

  This needs the S3 assetstore secrets `COS_ACCESS_KEY_ID` / `COS_SECRET_ACCESS_KEY`
  (the `skypilot/aws` env already lists the `s3` assetstore). The `report` output
  can stay `env://` — nothing downstream consumes it.

### Using your own data

Point `INPUT_LOCAL_DIR` at a directory of Parquet files. Each file needs a unique
document-ID column and a text column; name them with `DOC_ID_COLUMN` and
`DOC_CONTENT_COLUMN`:

```bash
gb build start -f build.yaml \
  --param INPUT_LOCAL_DIR=/path/to/my/parquet \
  --param DOC_ID_COLUMN=id \
  --param DOC_CONTENT_COLUMN=text
```

For data already in object storage, replace the `file_mounts` entry in the
`tokenize` target with a storage mount (a dict value with `source`/`mode` —
see [`DiGiT_Skypilot`](../DiGiT_Skypilot/build.yaml)), or use DPK's
`--data_s3_config` instead of `--data_local_config`.

### Using a gated or private tokenizer

`--tkn_tokenizer` accepts a path as well as a hub name. For a gated hub model,
pass a token through the tokenizer args and provide `HF_TOKEN` to the step.

## Verifying the output

The committed sample input has known-good results — it is DPK's own test fixture,
and these are the values DPK's `expected/` files assert:

| Stat | Expected |
|---|---|
| `source_files` | 5 |
| `result_files` | 3 |
| `skipped empty tables` | 2 |
| `num_rows` (documents) | 6 |
| `num_tokens` | 85 |

`validate`'s log line on success reads:

```
VALIDATION PASSED: 3 arrow file(s), 6 documents, 85 tokens
```

Two of the five inputs are deliberately empty, which is why 5 inputs yield 3
outputs — see [the input data README](../../../../samples/data/dpk-tokenization/README.md).

## Scaling up

The pure-Python runtime is single-node. For large corpora, DPK's Ray runtime
parallelizes across a cluster:

- **Module** — swap `dpk_tokenization2arrow.runtime` for
  `dpk_tokenization2arrow.ray.runtime` and add `--run_locally True` (single node,
  multi-core) or point it at a Ray cluster.
- **Install** — Ray extras: `pip install 'data-prep-toolkit-transforms[ray]'`.
- **Compute** — raise `NUM_GPUS_PER_NODE` (still 0 for tokenization),
  `TOTAL_MEMORY_PER_NODE`, and add `num_nodes` under `launcher_config.resources`.

Provisioning a multi-node Ray cluster on SLURM is non-trivial; on Kubernetes,
consider KubeRay instead.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Build hangs ~5 min after the job finishes | `POLL_INTERVAL_SECONDS` too high, or omitted so the 300s monitor default applies. |
| `pip install` fails on the worker | No outbound PyPI access. Pre-bake DPK into an image and set `command_config.image`. |
| `OSError: ... is not a local folder` for the tokenizer | Tokenizer name is wrong, or it is gated and needs `HF_TOKEN`. |
| `no .arrow files found` | Either the input dir had no Parquet files / all were empty (check the `tokenize` log's output tree), **or** `tokenize` succeeded but `validate` ran on a different node with an `env://` handoff — set `TOKENS_URI=s3://…` (see [Cross-node handoff](#cross-node-handoff)). |
| `token count mismatch` in validate | A real inconsistency between the token stream and its sidecars — inspect `validation.json` in the `report` artifact. |
| `space://steps/command` unresolvable | The target env is not of class `Skypilot`. Native `k8s`/`lsf`/`runpod` have no builtin `command` step. |

## See also

- [Input data provenance](../../../../samples/data/dpk-tokenization/README.md)
- [SkyPilot environments](../../../../docs/environments/skypilot.md)
- [`build.yaml` reference](../../../../docs/builds/build-yaml-reference.md)
- [Step resolution](../../../../docs/environments/step-resolution.md) — how `space://steps/command` routes per environment class
- [DPK tokenization transform](https://github.com/data-prep-kit/data-prep-kit/tree/dev/transforms/universal/tokenization)
