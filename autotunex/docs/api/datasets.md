# Datasets API

A **dataset** is a named reference to training data. This resource has full CRUD plus a
file upload, and a set of optional LLM-backed **intelligence** helpers. A job references a
dataset at submit time and requires it to be `ready`. This page documents the endpoints
under `/api/v1/datasets` and `/api/v1/datasets/intelligence`.

See [overview.md](overview.md) for shared conventions, [authentication.md](authentication.md)
for owner resolution, and [../concepts.md](../concepts.md) for the domain model.

## Ownership and scope

Reads and mutations are owner-scoped. By default a caller — admin included — sees its own
datasets **plus** the shared system tier: rows owned by the reserved system user
(`00000000-0000-0000-0000-000000000001`), the curated starter datasets every caller can
read and launch a job from. An admin widens to all owners per request with `scope=all`
(`own` | `all`, default `own`); a non-admin passing `scope=all` gets a **403**. `POST` and
`upload` are always own-scoped.

Mutations never widen to the shared tier, so `PUT` against a system-owned dataset returns
**404** — only an admin via `scope=all` may edit one. `upload` takes no `scope` parameter
at all: it is strictly own-only, so not even an admin can upload into a shared dataset, or
into another owner's.

`DELETE` against a system-owned dataset is refused outright, for **every** caller: a normal
user, an admin passing `scope=all`, and a caller whose own identity resolves to the system
user all get a **403** (`title`: `System Resource Protected`), because starter content is
shared by the whole deployment. The single exemption is an admin with an active
impersonation overlay onto the system user (`POST /auth/assume/{id}`). Mirrors
`configurations.md` exactly.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/datasets` | Create a dataset (metadata only) |
| `GET` | `/api/v1/datasets` | List datasets, newest first |
| `GET` | `/api/v1/datasets/{dataset_id}` | Get one dataset, optionally with a preview |
| `PUT` | `/api/v1/datasets/{dataset_id}` | Fully replace a dataset's metadata |
| `DELETE` | `/api/v1/datasets/{dataset_id}` | Delete a dataset |
| `POST` | `/api/v1/datasets/{dataset_id}/upload` | Upload the dataset's file(s) |
| `GET` | `/api/v1/datasets/hf/search` | Search HuggingFace for dataset repo ids |
| `GET` | `/api/v1/datasets/hf/splits` | Resolve a HuggingFace dataset's revision, configs and splits |
| `POST` | `/api/v1/datasets/hf/preview` | Preview a HuggingFace dataset's mapped rows |
| `POST` | `/api/v1/datasets/hf/import` | Import a HuggingFace dataset (returns `202`) |
| `POST` | `/api/v1/datasets/intelligence/parse-strategy` | Suggest a parsing strategy (LLM) |
| `POST` | `/api/v1/datasets/intelligence/suggest-mapping` | Suggest a column mapping (LLM) |
| `POST` | `/api/v1/datasets/intelligence/validate-strategy` | Dry-run a strategy (no LLM) |
| `GET` | `/api/v1/datasets/intelligence/formats` | List the dataset-type catalog |

There is no `PATCH`; updates are `PUT`-only full replacements.

---

## POST /api/v1/datasets

Create a dataset owned by the calling principal. Metadata only — no file yet. Returns
`201` with `DatasetRead`, `status: "empty"`. Unknown fields are rejected.

### Request body — `DatasetCreate`

| Field | Type | Required | Default | Constraints |
| --- | --- | --- | --- | --- |
| `name` | string | yes | — | 1–255 chars; may not contain `/`, `\`, or `..` |
| `description` | string \| null | no | `null` | Stored as `NULL` when omitted |
| `data_format` | string | no | `jsonl` | Validated to `jsonl` \| `csv` \| `parquet` (else `422`) |

```bash
curl -X POST https://example.com/api/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{ "name": "support-tickets", "description": "Q3 tickets", "data_format": "jsonl" }'
```

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Caller has no resolvable owner (unprovisioned) |
| `409` | The caller already owns a dataset with this `name` |
| `422` | `data_format` is unsupported, or the body otherwise fails validation |

---

## GET /api/v1/datasets

List the caller's datasets, newest first. Returns a `Page<DatasetRead>`.

### Query parameters

| Name | Type | Default | Constraints |
| --- | --- | --- | --- |
| `limit` | int | `20` | 1–100 |
| `offset` | int | `0` | ≥ 0 |
| `scope` | string | `own` | `own` \| `all` (admin only for `all`) |
| `q` | string | `none` | Case-insensitive substring filter. Matches the dataset name. |

### Response `200` — `Page<DatasetRead>`

`{ "items": DatasetRead[], "total": int, "limit": int, "offset": int }`. List responses
always report `preview` as `null` — a populated preview is a detail-endpoint option.

### Notable statuses

`403` if a non-admin requests `scope=all`.

---

## GET /api/v1/datasets/{dataset_id}

Return one dataset, optionally with a bounded row preview. Returns `DatasetRead`.

### Path & query parameters

| Name | In | Type | Default | Constraints |
| --- | --- | --- | --- | --- |
| `dataset_id` | path | UUID | — | Dataset id |
| `preview` | query | bool | `false` | Request a row preview |
| `preview_rows` | query | int | `10` | 1–100 |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

`preview` is populated when `preview=true` **and** the dataset has data to read — either
`status=ready`, or a non-empty `artifact_url` (a dataset registered out-of-band by the tuning
pipeline, which never flips `status` to `ready`). A backend failure while previewing degrades
`preview` to `null` and never fails the metadata read.

### The `DatasetRead` shape

| Field | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Dataset id |
| `user_id` | string | Owner's id |
| `name` | string | Dataset name (unique per owner) |
| `description` | string \| null | Free text |
| `data_format` | string | `jsonl` \| `csv` \| `parquet` |
| `status` | string | Lifecycle: `empty` \| `uploading` \| `importing` \| `ready` \| `error` |
| `status_detail` | string \| null | Extra detail, e.g. an error message |
| `train_file` | string | Generated from `name` as `<name>_train`; not writable |
| `train_records` | int \| null | Row count once processed |
| `train_file_size` | int \| null | Bytes once processed |
| `validation_file` | string | Generated from `name` as `<name>_validation`; not writable |
| `validation_records` | int \| null | Row count once processed |
| `validation_file_size` | int \| null | Bytes once processed |
| `artifact_id` | string \| null | Stored-artifact id (server-set) |
| `artifact_url` | string \| null | Stored-artifact location (server-set) |
| `hf_repo_id` | string \| null | Source Hub repo id; set only for a HuggingFace import, `null` otherwise |
| `hf_revision` | string \| null | The pinned parquet-branch commit SHA; set only for a HuggingFace import, `null` otherwise |
| `hf_config` | string \| null | The imported config name; set only for a HuggingFace import, `null` otherwise |
| `hf_split` | string \| null | The upstream train split imported; set only for a HuggingFace import, `null` otherwise |
| `hf_provenance` | object \| null | Column mapping, selected validation split, and per-split row/shard counts (see below); set only for a HuggingFace import, `null` otherwise |
| `associated_jobs` | `DatasetJobRef[]` | Compact refs to jobs using this dataset (owner-scoped) |
| `created_at` | datetime | ISO 8601 |
| `updated_at` | datetime | ISO 8601 |
| `preview` | object \| null | Present only when requested and readable (`ready` or `artifact_url`); see below |

**`DatasetJobRef`:** `{ id: UUID, experiment_name: string|null, status: string }`.

**`preview`** (a `DatasetPreview`):
`{ "train": [ {row}, ... ], "validation": [ {row}, ... ], "viewer_ready": bool }` — `train`
and `validation` are each a list of raw JSON rows bounded by `preview_rows`. `viewer_ready`
(default `true`) is `false` only when the HuggingFace dataset viewer was unavailable or
still precomputing, so empty `train`/`validation` there mean "not ready yet" rather than
"genuinely empty".

```json
{
  "id": "9f30...",
  "user_id": "a2c9...",
  "name": "support-tickets",
  "description": "Q3 tickets",
  "data_format": "jsonl",
  "status": "ready",
  "status_detail": null,
  "train_file": "support-tickets_train",
  "train_records": 1200,
  "train_file_size": 240000,
  "validation_file": "support-tickets_validation",
  "validation_records": 200,
  "validation_file_size": 40000,
  "artifact_id": "b7c1...",
  "artifact_url": "file:///data/9f30",
  "associated_jobs": [],
  "created_at": "2026-08-10T12:00:00Z",
  "updated_at": "2026-08-10T12:05:00Z",
  "preview": {
    "train": [{ "input": "...", "output": "..." }],
    "validation": [],
    "viewer_ready": true
  }
}
```

### Dataset status lifecycle

| Status | Meaning |
| --- | --- |
| `empty` | Created, no file uploaded yet |
| `uploading` | An upload is being processed off-request |
| `importing` | A HuggingFace import is being fetched off-request — the HF counterpart of `uploading`, kept distinct so the UI can say which |
| `ready` | File processed successfully; usable by a job and previewable |
| `error` | Processing failed (see `status_detail`) |

### Notable statuses

`403` (non-admin requesting `scope=all`), `404` (no such dataset, or not the caller's).

---

## PUT /api/v1/datasets/{dataset_id}

Fully replace a dataset's mutable **metadata** (name, description, format). Same body as
create (`DatasetCreate`). Returns `DatasetRead`.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `dataset_id` | path | UUID | — | Dataset id |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Non-admin requesting `scope=all` |
| `404` | No such dataset, or not the caller's |
| `409` | The new `name` collides with another of the caller's datasets |
| `422` | `data_format` is unsupported, or the body otherwise fails validation |

---

## DELETE /api/v1/datasets/{dataset_id}

Delete a dataset (and best-effort clean its stored files). Returns `204`.

A dataset owned by the reserved system user cannot be deleted by anyone — see *Ownership
and scope* above. `scope=all` does not override it.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `dataset_id` | path | UUID | — | Dataset id |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Non-admin requesting `scope=all` |
| `403` | The dataset belongs to the shared system tier (`title`: `System Resource Protected`); refused for every caller except an admin impersonating the system user |
| `404` | No such dataset, or not the caller's |
| `409` | A job still references this dataset |

---

## POST /api/v1/datasets/{dataset_id}/upload

Upload the dataset's file(s) with `multipart/form-data`. Cheap validation runs
synchronously; the heavy processing runs **off-request**. Returns `202` with `DatasetRead`
in `status: "uploading"` — poll `GET /datasets/{id}` for the terminal state (`ready` or
`error`).

Supports gzip-compressed bodies via the `Content-Encoding: gzip` request header.

### Path parameter

| Name | Type | Notes |
| --- | --- | --- |
| `dataset_id` | UUID | Dataset id |

### Multipart form fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `train_file` | file | yes | Training file; its extension sets the format (accepted set below) |
| `validation_file` | file | no | Optional validation file; its format must match the train file's |
| `validation_percentage` | int | no | Split a validation set from train; mutually exclusive with `validation_file` |
| `column_mapping` | string | no | JSON string, a flat `{target: source}` object |

Accepted extensions — anything else is a **415**. A trailing `.gz` on the *filename* is
stripped before this lookup, so `train.jsonl.gz` is accepted:

- `.jsonl`, `.json` → `jsonl`
- `.csv` → `csv`
- `.parquet`, `.pq` → `parquet`

```bash
curl -X POST https://example.com/api/v1/datasets/9f30.../upload \
  -F "train_file=@train.jsonl" \
  -F "validation_file=@val.jsonl" \
  -F 'column_mapping={"input":"question","output":"answer"}'
```

### Notable statuses

| Status | When |
| --- | --- |
| `404` | No such dataset, or not the caller's |
| `409` | The dataset is already `uploading` or `importing` |
| `413` | A file exceeds the configured cap (`AUTOTUNEX_DATASET_UPLOAD_MAX_BYTES`) |
| `415` | A file's extension is outside the supported set |
| `422` | Both a validation file and a percentage given, invalid `column_mapping` JSON, mismatched formats, or an empty file |

---

# HuggingFace imports

Import is available on the same-host bash standalone build, which keeps the dataset on
local disk behind a `file://` locator, and on non-standalone deployments whose dataset
storage pushes to HuggingFace (`dataset_storage_backend` of `huggingface`, or `auto`
with `llmb` on `PATH` and both token env vars set), which push it to a private repo
and record an `hf://` locator. Everywhere else, including standalone + LSF, every
route below returns the import-unavailable `503` and `GET /app-config` reports
`hf_import.available: false`. Setting `AUTOTUNEX_HF_IMPORT_ENABLED=false` (default
`true`) does the same in any deployment: all four `/datasets/hf/*` routes return the
import-unavailable `503`.

Use `GET /api/v1/datasets/hf/search?query=<text>` to find candidate repos. `query` is
required (1 character or more) and `limit` defaults to `20` (1–100). The response is a
plain list of repo id strings, most relevant first. This search is **always
anonymous** — public repos only, and the server's token is never sent — because no
single `repo_id` exists here to narrow that token against.

Use `GET /api/v1/datasets/hf/splits?repo_id=owner/name` to select a config and
splits. The response is `{ repo_id, revision, configs }`, where `configs` maps each
config name to its sorted split names, and `revision` is the immutable 40-character
commit SHA of HF's **converted parquet branch**, which may differ from the source
branch.

Both `POST /api/v1/datasets/hf/preview` and `POST /api/v1/datasets/hf/import`
require that `revision` alongside `repo_id`, `config`, `train_split`, and
`column_mapping` (target column → source column). Both accept an optional
`validation_split`. Import also requires `name` and accepts `description` or a
`validation_percentage` (1–50, mutually exclusive with `validation_split`).
Missing revisions, branch names, and abbreviated SHAs return **422** — and so, on
*both* routes, does a `config` or split absent from the pinned parquet branch, or a
selection whose first shards exceed `AUTOTUNEX_HF_IMPORT_MAX_BYTES`. Preview runs the
same planning and size-gate step as import, so it refuses those selections the same
way even though it downloads no dataset files.

Preview returns `revision`, `columns`, `raw_rows`, `mapped_rows`, `sampled`, and
`survived`. It samples at most 100 rows from HuggingFace's dataset viewer, which
serves the **latest** conversion and accepts no revision — so the sample may differ
from the snapshot `revision` pins, and the `revision` preview returns is the one an
import will read rather than the one the sample came from. The consequence worth
planning for: a column renamed between the pinned revision and the latest conversion
previews fine under its new name, and is then refused at import, when the mapping is
re-checked against the columns actually staged (below). Preview downloads no
dataset files and writes nothing to disk. The viewer cannot serve every dataset; when
it cannot, preview returns **503** naming *preview* (`Preview is unavailable for
<repo>. You can still import it without previewing.`) — the import path is
unaffected, and preview is optional.

Import does read the pinned snapshot, and returns **202** with status `importing`;
poll the dataset until it is `ready` or `error`. A moved upstream branch never
substitutes newer files for a selected revision. Two caps bound what lands:
`AUTOTUNEX_HF_IMPORT_MAX_BYTES` is one budget for the whole import, shared by the
train and validation splits, enforced as the bytes arrive, while
`AUTOTUNEX_HF_IMPORT_MAX_ROWS` applies to each selected split.

A mapping built against a stale preview cannot silently produce a truncated dataset:
import re-checks the mapping against the columns actually staged and refuses if
**any** non-blank mapped source is absent from that schema, naming both the missing
columns and the available ones. A separate check refuses a mapping whose every source
is blank — it names no column that could be reported missing — while a *partly* blank
mapping is accepted, matching the projection preview shows. Both checks run in the
background task after the `202`, so a refusal surfaces as the dataset turning `error`
with that message in `status_detail`, not as a failed request.

Imported datasets expose `hf_repo_id`, `hf_revision`, `hf_config`, `hf_split`,
and `hf_provenance`. The provenance object includes the column mapping, the selected
upstream `validation_split` (`null` when none was selected), and, for each of `train`
and `validation`:

| Field suffix | Meaning |
| --- | --- |
| `_original_rows` | Rows in downloaded shards; a lower bound when shards remain unread |
| `_retained_rows` | Rows retained from that source split after applying the import cap |
| `_unread_shards` | Number of shards skipped after reaching the row cap |
| `_truncated` | `true` if rows or shards were omitted; otherwise `false` |

Validation values are `null` when no upstream validation split was selected.
Older imports may lack the truncation fields; absence does not establish that
an import was complete. A successful regular file upload replacing an imported
dataset clears all HF provenance. The initial import's runner handoff preserves it.

### Notable statuses

Two of the 503s are worded differently on purpose, and the difference is actionable:
*"HuggingFace dataset import is not available in this deployment"* means an operator
has import switched off, so stop; *"HuggingFace could not be reached"* means the Hub
itself failed in a deployment that does support import, so retry. Collapsing the
second into the first would tell a caller in a correctly-configured deployment to
abandon an importable dataset, and send an operator to debug
`AUTOTUNEX_HF_IMPORT_*` settings that are already right — which is the whole reason
the second exception exists.

| Status | When |
| --- | --- |
| `403` | `import` only: the caller has no resolvable `users` row to own the new dataset (`title`: `Forbidden`) |
| `409` | `import` only: the caller already owns a dataset with this `name` |
| `422` | A malformed `revision` (a branch name or an abbreviated SHA), a `config` or split absent from the pinned parquet branch, or selected first shards exceeding `AUTOTUNEX_HF_IMPORT_MAX_BYTES` |
| `422` | `No tabular data found in <repo>.` — the parquet branch lists no configs or splits |
| `503` | `HuggingFace dataset import is not available in this deployment.` — refused on **all four** routes, before any Hub call |
| `503` | `HuggingFace could not be reached. Try again in a moment.` — any Hub-unreachable failure, on all four routes |
| `503` | `HuggingFace has not converted <repo> yet. Try again later, or upload a file.` — no parquet branch exists for the repo yet |
| `503` | `preview` only: `Preview is unavailable for <repo>. You can still import it without previewing.` — the viewer cannot sample this dataset (see above) |

---

# Dataset intelligence

Optional, LLM-backed helpers that suggest how to shape a raw sample into training pairs,
suggest a column mapping, or validate a strategy. Mounted under
`/api/v1/datasets/intelligence`. These require the LLM feature to be configured on the
server — when it is not, the LLM-backed routes return **503**. All routes require an
authenticated caller.

## POST /api/v1/datasets/intelligence/parse-strategy

Suggest how to turn a raw sample into `{input, output}` training pairs (calls the LLM).
Returns a `ParsingStrategy`.

### Request body — `ParseStrategyRequest`

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `sample` | array of objects **or** string | yes | — | The raw sample to analyze |
| `data_format` | string | no | `jsonl` | Format of the sample — one of `jsonl`, `csv`, `parquet`, `txt`, `xml`; anything else is a **422** |
| `custom_prompt` | string \| null | no | `null` | Extra guidance for the model |

### Response `200` — `ParsingStrategy`

| Field | Type | Notes |
| --- | --- | --- |
| `type` | string | `direct_mapping` \| `regex` \| `transformation` |
| `description` | string | Human-readable summary (default `""`) |
| `input_field` | string \| null | Source field for the input |
| `output_field` | string \| null | Source field for the output |
| `input_pattern` | string \| null | Regex for the input (regex strategy) |
| `output_pattern` | string \| null | Regex for the output (regex strategy) |
| `confidence` | float | 0.0–1.0 |
| `sample_extraction` | array of objects \| null | Worked examples of the extraction |

### Notable statuses

`422` (invalid request / unparseable sample), `502` (LLM backend error),
`503` (LLM not configured).

---

## POST /api/v1/datasets/intelligence/suggest-mapping

Suggest a flat `{target: source}` column mapping onto a training format (calls the LLM).
Returns a `ColumnMappingSuggestion`. The `column_mapping` is upload-ready — pipe it
straight into the `column_mapping` form field of an upload.

### Request body — `SuggestMappingRequest`

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `column_names` | array of strings | yes | — | The dataset's column names |
| `column_samples` | object (string → array of strings) | no | `{}` | Sample values per column |
| `sample_data` | array of objects | no | `[]` | Sample rows |
| `target_format` | string \| null | no | `null` | Desired training format |

### Response `200` — `ColumnMappingSuggestion`

| Field | Type | Notes |
| --- | --- | --- |
| `dataset_format` | string | The training-format catalog key chosen; expected to equal `target_format` when one was supplied (the LLM is instructed to use it; the server does not enforce it) |
| `tuning_type` | string | Inferred tuning type |
| `confidence` | float | 0.0–1.0 |
| `column_mapping` | object (string → string) | Flat `{target: source}`; unmapped targets are dropped |
| `column_confidence` | object (string → float) | Per-column confidence |
| `reasoning` | string | Model's rationale (default `""`) |

### Notable statuses

`422`, `502`, `503`.

---

## POST /api/v1/datasets/intelligence/validate-strategy

Dry-run a parsing strategy against a sample with **no LLM call**. Returns a
`ValidationResult`.

### Request body — `ValidateStrategyRequest`

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `strategy` | `ParsingStrategy` | yes | The strategy to test (same shape as the parse-strategy response) |
| `sample` | array of objects **or** string | yes | The sample to run it against |

### Response `200` — `ValidationResult`

| Field | Type | Notes |
| --- | --- | --- |
| `success` | bool | Whether the dry run succeeded |
| `parsed_count` | int | Rows successfully parsed (≥ 0) |
| `sample_results` | array of objects | Parsed sample outputs, first 5 at most |
| `errors` | array of strings | Per-row or overall errors |

`sample_results` holds at most the first 5 parsed pairs, whereas `parsed_count` is the full
count — the two are expected to differ once a sample parses more than 5 rows.

### Notable statuses

`422` (invalid request). No LLM is called, so no `502`/`503`.

---

## GET /api/v1/datasets/intelligence/formats

Return the dataset-type catalog, keyed by type name. The response is a plain JSON object.

### Response `200` — `object`

An opaque JSON object describing the supported dataset/training types.

### Notable statuses

`503` if the catalog provider is unavailable.

## See also

- [overview.md](overview.md) — base URL, pagination, error shape
- [authentication.md](authentication.md) — owners, admin, and `scope`
- [jobs.md](jobs.md) — a job requires a `ready` dataset
- [../concepts.md](../concepts.md) — the dataset concept and status lifecycle
- [../operations/configuration.md](../operations/configuration.md) — upload cap and LLM settings
