# Lineage Tracking

## Overview

Lineage tracking records the data provenance of builds, targets, and artifacts managed by gbserver. It captures what inputs were consumed, what outputs were produced, and the relationship between them — enabling reproducibility, auditing, and impact analysis.

Granite.Build uses its internal admin tables to persist all information related to build lineage. In addition, it can translate gbserver build/target/artifact data into [OpenLineage](https://openlineage.io/) events and emit them to a centralized lineage store. The code currently supports Weights & Biases via `WandBLineageStore`, and any other lineage store can be supported by implementing the `ILineageStore` interface.

---

## Interface: `ILineageStore`

Defined in `src/gbserver/lineage/jobstats.py`.

```python
class ILineageStore(ABC):
    def add_jobstats_for_build(self, storage, build_id: str) -> None: ...
    def add_jobstats_for_build_target(self, storage, build_id: str, target_id: str) -> None: ...
    def add_jobstats_for_original_artifact(self, artifact, sources) -> None: ...
    def create_jobstats_for_target(self, storage, targetrun, build=None) -> Tuple: ...
    def create_jobstats_for_original_artifact(self, artifact, sources): ...
```

The `add_*` methods persist lineage to the backend. The `create_*` methods build lineage data structures without persisting — used by API endpoints for read-only queries.

### Singleton accessor

```python
from gbserver.lineage.jobstats import get_lineage_store

store = get_lineage_store()  # returns the active ILineageStore instance
```

The backend is selected once on first call and cached for the process lifetime.

---

## Call Sites

Lineage is recorded at three points in the system:

### 1. Build target completion

**File:** `src/gbserver/buildwatcher/buildrunner.py`

When a target finishes with `Status.SUCCESS`, the build runner records lineage:

```python
get_lineage_store().add_jobstats_for_build_target(storage, build_id, target_id)
```

This call is non-blocking — exceptions are caught and logged without failing the build.

### 2. Artifact registration with origins

**File:** `src/gbserver/api/artifacts.py`

When an artifact is registered via the API with `origin_uris` (source artifacts), lineage is created to record the provenance:

```python
get_lineage_store().add_jobstats_for_original_artifact(artifact, input_artifacts)
```

### 3. Lineage API queries (read-only)

**File:** `src/gbserver/api/lineage.py`

The `GET /api/v1/lineage/build/{build_id}` and `GET /api/v1/lineage/target/{target_id}` endpoints call `create_jobstats_for_target()` to build lineage data on the fly without persisting it.

---

## WandB/OpenLineage Backend

**File:** `src/gbserver/lineage/wandb_jobstats.py`

`WandBLineageStore` translates gbserver's build/target/artifact model into [OpenLineage 2.0.2](https://openlineage.io/spec/2-0-2/) events and emits them to WandB via the `LineageService` abstraction.

### Data mapping

| gbserver concept | OpenLineage field |
|---|---|
| Target UUID | `run.runId` |
| Target name | `job.name` |
| `{space_name}/{build_name}` | `job.namespace` |
| Target status | `eventType` (SUCCESS→COMPLETE, FAILED→FAIL, etc.) |
| Target `finished_at` | `eventTime` |
| Input artifacts | `inputs[]` (Dataset objects) |
| Output artifacts | `outputs[]` (Dataset objects) |
| Build ID, target ID, username | `run.facets.tags` |
| Build `source_uri` | `run.facets.source_code.url` |
| Step configs | `run.facets.job_input_params` |
| Build description | `job.facets.documentation.description` |

Each artifact becomes an OpenLineage `Dataset` with:
- `namespace` = artifact URI (supports HuggingFace `hf://`, S3 `s3://`, and other URI schemes)
- `name` = artifact name (or UUID if unnamed)
- `facets` = `{artifact_id, artifact_uri, artifact_type}`

Supported artifact types for lineage: model, dataset, fileset, table, bucket. HuggingFace buckets (`hf://huggingface.co/buckets/org/name`) are logged with `type="bucket"` in W&B.

### Configuration

| Env var | Default | Purpose |
|---|---|---|
| `GBSERVER_LINEAGE_PROVIDER` | `wandb` | Backend for the LineageService factory |
| `GBSERVER_WANDB_API_KEY` | (empty) | WandB API key (secret) |
| `GBSERVER_WANDB_PROJECT` | `lineage-tracking` | WandB project name |
| `GBSERVER_WANDB_ENTITY` | (varies per env) | WandB entity/team (secret) |
| `GBSERVER_WANDB_BASE_URL` | `https://ibm.wandb.io` | WandB server URL |

### Dependencies

Requires the `wandb` optional dependency group. Install with `pip install .[wandb]`.

---

## OpenLineage REST API

In addition to the internal lineage storage, gbserver exposes OpenLineage endpoints for external consumers to ingest and query lineage events directly via WandB.

### Endpoints

All under `/api/v1/lineage/`:

| Method | Path | Purpose |
|---|---|---|
| POST | `/` | Ingest an OpenLineage event |
| GET | `/{run_id}` | Retrieve a lineage event by WandB run ID |
| POST | `/search` | Search events by tags |
| POST | `/artifact/runs` | Search events by artifact repo ID |

These endpoints use the `LineageService` abstraction directly (not `ILineageStore`).

### Event schema

Events follow the [OpenLineage RunEvent](https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent) specification:

```json
{
  "eventType": "COMPLETE",
  "eventTime": "2024-04-15T10:30:00.000Z",
  "run": { "runId": "uuid", "facets": { "tags": {} } },
  "job": { "namespace": "ns", "name": "job-name", "facets": {} },
  "inputs": [{ "namespace": "s3://bucket", "name": "data", "facets": {} }],
  "outputs": [{ "namespace": "s3://bucket", "name": "model", "facets": {} }],
  "producer": "gbserver"
}
```

### Source files

- Models: `src/gbserver/lineage/openlineage_models.py`
- Service interface + factory: `src/gbserver/lineage/openlineage_service.py`
- WandB service implementation: `src/gbserver/lineage/wandb_service.py`
- Utility functions (HuggingFace URL helpers): `src/gbserver/lineage/openlineage_utils.py`
- API endpoints: `src/gbserver/api/lineage.py`

---

## Source Layout

```
src/gbserver/lineage/
├── jobstats.py              # ILineageStore ABC + get_lineage_store() singleton
├── wandb_jobstats.py        # WandBLineageStore (OpenLineage → WandB)
├── openlineage_models.py    # Pydantic models for OpenLineage events
├── openlineage_service.py   # LineageService ABC + factory
├── openlineage_utils.py     # HuggingFace URL/URI helpers
└── wandb_service.py         # WandBLineageService implementation
```

---

## Testing

```shell
# Run all lineage tests (no infrastructure required)
pytest -s test/gbserver_test/lineage/test_openlineage_models.py \
          test/gbserver_test/lineage/test_openlineage_utils.py \
          test/gbserver_test/lineage/test_openlineage_service.py \
          test/gbserver_test/lineage/test_wandb_jobstats.py -v
```
