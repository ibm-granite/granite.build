# Lineage Tracking

## Overview

Lineage tracking records the data provenance of builds, targets, and artifacts managed by gbserver. It captures what inputs were consumed, what outputs were produced, and the relationship between them — enabling reproducibility, auditing, and impact analysis.

Two backends exist behind a common interface:

- **`LakehouseLineageStore`** — writes lineage records to the DMF Lakehouse `job_stats` table using the `dmf-lib` library (legacy, default)
- **`WandBLineageStore`** — translates gbserver build/target/artifact data into [OpenLineage](https://openlineage.io/) events and emits them to Weights & Biases (new)

A feature flag controls which backend is active at runtime.

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

The backend is selected once on first call based on the `lakehouse_lineage` feature flag and cached for the process lifetime.

---

## Feature Flag

The `lakehouse_lineage` feature flag is defined per-environment in `src/gbserver/types/gbserverenvconfig.py`:

| Environment | Default | Override env var |
|---|---|---|
| PROD | `True` | `GBSERVER_FEATURE_LAKEHOUSE_LINEAGE` |
| STAGING | `True` | `GBSERVER_FEATURE_LAKEHOUSE_LINEAGE` |
| DEV | `True` | `GBSERVER_FEATURE_LAKEHOUSE_LINEAGE` |
| STANDALONE | `True` | `GBSERVER_FEATURE_LAKEHOUSE_LINEAGE` |

- **True** (default): Lakehouse backend (`LakehouseLineageStore`)
- **False**: WandB/OpenLineage backend (`WandBLineageStore`)

To switch to the WandB backend, set `GBSERVER_FEATURE_LAKEHOUSE_LINEAGE=false` in the environment.

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

The `GET /api/v1/lineage/build/{build_id}` and `GET /api/v1/lineage/target/{target_id}` endpoints call `create_jobstats_for_target()` to build lineage data on the fly without persisting it. These endpoints are Lakehouse-specific.

---

## Lakehouse Backend

**File:** `src/gbserver/lineage/lakehouse_jobstats.py`

`LakehouseLineageStore` extends both `ILineageStore` and `BaseLakehouseStorage` (from `dmf-lib`). It writes `JobStats` records to the DMF `job_stats` table.

Key behaviors:
- Creates one `JobStats` record per output artifact (multiple for checkpoints)
- Handles non-Lakehouse artifact URIs by creating placeholder tables in Lakehouse
- Retries persistence up to 10 times with exponential backoff
- Provides `does_release_id_exist()` for querying existing records (not part of the interface)

### Dependencies

Requires the `lakehouse` optional dependency group (`dmf-lib`). When not installed, the module still imports but `_get_base_class()` returns a Pydantic BaseModel stub.

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
- `namespace` = artifact URI (supports HuggingFace `hf://` and Lakehouse `lh://` URIs)
- `name` = artifact name (or UUID if unnamed)
- `facets` = `{artifact_id, artifact_uri, artifact_type}`

Supported artifact types for lineage: model, dataset, fileset, table, bucket. HuggingFace buckets (`hf:///buckets/org/name`) are logged with `type="bucket"` in W&B.

### Configuration

| Env var | Default | Purpose |
|---|---|---|
| `GBSERVER_LINEAGE_PROVIDER` | `wandb` | Backend for the LineageService factory |
| `GBSERVER_WANDB_API_KEY` | (empty) | WandB API key (secret) |
| `GBSERVER_WANDB_PROJECT` | `lineage-tracking` | WandB project name |
| `GBSERVER_WANDB_ENTITY` | `dmf-testing` | WandB entity/team (secret, varies per env) |
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

These endpoints use the `LineageService` abstraction directly (not `ILineageStore`) and are available regardless of the `lakehouse_lineage` feature flag.

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
├── lakehouse_jobstats.py    # LakehouseLineageStore (DMF job_stats table)
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

# Lakehouse lineage tests (requires IBM infrastructure)
pytest -s test/gbserver_test/lineage/test_jobstats.py -v -m ibm
pytest -s test/gbserver_test/api/test_lineage.py -v -m ibm
```
