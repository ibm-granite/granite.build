# Dataset storage backends

A `StorageBackend` is the only place that knows *where* dataset bytes live.
The dataset service streams an upload to a local staging dir, then delegates:

- `persist(DatasetFiles) -> StorageLocator` — store the finalized files.
- `preview(DatasetRef, file, limit) -> list[dict]` — bounded rows for the view
  page; **must not load the whole file**.
- `delete(DatasetRef)` — remove all stored objects (idempotent).

The returned `StorageLocator` maps onto the existing `datasets.artifact_id` /
`datasets.artifact_url` columns — **no new column**. An object-storage backend
puts its URL (e.g. `s3://bucket/key`) in `artifact_url`.

Backends translate any underlying `fastapi.HTTPException` into the storage
contract errors (`StorageNotFound` / `StorageError`) so FastAPI types never leak
across the abstraction boundary.

## Built-ins
- `LocalStorageBackend` — disk under `UPLOAD_DIR`; locator is `(None, None)`.
- `GBStorageBackend` — `gb artifact push`; locator is the GB `(uuid, uri)`.
  Imported lazily by the factory (heavy gbcli/DB import chain).

## Adding a backend
1. Create `services/storage/<name>_backend.py` implementing `StorageBackend`.
2. Add a selection branch in `get_storage_backend() in services/storage/__init__.py`
   (import your backend lazily inside the branch if it has heavy deps).
3. Raise `StorageNotFound` / `StorageValidationError` / `StorageError` so the
   service maps them to 404 / 400 / 400.
Nothing else changes — the dataset service, routes, and frontend are untouched.
