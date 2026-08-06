# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Dataset service: create/list/get/delete + upload orchestration (single-shot,
auto-split, and chunked). Persistence delegated to a StorageBackend; LLM
parsing/mapping to DatasetIntelligence. Split from the former monolithic
dataset_service.py."""

import asyncio
import logging
import os
import shutil
from typing import Dict, List, Optional, Set
from uuid import UUID

import constants
import models as api
from fastapi import File, HTTPException, UploadFile
from services import db_service
from services.file import (
    CONFIG,
    FileValidator,
    count_records,
    remap_jsonl_file,
    save_raw_parquet_with_mapping,
    save_uploaded_file,
    stream_split_jsonl,
    stream_split_parquet,
    stream_to_disk,
)
from services.storage import (
    DatasetFiles,
    DatasetRef,
    StorageError,
    StorageNotFound,
    get_storage_backend,
)

from .intelligence import DatasetIntelligence

logger = logging.getLogger(__name__)

# Guard against duplicate concurrent uploads for the same dataset
# (e.g. ingress proxy retries due to timeout during gb artifact push)
_active_uploads: Set[str] = set()
_upload_events: Dict[str, asyncio.Event] = {}

# Staging directory for in-progress chunked uploads. Each upload_id accumulates
# its chunks into <UPLOAD_DIR>/.chunks/<upload_id>/ before being assembled.
_CHUNK_STAGING_DIR = os.path.join(CONFIG["UPLOAD_DIR"], ".chunks")

# Finalize-in-progress guard keyed by upload_id. A client/proxy that retries the
# final chunk while the first finalize is still running (GB push can be slow)
# would otherwise double-finalize; this rejects the duplicate fast instead of
# colliding with the dataset-level _with_upload_guard.
_finalizing_uploads: Set[str] = set()


class _DiskBackedUpload:
    """
    Minimal UploadFile-compatible wrapper around a file already on disk.

    Chunked uploads are assembled to a single file; this lets that assembled
    file flow through the existing upload methods (which expect an UploadFile
    exposing ``filename``, async ``read``/``seek``, and a ``.file`` handle for
    ``FileValidator.validate_file_size``) without buffering it in memory.
    """

    def __init__(self, path: str, filename: str):
        self._path = path
        self.filename = filename
        self.file = open(path, "rb")

    async def read(self, size: int = -1) -> bytes:
        return await asyncio.to_thread(self.file.read, size)

    async def seek(self, offset: int) -> None:
        await asyncio.to_thread(self.file.seek, offset)

    async def close(self) -> None:
        # Idempotent: safe to call more than once and never raises on close.
        if self.file is not None and not self.file.closed:
            await asyncio.to_thread(self.file.close)


class Dataset:
    def __init__(self, db: db_service.Database):
        self.db = db
        self._intelligence = DatasetIntelligence()

    async def push_dataset(self, dataset: api.DatasetInfo) -> api.DatasetInfo:
        if dataset.id is not None and await self.db.check_dataset_exists(dataset.id):
            # On update, guard against renaming onto another dataset this user already owns.
            existing = await self.db.get_dataset_by_name_and_user(
                dataset_name=dataset.name, user_id=dataset.user_id
            )
            if existing is not None and existing.get("id") != str(dataset.id):
                raise HTTPException(
                    status_code=409,
                    detail=f"A dataset named '{dataset.name}' already exists for this user.",
                )
            return await self.db.update_dataset(dataset=dataset)
        else:
            existing = await self.db.get_dataset_by_name_and_user(
                dataset_name=dataset.name, user_id=dataset.user_id
            )
            if existing is not None:
                # Distinguish an UNFINALIZED PLACEHOLDER from a real duplicate.
                #
                # After a page reload the client loses its in-memory dataset id, so it
                # calls createDataset again with the same name.  The DB row from the
                # pre-reload attempt already exists, but finalization (which sets
                # train_file_size / train_records via update_dataset_metadata) never
                # ran — it is still an empty placeholder.  Blocking this with 409 is
                # wrong: the user is simply resuming/retrying the same upload.
                #
                # Rule: a row is a placeholder when BOTH train_file_size AND
                # train_records are NULL (the DEFAULT for those columns; they are only
                # populated by _finalize_upload → update_dataset_metadata).  Reuse it
                # so the tus upload resumes into the same row and the client gets back
                # the correct dataset id without creating a duplicate DB entry.
                #
                # A row is finalized (and therefore a genuine name collision) when
                # train_file_size OR train_records is not NULL — keep the 409 there.
                is_placeholder = (
                    existing.get("train_file_size") is None
                    and existing.get("train_records") is None
                )
                if is_placeholder:
                    logger.info(
                        "Reusing unfinalized placeholder dataset %s for user %s "
                        "(resume after reload)",
                        existing.get("id"),
                        dataset.user_id,
                    )
                    return api.DatasetInfo(
                        id=UUID(str(existing["id"])),
                        user_id=existing.get("user_id"),
                        name=existing["name"],
                        description=existing.get("description") or "",
                    )
                raise HTTPException(
                    status_code=409,
                    detail=f"A dataset named '{dataset.name}' already exists for this user.",
                )
            # Prevent shadowing a SYSTEM_USER dataset of the same name for consistency
            # with the configurations flow.
            system_existing = await self.db.get_dataset_by_name_and_user(
                dataset_name=dataset.name, user_id=constants.SYSTEM_USER
            )
            if system_existing is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"'{dataset.name}' is a reserved system dataset name.",
                )
            return await self.db.insert_dataset(dataset=dataset)

    async def get_datasets(self, user_id: str) -> list[api.DatasetInfo]:
        return await self.db.get_datasets(user_id)

    def get_autotune_dataset_types(self) -> dict:
        return self._intelligence.get_autotune_dataset_types()

    async def generate_parsing_strategy(self, *args, **kwargs):
        return await self._intelligence.generate_parsing_strategy(*args, **kwargs)

    async def suggest_column_mapping(self, *args, **kwargs):
        return await self._intelligence.suggest_column_mapping(*args, **kwargs)

    async def get_dataset(self, id: str, user_id: str) -> api.DatasetInfo:
        dataset = await self.db.get_dataset(dataset_id=id, user_id=user_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="invalid dataset id")
        data_format = dataset.get("data_format", "jsonl")
        ext = data_format  # 'jsonl' or 'parquet'
        # Preview only: read at most 10 records per file so viewing a large
        # dataset never loads the whole file into memory (which OOM-killed the pod).
        PREVIEW_LIMIT = 10
        backend = get_storage_backend()
        ref = DatasetRef(
            dataset_id=dataset["id"],
            name=dataset["name"],
            data_format=data_format,
            artifact_url=dataset.get("artifact_url"),
        )
        try:
            dataset["train_data"] = await backend.preview(
                ref, f"{dataset['train_file']}.{ext}", limit=PREVIEW_LIMIT
            )
            dataset["validation_data"] = await backend.preview(
                ref, f"{dataset['validation_file']}.{ext}", limit=PREVIEW_LIMIT
            )
        except StorageNotFound as e:
            raise HTTPException(status_code=404, detail=str(e))
        except StorageError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return dataset

    async def _with_upload_guard(self, dataset_id: str, user_id: str, do_upload):
        """
        Deduplicate concurrent uploads for the same dataset (e.g. ingress proxy
        retries during a slow gb artifact push). A second request for an
        in-flight dataset waits for the original and returns its result rather
        than racing it. ``do_upload`` is a zero-arg coroutine factory.
        """
        if dataset_id in _active_uploads:
            logger.warning(
                f"Duplicate upload request for dataset {dataset_id}, "
                "waiting for original to complete"
            )
            event = _upload_events.get(dataset_id)
            if event:
                await event.wait()
            return await self.db.get_dataset(dataset_id=dataset_id, user_id=user_id)

        _active_uploads.add(dataset_id)
        _upload_events[dataset_id] = asyncio.Event()
        try:
            return await do_upload()
        finally:
            _active_uploads.discard(dataset_id)
            event = _upload_events.pop(dataset_id, None)
            if event:
                event.set()

    async def _finalize_upload(
        self,
        dataset_id: str,
        user_id: str,
        data: dict,
        metadata: dict,
    ) -> api.DatasetInfo:
        """
        Shared tail for every upload path: persist the saved files through the
        active storage backend, record the returned locator in metadata, persist
        metadata, and clean up on failure. Both the custom-validation and
        auto-split paths end here so the persist/DB/cleanup logic lives in
        exactly one place.
        """
        dataset_path = f"{CONFIG['UPLOAD_DIR']}/{data['id']}/{data['name']}"
        logger.debug("UPLOAD finalize: %s", dataset_path)
        try:
            backend = get_storage_backend()
            ext = metadata.get("data_format", "jsonl")
            files = DatasetFiles(
                dataset_id=str(data["id"]),
                name=data["name"],
                data_format=ext,
                local_dir=dataset_path,
                train_file=f"{data['train_file']}.{ext}",
                validation_file=f"{data['validation_file']}.{ext}",
            )
            locator = await backend.persist(files)
            metadata["artifact_id"] = locator.artifact_id
            metadata["artifact_url"] = locator.artifact_url
            return await self.db.update_dataset_metadata(
                id=dataset_id, user_id=user_id, metadata=metadata
            )
        except HTTPException:
            raise
        # Includes StorageError from backend.persist (e.g. GB push failure) -> rollback + HTTP 400.
        except Exception as e:
            logger.error(e)
            await self.delete_dataset(id=dataset_id, user_id=user_id)
            raise HTTPException(status_code=400, detail=f"Something went wrong: {e}")

    async def upload_datasets(
        self,
        dataset_id: str,
        user_id: str,
        train_file: UploadFile = File(...),
        validation_file: UploadFile = File(...),
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> api.DatasetInfo:
        """
        Upload dataset files (CSV, JSON, JSONL, Parquet, or Text format)
        """
        return await self._with_upload_guard(
            dataset_id,
            user_id,
            lambda: self._do_upload_datasets(
                dataset_id, user_id, train_file, validation_file, column_mapping
            ),
        )

    async def _do_upload_datasets(
        self,
        dataset_id: str,
        user_id: str,
        train_file: UploadFile,
        validation_file: UploadFile,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> api.DatasetInfo:
        files: List[UploadFile] = []
        metadata = {}
        data = await self.db.get_dataset(dataset_id=dataset_id, user_id=user_id)

        if not data:
            raise HTTPException(
                status_code=404, detail=f"Dataset {dataset_id} not found"
            )

        # Detect if parquet
        train_ext = (
            train_file.filename.split(".")[-1].lower()
            if train_file.filename
            else "jsonl"
        )
        is_parquet = train_ext == "parquet"

        train_file.filename = data["train_file"] + "." + train_ext
        files.append(train_file)

        val_ext = (
            validation_file.filename.split(".")[-1].lower()
            if validation_file.filename
            else train_ext
        )
        validation_file.filename = data["validation_file"] + "." + val_ext
        files.append(validation_file)

        for file in files:
            try:
                FileValidator.validate_file_size(file)
                file_format = FileValidator.validate_file_format(file.filename)

                if is_parquet and column_mapping:
                    # Parquet with column mapping: apply mapping server-side and save as parquet
                    file_path, record_count = await save_raw_parquet_with_mapping(
                        file,
                        filename=file.filename,
                        dataset_id=data["id"],
                        dataset_name=data["name"],
                        column_mapping=column_mapping,
                    )
                elif not is_parquet and column_mapping:
                    # JSONL with column mapping: stream to a temp file, then remap
                    # line-by-line into the final path (server-side equivalent of
                    # the old client-side mapping — no full in-memory load).
                    dir_path = os.path.join(
                        CONFIG["UPLOAD_DIR"], data["id"], data["name"]
                    )
                    tmp_path = os.path.join(dir_path, f".{file.filename}.raw")
                    final_path = os.path.join(dir_path, file.filename)
                    await stream_to_disk(file, tmp_path)
                    try:
                        record_count = await remap_jsonl_file(
                            tmp_path, final_path, column_mapping
                        )
                    finally:
                        await asyncio.to_thread(
                            lambda p=tmp_path: os.path.exists(p) and os.remove(p)
                        )
                    file_path = final_path
                else:
                    # No mapping: stream raw file to disk, count from the saved file.
                    file_path = await save_uploaded_file(
                        file, dataset_id=data["id"], dataset_name=data["name"]
                    )
                    record_count = await count_records(file_path, file_format)

                file_size = os.path.getsize(file_path)
                if file is train_file:
                    metadata["train_records"] = record_count
                    metadata["train_file_size"] = file_size
                elif file is validation_file:
                    metadata["validation_records"] = record_count
                    metadata["validation_file_size"] = file_size
            except HTTPException as he:
                raise he
            except Exception as e:
                logger.debug(f"Error processing file: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"An error occurred while processing the file: {str(e)}",
                )

        metadata["data_format"] = "parquet" if is_parquet else "jsonl"
        return await self._finalize_upload(dataset_id, user_id, data, metadata)

    async def upload_and_split_dataset(
        self,
        dataset_id: str,
        user_id: str,
        source_file: UploadFile,
        train_set_percentage: int = 80,
        column_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Upload a single file and split it into train/validation sets on the backend.
        This is optimized for large files.
        """
        return await self._with_upload_guard(
            dataset_id,
            user_id,
            lambda: self._do_upload_and_split_dataset(
                dataset_id, user_id, source_file, train_set_percentage, column_mapping
            ),
        )

    async def _do_upload_and_split_dataset(
        self,
        dataset_id: str,
        user_id: str,
        source_file: UploadFile,
        train_set_percentage: int = 80,
        column_mapping: Optional[Dict[str, str]] = None,
    ):
        metadata = {}
        data = await self.db.get_dataset(dataset_id=dataset_id, user_id=user_id)

        if not data:
            raise HTTPException(
                status_code=404, detail=f"Dataset {dataset_id} not found"
            )

        try:
            FileValidator.validate_file_size(source_file)
            file_format = FileValidator.validate_file_format(source_file.filename)
            is_parquet = file_format == "parquet"

            save_ext = (
                "parquet" if is_parquet else source_file.filename.split(".")[-1].lower()
            )
            dir_path = os.path.join(CONFIG["UPLOAD_DIR"], data["id"], data["name"])
            train_path = os.path.join(dir_path, data["train_file"] + "." + save_ext)
            validation_path = os.path.join(
                dir_path, data["validation_file"] + "." + save_ext
            )

            # Stream the source to disk first (bounded memory), then split it in a
            # second streaming pass — never materializing all records at once.
            # The dataset id seeds a reproducible-yet-shuffled train/val assignment.
            tmp_source = os.path.join(dir_path, f".source_upload.{save_ext}")
            await stream_to_disk(source_file, tmp_source)

            try:
                if is_parquet:
                    train_records, validation_records = await stream_split_parquet(
                        src_path=tmp_source,
                        train_path=train_path,
                        val_path=validation_path,
                        train_set_percentage=train_set_percentage,
                        seed=str(data["id"]),
                        column_mapping=column_mapping,
                    )
                else:
                    train_records, validation_records = await stream_split_jsonl(
                        src_path=tmp_source,
                        train_path=train_path,
                        val_path=validation_path,
                        train_set_percentage=train_set_percentage,
                        seed=str(data["id"]),
                    )
            finally:
                await asyncio.to_thread(
                    lambda: os.path.exists(tmp_source) and os.remove(tmp_source)
                )

            if train_records == 0 and validation_records == 0:
                raise HTTPException(
                    status_code=400,
                    detail="File is empty or contains no valid records",
                )

            metadata["train_records"] = train_records
            metadata["train_file_size"] = os.path.getsize(train_path)
            metadata["validation_records"] = validation_records
            metadata["validation_file_size"] = os.path.getsize(validation_path)
            metadata["data_format"] = "parquet" if is_parquet else "jsonl"

        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"An error occurred while processing the file: {str(e)}",
            )

        return await self._finalize_upload(dataset_id, user_id, data, metadata)

    async def append_chunk(
        self,
        dataset_id: str,
        auth_email: str,
        user,
        upload_id: Optional[str],
        chunk_index: int,
        chunk_count: int,
        train_file: UploadFile,
        validation_file: Optional[UploadFile],
        train_set_percentage: Optional[int],
        column_mapping: Optional[Dict[str, str]],
    ):
        """
        Accumulate one chunk of a chunked upload.

        Intermediate chunks ONLY append bytes to the staging file — no DB lookup,
        no dedup guard — so per-chunk overhead stays near-zero (that overhead × N
        chunks was what made large uploads crawl). On the final chunk the staging
        file is wrapped as a disk-backed upload and routed through the SAME upload
        logic as a single-shot request, so observable behavior (split vs custom
        validation, metadata, GB push, response) is identical.

        ``user`` is the user service; ``user_id`` is resolved lazily here only
        when finalizing. Returns ``None`` for intermediate chunks and the
        finalized DatasetInfo on the last chunk.
        """
        if not upload_id:
            raise HTTPException(status_code=400, detail="upload_id is required")

        stage_dir = os.path.join(_CHUNK_STAGING_DIR, upload_id)
        train_part = os.path.join(stage_dir, "train.part")

        # Append this train chunk to the staging file (bounded memory, no other
        # per-chunk work).
        await stream_to_disk(train_file, train_part, mode="ab")

        # Intermediate chunk: acknowledge and wait for the rest.
        if chunk_index < chunk_count - 1:
            return None

        # Final chunk. Guard against a retried/duplicate final chunk (e.g. proxy
        # timeout retry) double-finalizing the same upload.
        if upload_id in _finalizing_uploads:
            raise HTTPException(
                status_code=409,
                detail="This upload is already being finalized.",
            )
        _finalizing_uploads.add(upload_id)

        # Resolve user_id once, now that we actually need it.
        user_id = (await user.get_user(auth_email))["id"]

        original_train_name = train_file.filename or "train.jsonl"
        assembled_train = _DiskBackedUpload(train_part, original_train_name)
        try:
            if train_set_percentage is not None and validation_file is None:
                result = await self.upload_and_split_dataset(
                    dataset_id=dataset_id,
                    user_id=user_id,
                    source_file=assembled_train,  # type: ignore[arg-type]
                    train_set_percentage=train_set_percentage,
                    column_mapping=column_mapping,
                )
            elif validation_file is not None:
                result = await self.upload_datasets(
                    dataset_id=dataset_id,
                    user_id=user_id,
                    train_file=assembled_train,  # type: ignore[arg-type]
                    validation_file=validation_file,
                    column_mapping=column_mapping,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Either provide both train and validation files, or "
                        "provide train file with train_set_percentage"
                    ),
                )
        finally:
            await assembled_train.close()
            await asyncio.to_thread(shutil.rmtree, stage_dir, ignore_errors=True)
            _finalizing_uploads.discard(upload_id)
        return result

    @staticmethod
    async def cleanup_stale_chunk_uploads(max_age_minutes: int = 360) -> int:
        """
        Remove abandoned chunk-staging directories under <UPLOAD_DIR>/.chunks.

        A chunked upload only deletes its own staging dir on the FINAL chunk, so
        an upload that the client never finishes (disconnect, crash, navigate
        away) would otherwise leave its partial files on disk forever. A periodic
        sweep deletes staging dirs whose last modification is older than
        ``max_age_minutes``. Returns the number of directories removed.
        """
        import time

        def _sweep() -> int:
            if not os.path.isdir(_CHUNK_STAGING_DIR):
                return 0
            cutoff = time.time() - max_age_minutes * 60
            removed = 0
            for name in os.listdir(_CHUNK_STAGING_DIR):
                path = os.path.join(_CHUNK_STAGING_DIR, name)
                try:
                    if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                        shutil.rmtree(path, ignore_errors=True)
                        removed += 1
                except OSError:
                    # Best-effort: skip entries we can't stat/remove this pass.
                    continue
            return removed

        return await asyncio.to_thread(_sweep)

    async def delete_dataset(self, id: str, user_id: str) -> bool:
        dataset = await self.db.get_dataset(dataset_id=id, user_id=user_id)
        if not dataset:
            return False
        backend = get_storage_backend()
        try:
            await backend.delete(
                DatasetRef(
                    dataset_id=dataset["id"],
                    name=dataset["name"],
                    data_format=dataset.get("data_format", "jsonl"),
                )
            )
        except StorageNotFound:
            # Already gone in the backend — deletion is idempotent; proceed to drop the row.
            pass
        except StorageError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return await self.db.delete_dataset(id, user_id)
