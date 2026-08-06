# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Finalize dispatch for a completed tus file.

Records the completed file in the rendezvous; when it is the LAST file of the
upload group, wraps the staged file(s) as _DiskBackedUpload and runs the
UNCHANGED Dataset upload seam (the same methods append_chunk calls). Returns
the finalized dataset dict, or None when this file is not the last of its group.
"""

from __future__ import annotations

import logging
from typing import Optional

from .service import Dataset, _DiskBackedUpload
from .tus_metadata import UploadIntent
from .tus_rendezvous import UploadRendezvous

logger = logging.getLogger(__name__)


async def handle_completed_file(
    file_path: str,
    intent: UploadIntent,
    dataset: Dataset,
    rendezvous: UploadRendezvous,
    *,
    user_id: str,
) -> Optional[dict]:
    """Finalize a completed upload group.

    Identity (``user_id``) is resolved by the caller from the validated session,
    NEVER from client-supplied metadata.

    Once-only on SUCCESS: a duplicate completion of an already-finalized group
    no-ops. Retryable on FAILURE: if dispatch raises, the finalize claim is
    released so a tus-retried completion can re-attempt (the previous attempt
    rolled the dataset back).
    """
    group_complete = await rendezvous.record(
        intent.dataset_id,
        role=intent.role,
        expects=intent.expects,
        path=file_path,
        filename=intent.filename,
    )
    if not group_complete:
        return None  # waiting on sibling file(s)

    if not await rendezvous.claim_finalize(intent.dataset_id):
        return None  # another completion already finalized this group

    paths = rendezvous.paths(intent.dataset_id)
    names = rendezvous.filenames(intent.dataset_id)
    try:
        if intent.role == "source" or intent.expects == ["source"]:
            src = _DiskBackedUpload(paths["source"], intent.filename)
            try:
                return await dataset.upload_and_split_dataset(
                    dataset_id=intent.dataset_id,
                    user_id=user_id,
                    source_file=src,  # type: ignore[arg-type]
                    train_set_percentage=intent.train_set_percentage,
                    column_mapping=intent.column_mapping,
                )
            finally:
                await src.close()
        else:
            # Preserve each file's REAL client filename so downstream format
            # detection (extension-based) routes parquet vs jsonl correctly. The
            # hardcoded ".jsonl" fallbacks only apply if a role somehow arrived
            # without a filename.
            train = _DiskBackedUpload(paths["train"], names.get("train", "train.jsonl"))
            val = _DiskBackedUpload(
                paths["validation"], names.get("validation", "validation.jsonl")
            )
            try:
                return await dataset.upload_datasets(
                    dataset_id=intent.dataset_id,
                    user_id=user_id,
                    train_file=train,  # type: ignore[arg-type]
                    validation_file=val,  # type: ignore[arg-type]
                    column_mapping=intent.column_mapping,
                )
            finally:
                await train.close()
                await val.close()
    except Exception:
        # Failed finalize: release the once-only claim so a tus-retried completion
        # can re-attempt. The dataset was rolled back by the upload seam.
        await rendezvous.release_finalize(intent.dataset_id)
        raise
    finally:
        # Drop rendezvous file/lock state; on SUCCESS the dataset_id stays in
        # _finalized (once-only). On FAILURE we already released it above, so the
        # clear() below simply tidies the per-group buffers and a retry re-runs.
        rendezvous.clear(intent.dataset_id)
