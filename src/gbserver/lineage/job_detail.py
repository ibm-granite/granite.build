#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One job's full content, fetched from the store its index rows name.

The index rows are slim: they carry enough to draw the graph and a ``job_store``
saying where the rest lives (see :class:`JobStore`). This module is the read side of
that column -- one fetcher per store, each filling the same unified detail, so the
UI renders one shape whatever recorded the job.

A fetcher never raises for a store that cannot answer (a build pruned from this
server, a deleted W&B run, no access): it reports ``detail_available=False`` with a
reason, and the caller still gets everything the index itself recorded.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from gbserver.lineage.attributes import (
    PAYLOAD_EXECUTION_STATS,
    PAYLOAD_INPUT_PARAMS,
    PAYLOAD_OUTPUT_STATS,
    PAYLOAD_SOURCE_CODE,
    RETRIEVE,
    payload_detail,
)
from gbserver.storage.stored_lineage_row import JobStore
from gbserver.utils.redaction import redact_sensitive

logger = logging.getLogger(__name__)

PAYLOAD_KEYS = (
    PAYLOAD_INPUT_PARAMS,
    PAYLOAD_EXECUTION_STATS,
    PAYLOAD_OUTPUT_STATS,
    PAYLOAD_SOURCE_CODE,
)

# Called with a StoredBuild before its targets and steps are returned; raises when
# the caller may not read it. The API passes ``authorize_build_read_access``: step
# configs are exposed only to those who could read the build's own status.
AuthorizeBuild = Callable[[Any], None]


class JobDetailUnavailable(Exception):
    """The store has no detail for this job; the message says why."""


def _payload(source: Dict[str, Any]) -> Dict[str, Any]:
    """The four payloads from ``source``, with step configs redacted.

    The same masking as ``api.lineage.get_redacted_job_input_params``: every read
    path masks ``job_input_params`` unconditionally, whatever the writer did.
    """
    payload = {key: source[key] for key in PAYLOAD_KEYS if source.get(key)}
    if payload.get(PAYLOAD_INPUT_PARAMS):
        payload[PAYLOAD_INPUT_PARAMS] = redact_sensitive(payload[PAYLOAD_INPUT_PARAMS])
    return payload


def _fetch_lineage_job(job_id: str, retrieve: Dict[str, Any], ctx: "_Context"):
    job = ctx.admin_storage.lineage_job_storage.get_job(job_id)
    if job is None:
        raise JobDetailUnavailable(f"Job {job_id} is not in the lineage job table")
    return {"detail": _payload(payload_detail(job.attributes))}


def _fetch_target(job_id: str, retrieve: Dict[str, Any], ctx: "_Context"):
    """The target run and its steps, in the shape ``GET /builds/{id}/status`` uses."""
    storage = ctx.admin_storage
    target_run_uuid = str(retrieve.get("target_run_uuid") or "")
    build_id = str(retrieve.get("build_id") or "")
    target = (
        storage.target_storage.get_by_uuid(target_run_uuid) if target_run_uuid else None
    )
    if target is not None:
        build_id = target.build_id or build_id
    build = storage.build_storage.get_by_uuid(build_id) if build_id else None
    if build is None:
        raise JobDetailUnavailable(
            f"Build {build_id} is not on this server"
            if build_id
            else "No build recorded"
        )
    try:
        ctx.authorize_build(build)
    except Exception as e:
        raise JobDetailUnavailable("Not authorized to read this build") from e
    if target is None:
        raise JobDetailUnavailable(
            f"Target run {target_run_uuid} is not on this server"
        )

    steps = storage.step_storage.get_by_where({"target_id": target.uuid})
    build_dict = build.model_dump(mode="json")
    build_dict["build_archive"] = ""
    return {
        "build_id": build.uuid,
        "build": build_dict,
        "target": {
            **target.model_dump(mode="json"),
            "steps": [step.model_dump(mode="json") for step in steps],
        },
    }


def _fetch_wandb(job_id: str, retrieve: Dict[str, Any], ctx: "_Context"):
    """The W&B runs recorded for the job, merged; a local build's target if present.

    The W&B sink writes one run per output artifact, all carrying the same
    ``job_id`` and payloads, so the first run that has a payload speaks for all.
    """
    result: Dict[str, Any] = {}
    if retrieve.get("build_id") or retrieve.get("target_run_uuid"):
        try:
            result.update(_fetch_target(job_id, retrieve, ctx))
        except JobDetailUnavailable:
            pass

    runs = list(ctx.wandb_runs(job_id))
    if not runs:
        if result:
            return result
        raise JobDetailUnavailable("No W&B run found; it may have been deleted")
    payload: Dict[str, Any] = {}
    for run in runs:
        payload = _payload(dict(run.config or {}))
        if payload:
            break
    result["detail"] = payload
    result["origin_url"] = str(getattr(runs[0], "url", "") or "")
    return result


def _fetch_other(job_id: str, retrieve: Dict[str, Any], ctx: "_Context"):
    raise JobDetailUnavailable(
        "Recorded by an external system; only the index is known"
    )


_FETCHERS = {
    JobStore.LINEAGE_JOB: _fetch_lineage_job,
    JobStore.TARGETS: _fetch_target,
    JobStore.WANDB: _fetch_wandb,
    JobStore.OTHER: _fetch_other,
}


def _default_wandb_runs(job_id: str) -> List[Any]:
    # Deferred: W&B is only imported where a job actually lives there.
    # pylint: disable=import-outside-toplevel
    from gbserver.lineage.indexer import WandBLineageIndexer

    indexer = WandBLineageIndexer()
    return list(
        indexer._wandb_api().runs(  # pylint: disable=protected-access
            indexer._project_path(),  # pylint: disable=protected-access
            filters={"config.job_id": job_id},
            per_page=10,
        )
    )


class _Context:
    def __init__(
        self,
        admin_storage: Any,
        authorize_build: AuthorizeBuild,
        wandb_runs: Callable[[str], List[Any]],
    ) -> None:
        self.admin_storage = admin_storage
        self.authorize_build = authorize_build
        self.wandb_runs = wandb_runs


def fetch_job_detail(
    entry: Dict[str, Any],
    rows: List[Any],
    admin_storage: Any,
    authorize_build: AuthorizeBuild,
    wandb_runs: Optional[Callable[[str], List[Any]]] = None,
) -> Dict[str, Any]:
    """Extend a listing entry with the job's content from its store.

    Args:
        entry: the job's listing entry, as ``db_service._job_entry`` builds it.
        rows: the job's index rows; the first one's ``job_store`` and ``retrieve``
            decide where to look (every row of one job is written by one writer).
        admin_storage: the admin storage holding the job table, targets and builds.
        authorize_build: see :data:`AuthorizeBuild`.
        wandb_runs: lists the W&B runs of a job id; defaults to the configured W&B.
    """
    first = rows[0]
    try:
        job_store = JobStore(first.job_store)
    except ValueError:
        job_store = JobStore.OTHER
    retrieve = dict((first.attributes or {}).get(RETRIEVE) or {})
    ctx = _Context(admin_storage, authorize_build, wandb_runs or _default_wandb_runs)

    detail: Dict[str, Any] = {
        **entry,
        "job_store": job_store.value,
        "detail_available": True,
        "detail_error": None,
    }
    try:
        detail.update(_FETCHERS[job_store](entry["job_id"], retrieve, ctx))
    except JobDetailUnavailable as e:
        detail.update(detail_available=False, detail_error=str(e))
    except Exception as e:  # pylint: disable=broad-except
        logger.exception(
            "Fetching detail of job %s from %s failed", entry["job_id"], job_store.value
        )
        detail.update(
            detail_available=False,
            detail_error=f"Could not read {job_store.value}: {e}",
        )
    return detail
