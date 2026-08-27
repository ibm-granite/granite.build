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

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from gbcommon.types.constants import DEFAULT_GH_DOMAIN, is_public_github
from gbserver.lineage.jobstats import ILineageStore
from gbserver.lineage.openlineage_service import LineageService, LineageServiceFactory
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import (
    GB_JOB_STATS_DETAIL_CATEGORY,
    GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_JOB_NAME,
    GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_TYPE,
    GB_JOB_STATS_DETAIL_TYPE,
)
from gbserver.types.status import Status
from gbserver.utils.redaction import redact_sensitive, scrub_url_credentials
from gbserver.utils.utils import get_uuid

_LINEAGE_REPO_ORG = "ibm-granite" if is_public_github() else "granite-dot-build"
LINEAGE_PRODUCER_URL = f"https://{DEFAULT_GH_DOMAIN}/{_LINEAGE_REPO_ORG}/granite.build"
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

_STATUS_TO_EVENT_TYPE: Dict[Status, str] = {
    Status.SUCCESS: "COMPLETE",
    Status.FAILED: "FAIL",
    Status.RUNNING: "RUNNING",
    Status.PENDING: "START",
    Status.SUBMITTED: "START",
    Status.CANCELLED: "ABORT",
    Status.CANCEL_REQUESTED: "RUNNING",
    Status.INVALID: "FAIL",
}


def _lh_uri_to_namespace_and_name(uri: str) -> Optional[Tuple[str, str]]:
    from urllib.parse import urlparse

    from gbcommon.uri.lh import LhType, LhURI

    parse = urlparse(uri)
    if parse.scheme not in LhURI.get_supported_schemes():
        return None

    lh = LhURI(parse)
    namespace = lh.get_lh_namespace()
    lh_type = lh.get_lh_type()
    if lh_type == LhType.TABLE:
        name = lh.get_lh_table_name()
    elif lh_type == LhType.FILESET:
        name = f"{lh.get_lh_fileset_label()}-{lh.get_lh_fileset_version()}"
    elif lh_type == LhType.MODEL:
        name = f"{lh.get_lh_model_label()}-{lh.get_lh_model_revision()}"
    elif lh_type == LhType.DATASET:
        name = lh.get_lh_dataset_name()
    else:
        return None
    return namespace, name


def _build_target_artifact_reference(
    target_name: str,
    target_artifact_name: str,
    is_input: bool,
    index: int,
) -> str:
    in_or_out = "inputs" if is_input else "outputs"
    reference = f"{target_name}.{in_or_out}.{target_artifact_name}"
    if index >= 0:
        reference = f"{reference}[{index}]"
    return reference


def _artifact_to_lineage_entry(
    artifact: ArtifactRegistration,
    target_artifact_name: str = "",
    target_name: str = "",
    is_input: bool = True,
    index: int = -1,
) -> dict:
    from urllib.parse import urlparse

    from gbcommon.uri.hf import HfURI

    artifact_type = artifact.type
    if artifact.uri:
        from gbcommon.uri.uri import UnknownURIScheme
        from gbcommon.uri.utils import get_artifact_type

        try:
            artifact_type = get_artifact_type(artifact.uri)
        except UnknownURIScheme:
            pass

    namespace = artifact.uri
    name = artifact.name or target_artifact_name or artifact.uuid

    target_artifact_reference = _build_target_artifact_reference(
        target_name, target_artifact_name, is_input, index
    )

    facets: dict[str, Any] = {
        "artifact_id": artifact.uuid,
        "artifact_uri": artifact.uri,
        "artifact_type": artifact_type.name,
        "target_artifact_reference": target_artifact_reference,
        "gb-artifact-id": artifact.uuid,
        "gb-artifact-uri": artifact.uri,
        "gb-build-id": artifact.created_by_build_id,
        "gb-target-id": artifact.created_by_target_id,
        "gb-build-target-artifact": target_artifact_reference,
    }
    facets.update(artifact.model_dump(mode="json"))

    uri = artifact.uri
    parse = urlparse(uri)
    if parse.scheme in HfURI.get_supported_schemes():
        hf = HfURI(parse)
        parts = hf._parts()
        repo_id = f"{parts.owner}/{parts.repo}"
        namespace = parts.owner
        name = repo_id
    else:
        lh_result = _lh_uri_to_namespace_and_name(uri)
        if lh_result is not None:
            namespace, name = lh_result

    return {
        "namespace": namespace,
        "name": name,
        "uri": uri,
        "facets": facets,
    }


def _add_jobstats_mirror_fields(event: dict) -> None:
    # The REST jobstats endpoints expect a flat JobStats-shaped dict (with
    # release_id / job_details / sources / targets at the top level). wandb
    # stores these inside run.facets.job_details + inputs/outputs, so mirror
    # them for readers. wandb itself ignores unknown top-level keys.
    job_details = event.get("run", {}).get("facets", {}).get("job_details", {})
    event["release_id"] = job_details.get("release_id", "")
    event["job_details"] = job_details
    event["sources"] = event.get("inputs", [])
    event["targets"] = event.get("outputs", [])


# How long a "fully recorded in wandb" verdict stays trusted without re-asking
# wandb. The reconciler re-selects already-recorded targets on every scan by
# design (the watermark overlap window deliberately keeps the newest target in
# range), so in steady state the same candidate is verified every
# monitoring_interval — a network round-trip per scan to re-learn an unchanged
# fact. Caching the positive verdict collapses that to one call per TTL.
#
# Only *positive* verdicts are cached, and only for this long: a run deleted in
# wandb, or a target whose runs were only partially emitted, must eventually be
# noticed and re-recorded. The TTL bounds that staleness instead of making it
# permanent, and the cache is per-process, so a restart always re-verifies.
_RECORDED_CACHE_TTL_SECONDS = 6 * 60 * 60


class WandBLineageStore(ILineageStore):

    def __init__(self) -> None:
        self._service: LineageService = LineageServiceFactory.create("wandb")
        # (target uuid, expected run count) -> monotonic deadline after which the
        # verdict is re-checked. The expected count is part of the key, not just
        # the value: a verdict of "recorded" means "has all the runs we expected
        # *at the time we asked*", so a target that later grows an output (and so
        # expects more runs) must be re-checked rather than inherit the old
        # verdict. ``None`` is a distinct key, matching the presence-check
        # fallback for candidates with no expected count.
        # Monotonic, not wall-clock, so a system clock adjustment cannot expire
        # every entry at once or freeze them past the TTL.
        self._recorded_until: dict[tuple[str, Optional[int]], float] = {}

    def _build_events_for_target(
        self,
        storage: SingletonAdminStorage,
        build: StoredBuild,
        targetrun: StoredTargetRun,
    ) -> Tuple[List[dict], Dict[str, List[dict]]]:
        event_type = _STATUS_TO_EVENT_TYPE.get(targetrun.status, "OTHER")
        event_time = (
            targetrun.finished_at.isoformat()
            if targetrun.finished_at
            else (
                targetrun.started_at.isoformat()
                if targetrun.started_at
                else build.created_time.isoformat()
            )
        )

        inputs = []
        for target_artifact_name, uuid in targetrun.input_artifacts.items():
            artifact = storage.artifact_registry.get_by_uuid(uuid)
            if artifact and isinstance(artifact, ArtifactRegistration):
                inputs.append(
                    _artifact_to_lineage_entry(
                        artifact,
                        target_artifact_name,
                        target_name=targetrun.name,
                        is_input=True,
                        index=-1,
                    )
                )

        step_configs = []
        steps = storage.step_storage.get_by_where({"target_id": targetrun.uuid})
        for step in steps:
            # step.config is the rendered build.yaml input and step.metadata is
            # runtime data the step pushed (e.g. commit_hash). jobstats is readable
            # by any space member (not just the build owner/admin), so both are
            # emitted with secret-*named* keys masked via redact_sensitive, which
            # also scrubs userinfo@ credentials out of any URL-shaped value. The
            # definition_uri is scrubbed the same way so a credentialed BYOS clone
            # URL (git+ssh://token@... / https://token@...) cannot leak here.
            step_configs.append(
                {
                    "uri": scrub_url_credentials(step.definition_uri),
                    "config": redact_sensitive(step.config),
                    "metadata": redact_sensitive(step.metadata),
                }
            )

        started_at = (
            targetrun.started_at.isoformat() if targetrun.started_at else event_time
        )
        completed_at = (
            targetrun.finished_at.isoformat() if targetrun.finished_at else ""
        )

        base_event: Dict[str, Any] = {
            "eventType": event_type,
            "eventTime": event_time,
            "run": {
                "runId": targetrun.uuid,
                "facets": {
                    "tags": {
                        "build_id": build.uuid,
                        "target_id": targetrun.uuid,
                        "username": build.username,
                        "space_name": build.space_name,
                    },
                    "source_code": {
                        "url": build.source_uri,
                        "commit_hash": "",
                        "path": "",
                    },
                    "job_input_params": {"steps": step_configs},
                    "execution_stats": {},
                    "job_details": {
                        "job_id": targetrun.uuid,
                        "job_type": GB_JOB_STATS_DETAIL_TYPE,
                        "category": GB_JOB_STATS_DETAIL_CATEGORY,
                        "job_status": targetrun.status.name,
                        "job_started_at": started_at,
                        "job_completed_at": completed_at,
                        "release_id": targetrun.build_id,
                        "owner": build.username,
                        "job_output_stats": {},
                    },
                },
            },
            "job": {
                "namespace": f"{build.space_name}/{build.name}",
                "name": targetrun.name,
                "facets": {},
            },
            "producer": LINEAGE_PRODUCER_URL,
            "schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent",
        }

        if build.description:
            base_event["job"]["facets"]["documentation"] = {
                "description": build.description,
            }

        events_list: List[dict] = []
        events_dict: Dict[str, List[dict]] = {}

        # NOTE: the number of events emitted here (one per output artifact across
        # all output-artifact lists, or one "no-output" event below) must stay in
        # lockstep with lineage_reconciler.expected_run_count, which derives the
        # same count from the target in memory to detect partial records.
        for (
            target_artifact_name,
            output_artifact_list,
        ) in targetrun.output_artifacts.items():
            target_events: List[dict] = []
            include_index = len(output_artifact_list) > 1
            index = -1
            for output_uuid in output_artifact_list:
                if include_index:
                    index += 1
                artifact = storage.artifact_registry.get_by_uuid(output_uuid)
                outputs = []
                if artifact and isinstance(artifact, ArtifactRegistration):
                    outputs.append(
                        _artifact_to_lineage_entry(
                            artifact,
                            target_artifact_name,
                            target_name=targetrun.name,
                            is_input=False,
                            index=index,
                        )
                    )
                event = {
                    **base_event,
                    "inputs": inputs,
                    "outputs": outputs,
                }
                # Give each output-artifact event its own wandb run so
                # history rows are not collapsed when multiple events share
                # a single resumed run. Keeps counts aligned with the number
                # of output artifacts. The job_id in job_details still points
                # back to the logical target (targetrun.uuid).
                #
                # The id is a fresh random uuid, not derived from the target and
                # output uuids. Dedup is therefore carried entirely by the
                # target_id tag in run.facets.tags (see LineageService.
                # filter_unrecorded), which is why that tag must be present on
                # EVERY emitted event: a run without it is invisible to the
                # dedup query, cannot be counted toward expected_run_count, and
                # is unreclaimable -- no later scan can find it or replace it.
                #
                # Random is REQUIRED here, not incidental. Deriving the id from
                # the target/output (the scheme this replaced) means a run
                # DELETED in wandb can never be re-created: wandb refuses a
                # deleted run's id, and a derived id recomputes to that same
                # tombstoned value on every later scan, so the target becomes
                # permanently unrecordable. That happened with intentional
                # deletions; see commit 5824ae99 and the extended note in
                # WandBLineageService.filter_unrecorded, which also explains the
                # partial-record trade-off this buys and why it is accepted.
                #
                # Tag the run with the output artifact it represents. base_event
                # cannot carry this: its tags are shared by every event of the
                # target, while output_uuid identifies just this one. Run ids are
                # random and carry no output information, so without this tag the
                # only way to find the run for a given output is to fetch the
                # target's runs and inspect their outputs facet; with it the
                # lookup is a tag filter like the target_id ones above.
                #
                # Additive only: it does not affect dedup. filter_unrecorded
                # matches on "target_id=" tags and skips every other key, and
                # tags serialize generically as "k=v"
                # (WandBLineageService._process_event), so nothing else changes.
                event["run"] = {
                    **base_event["run"],
                    "runId": get_uuid(),
                    "facets": {
                        **base_event["run"]["facets"],
                        "tags": {
                            **base_event["run"]["facets"]["tags"],
                            "output_id": output_uuid,
                        },
                    },
                }
                _add_jobstats_mirror_fields(event)
                target_events.append(event)
            events_list.extend(target_events)
            events_dict[target_artifact_name] = target_events

        # A successful target with no output artifacts still represents a real
        # job run and must produce one event so its run is recorded — even when
        # it has no inputs either (e.g. a pure generation/compute target). Guard
        # only on the absence of output-artifact events, not on having inputs;
        # otherwise an artifact-less target emits nothing and the reconciler
        # silently marks it "recorded" without ever contacting the backend.
        if len(targetrun.output_artifacts) == 0:
            event = {
                **base_event,
                "inputs": inputs,
                "outputs": [],
                # Explicit random runId: inheriting base_event's would reuse
                # targetrun.uuid, the deterministic id this design replaced, and
                # a re-record would silently resume that one run instead of
                # writing a new one. The target_id tag comes along in
                # base_event["run"]["facets"]["tags"], keeping this event
                # dedupable like the per-output ones.
                "run": {**base_event["run"], "runId": get_uuid()},
            }
            _add_jobstats_mirror_fields(event)
            events_list.append(event)
            events_dict["no-output"] = [event]

        return events_list, events_dict

    def add_jobstats_for_build(
        self, storage: SingletonAdminStorage, build_id: str
    ) -> None:
        build = storage.build_storage.get_by_uuid(build_id)
        if build is None:
            raise ValueError(f"Build with id {build_id} was not found")
        assert isinstance(build, StoredBuild)

        targets = storage.target_storage.get_by_where({"build_id": build_id})
        count = 0
        for target in targets:
            assert isinstance(target, StoredTargetRun)
            self.__add_jobstats_for_target(storage, build, target)
            count += 1
        if count == 0:
            raise ValueError(f"Zero targets found in build with id {build_id}")

    def add_jobstats_for_build_target(
        self, storage: SingletonAdminStorage, build_id: str, target_id: str
    ) -> None:
        build = storage.build_storage.get_by_uuid(build_id)
        if build is None:
            raise ValueError(f"Build with id {build_id} was not found")
        assert isinstance(build, StoredBuild)

        targets = storage.target_storage.get_by_where(
            {"build_id": build_id, "uuid": target_id}
        )
        count = 0
        for target in targets:
            assert isinstance(target, StoredTargetRun)
            self.__add_jobstats_for_target(storage, build, target)
            count += 1
        if count == 0:
            raise ValueError(f"Zero targets found in build with id {build_id}")

    def __add_jobstats_for_target(
        self,
        storage: SingletonAdminStorage,
        build: StoredBuild,
        targetrun: StoredTargetRun,
    ) -> None:
        events, _ = self.create_jobstats_for_target(storage, targetrun, build)
        if not events:
            # No events means emit_event is never called, yet the caller
            # (reconciler) will still mark the target recorded — a silent no-op
            # that leaves nothing in the backend. Surface it rather than hide it.
            logger.warning(
                "No lineage events built for target %s (name=%s) in build %s; "
                "nothing emitted to the lineage backend",
                targetrun.uuid,
                targetrun.name,
                build.uuid,
            )
            return
        for event in events:
            self._service.emit_event(event)

    def add_jobstats_for_original_artifact(
        self,
        artifact: ArtifactRegistration,
        sources: list[ArtifactRegistration],
    ) -> None:
        event = self._build_event_for_artifact(artifact, sources)
        self._service.emit_event(event)

    def create_jobstats_for_target(
        self,
        storage: SingletonAdminStorage,
        targetrun: StoredTargetRun,
        build: Optional[StoredBuild] = None,
    ) -> Tuple[List[dict], Dict[str, List[dict]]]:
        if build is None:
            build_result = storage.build_storage.get_by_uuid(targetrun.build_id)
            if build_result is None:
                raise ValueError(
                    f"target's build could not be found under target's build id {targetrun.build_id}"
                )
            assert isinstance(build_result, StoredBuild)
            build = build_result

        if targetrun.build_id != build.uuid:
            raise ValueError(
                f"target's build id ({targetrun.build_id}) does not match that of the given build ({build.uuid})"
            )

        # Every SUCCESS run is a real run with its own outputs (in-place retry keeps
        # both the FAILED and the SUCCESS run in one build), so lineage is built
        # directly from the target's own outputs.
        return self._build_events_for_target(storage, build, targetrun)

    def create_jobstats_for_original_artifact(
        self,
        artifact: ArtifactRegistration,
        sources: list[ArtifactRegistration],
    ) -> dict:
        return self._build_event_for_artifact(artifact, sources)

    def count_release_ids(
        self, release_id: str, target_id: Optional[str] = None
    ) -> int:
        # One wandb run is created per (target, output artifact), so counting
        # runs tagged with this build_id (and optionally target_id) directly
        # yields the number of jobstats records without scanning run history.
        required = [f"target_id={target_id}"] if target_id else None
        return self._service.count_runs_by_tags(
            [f"build_id={release_id}"], required_tags=required
        )

    def does_release_id_exist(
        self,
        release_id: str,
        expected_count: int,
        target_id: Optional[str] = None,
    ) -> bool:
        count = self.count_release_ids(release_id, target_id)
        return count == expected_count

    def filter_unrecorded(
        self,
        target_ids: set[str],
        expected_counts: Optional[dict[str, int]] = None,
        on_query_error: Optional[Callable[[Exception], None]] = None,
    ) -> set[str]:
        # Drop candidates whose "fully recorded" verdict is still within its TTL,
        # so a steady-state scan that re-selects the same target does not re-ask
        # wandb every interval (see _RECORDED_CACHE_TTL_SECONDS).
        now = time.monotonic()
        # Sweep before the per-candidate lookups. Pruning only the keys touched
        # below cannot bound the dict: the checkpoint advances forward only, so
        # once a recorded target falls behind the scan's lower bound it is never
        # selected again and its entry would outlive its deadline for the life of
        # the process -- one dead tuple per (target, count) ever recorded. The
        # sweep is what makes an entry's lifetime the TTL rather than the
        # process's.
        self._prune_recorded_cache(now)
        counts = expected_counts or {}
        # A surviving entry is live by construction: the sweep above dropped every
        # expired deadline, so mere presence is the whole test.
        cached_recorded = {
            tid for tid in target_ids if (tid, counts.get(tid)) in self._recorded_until
        }
        to_check = target_ids - cached_recorded
        if cached_recorded:
            # Debug, not info: in steady state this fires every monitoring
            # interval with the same targets, which is the cache working as
            # intended rather than an event worth a line in the default log.
            logger.debug(
                "Skipping wandb dedup query for %d target(s) already known "
                "recorded (cached): %s",
                len(cached_recorded),
                sorted(cached_recorded),
            )
        if not to_check:
            return set()

        # Delegate to the service, which checks the candidates against wandb run
        # metadata. ``expected_counts`` lets it require a *full* set of runs per
        # target rather than mere presence (see ILineageStore.filter_unrecorded).
        # Never raises: it fails CLOSED, returning an empty set and reporting the
        # error through ``on_query_error``.
        #
        # The callback is wrapped rather than merely forwarded, because this layer
        # must know whether the query was answered before it caches anything. A
        # failed query now returns an EMPTY set, which is indistinguishable by
        # value from "every candidate is already recorded" -- and caching that
        # would turn one wandb outage into a TTL-long window in which real targets
        # are skipped as recorded. The flag is the only thing separating the two.
        query_failed = False

        def _note_failure(exc: Exception) -> None:
            nonlocal query_failed
            query_failed = True
            if on_query_error is not None:
                on_query_error(exc)

        unrecorded = self._service.filter_unrecorded(
            to_check, expected_counts, on_query_error=_note_failure
        )

        if query_failed:
            # Cache nothing: there is no verdict to cache. Returning the empty set
            # the service produced keeps this fail-closed -- the caller is expected
            # to abort the pass (it heard about the failure through on_query_error)
            # and retry, rather than read "nothing to record" as success.
            return unrecorded

        # Cache only the positive verdicts, and only for targets we actually asked
        # about. An unrecorded target is not cached: its verdict is expected to
        # change as soon as recording succeeds, and a stale negative would be
        # re-queried anyway.
        deadline = now + _RECORDED_CACHE_TTL_SECONDS
        newly_recorded = to_check - unrecorded
        for tid in newly_recorded:
            self._recorded_until[(tid, counts.get(tid))] = deadline
        if newly_recorded:
            # Info: this is the transition -- wandb was asked and answered "already
            # recorded", and the verdict is now cached for the TTL. Once per target
            # per TTL, not once per scan.
            logger.info(
                "wandb already holds lineage for %d target(s); caching the "
                "verdict for %ds: %s",
                len(newly_recorded),
                _RECORDED_CACHE_TTL_SECONDS,
                sorted(newly_recorded),
            )
        return unrecorded

    def _prune_recorded_cache(self, now: float) -> None:
        """Drop every "already recorded" verdict whose TTL has passed.

        Runs on each filter pass so the cache is bounded by the TTL rather than
        by how many targets the process has recorded over its lifetime.
        """
        expired = [
            key for key, deadline in self._recorded_until.items() if deadline <= now
        ]
        for key in expired:
            del self._recorded_until[key]

    def _build_event_for_artifact(
        self,
        artifact: ArtifactRegistration,
        sources: list[ArtifactRegistration],
    ) -> dict:
        use_index = len(sources) > 0
        inputs = []
        index = -1
        for src in sources:
            if use_index:
                index += 1
            inputs.append(
                _artifact_to_lineage_entry(
                    src,
                    target_artifact_name=src.name,
                    target_name=src.name,
                    is_input=True,
                    index=index,
                )
            )
        outputs = [
            _artifact_to_lineage_entry(
                artifact,
                target_artifact_name=artifact.name,
                target_name="pseudo-target",
                is_input=False,
                index=-1,
            )
        ]

        event_time = artifact.created_at.isoformat()

        job_input_params: Dict[str, Any] = {}
        if artifact.origin_uris:
            job_input_params["origin_uris"] = artifact.origin_uris
        if artifact.description:
            job_input_params["description"] = artifact.description

        event = {
            "eventType": "COMPLETE",
            "eventTime": event_time,
            "run": {
                "runId": artifact.uuid,
                "facets": {
                    "tags": {
                        "artifact_id": artifact.uuid,
                        # For registered-artifact jobstats the "release_id" is
                        # the artifact uuid itself — tag build_id with that so
                        # count_release_ids({artifact.uuid}) finds this run.
                        "build_id": artifact.uuid,
                        "target_id": artifact.created_by_target_id,
                        "username": artifact.username,
                        "space_name": artifact.space_name,
                    },
                    "source_code": {"url": "", "commit_hash": "", "path": ""},
                    "job_input_params": job_input_params,
                    "execution_stats": {},
                    "job_details": {
                        "job_id": artifact.uuid,
                        "job_type": GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_TYPE,
                        "category": GB_JOB_STATS_DETAIL_CATEGORY,
                        "job_status": artifact.status.name,
                        "job_started_at": event_time,
                        "job_completed_at": event_time,
                        "release_id": artifact.uuid,
                        "owner": artifact.username,
                        "job_output_stats": {},
                    },
                },
            },
            "job": {
                "namespace": artifact.space_name,
                "name": GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_JOB_NAME,
                "facets": {},
            },
            "inputs": inputs,
            "outputs": outputs,
            "producer": LINEAGE_PRODUCER_URL,
            "schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent",
        }
        _add_jobstats_mirror_fields(event)
        return event
