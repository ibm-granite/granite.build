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

from typing import Any, Dict, List, Optional, Tuple

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
    GBSERVER_LINEAGE_PROVIDER,
)
from gbserver.types.status import Status
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
        from gbcommon.uri.utils import get_artifact_type

        artifact_type = get_artifact_type(artifact.uri)

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


class WandBLineageStore(ILineageStore):

    def __init__(self) -> None:
        self._service: LineageService = LineageServiceFactory.create(
            GBSERVER_LINEAGE_PROVIDER
        )

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
            step_configs.append(
                {
                    "uri": step.definition_uri,
                    "config": step.config,
                    "config_dir": step.config_dir,
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
            "producer": "https://github.ibm.com/granite-dot-build/gbserver",
            "schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent",
        }

        if build.description:
            base_event["job"]["facets"]["documentation"] = {
                "description": build.description,
            }

        events_list: List[dict] = []
        events_dict: Dict[str, List[dict]] = {}

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
                job_id = base_event["run"]["facets"]["job_details"]["job_id"]
                event["run"] = {
                    **base_event["run"],
                    "runId": f"{job_id}-{output_uuid}",
                }
                _add_jobstats_mirror_fields(event)
                target_events.append(event)
            events_list.extend(target_events)
            events_dict[target_artifact_name] = target_events

        if len(targetrun.output_artifacts) == 0 and len(inputs) > 0:
            event = {
                **base_event,
                "inputs": inputs,
                "outputs": [],
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

        if targetrun.skipped_for_prerun_target_id:
            original = storage.target_storage.get_by_uuid(
                targetrun.skipped_for_prerun_target_id
            )
            if original is not None and isinstance(original, StoredTargetRun):
                targetrun = original.model_copy(
                    update={
                        "uuid": targetrun.uuid,
                        "build_id": targetrun.build_id,
                    }
                )
            else:
                logger.warning(
                    "Skipped target %s references unknown original %s",
                    targetrun.uuid,
                    targetrun.skipped_for_prerun_target_id,
                )

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
            "producer": "https://github.ibm.com/granite-dot-build/gbserver",
            "schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent",
        }
        _add_jobstats_mirror_fields(event)
        return event
