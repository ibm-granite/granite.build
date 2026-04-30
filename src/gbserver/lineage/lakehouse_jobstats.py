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

"""
Lakehouse-backed lineage storage using the job_stats table.
"""

from __future__ import annotations

import copy
import datetime
import json
from typing import Dict, List, Optional, Self, Tuple
from urllib.parse import urlparse

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential

from gbserver.utils.optional_imports import HAS_LAKEHOUSE

if HAS_LAKEHOUSE:
    from lakehouse.api import Datasource, JobDetails, JobStats, SourceCodeDetails
    from lakehouse.assets.table import Table
    from lakehouse.core import TableDetails

from gbcommon.uri.lh import LhType, LhURI
from gbcommon.uri.uri import URI
from gbserver.lineage.jobstats import ILineageStore
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.artifact_registry import IArtifactRegistry
from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.steprun_storage import IStoredStepRunStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.storage.target_run_storage import IStoredTargetRunStorage
from gbserver.types.artifact import ArtifactType
from gbserver.types.constants import (
    GB_JOB_STATS_DETAIL_CATEGORY,
    GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_JOB_NAME,
    GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_TYPE,
    GB_JOB_STATS_DETAIL_TYPE,
    LAKEHOUSE_ENVIRONMENT,
    PUBLIC_SPACE_LH_NAMESPACE,
)
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_uuid

logger = get_logger(__name__)


def _get_base_class():
    """Return BaseLakehouseStorage if lakehouse is available, else a stub."""
    if HAS_LAKEHOUSE:
        from gbserver.storage.lh.lh_storage import BaseLakehouseStorage

        return BaseLakehouseStorage
    # Return a minimal stub so the module can be imported
    from pydantic import BaseModel

    return BaseModel


class LakehouseLineageStore(ILineageStore, _get_base_class()):
    """Lakehouse-backed lineage storage using the job_stats table."""

    def count_release_ids(self: Self, release_id: str, target_id: Optional[str] = None) -> int:
        """Count the number of job_stats records for a release_id."""
        table = Table(lh=self.lh, namespace="dmf", table_name="job_stats")
        if target_id is None:
            row_filter = f"release_id = '{release_id}'"
        else:
            row_filter = f"release_id = '{release_id}' and job_id = '{target_id}'"
        results = table.to_json(row_filter=row_filter)
        if results is None or len(results) == 0:
            return 0
        obj = json.loads(results)
        assert isinstance(obj, list), f"invalid obj: {obj}"
        return len(obj)

    def does_release_id_exist(
        self: Self,
        release_id: str,
        expected_count: int,
        target_id: Optional[str] = None,
    ) -> bool:
        count = self.count_release_ids(release_id, target_id)
        return count == expected_count

    def add_jobstats_for_original_artifact(
        self: Self, artifact: ArtifactRegistration, sources: list[ArtifactRegistration]
    ) -> None:
        """Create and save the job stats entry in Lakehouse."""
        jobstats = self.create_jobstats_for_original_artifact(artifact, sources)
        self.__save_job_stats_with_retry(jobstats)

    def create_jobstats_for_original_artifact(
        self: Self, artifact: ArtifactRegistration, sources: list[ArtifactRegistration]
    ) -> JobStats:
        """Create a job stats entry for an artifact that was certified as being created from 0 or more artifacts using
        an undefined process
        """
        logger.info("Begin creating job stats from artifact %s", artifact.uuid)
        assert isinstance(artifact, ArtifactRegistration), f"invalid artifact: {artifact}"
        assert len(sources) > 0, f"invalid sources: {sources}"
        assert isinstance(sources[0], ArtifactRegistration), f"invalid sources: {sources}"

        job_details = self.__get_jobdetails_for_artifact(artifact)
        source_code_details = self.__get_source_code_details("")
        job_input_params = {}
        sources = self.__get_input_datasources_for_artifact(sources)

        target_name = "pseudo-target"
        target_artifact_name = artifact.name
        output_ds = self.__get_datasource_for_artifact(
            artifact, target_name, target_artifact_name, is_input=False, index=-1
        )
        stats = JobStats(
            release_id=artifact.uuid,
            job_details=job_details,
            source_code_details=source_code_details,
            sources=sources,
            targets=[output_ds],
            job_input_params=job_input_params,
            execution_stats={},
            job_output_stats={},
        )

        logger.info("Done creating jobs stats for artifact %s: %s", artifact.uuid, stats)
        return stats

    def add_jobstats_for_build(
        self: Self,
        storage: SingletonAdminStorage,
        build_id: str,
    ) -> None:
        """Add a JobStats entry for each target found in the build stored with the given build_id.
        All associated targets and artifacts must already be stored for lookup through the build.

        Args:
            storage(SingletonAdminStorage): admin storage holding builds, etc.
            build_id (str): uuid of a stored build.

        Raises:
            ValueError: if the build is not found or there are no targets found in storage for the given build.
        """
        logger.info(
            "Begin storing job stats for fully stored build+targets+artifacts with build id %s",
            build_id,
        )
        build_storage: IStoredBuildStorage = storage.build_storage
        targetrun_storage: IStoredTargetRunStorage = storage.target_storage
        build = build_storage.get_by_uuid(build_id)
        if build is None:
            raise ValueError(f"Build with id {build_id} was not found")
        row_filter = {"build_id": build_id}
        targets = targetrun_storage.get_by_where(row_filter)
        count = 0
        for target in targets:
            assert isinstance(build, StoredBuild)
            self.__add_jobstats_for_target(storage, build, target)
            count += 1
        if count == 0:
            raise ValueError(f"Zero targets found in build with id {build_id}")
        logger.info(
            "Done storing %s job stats record(s) for fully stored build+targets+artifacts with build id %s",
            count,
            build_id,
        )

    def add_jobstats_for_build_target(
        self: Self,
        storage: SingletonAdminStorage,
        build_id: str,
        target_id: str,
    ) -> None:
        """Add a JobStats entry for each target found in the build stored with the given build_id.
        All associated targets and artifacts must already be stored for lookup through the build.

        Args:
            storage(SingletonAdminStorage): admin storage holding builds, etc.
            build_id (str): uuid of a stored build.
            target_id (str): uuid of a stored target.

        Raises:
            ValueError: if the build is not found or there are no targets found in storage for the given build.
        """
        logger.info(
            "Begin storing job stats for fully stored build+targets+artifacts with build id %s target id %s",
            build_id,
            target_id,
        )
        build_storage: IStoredBuildStorage = storage.build_storage
        targetrun_storage: IStoredTargetRunStorage = storage.target_storage
        build = build_storage.get_by_uuid(build_id)
        if build is None:
            raise ValueError(f"Build with id {build_id} was not found")
        row_filter = {"build_id": build_id, "uuid": target_id}
        targets = targetrun_storage.get_by_where(row_filter)
        count = 0
        for target in targets:
            assert isinstance(build, StoredBuild), f"invalid build: {build}"
            self.__add_jobstats_for_target(storage, build, target)
            count += 1
        if count == 0:
            raise ValueError(f"Zero targets found in build with id {build_id}")
        logger.info(
            "Done storing %s job stats record(s) for fully stored build+targets+artifacts with build id %s target id %s",
            count,
            build_id,
            target_id,
        )

    def __add_jobstats_for_target(
        self: Self,
        storage: SingletonAdminStorage,
        build: StoredBuild,
        targetrun: StoredTargetRun,
    ) -> None:
        stats_list, _ = self.create_jobstats_for_target(storage, targetrun, build)
        logger.info("Begin storing job stats")
        for stats in stats_list:
            self.__save_job_stats_with_retry(stats)

    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def __save_job_stats_with_retry(self: Self, stats: JobStats):
        logger.info("Begin storing job stats=%s", stats)
        r = self.lh.get_lakehouse_api().save_job_stats(stats)
        logger.info("Done storing job stats, result=%s", r)

    def create_jobstats_for_target(
        self: Self,
        storage: SingletonAdminStorage,
        targetrun: StoredTargetRun,
        build: Optional[StoredBuild] = None,
    ) -> Tuple[List[JobStats], dict[str, list[JobStats]]]:
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
            original = storage.target_storage.get_by_uuid(targetrun.skipped_for_prerun_target_id)
            if original is not None:
                targetrun = original.model_copy(
                    update={
                        "uuid": targetrun.uuid,
                        "build_id": targetrun.build_id,
                    }
                )
            else:
                logger.warning(
                    "Skipped target %s references unknown original %s; no jobstats will be created",
                    targetrun.uuid,
                    targetrun.skipped_for_prerun_target_id,
                )

        logger.info(
            "Begin creating job stats from build id %s and target run named %s",
            build.uuid,
            targetrun.name,
        )
        job_details = self.__get_jobdetails_for_target(build, targetrun)
        source_code_details = self.__get_source_code_details(build.source_uri)
        job_input_params = self.__get_target_input_params(storage, targetrun)
        sources = self.__get_input_datasources_for_target(
            storage, targetrun, targetrun.input_artifacts
        )

        stats_list: List[JobStats] = []
        stats_dict: dict[str, list[JobStats]] = {}
        for (
            target_artifact_name,
            output_artifact_list,
        ) in targetrun.output_artifacts.items():
            assert isinstance(output_artifact_list, list)
            index = -1
            include_index = len(output_artifact_list) > 1
            target_stats_list: list[JobStats] = []
            for output_artifact_uuid in output_artifact_list:
                if include_index:
                    index += 1
                output_ds = self.__get_datasource_for_target_artifact(
                    storage,
                    targetrun,
                    target_artifact_name,
                    output_artifact_uuid,
                    is_input=False,
                    index=index,
                )
                stats = JobStats(
                    release_id=targetrun.build_id,
                    job_details=job_details,
                    source_code_details=source_code_details,
                    sources=sources,
                    targets=[output_ds],
                    job_input_params=job_input_params,
                    execution_stats={},
                    job_output_stats={},
                )
                target_stats_list.append(stats)
            stats_list.extend(target_stats_list)
            stats_dict[target_artifact_name] = target_stats_list

        if len(targetrun.output_artifacts) == 0 and len(sources) != 0:
            logger.info("Creating an empty-target job_stat because there's no output")
            stats = JobStats(
                release_id=targetrun.build_id,
                job_details=job_details,
                source_code_details=source_code_details,
                sources=sources,
                targets=[],
                job_input_params=job_input_params,
                execution_stats={},
                job_output_stats={},
            )
            stats_list.append(stats)
            stats_dict["no-output"] = [stats]

        logger.info("Done creating %s job stats: %s", len(stats_list), stats_list)
        return (stats_list, stats_dict)

    def __get_target_input_params(
        self: Self,
        storage: SingletonAdminStorage,
        targetrun: StoredTargetRun,
    ) -> Dict:
        step_storage: IStoredStepRunStorage = storage.step_storage
        params = {}
        if step_storage is None:
            return params
        steps = step_storage.get_by_where({"target_id": targetrun.uuid})
        step_list = []
        for step in steps:
            assert isinstance(step, StoredStepRun)
            value = {
                "uri": step.definition_uri,
                "config": step.config,
                "config_dir": step.config_dir,
            }
            step_list.append(value)
        params["steps"] = step_list
        return params

    def __get_jobdetails_for_target(
        self: Self, build: StoredBuild, targetrun: StoredTargetRun
    ) -> JobDetails:
        started_at = str(datetime.datetime.now().isoformat())
        started_at = "1972-01-02T00:00:00.000Z"
        completed_at = started_at
        if targetrun.started_at:
            started_at = str(targetrun.started_at.isoformat())
        if targetrun.finished_at:
            completed_at = str(targetrun.finished_at.isoformat())
        job_details = JobDetails(
            id=targetrun.uuid,
            name=targetrun.name,
            type=GB_JOB_STATS_DETAIL_TYPE,
            category=GB_JOB_STATS_DETAIL_CATEGORY,
            status=targetrun.status.name,
            started_at=started_at,
            completed_at=completed_at,
            owner=build.username,
        )
        return job_details

    def __get_jobdetails_for_artifact(self: Self, artifact: ArtifactRegistration) -> JobDetails:
        started_at = str(datetime.datetime.now().isoformat())
        started_at = str(datetime.datetime.now())
        started_at = "1972-01-02T00:00:00.000Z"
        completed_at = started_at
        job_details = JobDetails(
            id=artifact.uuid,
            name=GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_JOB_NAME,
            type=GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_TYPE,
            category=GB_JOB_STATS_DETAIL_CATEGORY,
            status=artifact.status.name,
            started_at=started_at,
            completed_at=completed_at,
            owner=artifact.username,
        )
        return job_details

    def __get_source_code_details(self: Self, source_uri: str) -> SourceCodeDetails:
        source_code_details = SourceCodeDetails(
            url=source_uri, commit_hash="", path=""  # TODO  # TODO
        )
        return source_code_details

    def __get_input_datasources_for_target(
        self: Self,
        storage: SingletonAdminStorage,
        targetrun: StoredTargetRun,
        artifact_uuids: dict[str, str],
    ) -> List[Datasource]:
        artifact_registry: IArtifactRegistry = storage.artifact_registry
        dlist = []
        for target_artifact_name, uuid in artifact_uuids.items():
            ds = self.__get_datasource_for_target_artifact(
                storage, targetrun, target_artifact_name, uuid, is_input=True
            )
            dlist.append(ds)
        return dlist

    def __get_input_datasources_for_artifact(
        self: Self,
        inputs: list[ArtifactRegistration],
    ) -> List[Datasource]:
        dlist = []
        use_index = len(inputs) > 0
        index = -1
        for curr_input in inputs:
            if use_index:
                index += 1
            ds = self.__get_datasource_for_artifact(
                curr_input, curr_input.name, curr_input.name, is_input=True, index=index
            )
            dlist.append(ds)
        return dlist

    def _get_table_specs(self: Self, artifact: ArtifactRegistration) -> Tuple[str, str, str, str]:
        uri = artifact.uri
        parse = urlparse(uri)

        lh_schemes = LhURI.get_supported_schemes()
        if parse.scheme not in lh_schemes:
            raise ValueError(f"URI scheme '{parse.scheme}' must be one of {lh_schemes}")
        lh = LhURI(parse)
        ns = lh.get_lh_namespace()
        table_name = lh.get_lh_table_name()
        table = ns + "." + table_name
        lh_type = lh.get_lh_type()
        version = ""
        match lh_type:
            case LhType.DATASET:
                lh_type = "dataset"
                name = lh.get_lh_dataset_name()
            case LhType.TABLE:
                lh_type = "table"
                name = table_name
            case LhType.MODEL:
                lh_type = "model"
                name = f"{lh.get_lh_model_label()}.{lh.get_lh_model_revision()}"
            case LhType.FILESET:
                lh_type = "fileset"
                name = lh.get_lh_fileset_label()
                version = lh.get_lh_fileset_version()
            case _:
                raise ValueError(f"Unrecognized LH asset type {lh_type}")
        return (lh_type, table, name, version)

    def __get_datasource_for_target_artifact(
        self: Self,
        storage: SingletonAdminStorage,
        targetrun: StoredTargetRun,
        target_artifact_name: str,
        artifact_uuid: str,
        is_input: bool,
        index: int = -1,
    ) -> Datasource:
        artifact_registry: IArtifactRegistry = storage.artifact_registry
        target_name = targetrun.name
        artifact = artifact_registry.get_by_uuid(artifact_uuid)
        assert isinstance(artifact, ArtifactRegistration)
        if artifact is None:
            raise ValueError(f"Could not find artifact with uuid {artifact_uuid}")
        return self.__get_datasource_for_artifact(
            artifact, target_name, target_artifact_name, is_input, index
        )

    def __get_datasource_for_artifact(
        self: Self,
        artifact: ArtifactRegistration,
        target_name: str,
        target_artifact_name: str,
        is_input: bool,
        index: int,
    ) -> Datasource:
        try:
            uri = URI.get_uri(artifact.uri)
        except Exception as e:
            logger.debug("failed to get uri from %s : %s", artifact.uri, e)
            uri = None
        if not uri or not isinstance(uri, LhURI):
            artifact = self.__create_registered_artifact_reference(artifact)
        extras = self.__get_datasource_extras(
            artifact, target_name, target_artifact_name, is_input, index
        )
        ds = self.__get_datasource_for_lh_artifact(artifact, extras)
        return ds

    def __get_datasource_extras(
        self: Self,
        artifact: ArtifactRegistration,
        target_name: str,
        target_artifact_name: str,
        is_input: bool,
        index: int,
    ) -> Dict[str, str]:
        in_or_out = "inputs" if is_input else "outputs"
        target_artifact_reference = target_name + "." + in_or_out + "." + target_artifact_name
        if index >= 0:
            target_artifact_reference = f"{target_artifact_reference}[{index}]"
        extras = {
            "gb-artifact-id": artifact.uuid,
            "gb-artifact-uri": artifact.uri,
            "gb-build-id": artifact.created_by_build_id,
            "gb-target-id": artifact.created_by_target_id,
            "gb-build-target-artifact": target_artifact_reference,
        }
        return extras

    def __get_datasource_for_lh_artifact(
        self: Self, artifact: ArtifactRegistration, extras: dict[str, str]
    ) -> Datasource:
        atype, table_name, name, version = self._get_table_specs(artifact)
        data_source = Datasource(
            type=atype,
            table=table_name,
            name=name,
            version=version,
            extra=extras,
            snapshot_id="",  # TODO
            path=[""],  # TODO
        )
        return data_source

    def __create_registered_artifact_reference(
        self: Self, artifact: ArtifactRegistration
    ) -> ArtifactRegistration:
        df = pd.DataFrame.from_dict(
            {
                "uuid": [artifact.uuid],
                "uri": [artifact.uri],
                "type": [artifact.type.name],
            }
        )
        art_name = artifact.name if artifact.name and len(artifact.name) > 0 else "unnamed"
        table_name = f"alias_{art_name}_{artifact.uuid.replace('-','_')}".lower()
        namespace = PUBLIC_SPACE_LH_NAMESPACE
        spaces = get_admin_storage().space_storage.get_by_where({"name": artifact.space_name})
        if not spaces is None and len(spaces) == 1:
            namespace = spaces[0].lakehouse_namespace
        else:
            logger.info(
                "Namespace for artifact not found for uri %s. Falling back to public",
                artifact.uri,
            )
        lhuri = LhURI.get_table_uri(
            table_name=table_name, lh_env=LAKEHOUSE_ENVIRONMENT, namespace=namespace
        )
        registry = get_admin_storage().artifact_registry

        new_artifact = registry.get_by_uri(lhuri)
        if new_artifact is not None:
            if isinstance(new_artifact, ArtifactRegistration):
                return new_artifact
            if len(new_artifact) > 0:
                assert len(new_artifact) == 1, f"unexpected new_artifact: {new_artifact}"
                new_artifact = new_artifact[0]
                return new_artifact

        table_details = TableDetails(
            namespace=namespace,
            name=table_name,
            is_public=(namespace == PUBLIC_SPACE_LH_NAMESPACE),
        )
        new_artifact = copy.deepcopy(artifact)
        new_artifact.uuid = get_uuid()
        new_artifact.uri = lhuri
        new_artifact.type = ArtifactType.UNDEFINED
        new_artifact.checksum = ""

        create_table = False
        try:
            registry.add(new_artifact)
            create_table = True
        except (
            Exception
        ):  # Really looking for IntegrityError, but don't want to expose the storage implementation
            # For aliasing Input URIs, we can get a race between builds that use the same input.
            # In this case we get an integrity error because they both try and register the same
            # aliasing lh:// uri in the same space which is not allowed per the registery.
            # So in this case, just find the previous registration and return it.
            # Seen with multi cpu buildwatcher test when using hf:// uris as inputs
            new_artifact = registry.get_by_uri(
                uri=new_artifact.uri, space_name=new_artifact.space_name
            )
            if not new_artifact:
                raise RuntimeError(
                    f"Got Exception storing alias, then couldn't find aliasing registration for {artifact.uri}"
                )
            assert isinstance(new_artifact, ArtifactRegistration)
            create_table = False

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        def create_table_with_retry():
            logger.debug(
                "Attempting to create lakehouse table %s.%s",
                namespace,
                table_name,
            )
            return Table.from_dataframe(lh=self.lh, df=df, table_details=table_details)

        if create_table:
            # If the artifact registration already existed, then this table was already created
            # so don't try again.
            logger.info(
                "Creating alias/placeholder table named %s.%s for uri %s",
                namespace,
                table_name,
                artifact.uri,
            )
            create_table_with_retry()
        else:
            logger.info(
                "Skipping creation of alias/placeholder table named %s.%s for uri %s, already exists.",
                namespace,
                table_name,
                artifact.uri,
            )

        return new_artifact
