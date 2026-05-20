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

import pytest

pytestmark = pytest.mark.ibm

from fastapi import status
from libgbtest.api.utils import AbstractAPITest
from libgbtest.storage.artifact_storage import ArtifactStorageTestSupport
from libgbtest.storage.build_storage import BuildStorageTestSupport
from libgbtest.storage.step_storage import StepStorageTestSupport
from libgbtest.storage.target_storage import TargetStorageTestSupport
from libgbtest.storage.utils import (
    StorageCollection,
    TargetSpec,
    connect_and_store_build,
)

from gbserver.api.lineage import TargetJobStatsResponse
from gbserver.types.status import Status

base_url = "api/v1/lineage"


class TestLineageAPI(AbstractAPITest):

    def test_get_target_jobstats_not_found(self):
        """Test that requesting JobStats for a non-existent target returns 404."""
        client = self.get_test_client()
        response = client.get(f"{base_url}/target/non-existent-uuid")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_target_jobstats(self):
        """Test retrieving JobStats for a valid target with output artifacts."""
        # Set up test support classes
        bsts = BuildStorageTestSupport()
        tsts = TargetStorageTestSupport()
        ssts = StepStorageTestSupport()
        asts = ArtifactStorageTestSupport()
        stc = StorageCollection(
            build_storage=self.storage.build_storage,
            artifact_registry=self.storage.artifact_registry,
            target_storage=self.storage.target_storage,
            step_storage=self.storage.step_storage,
        )

        # Create a completed build with a target and artifacts
        build = bsts._get_test_item(0)
        build.status = Status.SUCCESS
        target = tsts._get_test_item(0)
        target.status = Status.SUCCESS
        build.targets = [target.name]
        step = ssts._get_test_item(0)

        # Create input and output artifacts
        input_art0 = asts._get_test_item(0)
        input_art1 = asts._get_test_item(1)
        output_art2 = asts._get_test_item(2)

        targetspec = TargetSpec(
            target=target,
            step=step,
            input_artifacts=[input_art0, input_art1],
            output_artifacts=[output_art2],
        )

        # Store the build with its target and artifacts
        connect_and_store_build(build, targetspec, stc)

        # Make request to the lineage API
        client = self.get_test_client()
        response = client.get(f"{base_url}/target/{target.uuid}")

        # Verify successful response
        assert response.status_code == status.HTTP_200_OK
        resp_json = response.json()

        # Validate response structure
        assert "target_id" in resp_json
        assert resp_json["target_id"] == target.uuid
        assert "jobstats" in resp_json
        assert isinstance(resp_json["jobstats"], dict)

        # The jobstats dict should have entries for each output artifact name
        jobstats_dict = resp_json["jobstats"]
        # We expect at least one output artifact entry
        assert len(jobstats_dict) >= 1

        # Each entry should be a list of JobStats
        for artifact_name, stats_list in jobstats_dict.items():
            assert isinstance(stats_list, list)
            for stats in stats_list:
                # JobStats should have these key fields
                assert "release_id" in stats
                assert "job_details" in stats
                assert "sources" in stats
                assert "targets" in stats

    def test_get_target_jobstats_no_outputs(self):
        """Test retrieving JobStats for a target with no output artifacts."""
        # Set up test support classes
        bsts = BuildStorageTestSupport()
        tsts = TargetStorageTestSupport()
        ssts = StepStorageTestSupport()
        asts = ArtifactStorageTestSupport()
        stc = StorageCollection(
            build_storage=self.storage.build_storage,
            artifact_registry=self.storage.artifact_registry,
            target_storage=self.storage.target_storage,
            step_storage=self.storage.step_storage,
        )

        # Create a completed build with a target but only input artifacts (no outputs)
        build = bsts._get_test_item(1)
        build.status = Status.SUCCESS
        target = tsts._get_test_item(1)
        target.status = Status.SUCCESS
        build.targets = [target.name]
        step = ssts._get_test_item(1)

        # Create only input artifacts, no outputs
        input_art0 = asts._get_test_item(3)

        targetspec = TargetSpec(
            target=target,
            step=step,
            input_artifacts=[input_art0],
            output_artifacts=[],  # No output artifacts
        )

        # Store the build with its target and artifacts
        connect_and_store_build(build, targetspec, stc)

        # Make request to the lineage API
        client = self.get_test_client()
        response = client.get(f"{base_url}/target/{target.uuid}")

        # Verify successful response
        assert response.status_code == status.HTTP_200_OK
        resp_json = response.json()

        # Validate response structure
        assert "target_id" in resp_json
        assert resp_json["target_id"] == target.uuid
        assert "jobstats" in resp_json

        # For targets with inputs but no outputs, we expect a "no-output" key
        jobstats_dict = resp_json["jobstats"]
        assert "no-output" in jobstats_dict

    def test_get_build_jobstats_not_found(self):
        """Test that requesting JobStats for a non-existent build returns 404."""
        client = self.get_test_client()
        response = client.get(f"{base_url}/build/non-existent-uuid")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    # def test_get_build_jobstats(self):
    #     """Test retrieving JobStats for all targets in a build."""
    #     # Set up test support classes
    #     bsts = BuildStorageTestSupport()
    #     tsts = TargetStorageTestSupport()
    #     ssts = StepStorageTestSupport()
    #     asts = ArtifactStorageTestSupport()
    #     stc = StorageCollection(
    #         build_storage=self.storage.build_storage,
    #         artifact_registry=self.storage.artifact_registry,
    #         target_storage=self.storage.target_storage,
    #         step_storage=self.storage.step_storage,
    #     )

    #     # Create a completed build with multiple targets
    #     build = bsts._get_test_item(2)
    #     build.status = Status.SUCCESS

    #     # Create first target with artifacts
    #     target1 = tsts._get_test_item(2)
    #     target1.status = Status.SUCCESS
    #     step1 = ssts._get_test_item(2)
    #     input_art1 = asts._get_test_item(4)
    #     output_art1 = asts._get_test_item(5)
    #     targetspec1 = TargetSpec(
    #         target=target1,
    #         step=step1,
    #         input_artifacts=[input_art1],
    #         output_artifacts=[output_art1],
    #     )

    #     # Create second target with artifacts
    #     target2 = tsts._get_test_item(3)
    #     target2.status = Status.SUCCESS
    #     step2 = ssts._get_test_item(3)
    #     input_art2 = asts._get_test_item(6)
    #     output_art2 = asts._get_test_item(7)
    #     targetspec2 = TargetSpec(
    #         target=target2,
    #         step=step2,
    #         input_artifacts=[input_art2],
    #         output_artifacts=[output_art2],
    #     )

    #     build.targets = [target1.name, target2.name]

    #     # Store the build with its targets and artifacts
    #     connect_and_store_build(build, [targetspec1, targetspec2], stc)

    #     # Make request to the lineage API
    #     client = self.get_test_client()
    #     response = client.get(f"{base_url}/build/{build.uuid}")

    #     # Verify successful response
    #     assert response.status_code == status.HTTP_200_OK
    #     resp_json = response.json()

    #     # Validate response structure
    #     assert "build_id" in resp_json
    #     assert resp_json["build_id"] == build.uuid
    #     assert "targets" in resp_json
    #     assert isinstance(resp_json["targets"], list)

    #     # We should have JobStats for both targets
    #     targets_list = resp_json["targets"]
    #     assert len(targets_list) == 2

    #     # Validate each target response - now just a dict of artifact names to JobStats
    #     for jobstats_dict in targets_list:
    #         assert isinstance(jobstats_dict, dict)

    #         # Each target should have at least one output artifact entry
    #         assert len(jobstats_dict) >= 1

    #         # Validate JobStats structure
    #         for artifact_name, stats_list in jobstats_dict.items():
    #             assert isinstance(stats_list, list)
    #             for stats in stats_list:
    #                 assert "release_id" in stats
    #                 assert "job_details" in stats
    #                 assert "sources" in stats
    #                 assert "targets" in stats

    def test_get_build_jobstats_no_targets(self):
        """Test retrieving JobStats for a build with no targets."""
        # Set up test support classes
        bsts = BuildStorageTestSupport()
        stc = StorageCollection(
            build_storage=self.storage.build_storage,
            artifact_registry=self.storage.artifact_registry,
            target_storage=self.storage.target_storage,
            step_storage=self.storage.step_storage,
        )

        # Create a build with no targets
        build = bsts._get_test_item(4)
        build.status = Status.SUCCESS
        build.targets = []

        # Store just the build (no targets)
        stc.build_storage.add(build)

        # Make request to the lineage API
        client = self.get_test_client()
        response = client.get(f"{base_url}/build/{build.uuid}")

        # Verify successful response
        assert response.status_code == status.HTTP_200_OK
        resp_json = response.json()

        # Validate response structure
        assert "build_id" in resp_json
        assert resp_json["build_id"] == build.uuid
        assert "targets" in resp_json
        assert isinstance(resp_json["targets"], list)

        # Should have empty targets list
        assert len(resp_json["targets"]) == 0

    def test_get_build_jobstats(self):
        """Test retrieving JobStats for a build with multiple targets."""
        # Set up test support classes
        bsts = BuildStorageTestSupport()
        tsts = TargetStorageTestSupport()
        ssts = StepStorageTestSupport()
        asts = ArtifactStorageTestSupport()
        stc = StorageCollection(
            build_storage=self.storage.build_storage,
            artifact_registry=self.storage.artifact_registry,
            target_storage=self.storage.target_storage,
            step_storage=self.storage.step_storage,
        )

        # Create a completed build with multiple targets
        build = bsts._get_test_item(5)
        build.status = Status.SUCCESS

        # Create first target with artifacts
        target1 = tsts._get_test_item(5)
        target1.status = Status.SUCCESS
        step1 = ssts._get_test_item(5)
        input_art1 = asts._get_test_item(10)
        output_art1 = asts._get_test_item(11)
        targetspec1 = TargetSpec(
            target=target1,
            step=step1,
            input_artifacts=[input_art1],
            output_artifacts=[output_art1],
        )

        # Create second target with artifacts
        target2 = tsts._get_test_item(8)
        target2.status = Status.SUCCESS
        step2 = ssts._get_test_item(8)
        input_art2 = asts._get_test_item(14)
        output_art2 = asts._get_test_item(15)
        targetspec2 = TargetSpec(
            target=target2,
            step=step2,
            input_artifacts=[input_art2],
            output_artifacts=[output_art2],
        )

        # Create third target with artifacts
        target3 = tsts._get_test_item(9)
        target3.status = Status.SUCCESS
        step3 = ssts._get_test_item(9)
        input_art3 = asts._get_test_item(16)
        output_art3 = asts._get_test_item(17)
        targetspec3 = TargetSpec(
            target=target3,
            step=step3,
            input_artifacts=[input_art3],
            output_artifacts=[output_art3],
        )

        build.targets = [target1.name, target2.name, target3.name]

        # Store the build with its targets and artifacts
        connect_and_store_build(build, [targetspec1, targetspec2, targetspec3], stc)

        # Make request to the lineage API
        client = self.get_test_client()
        response = client.get(f"{base_url}/build/{build.uuid}")

        # Verify successful response
        assert response.status_code == status.HTTP_200_OK
        resp_json = response.json()

        # Validate response structure
        assert "build_id" in resp_json
        assert resp_json["build_id"] == build.uuid
        assert "targets" in resp_json
        assert isinstance(resp_json["targets"], list)

        # Should have exactly three targets
        targets_list = resp_json["targets"]
        assert len(targets_list) == 3

        # Validate each target response
        for jobstats_dict in targets_list:
            assert isinstance(jobstats_dict, dict)
            assert len(jobstats_dict) >= 1

            # Validate JobStats structure
            for artifact_name, stats_list in jobstats_dict.items():
                assert isinstance(stats_list, list)
                for stats in stats_list:
                    assert "release_id" in stats
                    assert "job_details" in stats
                    assert "sources" in stats
                    assert "targets" in stats

    def test_get_build_jobstats_no_outputs(self):
        """Test retrieving JobStats for a build where targets have no outputs."""
        # Set up test support classes
        bsts = BuildStorageTestSupport()
        tsts = TargetStorageTestSupport()
        ssts = StepStorageTestSupport()
        asts = ArtifactStorageTestSupport()
        stc = StorageCollection(
            build_storage=self.storage.build_storage,
            artifact_registry=self.storage.artifact_registry,
            target_storage=self.storage.target_storage,
            step_storage=self.storage.step_storage,
        )

        # Create a completed build with targets that have no outputs
        build = bsts._get_test_item(6)
        build.status = Status.SUCCESS

        # Create first target with only inputs (no outputs)
        target1 = tsts._get_test_item(6)
        target1.status = Status.SUCCESS
        step1 = ssts._get_test_item(6)
        input_art1 = asts._get_test_item(12)
        targetspec1 = TargetSpec(
            target=target1,
            step=step1,
            input_artifacts=[input_art1],
            output_artifacts=[],  # No outputs
        )

        # Create second target with only inputs (no outputs)
        target2 = tsts._get_test_item(7)
        target2.status = Status.SUCCESS
        step2 = ssts._get_test_item(7)
        input_art2 = asts._get_test_item(13)
        targetspec2 = TargetSpec(
            target=target2,
            step=step2,
            input_artifacts=[input_art2],
            output_artifacts=[],  # No outputs
        )

        build.targets = [target1.name, target2.name]

        # Store the build with its targets and artifacts
        connect_and_store_build(build, [targetspec1, targetspec2], stc)

        # Make request to the lineage API
        client = self.get_test_client()
        response = client.get(f"{base_url}/build/{build.uuid}")

        # Verify successful response
        assert response.status_code == status.HTTP_200_OK
        resp_json = response.json()

        # Validate response structure
        assert "build_id" in resp_json
        assert resp_json["build_id"] == build.uuid
        assert "targets" in resp_json
        assert isinstance(resp_json["targets"], list)

        # Should have two targets
        targets_list = resp_json["targets"]
        assert len(targets_list) == 2

        # Each target should have a "no-output" key since they have inputs but no outputs
        for jobstats_dict in targets_list:
            assert isinstance(jobstats_dict, dict)
            assert "no-output" in jobstats_dict
