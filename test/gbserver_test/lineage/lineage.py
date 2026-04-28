from abc import abstractmethod
from typing import Self

from gbserver_test.storage.test_artifact_storage import ArtifactStorageTestSupport
from gbserver_test.storage.test_build_storage import BuildStorageTestSupport
from gbserver_test.storage.test_step_storage import StepStorageTestSupport
from gbserver_test.storage.test_target_storage import TargetStorageTestSupport
from gbserver_test.test_utils import AbstractSingletonStorageUsingTest
from lakehouse.api import JobStats

from gbcommon.uri.lh import LhURI
from gbserver.lineage.lakehouse_jobstats import LakehouseLineageStore as JobStatsStorage
from gbserver.storage.singleton_storage import get_storage_factory


def get_test_support():
    ssts = StepStorageTestSupport()

    tsts = TargetStorageTestSupport()

    bsts = BuildStorageTestSupport()

    asts = ArtifactStorageTestSupport()

    return tsts, bsts, ssts, asts


class AbstractLineageTest(AbstractSingletonStorageUsingTest):
    """This class is created in anticipation of supporting tests for other lineage implementations, when
    1) We will have a lineage storage interface (currently JobStatsStorage class), and
    2) Storage independent build lineage-holding object (currently JobStats)
    """

    @abstractmethod
    def _get_tested_lineage_storage(self: Self):
        raise NotImplementedError()

    @classmethod
    def _get_storage_factory(cls):
        return get_storage_factory()

    def test_add_from_build_lh(self):
        self._helper_for_test_add_from_build(False)

    def test_add_from_build_non_lh(self):
        self._helper_for_test_add_from_build(True)

    def _helper_for_test_add_from_build(self, use_non_lh_artifact):
        build_storage = self.storage.build_storage
        target_storage = self.storage.target_storage
        step_storage = self.storage.step_storage
        artifact_registry = self.storage.artifact_registry

        tsts, bsts, ssts, asts = get_test_support()

        # Create the build that will hold our targets
        build = bsts._get_test_item(0)
        build_storage.add(build)
        model_table = "a_model_table"
        fileset_table = "a_fileset_table"

        # Create 1st target in our build
        targetrun0 = tsts._get_test_item(0)

        input_artifact0 = asts._get_test_item(0)
        input_artifact0.created_by_build_id = ""
        input_artifact0.created_by_target_id = ""
        input_artifact0.uri = (
            "env://foo.bar"
            if use_non_lh_artifact
            else LhURI.get_table_uri(table_name=input_artifact0.name)
        )
        artifact_registry.add(input_artifact0)

        input_artifact1 = asts._get_test_item(1)
        input_artifact1.created_by_build_id = ""
        input_artifact1.created_by_target_id = ""
        input_artifact1.uri = (
            "env://foo.bar/dataset_table"
            if use_non_lh_artifact
            else LhURI.get_dataset_uri(
                dataset_name=input_artifact1.name, table_name="dataset_table"
            )
        )
        artifact_registry.add(input_artifact1)

        input_artifact2 = asts._get_test_item(2)
        input_artifact2.created_by_build_id = ""
        input_artifact2.created_by_target_id = ""
        input_artifact2.uri = (
            "other://foo.bar/fset-123"
            if use_non_lh_artifact
            else LhURI.get_fileset_uri(
                table_name=fileset_table,
                fileset_label=input_artifact2.name,
                fileset_version="fset-123",
            )
        )
        artifact_registry.add(input_artifact2)

        output_artifact0 = asts._get_test_item(3)
        output_artifact0.name = targetrun0.name + "_output"
        output_artifact0.created_by_build_id = build.uuid
        output_artifact0.created_by_target_id = targetrun0.uuid
        output_artifact0.uri = (
            "more:/foo.bar/123"
            if use_non_lh_artifact
            else LhURI.get_model_uri(
                table_name=model_table,
                model_label=output_artifact0.name,
                model_revision="123",
            )
        )
        artifact_registry.add(output_artifact0)

        step0 = ssts._get_test_item(0)
        step0.build_id = build.uuid
        step0.target_id = targetrun0.uuid
        step0.config = {"a": 1, "b": "c"}
        step_storage.add(step0)

        targetrun0.build_id = build.uuid
        # targetrun0.input_artifact_ids = [input_artifact0.uuid, input_artifact1.uuid, input_artifact2.uuid]
        targetrun0.input_artifacts = {
            "in0": input_artifact0.uuid,
            "in1": input_artifact1.uuid,
            "in2": input_artifact2.uuid,
        }
        # targetrun0.output_artifact_ids = [output_artifact0.uuid]
        targetrun0.output_artifacts = {"out0": [output_artifact0.uuid]}
        target_storage.add(targetrun0)

        # jobstats = JobStatsStorage().create_job_stats(build_storage, targetrun0, artifact_registry)
        # jobstats = JobStatsStorage().create_job_stats(build_storage, targetrun0, step_storage, artifact_registry)
        # print(f"jobstats0={jobstats}")

        # create 2nd target
        targetrun1 = tsts._get_test_item(1)

        output_artifact1 = asts._get_test_item(4)
        output_artifact1.name = targetrun1.name + "_output"
        output_artifact1.created_by_build_id = build.uuid
        output_artifact1.created_by_target_id = targetrun1.uuid
        output_artifact1.uri = (
            "more:/foo.bar/123"
            if use_non_lh_artifact
            else LhURI.get_model_uri(
                table_name=model_table,
                model_label=output_artifact1.name,
                model_revision="abc",
            )
        )
        artifact_registry.add(output_artifact1)

        step1 = ssts._get_test_item(1)
        step1.build_id = build.uuid
        step1.target_id = targetrun1.uuid
        step1.config = {"d": 1, "e": "c"}
        step_storage.add(step1)

        targetrun1.build_id = build.uuid
        # targetrun1.input_artifact_ids = [input_artifact0.uuid, input_artifact1.uuid]
        targetrun1.input_artifacts = {
            "in0": input_artifact0.uuid,
            "in1": input_artifact1.uuid,
        }
        # targetrun1.output_artifact_ids = [output_artifact1.uuid]
        targetrun1.output_artifacts = {"out0": [output_artifact1.uuid]}
        target_storage.add(targetrun1)

        # create 3nd target that takes outputs of previous targets
        targetrun2 = tsts._get_test_item(2)

        output_artifact2 = asts._get_test_item(5)
        output_artifact2.name = targetrun2.name + "_output1"
        output_artifact2.created_by_build_id = build.uuid
        output_artifact2.created_by_target_id = targetrun2.uuid
        output_artifact2.uri = (
            "table:/foo.bar/xyz"
            if use_non_lh_artifact
            else LhURI.get_model_uri(
                table_name=model_table,
                model_label=output_artifact2.name,
                model_revision="xyz",
            )
        )
        artifact_registry.add(output_artifact2)

        output_artifact3 = asts._get_test_item(6)
        output_artifact3.name = targetrun2.name + "_output2a"
        output_artifact3.created_by_build_id = build.uuid
        output_artifact3.created_by_target_id = targetrun2.uuid
        output_artifact3.uri = (
            "model:/foo.bar/abc"
            if use_non_lh_artifact
            else LhURI.get_model_uri(
                table_name=model_table,
                model_label=output_artifact3.name,
                model_revision="abc",
            )
        )
        artifact_registry.add(output_artifact3)

        output_artifact4 = asts._get_test_item(7)
        output_artifact4.name = targetrun2.name + "_output2b"
        output_artifact4.created_by_build_id = build.uuid
        output_artifact4.created_by_target_id = targetrun2.uuid
        output_artifact4.uri = (
            "model:/foo.bar/abc"
            if use_non_lh_artifact
            else LhURI.get_model_uri(
                table_name=model_table,
                model_label=output_artifact4.name,
                model_revision="abc",
            )
        )
        artifact_registry.add(output_artifact4)

        step2 = ssts._get_test_item(2)
        step2.build_id = build.uuid
        step2.target_id = targetrun2.uuid
        step2.config = {"d": 1, "e": "c3"}
        step_storage.add(step2)

        targetrun2.build_id = build.uuid
        targetrun2.input_artifacts = {
            "in0": input_artifact0.uuid,
            "in1": output_artifact0.uuid,
            "in2": output_artifact1.uuid,
        }
        targetrun2.output_artifacts = {
            "out0": [output_artifact2.uuid],
            "out1": [output_artifact3.uuid, output_artifact4.uuid],
        }
        target_storage.add(targetrun2)

        # jobstats = JobStatsStorage().create_job_stats(build_storage, targetrun2, artifact_registry)
        # jobstats = JobStatsStorage().create_job_stats(build_storage, targetrun2, step_storage, artifact_registry)
        # print(f"jobstats2={jobstats}")
        # jobstats = JobStatsStorage().create_job_stats(build_storage, targetrun, artifact_registry)
        # JobStatsStorage().add_step(build_storage, targetrun, artifact_registry )
        lineage_storage = self._get_tested_lineage_storage()
        lineage_storage.add_jobstats_for_build(self.storage, build.uuid)

        output_count = 5
        assert lineage_storage.does_release_id_exist(
            release_id=build.uuid, expected_count=output_count
        ), f"Did not create {output_count} JobStats"

    def test_create_from_artifact(self):
        tsts, bsts, ssts, asts = get_test_support()

        # output = asts._get_test_item(0)
        # inputs = []
        # stats = JobStatsStorage.create_jobstats_for_original_artifact(output, inputs)
        # assert isinstance(stats,JobStats)
        # assert len(stats.sources) == 0
        # assert len(stats.targets) == 1

        storage = self._get_tested_lineage_storage()

        output = asts._get_test_item(0)
        inputs = [asts._get_test_item(1)]
        stats = storage.create_jobstats_for_original_artifact(output, inputs)
        assert isinstance(stats, JobStats)
        assert len(stats.sources) == 1
        assert len(stats.targets) == 1

        output = asts._get_test_item(0)
        inputs = [asts._get_test_item(1), asts._get_test_item(2)]
        stats = storage.create_jobstats_for_original_artifact(output, inputs)
        assert isinstance(stats, JobStats)
        assert len(stats.sources) == 2
        assert len(stats.targets) == 1

    def test_create_from_non_gb_artifact(self):
        tsts, bsts, ssts, asts = get_test_support()
        storage = self._get_tested_lineage_storage()

        output = asts._get_test_item(0)
        output.uri = "http://foo.bar"
        input = asts._get_test_item(1)
        input.uri = "env:///foo/bar"
        inputs = [input]
        stats = storage.create_jobstats_for_original_artifact(output, inputs)
        assert isinstance(stats, JobStats)
        assert len(stats.sources) == 1
        assert len(stats.targets) == 1
        # TODO: Should really make sure the placeholder artifacts got created,
