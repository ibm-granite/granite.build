from abc import abstractmethod
from typing import Self

from libgbtest.storage.artifact_storage import ArtifactStorageTestSupport
from libgbtest.storage.build_storage import BuildStorageTestSupport
from libgbtest.storage.step_storage import StepStorageTestSupport
from libgbtest.storage.target_storage import TargetStorageTestSupport
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbcommon.uri.lh import LhURI
from gbserver.lineage.lineage_reconciler import _expected_run_count
from gbserver.storage.singleton_storage import get_storage_factory


class _LineageBuildStorageTestSupport(BuildStorageTestSupport):
    """BuildStorageTestSupport for lineage tests.

    Lineage tests only need a build *record* to attach lineage to and never
    inspect its username, so we override the base class's GitHub-token/API
    lookup (intended for build-cancellation tests) with a fixed username.  This
    lets the lineage tests run without a GitHub token — e.g. in the live
    standalone/extended suite, where no token is configured.
    """

    def _get_build_user(self) -> str:
        return "lineage-test-user"


def get_test_support():
    ssts = StepStorageTestSupport()
    tsts = TargetStorageTestSupport()
    bsts = _LineageBuildStorageTestSupport()
    asts = ArtifactStorageTestSupport()
    return tsts, bsts, ssts, asts


class AbstractLineageTest(AbstractSingletonStorageUsingTest):
    """Base class for lineage store integration tests.

    Subclasses must implement _get_tested_lineage_storage() to return
    an ILineageStore instance.
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
        targetrun0.input_artifacts = {
            "in0": input_artifact0.uuid,
            "in1": input_artifact1.uuid,
            "in2": input_artifact2.uuid,
        }
        targetrun0.output_artifacts = {"out0": [output_artifact0.uuid]}
        target_storage.add(targetrun0)

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
        targetrun1.input_artifacts = {
            "in0": input_artifact0.uuid,
            "in1": input_artifact1.uuid,
        }
        targetrun1.output_artifacts = {"out0": [output_artifact1.uuid]}
        target_storage.add(targetrun1)

        # create 3rd target that takes outputs of previous targets
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

        lineage_storage = self._get_tested_lineage_storage()
        lineage_storage.add_jobstats_for_build(self.storage, build.uuid)

        output_count = 5
        assert lineage_storage.does_release_id_exist(
            release_id=build.uuid, expected_count=output_count
        ), f"Did not create {output_count} JobStats"

    def test_target_with_no_artifacts_still_emits_one_event(self):
        """A successful target with no inputs and no outputs must still record.

        Regression: the event builder only emitted from output artifacts, with a
        "no-output" fallback gated on having inputs. A target with neither (e.g.
        a pure generation/compute target) produced zero events, so recording was
        a silent backend no-op the reconciler still marked "recorded".
        """
        build_storage = self.storage.build_storage
        target_storage = self.storage.target_storage

        tsts, bsts, ssts, asts = get_test_support()

        build = bsts._get_test_item(0)
        build_storage.add(build)

        targetrun = tsts._get_test_item(0)
        targetrun.build_id = build.uuid
        targetrun.input_artifacts = {}
        targetrun.output_artifacts = {}
        target_storage.add(targetrun)

        lineage_storage = self._get_tested_lineage_storage()
        events, events_dict = lineage_storage.create_jobstats_for_target(
            self.storage, targetrun, build
        )

        assert len(events) == 1, "Expected exactly one event for artifact-less target"
        assert "no-output" in events_dict
        assert events[0].get("inputs", []) == []
        assert events[0].get("outputs", []) == []
        # The reconciler's in-memory count must match what the builder emits.
        assert _expected_run_count(targetrun) == len(events)

    def test_expected_run_count_matches_events_built(self):
        """``_expected_run_count`` must equal the events the builder emits.

        The reconciler derives a target's expected run count from the in-memory
        ``StoredTargetRun`` (to avoid a storage read) while the sink emits one run
        per built event. If the two ever diverge, a fully-recorded target could be
        seen as partial (harmless but wasteful re-record) or vice versa (the gap
        this guards against). This pins them together against real storage.
        """
        build_storage = self.storage.build_storage
        target_storage = self.storage.target_storage
        artifact_registry = self.storage.artifact_registry

        tsts, bsts, ssts, asts = get_test_support()

        build = bsts._get_test_item(0)
        build_storage.add(build)

        # Two output-artifact names, the second holding two artifacts -> 3 runs.
        out0 = asts._get_test_item(0)
        out1 = asts._get_test_item(1)
        out2 = asts._get_test_item(2)
        for a in (out0, out1, out2):
            artifact_registry.add(a)

        targetrun = tsts._get_test_item(0)
        targetrun.build_id = build.uuid
        targetrun.input_artifacts = {}
        targetrun.output_artifacts = {
            "out0": [out0.uuid],
            "out1": [out1.uuid, out2.uuid],
        }
        target_storage.add(targetrun)

        lineage_storage = self._get_tested_lineage_storage()
        events, _ = lineage_storage.create_jobstats_for_target(
            self.storage, targetrun, build
        )

        assert _expected_run_count(targetrun) == 3
        assert _expected_run_count(targetrun) == len(events)

    def test_filter_unrecorded_requires_full_run_count(self):
        """A partially-recorded target stays unrecorded until all its runs exist.

        Regression: ``filter_unrecorded`` marked a target recorded on the first
        run carrying its ``target_id`` tag. A target that emits N runs but crashed
        part-way through left 1..N-1 runs tagged, so it was filtered out and never
        re-recorded -> permanent partial lineage. It must count runs against the
        expected total instead.
        """
        build_storage = self.storage.build_storage
        target_storage = self.storage.target_storage
        artifact_registry = self.storage.artifact_registry

        tsts, bsts, ssts, asts = get_test_support()

        build = bsts._get_test_item(0)
        build_storage.add(build)

        out0 = asts._get_test_item(0)
        out1 = asts._get_test_item(1)
        for a in (out0, out1):
            artifact_registry.add(a)

        targetrun = tsts._get_test_item(0)
        targetrun.build_id = build.uuid
        targetrun.input_artifacts = {}
        targetrun.output_artifacts = {"out0": [out0.uuid], "out1": [out1.uuid]}
        target_storage.add(targetrun)

        store = self._get_tested_lineage_storage()
        events, _ = store.create_jobstats_for_target(self.storage, targetrun, build)
        assert len(events) == 2
        tid = targetrun.uuid
        expected = {tid: 2}

        # No runs yet: unrecorded regardless of expected count.
        assert store.filter_unrecorded({tid}, expected) == {tid}

        # Emit only the first of the two runs (simulates a crash mid-target).
        store._service.emit_event(events[0])
        # Count-based check: still unrecorded because 1 < 2 ...
        assert store.filter_unrecorded({tid}, expected) == {tid}
        # ... but the old presence-based fallback (no expected count) would have
        # wrongly considered it recorded — the exact masking this fix removes.
        assert store.filter_unrecorded({tid}, None) == set()

        # Emit the second run: now fully recorded under the count-based check.
        store._service.emit_event(events[1])
        assert store.filter_unrecorded({tid}, expected) == set()

    def test_create_from_artifact(self):
        tsts, bsts, ssts, asts = get_test_support()

        storage = self._get_tested_lineage_storage()

        output = asts._get_test_item(0)
        inputs = [asts._get_test_item(1)]
        stats = storage.create_jobstats_for_original_artifact(output, inputs)
        assert isinstance(stats, dict)
        assert len(stats.get("inputs", [])) == 1
        assert len(stats.get("outputs", [])) == 1

        output = asts._get_test_item(0)
        inputs = [asts._get_test_item(1), asts._get_test_item(2)]
        stats = storage.create_jobstats_for_original_artifact(output, inputs)
        assert isinstance(stats, dict)
        assert len(stats.get("inputs", [])) == 2
        assert len(stats.get("outputs", [])) == 1

    def test_create_from_non_gb_artifact(self):
        tsts, bsts, ssts, asts = get_test_support()
        storage = self._get_tested_lineage_storage()

        output = asts._get_test_item(0)
        output.uri = "http://foo.bar"
        input = asts._get_test_item(1)
        input.uri = "env:///foo/bar"
        inputs = [input]
        stats = storage.create_jobstats_for_original_artifact(output, inputs)
        assert isinstance(stats, dict)
        assert len(stats.get("inputs", [])) == 1
        assert len(stats.get("outputs", [])) == 1
        # TODO: Should really make sure the placeholder artifacts got created,
