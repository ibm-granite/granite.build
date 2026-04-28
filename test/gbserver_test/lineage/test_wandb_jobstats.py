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

import datetime
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage.openlineage_service import LineageService
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.artifact import ArtifactType
from gbserver.types.status import Status

pytestmark = pytest.mark.g4os


# ---------------------------------------------------------------------------
# Helpers for creating test data
# ---------------------------------------------------------------------------
def _make_build(
    uuid: str = "build-001",
    name: str = "test-build",
    space_name: str = "public",
    username: str = "testuser",
    description: str = "",
) -> StoredBuild:
    return StoredBuild(
        uuid=uuid,
        name=name,
        space_name=space_name,
        source_uri="https://github.example.com/org/repo",
        username=username,
        status=Status.SUCCESS,
        description=description,
    )


def _make_target(
    uuid: str = "target-001",
    build_id: str = "build-001",
    name: str = "train",
    input_artifacts: Optional[dict] = None,
    output_artifacts: Optional[dict] = None,
    skipped_for_prerun_target_id: str = "",
) -> StoredTargetRun:
    return StoredTargetRun(
        uuid=uuid,
        build_id=build_id,
        environment_uri="k8s://cluster/namespace",
        name=name,
        status=Status.SUCCESS,
        input_artifacts=input_artifacts or {},
        output_artifacts=output_artifacts or {},
        started_at=datetime.datetime(2024, 4, 15, 10, 0, 0),
        finished_at=datetime.datetime(2024, 4, 15, 11, 0, 0),
        skipped_for_prerun_target_id=skipped_for_prerun_target_id,
    )


def _make_artifact(
    uuid: str = "art-001",
    name: str = "my-model",
    uri: str = "s3://bucket/model",
    art_type: ArtifactType = ArtifactType.MODEL,
    space_name: str = "public",
    username: str = "testuser",
) -> ArtifactRegistration:
    return ArtifactRegistration(
        uuid=uuid,
        name=name,
        uri=uri,
        type=art_type,
        space_name=space_name,
        username=username,
    )


def _make_step(
    uuid: str = "step-001",
    build_id: str = "build-001",
    target_id: str = "target-001",
) -> StoredStepRun:
    return StoredStepRun(
        uuid=uuid,
        build_id=build_id,
        target_id=target_id,
        definition_uri="gbstep://train",
        config={"lr": 0.001},
        config_dir="/configs",
    )


def _make_mock_storage(
    build: StoredBuild,
    targets: list[StoredTargetRun],
    artifacts: dict[str, ArtifactRegistration],
    steps: Optional[list[StoredStepRun]] = None,
) -> SingletonAdminStorage:
    storage = MagicMock()

    storage.build_storage.get_by_uuid.side_effect = lambda uid: (
        build if uid == build.uuid else None
    )

    def get_targets_by_where(row_filter):
        result = targets
        if "build_id" in row_filter:
            result = [t for t in result if t.build_id == row_filter["build_id"]]
        if "uuid" in row_filter:
            result = [t for t in result if t.uuid == row_filter["uuid"]]
        return result

    storage.target_storage.get_by_where.side_effect = get_targets_by_where
    storage.target_storage.get_by_uuid.side_effect = lambda uid: next(
        (t for t in targets if t.uuid == uid), None
    )

    storage.artifact_registry.get_by_uuid.side_effect = lambda uid: artifacts.get(uid)

    storage.step_storage.get_by_where.side_effect = lambda row_filter: [
        s for s in (steps or []) if s.target_id == row_filter.get("target_id")
    ]

    return storage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestWandBLineageStore:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mock_service = MagicMock(spec=LineageService)
        with patch(
            "gbserver.lineage.wandb_jobstats.LineageServiceFactory"
        ) as mock_factory:
            mock_factory.create.return_value = self.mock_service
            from gbserver.lineage.wandb_jobstats import WandBLineageStore

            self.storage_impl = WandBLineageStore()
            yield

    def test_add_jobstats_for_build_target(self):
        input_art = _make_artifact("in-1", "input-data", "s3://bucket/data")
        output_art = _make_artifact("out-1", "output-model", "s3://bucket/model")
        build = _make_build()
        target = _make_target(
            input_artifacts={"data": "in-1"},
            output_artifacts={"model": ["out-1"]},
        )
        step = _make_step()
        storage = _make_mock_storage(
            build, [target], {"in-1": input_art, "out-1": output_art}, [step]
        )

        self.storage_impl.add_jobstats_for_build_target(
            storage, "build-001", "target-001"
        )

        self.mock_service.emit_event.assert_called_once()
        event = self.mock_service.emit_event.call_args[0][0]
        assert event["eventType"] == "COMPLETE"
        assert event["run"]["runId"] == "target-001"
        assert event["job"]["name"] == "train"
        assert event["job"]["namespace"] == "public/test-build"
        assert len(event["inputs"]) == 1
        assert len(event["outputs"]) == 1
        assert event["inputs"][0]["name"] == "input-data"
        assert event["inputs"][0]["facets"]["artifact_id"] == "in-1"
        assert event["outputs"][0]["name"] == "output-model"
        assert event["outputs"][0]["facets"]["artifact_id"] == "out-1"
        assert event["run"]["facets"]["tags"]["build_id"] == "build-001"
        assert (
            event["run"]["facets"]["job_input_params"]["steps"][0]["uri"]
            == "gbstep://train"
        )

    def test_add_jobstats_for_build(self):
        build = _make_build()
        target1 = _make_target(uuid="t1", output_artifacts={"model": ["art-1"]})
        target2 = _make_target(
            uuid="t2", name="eval", output_artifacts={"report": ["art-2"]}
        )
        art1 = _make_artifact("art-1", "m1", "s3://b/m1")
        art2 = _make_artifact("art-2", "m2", "s3://b/m2")
        storage = _make_mock_storage(
            build, [target1, target2], {"art-1": art1, "art-2": art2}
        )

        self.storage_impl.add_jobstats_for_build(storage, "build-001")

        assert self.mock_service.emit_event.call_count == 2

    def test_add_jobstats_for_build_not_found(self):
        storage = MagicMock()
        storage.build_storage.get_by_uuid.return_value = None

        with pytest.raises(ValueError, match="was not found"):
            self.storage_impl.add_jobstats_for_build(storage, "nonexistent")

    def test_add_jobstats_for_build_target_not_found(self):
        build = _make_build()
        storage = _make_mock_storage(build, [], {})

        with pytest.raises(ValueError, match="Zero targets found"):
            self.storage_impl.add_jobstats_for_build_target(
                storage, "build-001", "nonexistent"
            )

    def test_add_jobstats_for_original_artifact(self):
        output = _make_artifact("out-1", "registered-model", "s3://b/model")
        src1 = _make_artifact(
            "src-1", "dataset-a", "s3://b/data-a", ArtifactType.FILESET
        )
        src2 = _make_artifact(
            "src-2", "dataset-b", "s3://b/data-b", ArtifactType.FILESET
        )

        self.storage_impl.add_jobstats_for_original_artifact(output, [src1, src2])

        self.mock_service.emit_event.assert_called_once()
        event = self.mock_service.emit_event.call_args[0][0]
        assert event["eventType"] == "COMPLETE"
        assert event["run"]["runId"] == "out-1"
        assert event["job"]["name"] == "register"
        assert len(event["inputs"]) == 2
        assert len(event["outputs"]) == 1
        assert event["outputs"][0]["facets"]["artifact_id"] == "out-1"

    def test_create_jobstats_for_target(self):
        input_art = _make_artifact("in-1", "data", "s3://b/data")
        output_art = _make_artifact("out-1", "model", "s3://b/model")
        build = _make_build()
        target = _make_target(
            input_artifacts={"data": "in-1"},
            output_artifacts={"model": ["out-1"]},
        )
        storage = _make_mock_storage(
            build, [target], {"in-1": input_art, "out-1": output_art}
        )

        events_list, events_dict = self.storage_impl.create_jobstats_for_target(
            storage, target, build
        )

        self.mock_service.emit_event.assert_not_called()
        assert len(events_list) == 1
        assert "model" in events_dict
        assert len(events_dict["model"]) == 1
        assert events_list[0]["run"]["runId"] == "target-001"
        assert events_list[0]["inputs"][0]["name"] == "data"
        assert events_list[0]["outputs"][0]["name"] == "model"

    def test_create_jobstats_for_target_no_outputs(self):
        input_art = _make_artifact("in-1", "data", "s3://b/data")
        build = _make_build()
        target = _make_target(
            input_artifacts={"data": "in-1"},
            output_artifacts={},
        )
        storage = _make_mock_storage(build, [target], {"in-1": input_art})

        events_list, events_dict = self.storage_impl.create_jobstats_for_target(
            storage, target, build
        )

        assert len(events_list) == 1
        assert "no-output" in events_dict
        assert events_list[0]["outputs"] == []
        assert len(events_list[0]["inputs"]) == 1

    def test_create_jobstats_for_target_skipped(self):
        input_art = _make_artifact("in-1", "data", "s3://b/data")
        output_art = _make_artifact("out-1", "model", "s3://b/model")
        build = _make_build()
        original_target = _make_target(
            uuid="orig-target",
            input_artifacts={"data": "in-1"},
            output_artifacts={"model": ["out-1"]},
        )
        skipped_target = _make_target(
            uuid="skipped-target",
            skipped_for_prerun_target_id="orig-target",
        )
        storage = _make_mock_storage(
            build,
            [original_target, skipped_target],
            {"in-1": input_art, "out-1": output_art},
        )

        events_list, events_dict = self.storage_impl.create_jobstats_for_target(
            storage, skipped_target, build
        )

        assert len(events_list) == 1
        assert events_list[0]["run"]["runId"] == "skipped-target"
        assert "model" in events_dict

    def test_create_jobstats_for_target_build_not_found(self):
        target = _make_target(build_id="nonexistent")
        storage = MagicMock()
        storage.build_storage.get_by_uuid.return_value = None

        with pytest.raises(ValueError, match="could not be found"):
            self.storage_impl.create_jobstats_for_target(storage, target)

    def test_create_jobstats_for_target_build_mismatch(self):
        build = _make_build(uuid="other-build")
        target = _make_target(build_id="build-001")
        storage = MagicMock()

        with pytest.raises(ValueError, match="does not match"):
            self.storage_impl.create_jobstats_for_target(storage, target, build)

    def test_create_jobstats_for_original_artifact(self):
        output = _make_artifact("out-1", "model", "s3://b/model")
        src = _make_artifact("src-1", "data", "s3://b/data")

        result = self.storage_impl.create_jobstats_for_original_artifact(output, [src])

        self.mock_service.emit_event.assert_not_called()
        assert result["eventType"] == "COMPLETE"
        assert result["run"]["runId"] == "out-1"
        assert len(result["inputs"]) == 1
        assert len(result["outputs"]) == 1

    def test_status_mapping(self):
        build = _make_build()
        output_art = _make_artifact("out-1", "model", "s3://b/model")

        for gb_status, expected_event_type in [
            (Status.SUCCESS, "COMPLETE"),
            (Status.FAILED, "FAIL"),
            (Status.RUNNING, "RUNNING"),
            (Status.PENDING, "START"),
            (Status.CANCELLED, "ABORT"),
        ]:
            target = _make_target(output_artifacts={"model": ["out-1"]})
            target.status = gb_status
            storage = _make_mock_storage(build, [target], {"out-1": output_art})

            events_list, _ = self.storage_impl.create_jobstats_for_target(
                storage, target, build
            )
            assert events_list[0]["eventType"] == expected_event_type

    def test_hf_bucket_artifact(self):
        input_art = _make_artifact(
            "in-1",
            "my-bucket",
            "hf:///buckets/org/my-bucket",
            ArtifactType.BUCKET,
        )
        output_art = _make_artifact(
            "out-1",
            "trained-model",
            "hf:///models/org/trained-model",
            ArtifactType.MODEL,
        )
        build = _make_build()
        target = _make_target(
            input_artifacts={"data": "in-1"},
            output_artifacts={"model": ["out-1"]},
        )
        storage = _make_mock_storage(
            build, [target], {"in-1": input_art, "out-1": output_art}
        )

        events_list, _ = self.storage_impl.create_jobstats_for_target(
            storage, target, build
        )

        assert len(events_list) == 1
        event = events_list[0]
        assert event["inputs"][0]["name"] == "org/my-bucket"
        assert event["inputs"][0]["namespace"] == "org"
        assert event["inputs"][0]["uri"] == "hf:///buckets/org/my-bucket"
        assert event["inputs"][0]["facets"]["artifact_type"] == "BUCKET"
        assert event["outputs"][0]["name"] == "org/trained-model"
        assert event["outputs"][0]["uri"] == "hf:///models/org/trained-model"

    def test_lh_table_artifact(self):
        input_art = _make_artifact(
            "in-1",
            "",
            "lh://staging/granite_dot_build.public/tables/synth_data_5kokygr3",
            ArtifactType.TABLE,
        )
        output_art = _make_artifact("out-1", "model", "s3://b/model")
        build = _make_build()
        target = _make_target(
            input_artifacts={"data": "in-1"},
            output_artifacts={"model": ["out-1"]},
        )
        storage = _make_mock_storage(
            build, [target], {"in-1": input_art, "out-1": output_art}
        )

        events_list, _ = self.storage_impl.create_jobstats_for_target(
            storage, target, build
        )

        assert len(events_list) == 1
        event = events_list[0]
        assert event["inputs"][0]["name"] == "synth_data_5kokygr3"
        assert event["inputs"][0]["namespace"] == "granite_dot_build.public"

    def test_lh_fileset_artifact(self):
        input_art = _make_artifact(
            "in-1",
            "",
            "lh://prod/granite_dot_build.public/filesets/fileset_shared/gb_digit_data/20250319T171629",
            ArtifactType.FILESET,
        )
        output_art = _make_artifact("out-1", "model", "s3://b/model")
        build = _make_build()
        target = _make_target(
            input_artifacts={"data": "in-1"},
            output_artifacts={"model": ["out-1"]},
        )
        storage = _make_mock_storage(
            build, [target], {"in-1": input_art, "out-1": output_art}
        )

        events_list, _ = self.storage_impl.create_jobstats_for_target(
            storage, target, build
        )

        assert len(events_list) == 1
        event = events_list[0]
        assert event["inputs"][0]["name"] == "gb_digit_data-20250319T171629"
        assert event["inputs"][0]["namespace"] == "granite_dot_build.public"

    def test_lh_model_artifact(self):
        input_art = _make_artifact(
            "in-1",
            "",
            "lh://prod/granite_dot_build.public/models/model_shared/granite-2b-base/20250319T181102",
            ArtifactType.MODEL,
        )
        output_art = _make_artifact("out-1", "result", "s3://b/result")
        build = _make_build()
        target = _make_target(
            input_artifacts={"model": "in-1"},
            output_artifacts={"result": ["out-1"]},
        )
        storage = _make_mock_storage(
            build, [target], {"in-1": input_art, "out-1": output_art}
        )

        events_list, _ = self.storage_impl.create_jobstats_for_target(
            storage, target, build
        )

        assert len(events_list) == 1
        event = events_list[0]
        assert event["inputs"][0]["name"] == "granite-2b-base-20250319T181102"
        assert event["inputs"][0]["namespace"] == "granite_dot_build.public"
        assert event["inputs"][0]["facets"]["artifact_type"] == "MODEL"

    def test_build_description_in_job_facets(self):
        build = _make_build(description="Train granite model v3")
        output_art = _make_artifact("out-1", "model", "s3://b/model")
        target = _make_target(output_artifacts={"model": ["out-1"]})
        storage = _make_mock_storage(build, [target], {"out-1": output_art})

        events_list, _ = self.storage_impl.create_jobstats_for_target(
            storage, target, build
        )

        assert (
            events_list[0]["job"]["facets"]["documentation"]["description"]
            == "Train granite model v3"
        )

    def test_count_release_ids_no_target(self):
        self.mock_service.search_lineage_by_tags.return_value = (
            2,
            [
                {"run": {"facets": {"tags": {"build_id": "b1", "target_id": "t1"}}}},
                {"run": {"facets": {"tags": {"build_id": "b1", "target_id": "t2"}}}},
            ],
        )
        count = self.storage_impl.count_release_ids("b1")
        assert count == 2
        self.mock_service.search_lineage_by_tags.assert_called_once_with(
            ["build_id=b1"], limit=10000, offset=0
        )

    def test_count_release_ids_with_target(self):
        self.mock_service.search_lineage_by_tags.return_value = (
            2,
            [
                {"run": {"facets": {"tags": {"build_id": "b1", "target_id": "t1"}}}},
                {"run": {"facets": {"tags": {"build_id": "b1", "target_id": "t2"}}}},
            ],
        )
        count = self.storage_impl.count_release_ids("b1", target_id="t1")
        assert count == 1

    def test_count_release_ids_no_results(self):
        self.mock_service.search_lineage_by_tags.return_value = (0, [])
        count = self.storage_impl.count_release_ids("nonexistent")
        assert count == 0

    def test_does_release_id_exist_true(self):
        self.mock_service.search_lineage_by_tags.return_value = (
            1,
            [{"run": {"facets": {"tags": {"build_id": "b1", "target_id": "t1"}}}}],
        )
        assert self.storage_impl.does_release_id_exist("b1", 1, target_id="t1") is True

    def test_does_release_id_exist_false(self):
        self.mock_service.search_lineage_by_tags.return_value = (0, [])
        assert self.storage_impl.does_release_id_exist("b1", 3) is False


class TestFeatureFlagSelection:
    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        import gbserver.lineage.jobstats as mod

        mod.__dict__["_TestFeatureFlagSelection__JOBSTATS_STORAGE"] = None
        # Module-level __name starts with __ but no class mangling at module scope.
        # We need to use setattr on the module's global.
        old = getattr(mod, "_AbstractJobStatsStorage__JOBSTATS_STORAGE", None)
        # Reset via the actual global name used inside get_lineage_store
        mod.get_lineage_store.__globals__["__JOBSTATS_STORAGE"] = None
        yield
        mod.get_lineage_store.__globals__["__JOBSTATS_STORAGE"] = None

    def test_lakehouse_lineage_true_selects_lakehouse(self):
        import gbserver.lineage.jobstats as jobstats_mod
        from gbserver.lineage.lakehouse_jobstats import LakehouseLineageStore

        mock_config = MagicMock()
        mock_config.feature_flags = {"lakehouse_lineage": True}

        with patch("gbserver.types.constants.GB_ENVIRONMENT_CONFIG", mock_config):
            result = jobstats_mod.get_lineage_store()
            assert isinstance(result, LakehouseLineageStore)

    def test_lakehouse_lineage_false_selects_wandb(self):
        import gbserver.lineage.jobstats as jobstats_mod
        from gbserver.lineage.wandb_jobstats import WandBLineageStore

        mock_config = MagicMock()
        mock_config.feature_flags = {"lakehouse_lineage": False}

        with (
            patch("gbserver.types.constants.GB_ENVIRONMENT_CONFIG", mock_config),
            patch(
                "gbserver.lineage.wandb_jobstats.LineageServiceFactory"
            ) as mock_factory,
        ):
            mock_factory.create.return_value = MagicMock(spec=LineageService)
            result = jobstats_mod.get_lineage_store()
            assert isinstance(result, WandBLineageStore)
