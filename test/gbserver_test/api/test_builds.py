# from fastapi.testclient import TestClient

import tempfile
from base64 import b64encode
from pathlib import Path

import pytest

pytestmark = pytest.mark.ibm

from fastapi import Response, status
from gbserver_test.api.utils import AbstractAPITest
from gbserver_test.constants import (
    GBTEST_ADMIN_GITHUB_TOKEN,
    GBTEST_NON_ADMIN_GITHUB_TOKEN,
    GBTEST_SPACE_NAME,
)
from gbserver_test.storage.test_artifact_storage import ArtifactStorageTestSupport
from gbserver_test.storage.test_build_storage import BuildStorageTestSupport
from gbserver_test.storage.test_step_storage import StepStorageTestSupport
from gbserver_test.storage.test_target_storage import TargetStorageTestSupport
from gbserver_test.storage.utils import (
    StorageCollection,
    TargetSpec,
    connect_and_store_build,
)

from gbserver.api.builds import (
    BuildStatus2,
    BuildStatusResponse2,
    BuildSubmitRequest,
    BuildSubmitResponse,
    BuildUpdateRequest,
    BuildUpdateResponse,
    BuildValidateRequest,
    CancelBuildResponse,
    CountBuildsResponse,
    GetBuildResponse,
    ListBuildResponse,
    TargetRecord2,
)
from gbserver.api.utils import ListAppendOrSet
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import SYSTEM_TAG_PREFIX
from gbserver.types.status import Status
from gbserver.utils.archive import create_archive_bytes


def store_finished_build(
    bsts: BuildStorageTestSupport,
    tsts: TargetStorageTestSupport,
    ssts: StepStorageTestSupport,
    asts: ArtifactStorageTestSupport,
    stc: StorageCollection,
    status: Status,
) -> tuple[str, StoredBuild, TargetSpec]:
    """Create a build with 1 target and 2 inputs and 1 output, if status=SUCCESS.

    Args:
        bsts (BuildStorageTestSupport): _description_
        tsts (TargetStorageTestSupport): _description_
        asts (ArtifactStorageTestSupport): _description_
        stc (StorageCollection): _description_
        started (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    build = bsts._get_test_item(0)
    build.status = status
    step = ssts._get_test_item(0)
    if status == Status.SUCCESS:
        target = tsts._get_test_item(0)
        target.status = status
        build.targets = [target.name]
        input_art0 = asts._get_test_item(0)
        input_art1 = asts._get_test_item(1)
        output_art2 = asts._get_test_item(2)
        targetspec = TargetSpec(
            target=target,
            step=step,
            input_artifacts=[input_art0, input_art1],
            output_artifacts=[output_art2],
        )
    elif status in [
        Status.PENDING,
        Status.RUNNING,
        Status.FAILED,
        Status.INVALID,
        Status.CANCELLED,
        Status.CANCEL_REQUESTED,
    ]:
        targetspec = []
    else:
        assert False, f"Status {status} not supported yet."

    connect_and_store_build(build, targetspec, stc)
    return build.uuid, build, targetspec


# def test_get_build_status():
#     bsts = BuildStorageTestSupport()
#     ssts = StepStorageTestSupport()
#     asts = ArtifactStorageTestSupport()

#     # Create and store the build
#     stc = storage_utils.StorageCollection(build_storage=bsts.get_empty_test_storage(),
#                                           artifact_registry=asts.get_empty_test_storage(),
#                                           step_storage=ssts.get_empty_test_storage(),
#     )

#     build_uuid = store_finished_build(bsts,ssts, asts, stc)

#     # configure the api to use our test storage
#     api.artifact_storage = stc.artifact_registry
#     api.build_storage = stc.build_storage
#     api.step_storage = stc.step_storage

#     status = api.get_build_status(build_uuid)
#     print(f"status={status}")

#     # CLean up the storage
#     bsts.get_empty_test_storage()
#     ssts.get_empty_test_storage()
#     asts.get_empty_test_storage()


base_url = "api/v1/builds"


class TestBuildAPI(AbstractAPITest):

    def test_get_status_and_cancel(self):
        # Set up the rest api methods to use our test storage
        bsts = BuildStorageTestSupport()
        tsts = TargetStorageTestSupport()
        ssts = StepStorageTestSupport()
        asts = ArtifactStorageTestSupport()
        stc = StorageCollection(  # This class should eventually go away in favor of the singleton_storage module.
            build_storage=self.storage.build_storage,
            artifact_registry=self.storage.artifact_registry,
            target_storage=self.storage.target_storage,
            step_storage=self.storage.step_storage,
        )
        # # configure the api to use our test storage
        # singleton_storage.set_storage(None, stc.build_storage, stc.target_storage, None, stc.artifact_registry, None)

        # completed = False
        self.run_status_test(bsts, tsts, ssts, asts, stc, tested_status=Status.SUCCESS)
        self.run_status_test(bsts, tsts, ssts, asts, stc, tested_status=Status.PENDING)
        self.run_status_test(bsts, tsts, ssts, asts, stc, tested_status=Status.RUNNING)
        self.run_status_test(bsts, tsts, ssts, asts, stc, tested_status=Status.INVALID)
        self.run_status_test(
            bsts, tsts, ssts, asts, stc, tested_status=Status.CANCELLED
        )
        self.run_status_test(
            bsts, tsts, ssts, asts, stc, tested_status=Status.CANCEL_REQUESTED
        )

    def run_status_test(
        self,
        bsts: BuildStorageTestSupport,
        tsts: TargetStorageTestSupport,
        ssts: StepStorageTestSupport,
        asts: ArtifactStorageTestSupport,
        stc: StorageCollection,
        tested_status: Status,
    ):

        build_uuid, build, targetspec = store_finished_build(
            bsts, tsts, ssts, asts, stc, status=tested_status
        )

        client = self.get_test_client()
        response = client.get(f"{base_url}")
        assert response.status_code == 200

        response = client.get(f"{base_url}/{build_uuid}")
        assert response.status_code == 200

        # Test status
        response = client.get(f"{base_url}/{build_uuid}/status")
        assert response.status_code == 200
        resp_json = response.json()
        print(f"response={resp_json}")
        resp: BuildStatusResponse2 = BuildStatusResponse2.model_validate(resp_json)
        status = resp.status
        assert isinstance(status, BuildStatus2)

        assert status.build.status == tested_status
        if tested_status == Status.SUCCESS:
            # Validate names of targets run
            assert status.build.targets == [targetspec.target.name]
            # Validate the number of targets run
            assert len(status.target_runs) == 1
            # Validate the 1 target that was run
            target_run: TargetRecord2 = status.target_runs[0]
            assert isinstance(target_run, TargetRecord2)
            stored_target = target_run.target
            assert isinstance(stored_target, StoredTargetRun)
            assert stored_target.build_id == build_uuid
            assert stored_target.uuid == targetspec.target.uuid
            assert len(target_run.input_artifacts) == len(targetspec.input_artifacts)
            self._verify_artifacts(target_run.input_artifacts, build, targetspec)
            assert len(target_run.output_artifacts) == len(targetspec.output_artifacts)
            self._verify_artifacts(target_run.output_artifacts, build, targetspec)
            # Validate the step within the target
            assert (
                len(target_run.steps) == 1
            )  # Only 1 step allowed in TargetSpec for now
            stored_step = target_run.steps[0]
            assert isinstance(stored_step, StoredStepRun)
            assert stored_step.build_id == build.uuid
            assert stored_step.uuid == targetspec.step.uuid
            assert stored_step.target_id == targetspec.target.uuid
        elif tested_status in [
            Status.PENDING,
            Status.RUNNING,
            Status.FAILED,
            Status.INVALID,
            Status.CANCELLED,
            Status.CANCEL_REQUESTED,
        ]:
            assert len(status.target_runs) == 0
        else:
            assert False, "Unexpected build status"

        # Test cancel build.
        # NOTE: There are no running builds in this test.  We're just making sure the api sets the right status in the build.
        response = client.delete(f"{base_url}/{build_uuid}")
        if not tested_status.is_cancellable():
            assert response.status_code == 412
        else:
            assert response.status_code == 200
            resp_json = response.json()
            print(f"response={resp_json}")
            resp: CancelBuildResponse = CancelBuildResponse.model_validate(resp_json)
            if tested_status == Status.SUCCESS:
                assert resp.canceled.status == Status.SUCCESS
            elif tested_status == Status.PENDING:
                assert resp.canceled.status == Status.CANCELLED
            elif tested_status == Status.RUNNING:
                assert resp.canceled.status == Status.CANCEL_REQUESTED

    def _verify_artifacts(
        self,
        artifacts: list[ArtifactRegistration],
        build: StoredBuild,
        targetspec: TargetSpec,
    ):
        target: StoredTargetRun = targetspec.target
        step: StoredStepRun = targetspec.step
        for art in artifacts:
            assert isinstance(art, ArtifactRegistration)
            assert art.created_by_build_id == target.build_id
            assert art.created_by_target_id == target.uuid
            # assert art.created_by_step_id == step.uuid  # This field is deprecated
            assert art.username == build.username

    def test_list_pagination(self):
        storage = self.storage.build_storage
        bsts = BuildStorageTestSupport()
        # Add pages of items
        size = 2
        pages = 3
        total = size * pages
        items = []
        for i in range(total):
            item = bsts._get_test_item(i)
            storage.add(item)
            items.append(item)

        client = self.get_test_client()
        for page in range(pages):
            uri = f"{base_url}?page_index={page}&page_size={size}"
            response = client.get(uri)
            # assert response.status_code == 200
            # resp_json = response.json()
            # resp : ListBuildResponse = ListBuildResponse.model_validate(resp_json)
            # results = resp.builds
            results = self._get_build_list_response(response)
            begin_index = page * size
            end_index = begin_index + size  # Exclusive
            expected = items[begin_index:end_index]
            assert len(results) == size
            for i in range(size):
                assert expected[i] == results[i]

    def test_list_sorting(self):
        bsts = BuildStorageTestSupport()
        # Get the items to use from the sub-class
        sort_column, items = bsts._get_ascending_sorted_test_items(3)

        # Insert the items in an unsorted order.
        items_to_add = []
        storage = self.storage.build_storage
        for index in [0, 2, 1]:
            items_to_add.append(items[index])
        storage.add(items_to_add)

        client = self.get_test_client()

        # search via ascending
        uri = f"{base_url}?page_index=0&page_size={len(items)}&sort={sort_column}:asc"
        response = client.get(uri)
        results = self._get_build_list_response(response)
        self._verify_get_results(items, results)

        # search via descending
        items.reverse()
        uri = f"{base_url}?sort={sort_column}:desc"
        response = client.get(uri)
        results = self._get_build_list_response(response)
        self._verify_get_results(items, results)

    def _get_build_submit_response(self, response: Response) -> BuildSubmitResponse:
        assert response.status_code == 200
        resp_json = response.json()
        resp: BuildSubmitResponse = BuildSubmitResponse.model_validate(resp_json)
        return resp

    def _get_update_build_response(self, response: Response) -> StoredBuild:
        assert (
            response.status_code == 200
        ), f"Failed response content={str(response.content)}"
        resp_json = response.json()
        resp: BuildUpdateResponse = BuildUpdateResponse.model_validate(resp_json)  # type: ignore
        build = resp.build
        return build

    def _get_build_list_response(self, response: Response) -> list[StoredBuild]:
        assert response.status_code == 200
        resp_json = response.json()
        resp: ListBuildResponse = ListBuildResponse.model_validate(resp_json)
        builds = resp.builds
        return builds

    def _get_build_count_response(self, response: Response) -> int:
        assert response.status_code == 200
        resp_json = response.json()
        resp: CountBuildsResponse = CountBuildsResponse.model_validate(resp_json)
        return resp.count

    def _get_build_response(self, response: Response) -> StoredBuild:
        assert response.status_code == 200
        resp_json = response.json()
        resp: GetBuildResponse = GetBuildResponse.model_validate(resp_json)
        return resp.build

    def _validate_build_submit_response(
        self, response: Response, req: BuildSubmitRequest
    ) -> StoredBuild:
        bs_response = self._get_build_submit_response(response)
        build_id = bs_response.build_id
        build = self.storage.build_storage.get_by_uuid(build_id)
        assert isinstance(build, StoredBuild)
        assert req.name == build.name
        assert req.build_archive == build.build_archive
        assert req.space_name == build.space_name
        assert req.username == build.username
        assert req.targets == build.targets
        assert req.description == build.description
        assert req.tags == build.tags
        return build

    def test_admin_submit(self):
        self._helper_for_test_submit(True)

    def test_non_admin_submit(self):
        self._helper_for_test_submit(False)

    def _helper_for_test_submit(self, as_admin: bool):

        admin_token = GBTEST_ADMIN_GITHUB_TOKEN
        non_admin_token = GBTEST_NON_ADMIN_GITHUB_TOKEN
        if as_admin:
            if admin_token is None:
                pytest.skip(reason="No github admin token available in the environment")
            token = admin_token
        else:
            if non_admin_token is None:
                pytest.skip(
                    reason="No github non-admin token available in the environment"
                )
            token = non_admin_token

        client = self.get_test_client(token=token)
        headers = {
            "content-type": "application/json",
        }
        username0 = self.get_gh_username(token=token)
        space_name0 = GBTEST_SPACE_NAME

        # Register a single artifact with or without system tags
        current_description = "de1"
        for using_sys_tags in [True, False]:
            # First time, try with system tags and if not an admin, then expect an exception
            current_tags = (
                [SYSTEM_TAG_PREFIX + "tag1"] if using_sys_tags else ["tag1"]
            )  # Try again w/o system tags
            req = BuildSubmitRequest(
                name="name0",
                space_name=space_name0,
                build_archive="xyz",
                username=username0,
                targets=["a", "b"],
                description=current_description,
                tags=current_tags,
            )
            response = client.post(
                f"{base_url}", data=req.model_dump_json(), headers=headers
            )
            if as_admin or not using_sys_tags:
                build0 = self._validate_build_submit_response(response, req)
                break  # don't need to retry when running as admin
            else:
                assert (
                    not as_admin and using_sys_tags
                ), "Should only get here on this condition"
                assert (
                    response.status_code == status.HTTP_401_UNAUTHORIZED
                ), "Non-Admin should not be able to submit with system tags"

        # Append tags and test
        appended_tags = [SYSTEM_TAG_PREFIX + "tag2"]
        tags_req = ListAppendOrSet(append=appended_tags)
        req = BuildUpdateRequest(tags=tags_req)
        response = client.put(
            f"{base_url}/{build0.uuid}/update",
            data=req.model_dump_json(),
            headers=headers,
        )
        if as_admin:
            build = self._get_update_build_response(response)
            current_tags.extend(appended_tags)
            assert build.tags == current_tags
            assert build.description == current_description
        else:
            assert (
                response.status_code == status.HTTP_401_UNAUTHORIZED
            ), "Non-Admin should not be able to append system tags"

        # Set tags and test
        set_tags = [SYSTEM_TAG_PREFIX + "tag3"]
        tags_req = ListAppendOrSet(set=set_tags)
        req = BuildUpdateRequest(tags=tags_req)
        response = client.put(
            f"{base_url}/{build0.uuid}/update",
            data=req.model_dump_json(),
            headers=headers,
        )
        if as_admin:
            build = self._get_update_build_response(response)
            current_tags = set_tags
            assert build.tags == current_tags
            assert build.description == current_description
        else:
            assert (
                response.status_code == status.HTTP_401_UNAUTHORIZED
            ), "Non-Admin should not be able to set system tags"

        # Try deleting a system tag as a non-admin
        if as_admin:
            client = self.get_test_client(
                token=non_admin_token
            )  # switch to the non-admin client
            set_tags = []
            tags_req = ListAppendOrSet(set=set_tags)
            req = BuildUpdateRequest(tags=tags_req)
            response = client.put(
                f"{base_url}/{build0.uuid}/update",
                data=req.model_dump_json(),
                headers=headers,
            )
            assert (
                response.status_code == status.HTTP_401_UNAUTHORIZED
            ), "Non-Admin should not be able to set system tags"

    def test_build_get(self):
        # Get the test_gb_artifact table storage, empty it and have the rest api use it.
        build_storage = self.storage.build_storage
        support = BuildStorageTestSupport()
        client = self.get_test_client()

        # Make sure there are no items
        response = client.get(f"{base_url}")
        builds = self._get_build_list_response(response)
        assert len(builds) == 0

        # Add 3 items
        item0 = support._get_test_item(0)
        item1 = support._get_test_item(1)
        item2 = support._get_test_item(2)
        build_storage.add([item0, item1, item2])

        # Test get all
        response = client.get(f"{base_url}")
        builds = self._get_build_list_response(response)
        assert len(builds) == 3

        # Test get by uuid
        response = client.get(f"{base_url}/non-uuid")
        assert response.status_code == 404

        # Test get by uuid
        response = client.get(f"{base_url}/{item1.uuid}")
        build = self._get_build_response(response)
        assert build.username == item1.username

        # test get by where
        response = client.get(f"{base_url}?username={item1.username}")
        artifacts = self._get_build_list_response(response)
        assert len(artifacts) == 3  # All builds have the same username

        # test get by where
        response = client.get(f"{base_url}?space_name={item1.space_name}")
        artifacts = self._get_build_list_response(response)
        assert len(artifacts) == 1
        build = artifacts[0]
        assert build.username == item1.username

        # test get by where
        response = client.get(f"{base_url}?source_uri={item1.source_uri}")
        artifacts = self._get_build_list_response(response)
        assert len(artifacts) == 1
        build = artifacts[0]
        assert build.username == item1.username

        # test get by where matching tag
        response = client.get(f"{base_url}?tag={item1.tags[0]}")
        artifacts = self._get_build_list_response(response)
        assert len(artifacts) == 1
        build = artifacts[0]
        assert build.username == item1.username

        # test get by where non-matching tags
        response = client.get(f"{base_url}?tag={item1.tags[0]}&tag=not-present")
        artifacts = self._get_build_list_response(response)
        assert len(artifacts) == 0

    def test_count(self):
        """Test the /count endpoint returns the correct number of builds."""
        build_storage = self.storage.build_storage
        support = BuildStorageTestSupport()
        client = self.get_test_client()

        # Count on empty storage should return 0
        response = client.get(f"{base_url}/count")
        count = self._get_build_count_response(response)
        assert count == 0, "Expected count of 0 on empty storage"

        # Add items and verify count increases
        item0 = support._get_test_item(0)
        build_storage.add(item0)
        response = client.get(f"{base_url}/count")
        count = self._get_build_count_response(response)
        assert count == 1, "Expected count of 1 after adding first item"

        # Add more items
        item1 = support._get_test_item(1)
        item2 = support._get_test_item(2)
        build_storage.add([item1, item2])
        response = client.get(f"{base_url}/count")
        count = self._get_build_count_response(response)
        assert count == 3, "Expected count of 3 after adding two more items"

        # Delete an item and verify count decreases
        build_storage.delete(item0.uuid)
        response = client.get(f"{base_url}/count")
        count = self._get_build_count_response(response)
        assert count == 2, "Expected count of 2 after deleting one item"

    def test_count_with_where(self):
        """Test the /count endpoint with filter parameters."""
        build_storage = self.storage.build_storage
        support = BuildStorageTestSupport()
        client = self.get_test_client()

        # Add items with different usernames
        item0 = support._get_test_item(0)
        item1 = support._get_test_item(1)
        item2 = support._get_test_item(2)
        build_storage.add([item0, item1, item2])

        # Count all should return 3
        # response = client.get(f"{base_url}/count")
        # count = self._get_build_count_response(response)
        # assert count == 3, "Expected count of 3 for all items"

        # Count with username filter
        response = client.get(f"{base_url}/count?name={item0.name}")
        count = self._get_build_count_response(response)
        assert count == 1, "Expected count of 1 when filtering by username"

        # Count with non-matching username filter
        response = client.get(f"{base_url}/count?name=non-existent-build")
        count = self._get_build_count_response(response)
        assert count == 0, "Expected count of 0 when no items match"

    def _get_validate_test_data_dir(self) -> Path:
        """Get the directory containing validation test data."""
        src_file_dir = Path(__file__).resolve().parent
        # Navigate from test/gbserver_test/api to test-data/gbserver_test/api/builds/validate
        test_data_dir = (
            src_file_dir.parent.parent.parent
            / "test-data"
            / "gbserver_test"
            / "api"
            / "builds"
            / "validate"
        )
        assert test_data_dir.is_dir(), f"Test data directory not found: {test_data_dir}"
        return test_data_dir

    def _create_build_archive_from_yaml(self, build_yaml_path: Path) -> str:
        """Create a base64-encoded zip archive from a build.yaml file.

        Args:
            build_yaml_path: Path to the build.yaml file.

        Returns:
            Base64-encoded string of the zip archive.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir) / "build"
            build_dir.mkdir()
            # Copy the yaml file as build.yaml in the archive
            dest_path = build_dir / "build.yaml"
            dest_path.write_text(build_yaml_path.read_text())
            archive_bytes = create_archive_bytes(build_dir)
            return b64encode(archive_bytes).decode("utf-8")

    def _helper_validate_build_yaml_files(self, test_dir: Path, expect_valid: bool):
        """Helper to test validation of build.yaml files in a directory.

        Args:
            test_dir: Directory containing build.yaml files to test.
            expect_valid: If True, expect all files to be valid (200). If False, expect invalid (422).
        """
        yaml_files = list(test_dir.glob("*.yaml")) + list(test_dir.glob("*.yml"))
        if len(yaml_files) == 0:
            pytest.skip(f"No yaml files found in {test_dir}")

        client = self.get_test_client()
        headers = {"content-type": "application/json"}
        username = self.get_gh_username()

        for yaml_file in yaml_files:
            build_archive = self._create_build_archive_from_yaml(yaml_file)
            req = BuildValidateRequest(
                build_archive=build_archive,
                space_name=GBTEST_SPACE_NAME,
                username=username,
            )
            response = client.post(
                f"{base_url}/validate", json=req.model_dump(), headers=headers
            )

            if expect_valid:
                assert (
                    response.status_code == status.HTTP_200_OK
                ), f"Expected {yaml_file.name} to be valid, but got {response.status_code}: {response.json()}"
            else:
                assert (
                    response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
                ), f"Expected {yaml_file.name} to be invalid, but got {response.status_code}"
                resp_json = response.json()
                assert (
                    "errors" in resp_json
                ), f"Expected 'errors' in response for {yaml_file.name}"

    def test_build_validation_valid(self):
        """Test that all build.yaml files in the valid directory pass validation."""
        test_dir = self._get_validate_test_data_dir() / "valid"
        self._helper_validate_build_yaml_files(test_dir, expect_valid=True)

    def test_build_validation_invalid(self):
        """Test that all build.yaml files in the invalid directory fail validation."""
        test_dir = self._get_validate_test_data_dir() / "invalid"
        self._helper_validate_build_yaml_files(test_dir, expect_valid=False)

    def test_get_status_failed(self):
        """Verify that a FAILED build is returned correctly by the status API."""
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
        self.run_status_test(bsts, tsts, ssts, asts, stc, tested_status=Status.FAILED)
