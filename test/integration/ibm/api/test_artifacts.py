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

import json
from typing import Optional
from urllib.parse import quote

import pytest

pytestmark = pytest.mark.ibm

# REST API
from fastapi import Response, status
from fastapi.testclient import TestClient
from git import Union
from libgbtest.api.utils import AbstractAPITest
from libgbtest.constants import (
    GBTEST_ADMIN_GITHUB_TOKEN,
    GBTEST_NON_ADMIN_GITHUB_TOKEN,
    GBTEST_SPACE_NAME,
)
from libgbtest.storage.artifact_storage import ArtifactStorageTestSupport

from gbcommon.uri.hf import HfType, HfURI
from gbcommon.uri.lh import LhURI
from gbserver.api.artifacts import (
    ArtifactDatasetRequest,
    ArtifactFilesetlRequest,
    ArtifactModelRequest,
    ArtifactTableRequest,
    ArtifactUpdateRequest,
    ArtifactUpdateResponse,
    ChangeArchiveResponse,
    DecodedHfURIResponse,
    DecodedURIResponse,
    GetArtifactResponse,
    ListAppendOrSet,
    ListArtifactsResponse,
    RegisterArtifactResponse,
)
from gbserver.api.auth import get_gh_user
from gbserver.lineage.jobstats import get_lineage_store
from gbserver.storage.artifact_registration import (
    ArtifactRegistration,
    ArtifactRegistrationStatus,
)
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.types.constants import PUBLIC_SPACE_NAME, SYSTEM_TAG_PREFIX

base_url = "api/v1/artifacts"


class TestArtifactAPI(AbstractAPITest):

    def test_artifact_get(self):
        # Get the test_gb_artifact table storage, empty it and have the rest api use it.
        art_storage = self.storage.artifact_registry
        # singleton_storage.set_storage(None, None, None, None, art_storage, None)
        support = ArtifactStorageTestSupport()
        client = self.get_test_client()

        # Make sure there are no items
        response = client.get(f"{base_url}")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 0

        # Add 2 items
        a0 = support._get_test_item(0)
        a1 = support._get_test_item(1)
        a2 = support._get_test_item(2)
        a2.is_archived = True
        art_storage.add([a0, a1, a2])

        # Test get all
        response = client.get(f"{base_url}")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 3

        # Test get by uuid
        response = client.get(f"{base_url}/non-uuid")
        assert response.status_code == 404

        # Test get by uuid
        response = client.get(f"{base_url}/{a1.uuid}")
        assert response.status_code == 200
        resp_json = response.json()
        resp: GetArtifactResponse = GetArtifactResponse.model_validate(resp_json)
        art: ArtifactRegistration = resp.artifact
        assert art != None
        assert art.username == a1.username

        # test get by where
        response = client.get(f"{base_url}?username={a1.username}")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 1
        art = artifacts[0]
        assert art.username == a1.username

        # test get by where
        response = client.get(f"{base_url}?build_id={a1.created_by_build_id}")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 1
        art = artifacts[0]
        assert art.username == a1.username

        # test get by where
        response = client.get(f"{base_url}?uri={a1.uri}")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 1
        art = artifacts[0]
        assert art.username == a1.username

        # test get by where
        response = client.get(f"{base_url}?is_archived=True")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 1
        art = artifacts[0]
        assert art.username == a2.username

        # test get by where with checksum
        response = client.get(f"{base_url}?checksum={a1.checksum}")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 1
        art = artifacts[0]
        assert art.username == a1.username

        # test get by where matching tag
        response = client.get(f"{base_url}?tag={a1.tags[0]}")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 1
        art = artifacts[0]
        assert art.username == a1.username

        # test get by where non-matching tags
        response = client.get(f"{base_url}?tag={a1.tags[0]}&tag=not-present")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 0

    def _get_artifact_registration_from_registration_response(
        self, response: Response
    ) -> ArtifactRegistration:
        assert response.status_code == 200
        resp_json = response.json()
        resp: RegisterArtifactResponse = RegisterArtifactResponse.model_validate(
            resp_json
        )
        artifact = resp.registered
        return artifact

    def _get_artifact_archive_response(
        self, response: Response
    ) -> ChangeArchiveResponse:
        assert response.status_code == 200
        resp_json = response.json()
        resp: ChangeArchiveResponse = ChangeArchiveResponse.model_validate(resp_json)
        return resp

    def _get_artifact_list_response(
        self, response: Response
    ) -> list[ArtifactRegistration]:
        assert response.status_code == 200
        resp_json = response.json()
        resp: ListArtifactsResponse = ListArtifactsResponse.model_validate(resp_json)
        artifacts = resp.artifacts
        return artifacts

    def _get_update_artifact_response(self, response: Response) -> ArtifactRegistration:
        assert (
            response.status_code == 200
        ), f"Failed response content={str(response.content)}"
        resp_json = response.json()
        resp: ArtifactUpdateResponse = ArtifactUpdateResponse.model_validate(resp_json)  # type: ignore
        artifact = resp.artifact
        return artifact

    def _validate_artifact_registration_response(
        self,
        response: Response,
        req: Union[
            ArtifactTableRequest,
            ArtifactModelRequest,
            ArtifactFilesetlRequest,
            ArtifactDatasetRequest,
        ],
    ) -> ArtifactRegistration:
        artifact = self._get_artifact_registration_from_registration_response(response)
        assert isinstance(artifact, ArtifactRegistration)
        if isinstance(req, ArtifactTableRequest):
            expected_uri = LhURI.get_table_uri(
                table_name=req.table_name, namespace=req.namespace
            )
        elif isinstance(req, ArtifactModelRequest):
            expected_uri = LhURI.get_model_uri(
                table_name=req.table_name,
                namespace=req.namespace,
                model_label=req.model_label,
                model_revision=req.model_revision,
            )
        elif isinstance(req, ArtifactFilesetlRequest):
            expected_uri = LhURI.get_fileset_uri(
                table_name=req.table_name,
                namespace=req.namespace,
                fileset_label=req.fileset_label,
                fileset_version=req.fileset_version,
            )
        elif isinstance(req, ArtifactDatasetRequest):
            expected_uri = LhURI.get_dataset_uri(
                table_name=req.table_name,
                namespace=req.namespace,
                dataset_name=req.dataset_name,
            )
        assert expected_uri == artifact.uri, f"Did not get expected uri {expected_uri}"
        assert artifact.tags == req.tags
        assert artifact.space_name == req.space_name
        # assert artifact.namespace == req.namespace
        # assert artifact.table_name == req.table_name
        name = (
            req.name if req.name and len(req.name) > 0 else req.table_name
        )  # API uses the table name when name is not provided.
        assert artifact.name == name
        # asesrt artifact.lh_env == req.lh_env
        assert artifact.tags == req.tags
        assert artifact.certified_no_restrictions == req.certified_no_restrictions
        assert artifact.origin_uris == req.origin_uris
        assert artifact.description == req.description
        return artifact

    def _XYZvalidate_artifact_registration_response(
        self, response: Response, expected_uri
    ):
        artifact = self._get_artifact_registration_from_registration_response(response)
        uri = artifact.uri
        assert expected_uri == uri, f"Did not get expected uri {expected_uri}"

    def test_artifact_register(self):
        # # Get the test_gb_artifact table storage, empty it and have the rest api use it.
        # art_storage = self.storage.artifact_registry
        # # only art_storage
        # singleton_storage.set_storage(None, None, None, None, art_storage, None)

        tablename0 = "tablename0"
        namespace0 = "namespace0"
        dataset_name0 = "ds0"
        model_label0 = "ml0"
        model_revision0 = "mr0"
        space_name0 = "sn0"
        tablename1 = "tablename1"
        namespace1 = "namespace1"
        dataset_name1 = "ds1"
        model_label1 = "ml1"
        model_revision1 = "mr1"
        space_name1 = "sn1"
        fileset_label1 = "fileset1"
        fileset_version1 = "fileset_v1"
        username0 = "un0"
        username1 = "un1"

        client = self.get_test_client()
        headers = {
            "content-type": "application/json",
        }

        # Register the artifacts
        req = ArtifactTableRequest(
            username=username0,
            space_name=space_name0,
            namespace=namespace0,
            table_name=tablename0,
            certified_no_restrictions=True,
        )
        response = client.post(
            f"{base_url}/lh/table", data=req.model_dump_json(), headers=headers
        )
        self._validate_artifact_registration_response(response, req)

        req = ArtifactDatasetRequest(
            username=username1,
            space_name=space_name1,
            namespace=namespace1,
            table_name=tablename1,
            dataset_name=dataset_name1,
            certified_no_restrictions=True,
        )
        response = client.post(
            f"{base_url}/lh/dataset", data=req.model_dump_json(), headers=headers
        )
        self._validate_artifact_registration_response(response, req)

        req = ArtifactModelRequest(
            username=username1,
            space_name=space_name1,
            namespace=namespace1,
            table_name=tablename1,
            model_label=model_label1,
            model_revision=model_revision1,
            certified_no_restrictions=True,
        )
        response = client.post(
            f"{base_url}/lh/model", data=req.model_dump_json(), headers=headers
        )
        self._validate_artifact_registration_response(response, req)

        req = ArtifactFilesetlRequest(
            username=username1,
            space_name=space_name1,
            namespace=namespace1,
            table_name=tablename1,
            fileset_label=fileset_label1,
            fileset_version=fileset_version1,
            certified_no_restrictions=True,
        )
        response = client.post(
            f"{base_url}/lh/fileset", data=req.model_dump_json(), headers=headers
        )
        self._validate_artifact_registration_response(response, req)

        response = client.get(f"{base_url}")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 4

        # Test get by uuid
        # response = client.get(f"{base_url}/{a1.uuid}")    # TODO: Why is this one failing?  it works above?
        response = client.get(f"{base_url}?username={username0}")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 1
        art: ArtifactRegistration = artifacts[0]
        assert art.username == username0

    def test_artifact_archival(self):
        # # Get the test_gb_artifact table storage, empty it and have the rest api use it.
        # art_storage = self.storage.artifact_registry
        # # only art_storage
        # singleton_storage.set_storage(None, None, None, None, art_storage, None)

        tablename0 = "tablename0"
        namespace0 = "namespace0"
        dataset_name0 = "ds0"
        model_label0 = "ml0"
        model_revision0 = "mr0"
        space_name0 = "sn0"
        tablename1 = "tablename1"
        namespace1 = "namespace1"
        dataset_name1 = "ds1"
        model_label1 = "ml1"
        model_revision1 = "mr1"
        space_name1 = "sn1"
        fileset_label1 = "fileset1"
        fileset_version1 = "fileset_v1"
        username0 = "un0"
        username1 = "un1"

        client = self.get_test_client()
        headers = {
            "content-type": "application/json",
        }

        # Register the artifacts
        req = ArtifactTableRequest(
            username=username0,
            space_name=space_name0,
            namespace=namespace0,
            table_name=tablename0,
            certified_no_restrictions=True,
        )
        response = client.post(
            f"{base_url}/lh/table", data=req.model_dump_json(), headers=headers
        )
        artifact0 = self._get_artifact_registration_from_registration_response(response)
        assert artifact0.is_archived == False

        req = ArtifactDatasetRequest(
            username=username1,
            space_name=space_name1,
            namespace=namespace1,
            table_name=tablename1,
            dataset_name=dataset_name1,
            certified_no_restrictions=True,
        )
        response = client.post(
            f"{base_url}/lh/dataset", data=req.model_dump_json(), headers=headers
        )
        artifact1 = self._get_artifact_registration_from_registration_response(response)
        assert artifact1.is_archived == False

        # Archive artifact0
        response = client.put(f"{base_url}/{artifact0.uuid}/archive")
        car = self._get_artifact_archive_response(response)
        assert car.artifact.uuid == artifact0.uuid
        assert car.was_archived == False
        assert car.artifact.is_archived == True

        # Now search for archived and expect only artifact0
        response = client.get(f"{base_url}?is_archived=True")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 1
        assert artifacts[0].uuid == artifact0.uuid

        # Now search for unarchived and expect only artifact1
        response = client.get(f"{base_url}?is_archived=False")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 1
        assert artifacts[0].uuid == artifact1.uuid

        # UnArchive artifact0
        response = client.put(f"{base_url}/{artifact0.uuid}/unarchive")
        car = self._get_artifact_archive_response(response)
        assert car.artifact.uuid == artifact0.uuid
        assert car.was_archived == True
        assert car.artifact.is_archived == False

        # Now search for unarchived and expect both
        response = client.get(f"{base_url}?is_archived=False")
        artifacts = self._get_artifact_list_response(response)
        assert len(artifacts) == 2

    def test_uncertifed_registration(self):
        tablename0 = "tablename0"
        namespace0 = "namespace0"
        dataset_name0 = "ds0"
        model_label0 = "ml0"
        model_revision0 = "mr0"
        space_name0 = "sn0"
        username0 = "un0"

        client = self.get_test_client()
        headers = {
            "content-type": "application/json",
        }

        # Register the artifacts
        req = ArtifactTableRequest(
            username=username0,
            space_name=space_name0,
            namespace=namespace0,
            table_name=tablename0,
            certified_no_restrictions=False,
        )
        response = client.post(
            f"{base_url}/lh/table", data=req.model_dump_json(), headers=headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_artifact_register_with_origins(self):
        client = self.get_test_client()
        headers = {
            "content-type": "application/json",
        }

        # Register some input artifacts
        origins = []
        origin_uris = []
        for index in range(2):
            req = ArtifactTableRequest(
                username=f"user",
                space_name="space",
                namespace="namespace",
                table_name=f"table{index}",
                certified_no_restrictions=True,
            )
            response = client.post(
                f"{base_url}/lh/table", data=req.model_dump_json(), headers=headers
            )
            artifact = self._get_artifact_registration_from_registration_response(
                response
            )
            origins.append(artifact)
            origin_uris.append(artifact.uri)

        # Now make the request we're testing - to create a jobstats for an artifact with origins
        req = ArtifactTableRequest(
            username=f"user",
            space_name="space",
            namespace="namespace",
            table_name=f"derived_table",
            origin_uris=origin_uris,
        )
        response = client.post(
            f"{base_url}/lh/table", data=req.model_dump_json(), headers=headers
        )
        artifact = self._get_artifact_registration_from_registration_response(response)

        # JobStats record with a release_id equal to the output artifact id should have been created.
        job_storage = get_lineage_store()
        # assert not job_storage.does_release_id_exist("should-not-exist")
        assert job_storage.does_release_id_exist(artifact.uuid, 1)

        # Now try and register another artifact with an input that does not exist.
        origin_uris.append("lh://unregistered")
        req = ArtifactTableRequest(
            username=f"user",
            space_name="space",
            namespace="namespace",
            table_name=f"derived_table_should_fail",
            origin_uris=origin_uris,
        )
        response = client.post(
            f"{base_url}/lh/table", data=req.model_dump_json(), headers=headers
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def _get_uri_decoding(
        self, response: Response
    ) -> Union[DecodedURIResponse, DecodedHfURIResponse]:
        assert response.status_code == 200
        resp_json = response.json()
        if "owner" in resp_json:
            return DecodedHfURIResponse.model_validate(resp_json)
        return DecodedURIResponse.model_validate(resp_json)

    def _validate_decoding(
        self,
        decoding: Union[DecodedURIResponse, DecodedHfURIResponse],
        expected: dict[str, str],
    ):
        d = dict(decoding)
        # Only test values in the expected dictionary.
        for key, value in expected.items():
            d_value = d.get(key, None)
            assert d_value == value

    def _validate_uri_decode_result(
        self, client: TestClient, uri_or_id: str, expected: dict[str, str], by_id=False
    ):
        if by_id:
            response = client.get(f"{base_url}/decode?id={uri_or_id}")
        else:
            encoded_uri = quote(uri_or_id, safe="")
            response = client.get(f"{base_url}/decode?uri={encoded_uri}")
        decoding = self._get_uri_decoding(response)
        self._validate_decoding(decoding, expected)

    def test_artifact_uri_decoder_by_uri(self):
        client = self.get_test_client()
        namespace = "test_namespace"
        table_name = "test_tablename"
        lh_env = "lh_env"

        uri = LhURI.get_table_uri(
            namespace=namespace, table_name=table_name, lh_env=lh_env
        )
        expected = {
            "namespace": namespace,
            "table_name": table_name,
            "uri": uri,
            "type": "table",
        }
        self._validate_uri_decode_result(client, uri, expected)

        model_label = "test_model_label"
        model_revision = "test_model_revision"
        uri = LhURI.get_model_uri(
            namespace=namespace,
            table_name=table_name,
            lh_env=lh_env,
            model_label=model_label,
            model_revision=model_revision,
        )
        expected = {
            "namespace": namespace,
            "table_name": table_name,
            "uri": uri,
            "type": "model",
            "model_label": model_label,
            "model_revision": model_revision,
        }
        self._validate_uri_decode_result(client, uri, expected)

        fileset_label = "test_fileset_label"
        fileset_version = "test_fileset_version"
        uri = LhURI.get_fileset_uri(
            namespace=namespace,
            table_name=table_name,
            lh_env=lh_env,
            fileset_label=fileset_label,
            fileset_version=fileset_version,
        )
        expected = {
            "namespace": namespace,
            "table_name": table_name,
            "uri": uri,
            "type": "fileset",
            "fileset_label": fileset_label,
            "fileset_version": fileset_version,
        }
        self._validate_uri_decode_result(client, uri, expected)

        dataset_name = "test_dataset_name"
        uri = LhURI.get_dataset_uri(
            namespace=namespace,
            table_name=table_name,
            lh_env=lh_env,
            dataset_name=dataset_name,
        )
        expected = {
            "namespace": namespace,
            "table_name": table_name,
            "uri": uri,
            "type": "dataset",
            "dataset_name": dataset_name,
        }
        self._validate_uri_decode_result(client, uri, expected)

    def test_artifact_uri_decoder_by_id(self):
        client = self.get_test_client()
        headers = {
            "content-type": "application/json",
        }

        namespace = "test_namespace"
        table_name = "test_tablename"
        username = "test_username"
        space_name = "test_spacename"

        # Register an artifact
        req = ArtifactTableRequest(
            username=username,
            space_name=space_name,
            namespace=namespace,
            table_name=table_name,
            certified_no_restrictions=True,
        )
        response = client.post(
            f"{base_url}/lh/table", data=req.model_dump_json(), headers=headers
        )
        artifact = self._get_artifact_registration_from_registration_response(response)

        # Now decode its uri via its uuid
        expected = {
            "namespace": namespace,
            "table_name": table_name,
            "uri": artifact.uri,
            "type": "table",
        }
        self._validate_uri_decode_result(client, artifact.uuid, expected, by_id=True)

    def test_artifact_update_by_non_admin(self):

        tablename0 = "tablename0"
        namespace0 = "namespace0"
        dataset_name0 = "ds0"
        model_label0 = "ml0"
        model_revision0 = "mr0"
        space_name0 = GBTEST_SPACE_NAME
        username0 = self.get_gh_username()

        client = self.get_test_client()
        headers = {
            "content-type": "application/json",
        }

        # Register an artifact
        current_tags = ["tags1"]
        current_description = "de1"
        req = ArtifactTableRequest(
            username=username0,
            space_name=space_name0,
            namespace=namespace0,
            table_name=tablename0,
            certified_no_restrictions=True,
            description=current_description,
            tags=current_tags,
        )
        response = client.post(
            f"{base_url}/lh/table", data=req.model_dump_json(), headers=headers
        )
        artifact0 = self._get_artifact_registration_from_registration_response(response)
        assert artifact0.tags == current_tags
        assert artifact0.description == current_description

        # Update the description and test
        new_description = "de1"
        req = ArtifactUpdateRequest(description=new_description)
        response = client.put(
            f"{base_url}/{artifact0.uuid}/update",
            data=req.model_dump_json(),
            headers=headers,
        )
        art = self._get_update_artifact_response(response)
        assert art.description == new_description
        assert art.tags == current_tags
        current_description = new_description

        # Append tags and test
        appended_tags = ["tag2"]
        tags_req = ListAppendOrSet(append=appended_tags)
        req = ArtifactUpdateRequest(tags=tags_req)
        response = client.put(
            f"{base_url}/{artifact0.uuid}/update",
            data=req.model_dump_json(),
            headers=headers,
        )
        art = self._get_update_artifact_response(response)
        current_tags.extend(appended_tags)
        assert art.tags == current_tags
        assert art.description == current_description

        # Set tags and test
        set_tags = ["tag3"]
        tags_req = ListAppendOrSet(set=set_tags)
        req = ArtifactUpdateRequest(tags=tags_req)
        response = client.put(
            f"{base_url}/{artifact0.uuid}/update",
            data=req.model_dump_json(),
            headers=headers,
        )
        art = self._get_update_artifact_response(response)
        current_tags = set_tags
        assert art.tags == current_tags
        assert art.description == current_description

        # Set and append tags and expect an error
        tags_req = ListAppendOrSet(set=set_tags, append=appended_tags)
        req = ArtifactUpdateRequest(tags=tags_req)
        response = client.put(
            f"{base_url}/{artifact0.uuid}/update",
            data=req.model_dump_json(),
            headers=headers,
        )
        assert response.status_code == 400

        # Set tags and description
        set_tags = ["tag3"]
        current_description = "de3"
        tags_req = ListAppendOrSet(set=set_tags)
        req = ArtifactUpdateRequest(description=current_description, tags=tags_req)
        response = client.put(
            f"{base_url}/{artifact0.uuid}/update",
            data=req.model_dump_json(),
            headers=headers,
        )
        art = self._get_update_artifact_response(response)
        current_tags = set_tags
        assert art.tags == current_tags
        assert art.description == current_description

    def test_artifact_update_system_tags(self):
        self._helper_artifact_update_system_tags(True)
        self._helper_artifact_update_system_tags(False)

    def _helper_artifact_update_system_tags(self, as_admin: bool):
        admin_token = GBTEST_ADMIN_GITHUB_TOKEN
        non_admin_token = GBTEST_NON_ADMIN_GITHUB_TOKEN
        if as_admin:
            if admin_token is None:
                pytest.skip(reason="No github admin token available in the environment")
            token = admin_token
            tablename0 = "admintable"
        else:
            if non_admin_token is None:
                pytest.skip(
                    reason="No github non-admin token available in the environment"
                )
            token = non_admin_token
            tablename0 = "nonadmintable"

        client = self.get_test_client(token=token)
        username0 = self.get_gh_username(token=token)
        namespace0 = "namespace0"
        space_name0 = GBTEST_SPACE_NAME

        # Register the admin user in space_user_storage so StorageSpaceAccessManager
        # recognizes them as a super-admin (admin of the public space).
        if as_admin:
            user, _ = get_gh_user(token)
            assert user is not None
            self.storage.space_user_storage.add(
                StoredSpaceUser(
                    space_name=PUBLIC_SPACE_NAME, username=user.email, role="admin"
                )
            )

        headers = {
            "content-type": "application/json",
        }

        # Register a single artifact with or without system tags
        current_description = "de1"
        for using_sys_tags in [True, False]:
            # First time, try with system tags and if not an admin, then expect an exception
            current_tags = (
                [SYSTEM_TAG_PREFIX + "tag1"] if using_sys_tags else ["tag1"]
            )  # Try again w/o system tags
            req = ArtifactTableRequest(
                username=username0,
                space_name=space_name0,
                namespace=namespace0,
                table_name=tablename0,
                certified_no_restrictions=True,
                description=current_description,
                tags=current_tags,
            )
            response = client.post(
                f"{base_url}/lh/table", data=req.model_dump_json(), headers=headers
            )
            if as_admin or not using_sys_tags:
                artifact0 = self._get_artifact_registration_from_registration_response(
                    response
                )
                assert artifact0.tags == current_tags
                assert artifact0.description == current_description
                break  # don't need to retry when running as admin
            else:
                assert (
                    not as_admin and using_sys_tags
                ), "Should only get here on this condition"
                assert (
                    response.status_code == status.HTTP_401_UNAUTHORIZED
                ), "Non-Admin should not be able to create with system tags"

        # Append tags and test
        appended_tags = [SYSTEM_TAG_PREFIX + "tag2"]
        tags_req = ListAppendOrSet(append=appended_tags)
        req = ArtifactUpdateRequest(tags=tags_req)
        response = client.put(
            f"{base_url}/{artifact0.uuid}/update",
            data=req.model_dump_json(),
            headers=headers,
        )
        if as_admin:
            art = self._get_update_artifact_response(response)
            current_tags.extend(appended_tags)
            assert art.tags == current_tags
            assert art.description == current_description
        else:
            assert (
                response.status_code == status.HTTP_401_UNAUTHORIZED
            ), "Non-Admin should not be able to append system tags"

        # Set tags and test
        set_tags = [SYSTEM_TAG_PREFIX + "tag3"]
        tags_req = ListAppendOrSet(set=set_tags)
        req = ArtifactUpdateRequest(tags=tags_req)
        response = client.put(
            f"{base_url}/{artifact0.uuid}/update",
            data=req.model_dump_json(),
            headers=headers,
        )
        if as_admin:
            art = self._get_update_artifact_response(response)
            current_tags = set_tags
            assert art.tags == current_tags
            assert art.description == current_description
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
            req = ArtifactUpdateRequest(tags=tags_req)
            response = client.put(
                f"{base_url}/{artifact0.uuid}/update",
                data=req.model_dump_json(),
                headers=headers,
            )
            assert (
                response.status_code == status.HTTP_401_UNAUTHORIZED
            ), "Non-Admin should not be able to set system tags"

    def __get_fileset_request(
        self, index: int, checksum: Optional[str] = None
    ) -> ArtifactFilesetlRequest:
        if checksum is None:
            checksum = str(index)
        req = ArtifactFilesetlRequest(
            username=f"username{index}",
            space_name=f"space_name{index}",
            namespace=f"namespace{index}",
            table_name=f"tablename{index}",
            fileset_label=f"fileset_label{index}",
            fileset_version=f"fileset_version{index}",
            certified_no_restrictions=True,
            checksum=checksum,
        )
        return req

    def test_checksum_semantics(self):

        client = self.get_test_client()
        headers = {
            "content-type": "application/json",
        }

        # Register the fileset number 0, with empty checksum (not enforced)
        req = self.__get_fileset_request(0, "")
        response = client.post(
            f"{base_url}/lh/fileset", data=req.model_dump_json(), headers=headers
        )
        self._validate_artifact_registration_response(response, req)

        # Register the fileset number 1, without checksum
        req = self.__get_fileset_request(1, "")
        response = client.post(
            f"{base_url}/lh/fileset", data=req.model_dump_json(), headers=headers
        )
        self._validate_artifact_registration_response(response, req)

        # Register the fileset number 2 with checksum
        req = self.__get_fileset_request(2, "2")
        response = client.post(
            f"{base_url}/lh/fileset", data=req.model_dump_json(), headers=headers
        )
        artifact2 = self._validate_artifact_registration_response(response, req)

        # Register the fileset number 3 with the same checksum as item 2
        req = self.__get_fileset_request(3, "2")
        response = client.post(
            f"{base_url}/lh/fileset", data=req.model_dump_json(), headers=headers
        )
        assert response.status_code == 409
        content = json.loads(response.content)
        detail = content["detail"]
        assert detail["uri"] == artifact2.uri, "Conflicting URI is not the expected one"
        assert (
            detail["uuid"] == artifact2.uuid
        ), "Conflicting UID is not the expected one"

    def test_update_status(self):
        if GBTEST_ADMIN_GITHUB_TOKEN is None or GBTEST_NON_ADMIN_GITHUB_TOKEN is None:
            pytest.skip(reason="No github admin token available in the environment")

        # Register the admin user as super-admin so StorageSpaceAccessManager grants access
        admin_user, _ = get_gh_user(GBTEST_ADMIN_GITHUB_TOKEN)
        assert admin_user is not None
        self.storage.space_user_storage.add(
            StoredSpaceUser(
                space_name=PUBLIC_SPACE_NAME, username=admin_user.email, role="admin"
            )
        )

        admin_client = self.get_test_client(token=GBTEST_ADMIN_GITHUB_TOKEN)
        non_admin_client = self.get_test_client(token=GBTEST_NON_ADMIN_GITHUB_TOKEN)
        headers = {
            "content-type": "application/json",
        }

        # Register the fileset number 0, with PENDING status
        req = self.__get_fileset_request(0, "")
        req.status = ArtifactRegistrationStatus.PENDING
        response = admin_client.post(
            f"{base_url}/lh/fileset", data=req.model_dump_json(), headers=headers
        )
        registration = self._validate_artifact_registration_response(response, req)
        assert registration.status == ArtifactRegistrationStatus.PENDING

        req = ArtifactUpdateRequest(status=ArtifactRegistrationStatus.SUCCESS)
        response = non_admin_client.put(
            f"{base_url}/{registration.uuid}/update",
            data=req.model_dump_json(),
            headers=headers,
        )
        assert response.status_code == 401  # Not authorized

        response = admin_client.put(
            f"{base_url}/{registration.uuid}/update",
            data=req.model_dump_json(),
            headers=headers,
        )
        art = self._get_update_artifact_response(response)
        assert art.status == ArtifactRegistrationStatus.SUCCESS

    def test_artifact_uri_decoder_hf_model(self):
        """Test /decode with HF model URI"""
        client = self.get_test_client()
        uri = HfURI.from_parts(
            owner="ibm-granite",
            repo="granite-3b",
            hf_type=HfType.MODEL,
            revision="main",
        )
        uri_str = uri.get_uristr(uri)
        expected = {
            "owner": "ibm-granite",
            "repo": "granite-3b",
            "revision": "main",
            "host": "huggingface.co",
        }
        self._validate_uri_decode_result(client, uri_str, expected)

    def test_artifact_uri_decoder_hf_dataset(self):
        """Test /decode with HF dataset URI"""
        client = self.get_test_client()
        uri = HfURI.from_parts(
            owner="wikitext",
            repo="wikitext-103-v1",
            hf_type=HfType.DATASET,
        )
        uri_str = uri.get_uristr(uri)
        expected = {
            "owner": "wikitext",
            "repo": "wikitext-103-v1",
            "hf_type": "dataset",
            "revision": "main",
        }
        self._validate_uri_decode_result(client, uri_str, expected)

    def test_hf_model_reg_name(self):
        """Test that HF model registration uses model_id as name when name is empty"""
        client = self.get_test_client()
        headers = {"content-type": "application/json"}
        req_data = {
            "model_id": "test-model",
            "space_name": GBTEST_SPACE_NAME,
            "username": self.get_gh_username(),
            "organization": "ibm-research",
            "certified_no_restrictions": True,
        }
        response = client.post(f"{base_url}/hf/model", json=req_data, headers=headers)
        assert response.status_code == 200
        artifact = response.json()["registered"]
        assert artifact["name"] == "test-model"

    def test_hf_dataset_reg_name(self):
        """Test that HF dataset registration uses dataset_id as name when name is empty"""
        client = self.get_test_client()
        headers = {"content-type": "application/json"}
        req_data = {
            "dataset_id": "test-dataset",
            "space_name": GBTEST_SPACE_NAME,
            "username": self.get_gh_username(),
            "organization": "ibm-research",
            "certified_no_restrictions": True,
        }
        response = client.post(f"{base_url}/hf/dataset", json=req_data, headers=headers)
        assert response.status_code == 200
        artifact = response.json()["registered"]
        assert artifact["name"] == "test-dataset"

    def test_hf_bucket_reg_name(self):
        """Test that HF bucket registration uses bucket_id as name when name is empty"""
        client = self.get_test_client()
        headers = {"content-type": "application/json"}
        req_data = {
            "bucket_id": "test-bucket",
            "space_name": GBTEST_SPACE_NAME,
            "username": self.get_gh_username(),
            "organization": "ibm-research",
            "certified_no_restrictions": True,
        }
        response = client.post(f"{base_url}/hf/bucket", json=req_data, headers=headers)
        assert response.status_code == 200
        artifact = response.json()["registered"]
        assert artifact["name"] == "test-bucket"

    def test_artifact_uri_decoder_hf_bucket(self):
        """Test /decode with HF bucket URI"""
        client = self.get_test_client()
        uri = HfURI.from_parts(
            owner="ibm-granite",
            repo="test-bucket",
            hf_type=HfType.BUCKET,
        )
        uri_str = uri.get_uristr(uri)
        expected = {
            "owner": "ibm-granite",
            "repo": "test-bucket",
            "hf_type": "bucket",
        }
        self._validate_uri_decode_result(client, uri_str, expected)
