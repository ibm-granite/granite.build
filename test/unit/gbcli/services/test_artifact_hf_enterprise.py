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

"""Unit tests for the HF Enterprise/non-Enterprise split in ``upload_to_hf``.

A non-Enterprise org must upload without a resource group; an Enterprise org
must still require one. HFRegistry is mocked — no HF calls are made.
"""

from unittest.mock import patch

import pytest

from gbcli.services.service_artifact import upload_to_hf

pytestmark = pytest.mark.standalone

_ENTERPRISE = ["ibm-research", "ibm-granite"]


def _run_upload(tmp_path, org, resource_group_id=None, artifact_type="model"):
    src = tmp_path / "model.bin"
    src.write_bytes(b"weights")
    with (
        patch(
            "gbcli.services.service_artifact.HF_ENTERPRISE_ORGANIZATIONS",
            _ENTERPRISE,
        ),
        patch("gbcli.services.service_artifact.HFRegistry") as registry_cls,
    ):
        registry_cls.return_value.upload_artifact.return_value = "uploaded"
        result = upload_to_hf(
            hf_token="tok",
            path_name=str(src),
            artifact_name="my-model",
            type=artifact_type,
            hf_organization=org,
            resource_group_id=resource_group_id,
        )
    return result, registry_cls


class TestUploadToHfNonEnterprise:
    def test_uploads_without_resource_group(self, tmp_path):
        """A non-Enterprise org no longer needs a resource group id."""
        result, registry_cls = _run_upload(tmp_path, org="my-user")

        assert result == "uploaded"
        assert registry_cls.call_args.kwargs["resource_group_id"] is None
        assert registry_cls.call_args.kwargs["organization"] == "my-user"

    def test_bucket_type_uploads_without_resource_group(self, tmp_path):
        """The bucket branch (create_bucket) also accepts no resource group."""
        result, registry_cls = _run_upload(
            tmp_path, org="my-user", artifact_type="bucket"
        )

        assert result == "uploaded"
        assert registry_cls.call_args.kwargs["resource_group_id"] is None
        assert (
            registry_cls.return_value.upload_artifact.call_args.kwargs["artifact_type"]
            == "bucket"
        )

    def test_fileset_maps_to_bucket(self, tmp_path):
        """-t fileset also reaches create_bucket, so it must work too."""
        _, registry_cls = _run_upload(tmp_path, org="my-user", artifact_type="fileset")

        assert (
            registry_cls.return_value.upload_artifact.call_args.kwargs["artifact_type"]
            == "bucket"
        )


class TestUploadToHfEnterprise:
    def test_missing_resource_group_still_raises(self, tmp_path):
        """An Enterprise org must still supply a resource group id."""
        with pytest.raises(Exception, match="No HuggingFace resource group id"):
            _run_upload(tmp_path, org="ibm-research")

    def test_forwards_resource_group(self, tmp_path):
        _, registry_cls = _run_upload(
            tmp_path, org="ibm-research", resource_group_id="rg-123"
        )

        assert registry_cls.call_args.kwargs["resource_group_id"] == "rg-123"
