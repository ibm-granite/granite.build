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

"""Unit tests for the PROD artifact-registration gate in ``register_artifact``.

The gate admits an artifact in the PROD environment iff its URI type is
production-safe (``URI.is_prod_safe()``). HfURI is always safe; LhURI is safe
only when it points at the production Lakehouse host; all other URI types are
rejected. These tests exercise the gate directly without any storage or secret
infrastructure — rejection paths raise before storage is touched, and the one
allow path mocks the downstream storage.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, status

from gbserver.api import artifacts as artifacts_module
from gbserver.api.artifacts import register_artifact
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.types.artifact import ArtifactType


def _registration(uri: str, art_type: ArtifactType = ArtifactType.MODEL):
    return ArtifactRegistration(
        type=art_type,
        uri=uri,
        space_name="sn",
        username="un",
    )


def _call_gate(uri: str, art_type: ArtifactType = ArtifactType.MODEL):
    """Invoke register_artifact in a PROD environment with a dummy request."""
    request = MagicMock()
    with patch.object(artifacts_module, "GB_ENVIRONMENT", "PROD"):
        return register_artifact(request, _registration(uri, art_type))


class TestRegisterArtifactProdGate:
    def test_hf_uri_allowed_in_prod(self):
        """HF artifacts register in PROD (the feature this gate enables)."""
        with patch.object(artifacts_module, "is_super_admin", return_value=True):
            with patch.object(artifacts_module, "get_admin_storage") as mock_storage:
                mock_storage.return_value.artifact_registry.add = MagicMock()
                resp = _call_gate("hf://huggingface.co/models/ibm-granite/granite")

        assert resp.registered.uri == "hf://huggingface.co/models/ibm-granite/granite"

    def test_lh_uri_prod_host_allowed_in_prod(self):
        """LhURI pointing at the production Lakehouse host is allowed in PROD."""
        with patch.object(artifacts_module, "is_super_admin", return_value=True):
            with patch.object(artifacts_module, "get_admin_storage") as mock_storage:
                mock_storage.return_value.artifact_registry.add = MagicMock()
                resp = _call_gate("lh://prod/namespace0/models/table0/label0/rev0")

        assert resp.registered.uri.startswith("lh://prod/")

    def test_lh_uri_non_prod_host_rejected_in_prod(self):
        """LhURI at a non-prod host is rejected in PROD (regression guard)."""
        with pytest.raises(HTTPException) as exc:
            _call_gate("lh://staging/namespace0/models/table0/label0/rev0")
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST

    def test_file_uri_rejected_in_prod(self):
        """A file:// artifact has no prod-safety notion and is rejected in PROD.

        Pins the default-deny behavior: only explicitly prod-safe URI types are
        admitted, so non-environment URI types stay blocked in PROD.
        """
        with pytest.raises(HTTPException) as exc:
            _call_gate("file:///tmp/some/artifact")
        assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
