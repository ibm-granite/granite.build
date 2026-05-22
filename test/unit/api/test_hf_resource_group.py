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

"""Unit tests for GET /hf/resource-group endpoint (mocked, no HF API calls)."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from gbserver.api.artifacts import artifacts_api

client = TestClient(artifacts_api)


class TestHFResourceGroupEndpoint:
    def test_resolve_success(self):
        with (
            patch("gbserver.api.artifacts.get_hf_token", return_value="fake-token"),
            patch(
                "gbserver.api.artifacts.HfURI.resolve_resource_group_id_for_org",
                return_value="prod-id-123",
            ),
        ):
            resp = client.get(
                "/hf/resource-group",
                params={
                    "space_name": "public",
                    "organization": "ibm-research",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resource_group_id"] == "prod-id-123"
        assert "gbspace-public" in data["resource_group_name"]

    def test_resolve_not_found(self):
        with (
            patch("gbserver.api.artifacts.get_hf_token", return_value="fake-token"),
            patch(
                "gbserver.api.artifacts.HfURI.resolve_resource_group_id_for_org",
                return_value=None,
            ),
        ):
            resp = client.get(
                "/hf/resource-group",
                params={
                    "space_name": "nonexistent",
                    "organization": "ibm-research",
                },
            )

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_resolve_value_error(self):
        with (
            patch("gbserver.api.artifacts.get_hf_token", return_value="fake-token"),
            patch(
                "gbserver.api.artifacts.HfURI.resolve_resource_group_id_for_org",
                side_effect=ValueError("Could not resolve resource group id"),
            ),
        ):
            resp = client.get(
                "/hf/resource-group",
                params={
                    "space_name": "broken",
                    "organization": "ibm-research",
                },
            )

        assert resp.status_code == 404
        assert "Could not resolve" in resp.json()["detail"]

    def test_missing_space_name_returns_422(self):
        resp = client.get(
            "/hf/resource-group",
            params={"organization": "ibm-research"},
        )
        assert resp.status_code == 422

    def test_missing_organization_returns_422(self):
        resp = client.get(
            "/hf/resource-group",
            params={"space_name": "public"},
        )
        assert resp.status_code == 422

    def test_empty_space_name_returns_400(self):
        resp = client.get(
            "/hf/resource-group",
            params={
                "space_name": "",
                "organization": "ibm-research",
            },
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_missing_token_returns_503(self):
        with patch("gbserver.api.artifacts.get_hf_token", return_value=None):
            resp = client.get(
                "/hf/resource-group",
                params={
                    "space_name": "public",
                    "organization": "ibm-research",
                },
            )

        assert resp.status_code == 503
        assert "not configured" in resp.json()["detail"].lower()

    def test_staging_env_uses_suffixed_name(self, monkeypatch):
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "STAGING")
        with (
            patch("gbserver.api.artifacts.get_hf_token", return_value="fake-token"),
            patch(
                "gbserver.api.artifacts.HfURI.resolve_resource_group_id_for_org",
                return_value="staging-id",
            ),
        ):
            resp = client.get(
                "/hf/resource-group",
                params={
                    "space_name": "public",
                    "organization": "ibm-research",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["resource_group_name"] == "gbspace-public-staging"
        assert data["resource_group_id"] == "staging-id"
