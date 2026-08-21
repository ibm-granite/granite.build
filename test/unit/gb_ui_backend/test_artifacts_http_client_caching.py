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

"""Regression test for a code-review observation: list_artifacts/
describe_artifact opened a fresh httpx.AsyncClient (and its connection pool)
on every single call, rather than reusing one cached client the way
cloud_logs.py's CloudLogsClient does.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gb_ui_backend.services.chat_agents import dashboard_tools


@pytest.fixture(autouse=True)
def _reset_cached_client():
    dashboard_tools._artifacts_http_client = None
    yield
    dashboard_tools._artifacts_http_client = None


class _FakeHttpClient:
    instances_created = 0

    def __init__(self, *_args, **_kwargs):
        _FakeHttpClient.instances_created += 1

    async def get(self, *_args, **_kwargs):
        response = SimpleNamespace(status_code=200)
        response.json = lambda: {"artifacts": [], "artifact": {}}
        response.raise_for_status = lambda: None
        return response


@pytest.mark.asyncio
class TestArtifactsHttpClientCaching:
    async def test_the_same_client_is_reused_across_list_and_describe_calls(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            dashboard_tools,
            "get_config",
            lambda: SimpleNamespace(gbserver_url="http://fake"),
        )
        _FakeHttpClient.instances_created = 0
        monkeypatch.setattr(dashboard_tools.httpx, "AsyncClient", _FakeHttpClient)

        await dashboard_tools.list_artifacts()
        await dashboard_tools.describe_artifact("a1")
        await dashboard_tools.list_artifacts(build_id="b1")

        assert _FakeHttpClient.instances_created == 1

    async def test_get_artifacts_http_client_returns_the_same_object_every_call(self):
        first = dashboard_tools._get_artifacts_http_client()
        second = dashboard_tools._get_artifacts_http_client()

        assert first is second
