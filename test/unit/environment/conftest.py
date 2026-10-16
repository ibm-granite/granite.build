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

"""Shared fixtures for the environment unit tests."""

import pytest


@pytest.fixture
def real_apply_cloud_config():
    """Opt a test out of :func:`_no_cloud_config_writes` (it patches what it needs)."""


@pytest.fixture(autouse=True)
def _no_cloud_config_writes(request, monkeypatch):
    """Keep unit-test launches from rewriting the developer's ~/.sky/config.yaml.

    Every Skypilot launch applies its env's cloud_config last-writer-wins, which
    writes the real home directory's SkyPilot config and reloads the client. A
    unit test that launches (with sky mocked) must not do either, so this makes
    ``_apply_cloud_config`` a no-op unless the test requests
    ``real_apply_cloud_config``.
    """
    if "real_apply_cloud_config" in request.fixturenames:
        return
    monkeypatch.setattr(
        "gbserver.environment.skypilot._apply_cloud_config",
        lambda env_name, cloud_config: None,
    )
