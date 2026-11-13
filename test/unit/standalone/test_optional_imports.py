# Copyright Granite.Build Authors
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

"""Tests that optional IBM/LH/RabbitMQ packages can be absent without crashing imports.

These tests verify the import guards added in issue #6 (architecture sections 1.4-1.10).
They run with the full test venv (all packages installed) and verify guard flags
and lazy-import patterns are properly set up.
"""


class TestStandaloneImports:
    """Verify that key modules have proper import guards for optional dependencies."""

    def test_singleton_storage_importable(self):
        """Section 1.4: singleton_storage uses lazy LH/shadowed imports."""
        from gbserver.storage.singleton_storage import get_storage_factory

        assert callable(get_storage_factory)

    def test_root_api_importable(self):
        """Section 1.7: root_api imports succeed."""
        from gbserver.api.root_api import root_api

        assert root_api is not None

    def test_secrets_api_no_toplevel_ibm_import(self):
        """Section 1.7: secrets.py does not import IBM SDK at module level."""
        import importlib

        mod = importlib.import_module("gbserver.api.secrets")
        # The class should NOT be in module globals (lazy-imported via helper)
        assert "IbmcloudSpaceSecretManagerAdmin" not in dir(mod)

    def test_secrets_api_has_lazy_helper(self):
        """Section 1.7: secrets.py has _get_ibm_secret_manager_admin helper."""
        from gbserver.api.secrets import _get_ibm_secret_manager_admin

        assert callable(_get_ibm_secret_manager_admin)

    def test_command_build_no_toplevel_lh_import(self):
        """Section 1.10: command_build.py does not import LhSpaceStorage at module level."""
        import importlib

        mod = importlib.import_module("gbserver.commands.command_build")
        assert "LhSpaceStorage" not in dir(mod)

    def test_buildwatcher_no_toplevel_buildrunnerjob(self):
        """BuildRunnerJob (kubernetes_asyncio) is not imported at module level."""
        import importlib

        mod = importlib.import_module("gbserver.buildwatcher.buildwatcher")
        assert "BuildRunnerJob" not in dir(mod)
