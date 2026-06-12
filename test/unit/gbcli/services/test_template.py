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

"""Unit tests for gbcli template service.

Ported from gbcli-upstream test/unit_tests/services/test_template.py. The repo tree
listing is mocked — no IBM infrastructure or network required.

Adapted to the current ``list_templates(github_token, space=None,
template_repo=None, ...)`` signature: ``github_token`` is now a required positional,
the branch comes from ``gb_environment_config().space_config_branch_name`` (explicit
repo) or ``.branch_assets`` (default assets repo), and the rendered ``description``
URL is built from ``DEFAULT_GH_DOMAIN`` plus the org/name parsed out of the repo URL.
"""

from unittest.mock import patch

import pytest

from gbcli.services.service_template import list_templates
from gbcli.utils.gbconstants import (
    ASSETS_REPO_NAME,
    ASSETS_REPO_ORG,
    gb_environment_config,
)
from gbcommon.types.constants import DEFAULT_GH_DOMAIN

pytestmark = pytest.mark.standalone

_FAKE_TOKEN = "fake-token"


class TestTemplate:
    @patch("gbcli.services.service_template.list_repo_tree")
    def test_list_templates_with_template_repo(self, mock_list_repo_tree):
        """List templates from an explicit template_repo (no callback)."""
        # github.com/<org>/<name> -> org/name parsed from split("/")[3:]
        template_repo = "https://github.com/ibm-watsonx/watsonx-code-assistant-assets"
        org = "ibm-watsonx"
        name = "watsonx-code-assistant-assets"
        branch_name = gb_environment_config().space_config_branch_name

        mock_list_repo_tree.return_value = {
            "tree": [
                {
                    "path": "templates/python-flask-template",
                    "type": "tree",
                },
                {
                    "path": "templates/python-flask-template/build.yaml",
                    "type": "blob",
                },
            ]
        }

        expected_output = [
            {
                "template_name": "python-flask-template",
                "description": (
                    f"https://{DEFAULT_GH_DOMAIN}/{org}/{name}/tree/{branch_name}/"
                    "templates/python-flask-template"
                ),
            }
        ]

        actual_output = list_templates(_FAKE_TOKEN, template_repo=template_repo)
        assert actual_output == expected_output

    @patch("gbcli.services.service_template.list_repo_tree")
    def test_list_templates(self, mock_list_repo_tree):
        """List templates from the default assets repo (no callback)."""
        branch_name = gb_environment_config().branch_assets

        mock_list_repo_tree.return_value = {
            "tree": [
                {
                    "path": "templates/python-flask-template",
                    "type": "tree",
                },
                {
                    "path": "templates/python-flask-template/build.yaml",
                    "type": "blob",
                },
            ]
        }

        expected_output = [
            {
                "template_name": "python-flask-template",
                "description": (
                    f"https://{DEFAULT_GH_DOMAIN}/{ASSETS_REPO_ORG}/{ASSETS_REPO_NAME}"
                    f"/tree/{branch_name}/templates/python-flask-template"
                ),
            }
        ]

        actual_output = list_templates(_FAKE_TOKEN)
        assert actual_output == expected_output
