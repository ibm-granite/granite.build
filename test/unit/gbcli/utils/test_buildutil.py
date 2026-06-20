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

"""Unit tests for gbcli buildutil.get_yaml_patches_in_steps. Pure function with
get_yaml_diff mocked -- no IBM infrastructure required.
"""

import unittest
from unittest.mock import patch

import pytest

from gbcli.utils import buildutil

pytestmark = pytest.mark.standalone


class TestGetYamlPatchesInSteps(unittest.TestCase):
    def test_sets_updated_yaml_when_patch_applies(self):
        """A successful patch must populate validation['updated_yaml'].

        Regression test: the guard previously read an unbound ``updated_yaml``
        local, raising UnboundLocalError on the first iteration.
        """
        validations = [{"json_patch": [{"op": "add", "path": "/x", "value": 1}]}]
        with patch.object(
            buildutil, "get_yaml_diff", return_value={"x": 1, "version": 1}
        ):
            buildutil.get_yaml_patches_in_steps({"version": 1}, validations)

        self.assertIn("updated_yaml", validations[0])
        self.assertIn("x: 1", validations[0]["updated_yaml"])

    def test_no_updated_yaml_when_patch_is_empty(self):
        """An empty diff leaves updated_yaml unset and does not raise."""
        validations = [{"json_patch": []}]
        with patch.object(buildutil, "get_yaml_diff", return_value={}):
            buildutil.get_yaml_patches_in_steps({"version": 1}, validations)

        self.assertNotIn("updated_yaml", validations[0])


if __name__ == "__main__":
    unittest.main()
