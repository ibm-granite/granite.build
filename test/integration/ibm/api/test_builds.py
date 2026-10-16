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

"""Build API tests on GitHub-token auth.

The test bodies live in test/integration/standalone/api/test_builds.py
(BuildAPITests), which runs them on apikey auth; this copy runs them against
real GitHub identities and validates builds against the IBM public space.
"""

from pathlib import Path

import pytest
from integration.standalone.api.test_builds import BuildAPITests
from libgbtest.api.utils import AbstractAPITest
from libgbtest.buildrunner.buildtest import get_test_data_dir_for

pytestmark = pytest.mark.ibm


class TestBuildAPI(BuildAPITests, AbstractAPITest):
    """Build API tests on GitHub-token auth."""

    def _get_validate_test_data_dir(self) -> Path:
        """Validation fixtures that resolve against the IBM public space."""
        test_data_dir = get_test_data_dir_for(__file__) / "builds" / "validate"
        assert test_data_dir.is_dir(), f"Test data directory not found: {test_data_dir}"
        return test_data_dir
