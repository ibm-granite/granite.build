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

"""Space API tests on GitHub-token auth.

The test bodies live in test/integration/standalone/api/test_spaces.py
(SpacesAPITests, SpaceMembersAPITests), which runs them on apikey auth.
"""

import pytest
from integration.standalone.api.test_spaces import (
    SpaceMembersAPITests,
    SpacesAPITests,
)
from libgbtest.api.utils import AbstractAPITest

pytestmark = pytest.mark.ibm


class TestSpacesAPI(SpacesAPITests, AbstractAPITest):
    """Space API tests on GitHub-token auth."""


class TestSpaceMembersAPI(SpaceMembersAPITests, AbstractAPITest):
    """Space member API tests on GitHub-token auth."""
