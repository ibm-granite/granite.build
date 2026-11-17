#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

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
