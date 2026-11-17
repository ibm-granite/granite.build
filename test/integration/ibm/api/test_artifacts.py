#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Artifact API tests on GitHub-token auth.

The test bodies live in test/integration/standalone/api/test_artifacts.py
(ArtifactAPITests), which runs them on apikey auth; this copy runs them against
real GitHub identities (admin and non-admin tokens in live mode).
"""

import pytest
from integration.standalone.api.test_artifacts import ArtifactAPITests
from libgbtest.api.utils import AbstractAPITest

pytestmark = pytest.mark.ibm


class TestArtifactAPI(ArtifactAPITests, AbstractAPITest):
    """Artifact API tests on GitHub-token auth."""
