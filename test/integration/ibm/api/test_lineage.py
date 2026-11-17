#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Lineage API tests on GitHub-token auth, against the real WandB lineage service.

The test bodies live in test/integration/standalone/api/test_lineage.py
(LineageAPITests), which runs them on apikey auth with the WandB service
stubbed out.
"""

import pytest
from integration.standalone.api.test_lineage import LineageAPITests
from libgbtest.api.utils import AbstractAPITest

from gbserver.types.constants import ENV_VAR_PREFIX

pytestmark = pytest.mark.ibm


class TestLineageAPI(LineageAPITests, AbstractAPITest):
    """Lineage API tests on GitHub-token auth with the real lineage store."""

    def _server_env(self) -> dict[str, str]:
        """Pin the WandB lineage store.

        Under GB_ENVIRONMENT=STANDALONE the provider would default to "none", and
        the no-op store would silently skip the records_centralized_lineage-guarded
        assertions.

        Returns:
            The base env, with GBSERVER_LINEAGE_PROVIDER=wandb.
        """
        return {**super()._server_env(), f"{ENV_VAR_PREFIX}_LINEAGE_PROVIDER": "wandb"}
