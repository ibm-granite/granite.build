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
