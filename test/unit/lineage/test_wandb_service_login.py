# SPDX-License-Identifier: Apache-2.0

"""WandBLineageService logs in without wandb's network check.

wandb 0.31.2 (the gbserver image's) changed ``wandb.login``'s ``verify`` default
from False to True: login then authenticates over the network and raises
AuthenticationError on any failure, an unreachable host included. The service is
built by the lineage watcher and the API paths, so it keeps 0.27.2's behaviour.
"""

from unittest.mock import patch

from gbserver.lineage.wandb_service import WandBLineageService
from gbserver.types.constants import GBSERVER_WANDB_API_KEY, GBSERVER_WANDB_BASE_URL


def test_login_passes_verify_false():
    with patch("gbserver.lineage.wandb_service.wandb") as fake:
        WandBLineageService()
    fake.login.assert_called_once_with(
        key=GBSERVER_WANDB_API_KEY, host=GBSERVER_WANDB_BASE_URL, verify=False
    )
