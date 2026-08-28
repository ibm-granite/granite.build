"""Construct K8s exactly as the fixture does, and print any exception fully."""
import asyncio
import traceback
from unittest.mock import MagicMock, patch

env_config = MagicMock()
env_config.config = {"namespace": "gb-test"}
env_config.type = "K8s"


def fake_init(self, *_args, **_kwargs):
    self.config = env_config
    self.secrets = {}


try:
    from gbserver.environment.k8s import K8s

    with patch("gbserver.environment.environment.Environment.__init__", new=fake_init):
        k8s = K8s(
            event_q=asyncio.Queue(),
            environment_config=env_config,
            node_health_tracker=MagicMock(),
        )
    print("CONSTRUCTED OK:", type(k8s).__name__)
except Exception:
    print("FAILED:")
    traceback.print_exc()
