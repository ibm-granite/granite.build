# Copyright Granite.Build Authors
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

"""fm-tune drives verl's v0 RayPPOTrainer on purpose; fail loudly if a verl bump removes it.

verl 0.9 defaults to the V1 trainer (``trainer.use_v1``), which routes rollouts
through TransferQueue. TransferQueue's controller is a cluster-wide *named* Ray
actor ("TransferQueueController" in namespace "transfer_queue") and ``tq.init()``
attaches to an existing one, so concurrent RL HPO trials in one Ray job would
share a single queue (mixing rollout data across trials) and the first trial's
``tq.close()`` would tear it down under the rest. Until TransferQueue can be
isolated per trial, driver_multi_verl.py stays on v0. If this test fails, verl
dropped v0: the port to V1 has to solve that isolation first (see CLAUDE.md,
"verl Integration").
"""

import pytest

pytest.importorskip("verl")


def test_v0_trainer_and_engine_workers_still_exist():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role  # noqa: F401
    from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker  # noqa: F401

    assert hasattr(RayPPOTrainer, "init_workers") and hasattr(RayPPOTrainer, "fit")


def test_transfer_queue_controller_is_still_a_global_singleton():
    """When this fails, TQ may support per-trial isolation and the V1 port is unblocked."""
    import inspect

    tq_interface = pytest.importorskip("transfer_queue.interface")
    src = inspect.getsource(tq_interface)
    assert 'name="TransferQueueController"' in src and 'namespace="transfer_queue"' in src
