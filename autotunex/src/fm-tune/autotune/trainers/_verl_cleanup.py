# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
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

"""Release every Ray actor and placement group a verl ``RayPPOTrainer`` created.

``RayPPOTrainer`` has no shutdown method, and between HPO trials anything it
leaves alive keeps holding GPU memory (vLLM servers, FSDP workers) or GPU
*reservations* (placement groups survive actor death: the next trial then fails
with ``Total available GPUs 0 is less than total desired GPUs N``).

verl 0.7 kept every actor in four worker groups (``actor_rollout_wg``,
``critic_wg``, ``ref_policy_wg``, ``reward_wg``), so the old cleanup killed
those by name. verl 0.9 also starts actors *outside* the worker groups -- the
vLLM servers behind ``llm_server_manager``, ``reward_loop_manager``'s
``RewardLoopWorker`` actors, ``async_rollout_manager``'s ``AgentLoopWorker``
actors -- none with a shutdown method, and their attribute names change between
releases. So instead of naming them, :func:`collect_ray_resources` walks the
trainer's object graph and returns every actor handle and placement group it can
reach. It imports nothing from verl, which keeps it unit-testable without GPUs.

Only objects reachable from *this* trainer are touched: concurrent HPO trials
share one Ray job, so a "kill every actor in the job" sweep would take down
sibling trials.
"""

import logging
import time
from typing import Any, Callable, List, Tuple

logger = logging.getLogger(__name__)

# Object-graph depth bound. verl 0.9.1's deepest handle (trainer ->
# async_rollout_manager -> agent_loop_workers -> [i]) is depth 3; vLLM servers
# sit ~5 deep. 8 leaves headroom without walking into torch/omegaconf internals.
_MAX_DEPTH = 8

# Leaf types that never hold Ray handles and are expensive or unsafe to walk.
_SKIP_TYPE_PREFIXES = (
    "torch.",
    "omegaconf.",
    "numpy.",
    "transformers.",
    "tokenizers.",
    "pandas.",
    "pyarrow.",
    "datasets.",
    "hydra.",
    # Ray internals: handles are matched before this check, so skipping these never misses one.
    "ray.",
)


def collect_ray_resources(
    root: Any,
    is_actor: Callable[[Any], bool],
    is_placement_group: Callable[[Any], bool],
    max_depth: int = _MAX_DEPTH,
) -> Tuple[List[Any], List[Any]]:
    """Return ``(actors, placement_groups)`` reachable from ``root``, each listed once."""
    actors: List[Any] = []
    pgs: List[Any] = []
    seen = set()

    def visit(obj: Any, depth: int) -> None:
        if obj is None or depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        if is_actor(obj):
            actors.append(obj)
            return
        if is_placement_group(obj):
            pgs.append(obj)
            return
        if isinstance(obj, (str, bytes, int, float, bool)):
            return
        if type(obj).__module__.startswith(_SKIP_TYPE_PREFIXES):
            return
        if isinstance(obj, dict):
            children = list(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            children = list(obj)
        else:
            children = list(getattr(obj, "__dict__", {}).values())
        for child in children:
            visit(child, depth + 1)

    visit(root, 0)
    return actors, pgs


def cleanup_ray_resources(trainer: Any) -> None:
    """Kill every actor and remove every placement group reachable from ``trainer``.

    Never raises: it runs in a ``finally`` after ``trainer.fit()``, where an
    exception would mask the training error being propagated.
    """
    try:
        import ray
        from ray.actor import ActorHandle
        from ray.util.placement_group import PlacementGroup

        actors, pgs = collect_ray_resources(
            trainer,
            is_actor=lambda o: isinstance(o, ActorHandle),
            is_placement_group=lambda o: isinstance(o, PlacementGroup),
        )
    except Exception as e:  # noqa: BLE001 -- cleanup must not mask the real error
        logger.warning(f"[AutoTune] verl cleanup: could not collect Ray resources: {e!r}")
        return

    killed = 0
    for actor in actors:
        try:
            ray.kill(actor)
            killed += 1
        except Exception:  # noqa: BLE001 -- already dead
            pass
    removed = 0
    for pg in pgs:
        try:
            ray.util.remove_placement_group(pg)
            removed += 1
        except Exception:  # noqa: BLE001 -- already removed
            pass

    if killed or removed:
        time.sleep(2)  # let Ray reclaim GPU / shared-memory resources before the next trial
    logger.info(f"[AutoTune] verl cleanup: killed {killed} actor(s), removed {removed} placement group(s)")
