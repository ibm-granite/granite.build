"""Pins autotune/trainers/_verl_cleanup.py: find every Ray resource a verl trainer holds."""

from autotune.trainers._verl_cleanup import collect_ray_resources


class Actor:
    pass


class PG:
    pass


class WorkerGroup:
    def __init__(self, n):
        self._workers = [Actor() for _ in range(n)]


class Pool:
    def __init__(self):
        self.pgs = [PG(), PG()]


def _collect(root, **kw):
    return collect_ray_resources(
        root, is_actor=lambda o: isinstance(o, Actor), is_placement_group=lambda o: isinstance(o, PG), **kw
    )


def _trainer():
    """A stand-in shaped like verl 0.9.1's RayPPOTrainer after init_workers()."""

    class Manager:
        pass

    t = Manager()
    t.actor_rollout_wg = WorkerGroup(2)
    t.ref_policy_wg = t.actor_rollout_wg  # fused ref: same group, must not double count
    t.critic_wg = WorkerGroup(2)
    t.resource_pool_manager = Manager()
    t.resource_pool_manager.resource_pool_dict = {"global_pool": Pool()}
    t.reward_loop_manager = Manager()
    t.reward_loop_manager.reward_loop_workers = [Actor()]
    t.async_rollout_manager = Manager()
    t.async_rollout_manager.agent_loop_workers = [Actor(), Actor()]
    t.llm_server_manager = Manager()
    t.llm_server_manager.replicas = [Manager()]
    t.llm_server_manager.replicas[0].servers = {"0": Actor()}
    t.config = {"lr": 1e-5, "name": "x"}
    return t


def test_finds_worker_groups_managers_and_placement_groups():
    actors, pgs = _collect(_trainer())
    # 2 actor/ref + 2 critic + 1 reward loop + 2 agent loop + 1 vLLM server
    assert len(actors) == 8
    assert len(pgs) == 2


def test_each_resource_listed_once():
    actors, _ = _collect(_trainer())
    assert len({id(a) for a in actors}) == len(actors)


def test_cycles_terminate():
    t = _trainer()
    t.self_ref = t
    t.critic_wg.owner = t
    actors, _ = _collect(t)
    assert len(actors) == 8


def test_depth_bound():
    class Box:
        pass

    root = cur = Box()
    for _ in range(12):
        cur.child = Box()
        cur = cur.child
    cur.actor = Actor()
    assert _collect(root, max_depth=8) == ([], [])
    assert len(_collect(root, max_depth=20)[0]) == 1
