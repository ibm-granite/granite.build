"""Wiring tests for autotune.trainers.verl_overrides.

They build fm-tune's verl override dict from a trial config shaped the way
AutotuneOptimizer.setup_pipeline builds it (sampled values at the top level,
fixed sections nested), using the shipped autotune.yaml, and check that each
value lands on the verl key verl 0.7.1 reads. The module under test imports
no torch/verl/hydra/omegaconf, so these run on any platform.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from autotune.config import AutotuneConfig
from autotune.trainers.verl_overrides import build_verl_overrides

_SHIPPED_YAML = Path(__file__).resolve().parent.parent / "autotune" / "configs" / "autotune.yaml"


@pytest.fixture(scope="module")
def shipped():
    cfg = AutotuneConfig()
    cfg.load(str(_SHIPPED_YAML))
    return cfg


def _hyperparams(shipped, algo):
    return shipped.get_tuner_rl_config_dict(algo)["hyperparams"]


def _build(shipped, algo, sampled=None, drop=(), rl_overrides=None):
    """Build the overrides for one trial of ``algo``.

    ``sampled`` replaces search-space defaults (or adds extra tuned keys),
    ``drop`` removes params from the search space (i.e. "not swept"), and
    ``rl_overrides`` edits the fixed training_rl_config section.
    """
    hp = _hyperparams(shipped, algo)
    trial = {k: spec["default"] for k, spec in hp.items() if k not in drop}
    trial.update(sampled or {})

    training_config = deepcopy(shipped.get_training_config_dict())
    training_config.update({"model_name_or_path": "dummy/model", "output_dir": "/tmp/out", "rl_algorithm": algo})
    training_rl_config = deepcopy(shipped.get_training_rl_config_dict())
    training_rl_config.update(rl_overrides or {})

    # Same as driver_multi_verl.train_driver_multi_gpu: every sampled param.
    train_kwargs = dict(trial)
    train_kwargs["num_train_epochs"] = 1

    return build_verl_overrides(
        training_config=training_config,
        training_rl_config=training_rl_config,
        train_kwargs=train_kwargs,
        train_file="train.parquet",
        eval_file="val.parquet",
        num_workers=2,
        rl_algorithm=algo,
    )


def _get(d, dotted):
    for key in dotted.split("."):
        d = d[key]
    return d


@pytest.mark.parametrize("algo", ["ppo", "grpo", "dapo"])
def test_builds_every_top_level_section(shipped, algo):
    overrides = _build(shipped, algo)
    assert set(overrides) == {"data", "actor_rollout_ref", "critic", "reward", "algorithm", "trainer"}
    assert (
        _get(overrides, "actor_rollout_ref.actor.optim.lr") == _hyperparams(shipped, algo)["learning_rate"]["default"]
    )


# Every param in a shipped online-RL search space must either reach the verl key
# listed here, or be listed in PENDING_DECISION. A new search-space param that
# nobody wires fails test_every_swept_param_is_accounted_for.
WIRED = {
    "learning_rate": "actor_rollout_ref.actor.optim.lr",
    "per_device_train_batch_size": "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "clip_range": "actor_rollout_ref.actor.clip_ratio_low",
    "entropy_coef": "actor_rollout_ref.actor.entropy_coeff",
    "rollout_n": "actor_rollout_ref.rollout.n",
    "rollout_temperature": "actor_rollout_ref.rollout.temperature",
}

# Swept but not yet applied by verl, pending stakeholder decisions:
#   kl_coef — whether KL is on by default, and for PPO whether it goes in the
#     reward or the actor loss. It reaches algorithm.kl_ctrl.kl_coef, which
#     verl reads only when algorithm.use_kl_in_reward is on (it is off).
#   gradient_accumulation_steps — make it real or drop it from the search space.
PENDING_DECISION = {"kl_coef", "gradient_accumulation_steps"}

ONLINE_RL = ["ppo", "grpo", "dapo"]


def _non_default(spec):
    for v in spec["values"]:
        if v != spec["default"]:
            return v
    raise AssertionError(f"no non-default value in {spec['values']}")


@pytest.mark.parametrize("algo", ONLINE_RL)
def test_every_swept_param_is_accounted_for(shipped, algo):
    unaccounted = set(_hyperparams(shipped, algo)) - set(WIRED) - PENDING_DECISION
    assert not unaccounted, f"{algo}: swept but not wired to verl: {sorted(unaccounted)}"


@pytest.mark.parametrize("algo", ONLINE_RL)
def test_every_wired_param_reaches_its_verl_key(shipped, algo):
    hp = _hyperparams(shipped, algo)
    sampled = {k: _non_default(spec) for k, spec in hp.items()}
    overrides = _build(shipped, algo, sampled)
    for param in sorted(set(hp) & set(WIRED)):
        assert _get(overrides, WIRED[param]) == sampled[param], f"{algo}: {param} -> {WIRED[param]}"


@pytest.mark.parametrize("algo, kl_coef", [("grpo", 0.1), ("dapo", 0.0), ("ppo", 0.0)])
def test_kl_coef_reaches_kl_ctrl_and_keeps_zero(shipped, algo, kl_coef):
    overrides = _build(shipped, algo, {"kl_coef": kl_coef})
    assert _get(overrides, "algorithm.kl_ctrl.kl_coef") == kl_coef


@pytest.mark.parametrize("algo", ["grpo", "dapo"])
def test_unswept_rollout_values_come_from_training_rl_config(shipped, algo):
    overrides = _build(
        shipped,
        algo,
        drop=("rollout_n", "rollout_temperature"),
        rl_overrides={"rollout_n": 7, "rollout_temperature": 0.9},
    )
    assert _get(overrides, "actor_rollout_ref.rollout.n") == 7
    assert _get(overrides, "actor_rollout_ref.rollout.temperature") == 0.9


def test_ppo_rollout_n_is_always_one(shipped):
    overrides = _build(shipped, "ppo", {"rollout_n": 8}, rl_overrides={"rollout_n": 7})
    assert _get(overrides, "actor_rollout_ref.rollout.n") == 1


@pytest.mark.parametrize(
    "sampled, drop, expected",
    [
        ({"entropy_coef": 0.1, "entropy_coeff": 0.05}, (), 0.1),  # search-space name wins
        ({"entropy_coeff": 0.05}, ("entropy_coef",), 0.05),  # legacy name still works alone
        ({"entropy_coef": 0.0, "entropy_coeff": 0.05}, (), 0.0),  # explicit 0.0 is kept
    ],
)
def test_ppo_entropy_coef(shipped, sampled, drop, expected):
    overrides = _build(shipped, "ppo", sampled, drop=drop)
    assert _get(overrides, "actor_rollout_ref.actor.entropy_coeff") == expected


def test_dapo_overlong_settings_go_where_the_dapo_reward_manager_reads_them(shipped):
    overrides = _build(shipped, "dapo", rl_overrides={"overlong_buffer_len": 64, "overlong_penalty_factor": 2.0})
    reward_kwargs = overrides["reward"]["reward_kwargs"]
    assert reward_kwargs["max_resp_len"] == shipped.get_training_rl_config_dict()["max_response_length"]
    assert reward_kwargs["overlong_buffer_cfg"] == {"enable": True, "len": 64, "penalty_factor": 2.0, "log": False}
    assert "overlong_buffer_cfg" not in overrides["algorithm"]


def test_dapo_shipped_overlong_defaults(shipped):
    cfg = _build(shipped, "dapo")["reward"]["reward_kwargs"]["overlong_buffer_cfg"]
    assert (cfg["len"], cfg["penalty_factor"]) == (128, 1.0)


# The overlong penalty is added to the score that HPO ranks DAPO trials by
# (critic/score/mean -> loss), so sweeping it would rank trials on differently
# shaped rewards and favour the weakest penalty. Keep it in training_rl_config.
@pytest.mark.parametrize("yaml_path", sorted(_SHIPPED_YAML.parent.glob("*.yaml")), ids=lambda p: p.name)
def test_dapo_overlong_penalty_is_not_swept(yaml_path):
    cfg = AutotuneConfig()
    cfg.load(str(yaml_path))
    swept = set(cfg.get_tuner_rl_config_dict("dapo")["hyperparams"])
    assert not swept & {"overlong_buffer_len", "overlong_penalty_factor"}
    assert {"overlong_buffer_len", "overlong_penalty_factor"} <= set(cfg.get_training_rl_config_dict())


def test_dapo_overlong_buffer_longer_than_max_response_raises(shipped):
    with pytest.raises(ValueError, match=r"overlong_buffer_len \(512\).*max_response_length \(256\)"):
        _build(shipped, "dapo", rl_overrides={"overlong_buffer_len": 512, "max_response_length": 256})


@pytest.mark.parametrize("algo", ["ppo", "grpo"])
def test_non_dapo_has_no_reward_kwargs(shipped, algo):
    assert "reward_kwargs" not in _build(shipped, algo)["reward"]


def test_ppo_clip_range_sets_every_clip_bound_verl_reads(shipped):
    # verl 0.7.1's policy loss reads clip_ratio_low/high (default 0.2) before clip_ratio.
    actor = _build(shipped, "ppo", {"clip_range": 0.3})["actor_rollout_ref"]["actor"]
    assert (actor["clip_ratio"], actor["clip_ratio_low"], actor["clip_ratio_high"]) == (0.3, 0.3, 0.3)
