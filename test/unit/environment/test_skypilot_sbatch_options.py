"""Per-step SLURM ``sbatch_options`` passthrough for SkyPilot launches.

Covers the layered resolver (``Skypilot._resolve_sbatch_options``) and the
end-to-end merge that lands the directives in
``sky.Resources(_cluster_config_overrides={"slurm": {"sbatch_options": ...}})``,
including the SLURM-only gating (a no-op WARNING on other clouds) and
coexistence with a ``docker`` override.
"""

import logging

import pytest

# Shared launch-mock scaffolding (see test_skypilot_slurm.py, which uses the
# same helpers) lives in libgbtest so it doesn't drift across test files.
from libgbtest.environments.skypilot_mocks import _launch_and_get_resources, _make_env

from gbserver.environment.skypilot import Skypilot


async def _overrides_for(env: Skypilot, launch_id: str, **launch_kwargs):
    """Launch under a mocked ``sky`` and return just the
    ``_cluster_config_overrides`` kwarg passed to ``sky.Resources`` (``None``
    when no overrides applied).

    :param env: the Skypilot environment under test.
    :param launch_id: unique id for this launch (arms the ready event).
    :param launch_kwargs: forwarded to ``launch_skypilot``.
    :returns: the ``_cluster_config_overrides`` value, or ``None``.
    """
    kwargs = await _launch_and_get_resources(env, launch_id, **launch_kwargs)
    return kwargs["_cluster_config_overrides"]


# ---------------------------------------------------------------------------
# Pure resolver: _resolve_sbatch_options (env < step < build, merged per key)
# ---------------------------------------------------------------------------
class TestResolveSbatchOptions:
    def test_empty_when_no_layer_sets_it(self):
        env = _make_env({"default_cloud": "slurm"})
        assert env._resolve_sbatch_options({}, {}) == {}

    def test_env_default_used_when_step_and_build_unset(self):
        env = _make_env({"default_cloud": "slurm", "sbatch_options": {"qos": "normal"}})
        assert env._resolve_sbatch_options({}, {}) == {"qos": "normal"}

    def test_per_key_merge_across_all_three_layers(self):
        env = _make_env({"default_cloud": "slurm", "sbatch_options": {"qos": "normal"}})
        merged = env._resolve_sbatch_options(
            {"sbatch_options": {"time": 60, "qos": "high"}},  # step.yaml
            {"launcher_config": {"sbatch_options": {"time": 30}}},  # build.yaml
        )
        # build wins on `time`; step's `qos` survives (build didn't set it);
        # env's `qos` is overridden by the step.
        assert merged == {"time": 30, "qos": "high"}

    def test_null_keys_coerced_to_empty(self):
        # A bare (present-but-null) YAML key parses to None; every layer must
        # tolerate it rather than crash the merge with a None operand.
        env = _make_env({"default_cloud": "slurm", "sbatch_options": None})
        assert (
            env._resolve_sbatch_options(
                {"sbatch_options": None},  # bare `sbatch_options:` in step.yaml
                {"launcher_config": None},  # bare `launcher_config:` in build.yaml
            )
            == {}
        )


# ---------------------------------------------------------------------------
# End-to-end merge into sky.Resources(_cluster_config_overrides=...)
# ---------------------------------------------------------------------------
class TestSbatchOptionsLaunch:
    @pytest.mark.asyncio
    async def test_step_level_reaches_sbatch_options(self):
        # gbserver forwards the map verbatim; keys use documented pass-through
        # directives (SkyPilot itself drops protected keys like `gres` — that
        # happens downstream, not here). See skypilot-slurm.md#sbatch_options.
        env = _make_env({"default_cloud": "slurm"})
        overrides = await _overrides_for(
            env,
            "sb-step",
            launcher_config={
                "run": "hostname",
                "resources": {},
                "sbatch_options": {"time": 240, "qos": "high"},
            },
            config={},
        )
        assert overrides["slurm"]["sbatch_options"] == {"time": 240, "qos": "high"}

    @pytest.mark.asyncio
    async def test_env_default_applies_when_step_unset(self):
        env = _make_env({"default_cloud": "slurm", "sbatch_options": {"qos": "normal"}})
        overrides = await _overrides_for(
            env,
            "sb-env",
            launcher_config={"run": "hostname", "resources": {}},
            config={},
        )
        assert overrides["slurm"]["sbatch_options"] == {"qos": "normal"}

    @pytest.mark.asyncio
    async def test_build_beats_step_beats_env_per_key(self):
        env = _make_env({"default_cloud": "slurm", "sbatch_options": {"qos": "normal"}})
        overrides = await _overrides_for(
            env,
            "sb-prec",
            launcher_config={
                "run": "hostname",
                "resources": {},
                "sbatch_options": {"time": 60, "qos": "high"},
            },
            config={"launcher_config": {"sbatch_options": {"time": 30}}},
        )
        assert overrides["slurm"]["sbatch_options"] == {"time": 30, "qos": "high"}

    @pytest.mark.asyncio
    async def test_merges_with_docker_without_clobbering(self):
        env = _make_env({"default_cloud": "slurm"})
        overrides = await _overrides_for(
            env,
            "sb-docker",
            launcher_config={
                "run": "hostname",
                "resources": {},
                "sbatch_options": {"time": 60},
                "docker": {"run_options": ["--shm-size=1g"]},
            },
            config={},
        )
        assert overrides["slurm"]["sbatch_options"] == {"time": 60}
        assert overrides["docker"] == {"run_options": ["--shm-size=1g"]}

    @pytest.mark.asyncio
    async def test_non_slurm_cloud_is_noop_and_warns(self, caplog):
        # Explicitly set on the (non-SLURM) step -> WARNING, so the author knows
        # the field they just wrote is ignored.
        env = _make_env({"default_cloud": "kubernetes"})
        with caplog.at_level(logging.WARNING):
            overrides = await _overrides_for(
                env,
                "sb-k8s",
                launcher_config={
                    "run": "hostname",
                    "resources": {},
                    "sbatch_options": {"time": 60},
                },
                config={},
            )
        # No per-task channel off SLURM: the field is a documented no-op.
        assert overrides is None
        assert any("not SLURM" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_non_slurm_inherited_env_default_does_not_warn(self, caplog):
        # An env-wide default on a non-SLURM step the author never touched is a
        # passive no-op: DEBUG, not WARNING (otherwise every k8s/aws step of a
        # SLURM-default env would nag).
        env = _make_env(
            {"default_cloud": "kubernetes", "sbatch_options": {"qos": "normal"}}
        )
        with caplog.at_level(logging.WARNING):
            overrides = await _overrides_for(
                env,
                "sb-inherit",
                launcher_config={"run": "hostname", "resources": {}},
                config={},
            )
        assert overrides is None
        assert not any("not SLURM" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_overrides_when_nothing_set(self):
        env = _make_env({"default_cloud": "slurm"})
        overrides = await _overrides_for(
            env,
            "sb-none",
            launcher_config={"run": "hostname", "resources": {}},
            config={},
        )
        # Neither docker nor sbatch_options -> no cluster_config_overrides at all.
        assert overrides is None


# ---------------------------------------------------------------------------
# Interaction with the build-tracking --comment (from run_metadata). The
# launcher applies the comment via apply_slurm_comment_override BEFORE the
# per-step sbatch_options deep-merge, so the two must land together under
# slurm.sbatch_options without clobbering. These guard that merge order.
# ---------------------------------------------------------------------------
class TestSbatchOptionsWithMetadataComment:
    @pytest.mark.asyncio
    async def test_comment_and_sbatch_options_coexist(self):
        # Build-tracking comment (from run_metadata) and per-step sbatch
        # directives coexist under slurm.sbatch_options; neither is dropped.
        env = _make_env({"default_cloud": "slurm"})
        overrides = await _overrides_for(
            env,
            "sb-comment-coexist",
            launcher_config={
                "run": "hostname",
                "resources": {},
                "sbatch_options": {"time": 60, "qos": "high"},
            },
            config={},
            run_metadata={"build_id": "b1", "target_name": "My Target"},
        )
        assert overrides["slurm"]["sbatch_options"] == {
            "time": 60,
            "qos": "high",
            "comment": "build_id=b1;target=My_Target",
        }

    @pytest.mark.asyncio
    async def test_explicit_sbatch_comment_overrides_metadata(self):
        # An explicit `comment` in sbatch_options is merged last, so it wins
        # over the build-tracking comment (documented precedence).
        env = _make_env({"default_cloud": "slurm"})
        overrides = await _overrides_for(
            env,
            "sb-comment-override",
            launcher_config={
                "run": "hostname",
                "resources": {},
                "sbatch_options": {"comment": "operator-note"},
            },
            config={},
            run_metadata={"build_id": "b1"},
        )
        assert overrides["slurm"]["sbatch_options"]["comment"] == "operator-note"
