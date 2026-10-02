"""When a failing job-status poll means the cluster is gone, and when it does not.

The rule used to be three failed polls, or ONE poll saying "does not exist". At a
30 s poll interval that is a ~90 s SSH blip, and on BlueVela it tore down healthy
training runs: gbserver ran ``sky.down`` and LSF recorded ``TERM_OWNER`` against a
job that was still training. Now the failures have to last a grace period, and on
LSF a direct ``bjobs`` check gets the last word until a hard ceiling.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment import skypilot as skypilot_mod
from gbserver.environment.skypilot import Skypilot
from gbserver.types.environmentconfig import EnvironmentConfig
from gbserver.types.errors import WorkloadFailedException


@pytest.fixture
def lsf_env():
    config = EnvironmentConfig(
        name="test-lsf", type="Skypilot", config={"default_cloud": "lsf"}
    )
    env = Skypilot(event_q=asyncio.Queue(), environment_config=config)
    env._cluster_names["l1"] = "gb-train-l1"
    env._job_ids["l1"] = 1
    env._ssh_hpc_launches.add("l1")
    return env


@pytest.fixture
def cloud_env():
    """A launch on a non-HPC cloud (AWS, Kubernetes): no SSH login node."""
    config = EnvironmentConfig(
        name="test-aws", type="Skypilot", config={"default_cloud": "aws"}
    )
    env = Skypilot(event_q=asyncio.Queue(), environment_config=config)
    env._cluster_names["l1"] = "gb-train-l1"
    env._job_ids["l1"] = 1
    return env


class _Clock:
    """A monotonic clock each poll advances by ``step`` seconds."""

    def __init__(self, step):
        self.now = 1000.0
        self.step = step

    def __call__(self):
        return self.now

    def tick(self):
        self.now += self.step


def _sky(outcomes, clock):
    """A fake ``sky`` whose job_status replays ``outcomes``: an Exception raises,
    anything else is returned as the job's status."""
    sky = MagicMock()
    succeeded = MagicMock()
    succeeded.is_terminal.return_value = True
    succeeded.__str__ = lambda self: "JobStatus.SUCCEEDED"
    failed = MagicMock()
    failed.is_terminal.return_value = True
    failed.__str__ = lambda self: "JobStatus.FAILED"
    sky.JobStatus.SUCCEEDED = succeeded
    sky.JobStatus.FAILED = failed
    queue = list(outcomes)

    def job_status(*_a, **_k):
        clock.tick()
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    sky.job_status.side_effect = job_status
    sky.get.side_effect = lambda status: {1: status}
    return sky, succeeded


async def _poll(env, sky, clock, probe=None, **monitor):
    with (
        patch.object(skypilot_mod, "sky", sky),
        patch.object(skypilot_mod, "HAS_SKYPILOT", True),
        patch.object(skypilot_mod.time, "monotonic", clock),
        patch.object(
            skypilot_mod, "_lsf_job_alive", probe or MagicMock(return_value=None)
        ),
        patch.object(env, "_download_and_parse_logs", MagicMock()),
    ):
        await env._poll_skypilot_job(launch_id="l1", poll_interval=0, **monitor)


SSH = RuntimeError("Command ... failed with return code 255.")
MISSING = RuntimeError("Cluster 'gb-train-l1' does not exist.")


class TestGracePeriod:
    @pytest.mark.asyncio
    async def test_one_does_not_exist_is_not_final(self, lsf_env):
        """The old rule failed the step on this first poll."""
        clock = _Clock(step=30)
        sky, ok = _sky([MISSING, MISSING, None], clock)
        sky.get.side_effect = lambda status: {1: ok}
        # Recovers on the third poll: the step succeeds, nothing is torn down.
        await _poll(lsf_env, sky, clock)
        assert sky.job_status.call_count == 3

    @pytest.mark.asyncio
    async def test_failures_inside_the_grace_do_not_fail_the_step(self, lsf_env):
        """Twenty failures 30 s apart are ten minutes: inside the 15-minute
        default, so the run the old rule killed at failure three survives."""
        clock = _Clock(step=30)
        sky, ok = _sky([SSH] * 20 + [None], clock)
        sky.get.side_effect = lambda status: {1: ok}
        probe = MagicMock(return_value=False)
        await _poll(lsf_env, sky, clock, probe=probe)
        assert sky.job_status.call_count == 21
        probe.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_lsf_cluster_is_gone_after_the_grace(self, lsf_env):
        """No LSF cluster recorded -> no probe -> FAILED once the grace is over."""
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 100, clock)
        with pytest.raises(WorkloadFailedException):
            await _poll(lsf_env, sky, clock, poll_failure_grace_seconds=300)
        # 300 s of failures at 60 s per poll: the step fails on about the sixth,
        # not the third.
        assert 5 <= sky.job_status.call_count <= 7

    @pytest.mark.asyncio
    async def test_at_least_three_failures_even_with_zero_grace(self, lsf_env):
        clock = _Clock(step=1)
        sky, _ = _sky([SSH] * 10, clock)
        with pytest.raises(WorkloadFailedException):
            await _poll(lsf_env, sky, clock, poll_failure_grace_seconds=0)
        assert sky.job_status.call_count == 3


class TestOtherClouds:
    """Off SLURM/LSF a lost cluster is usually a real preemption: the
    RetryHandler should see FAILED at once, not after the 15-minute grace."""

    @pytest.mark.asyncio
    async def test_does_not_exist_is_still_final(self, cloud_env):
        clock = _Clock(step=30)
        sky, _ = _sky([MISSING] * 10, clock)
        probe = MagicMock(return_value=True)
        with pytest.raises(WorkloadFailedException):
            await _poll(cloud_env, sky, clock, probe=probe)
        assert sky.job_status.call_count == 1
        probe.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_grace_by_default(self, cloud_env):
        """Three failures 30 s apart are 90 s: final, as before this change."""
        clock = _Clock(step=30)
        sky, _ = _sky([SSH] * 10, clock)
        with pytest.raises(WorkloadFailedException):
            await _poll(cloud_env, sky, clock)
        assert sky.job_status.call_count == 3

    @pytest.mark.asyncio
    async def test_an_explicit_grace_still_applies(self, cloud_env):
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 100, clock)
        with pytest.raises(WorkloadFailedException):
            await _poll(cloud_env, sky, clock, poll_failure_grace_seconds=300)
        assert 5 <= sky.job_status.call_count <= 7


class TestLsfProbe:
    @pytest.fixture(autouse=True)
    def _lsf(self, lsf_env):
        lsf_env._lsf_clusters["l1"] = "bluevela"

    @pytest.mark.asyncio
    async def test_lsf_saying_gone_is_final_right_after_the_grace(self, lsf_env):
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 100, clock)
        probe = MagicMock(return_value=False)
        with pytest.raises(WorkloadFailedException):
            await _poll(
                lsf_env, sky, clock, probe=probe, poll_failure_grace_seconds=300
            )
        probe.assert_called_once_with("bluevela", "gb-train-l1")

    @pytest.mark.asyncio
    async def test_lsf_saying_alive_keeps_the_cluster_until_the_ceiling(self, lsf_env):
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 500, clock)
        probe = MagicMock(return_value=True)
        with pytest.raises(WorkloadFailedException):
            await _poll(
                lsf_env,
                sky,
                clock,
                probe=probe,
                poll_failure_grace_seconds=300,
                poll_failure_max_seconds=3600,
            )
        # Held for the whole hour, not the five-minute grace...
        assert 59 <= sky.job_status.call_count <= 61
        # ...and bjobs was asked at most every five minutes, not every poll.
        assert 10 <= probe.call_count <= 12

    @pytest.mark.asyncio
    async def test_lsf_unreachable_is_treated_like_alive(self, lsf_env):
        """SSH down to the login node too: no evidence the job is dead."""
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 500, clock)
        probe = MagicMock(return_value=None)
        with pytest.raises(WorkloadFailedException):
            await _poll(
                lsf_env,
                sky,
                clock,
                probe=probe,
                poll_failure_grace_seconds=300,
                poll_failure_max_seconds=1800,
            )
        assert 29 <= sky.job_status.call_count <= 31

    @pytest.mark.asyncio
    async def test_recovery_resets_the_clock(self, lsf_env):
        """Two ten-minute outages separated by one good poll never add up to the
        15-minute grace, so LSF is never even asked."""
        clock = _Clock(step=60)
        running = MagicMock()
        running.is_terminal.return_value = False
        running.__str__ = lambda self: "JobStatus.RUNNING"
        sky, ok = _sky([SSH] * 10 + [running] + [SSH] * 10 + [None], clock)
        sky.get.side_effect = lambda status: {1: status if status is running else ok}
        probe = MagicMock(return_value=False)
        await _poll(lsf_env, sky, clock, probe=probe)
        assert sky.job_status.call_count == 22
        probe.assert_not_called()


class TestLsfJobAlive:
    """The probe itself, against a fake LsfClient."""

    def _run(self, states=None, error=None):
        client = MagicMock()
        if error is not None:
            client.get_jobs_state_by_name.side_effect = error
        else:
            client.get_jobs_state_by_name.return_value = states
        ssh_cfg = MagicMock()
        ssh_cfg.lookup.return_value = {"hostname": "login4", "user": "u"}
        with (
            patch("sky.adaptors.lsf.LsfClient", return_value=client),
            patch("sky.provision.lsf.utils.get_lsf_ssh_config", return_value=ssh_cfg),
        ):
            result = skypilot_mod._lsf_job_alive("bluevela", "gb-train-l1")
        return result, client

    def test_running_job_is_alive(self):
        result, client = self._run(["RUN"])
        assert result is True
        # SkyPilot's LSF job name is <cluster>-<user hash>.
        client.get_jobs_state_by_name.assert_called_once_with("gb-train-l1-*")

    def test_unknown_host_state_is_alive(self):
        """UNKWN is LSF losing contact with the execution host -- the network
        trouble the grace rides out -- not the job ending."""
        assert self._run(["UNKWN"])[0] is True

    def test_provisioning_job_is_alive(self):
        assert self._run(["PROV"])[0] is True

    def test_no_job_is_gone(self):
        assert self._run([])[0] is False

    def test_finished_job_is_gone(self):
        assert self._run(["EXIT"])[0] is False

    def test_ssh_failure_is_unknown(self):
        assert self._run(error=RuntimeError("return code 255"))[0] is None
