"""When a failing job-status poll means the cluster is gone, and when it does not.

The rule used to be three failed polls, or ONE poll saying "does not exist". At a
30 s poll interval that is a ~90 s SSH blip, and on BlueVela it tore down healthy
training runs: gbserver ran ``sky.down`` and LSF recorded ``TERM_OWNER`` against a
job that was still training. Now the failures have to last a grace period, and on
LSF a direct ``bjobs`` check gets the last word until a hard ceiling.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.environment import skypilot as skypilot_mod
from gbserver.environment.skypilot import Skypilot
from gbserver.types.buildevent import BuildEventType, EntityRunMetadata
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
def slurm_env():
    config = EnvironmentConfig(
        name="test-slurm", type="Skypilot", config={"default_cloud": "slurm"}
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


async def _poll(env, sky, clock, probe=None, kill=None, **monitor):
    # Only the poll-failure tracker's clock is faked: asyncio's loop.time()
    # still reads the real time.monotonic, so timeouts in the loop behave.
    with (
        patch.object(skypilot_mod, "sky", sky),
        patch.object(skypilot_mod, "HAS_SKYPILOT", True),
        patch.object(skypilot_mod, "_poll_failure_clock", clock),
        patch.object(
            skypilot_mod, "_lsf_job_alive", probe or MagicMock(return_value=None)
        ),
        patch.object(
            skypilot_mod, "_lsf_cancel_job", kill or MagicMock(return_value=True)
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

    @pytest.mark.asyncio
    async def test_slurm_does_not_exist_is_not_final(self, slurm_env):
        clock = _Clock(step=30)
        sky, ok = _sky([MISSING, MISSING, None], clock)
        sky.get.side_effect = lambda status: {1: ok}
        await _poll(slurm_env, sky, clock)
        assert sky.job_status.call_count == 3

    @pytest.mark.asyncio
    async def test_slurm_is_gone_after_the_grace_without_asking_lsf(self, slurm_env):
        clock = _Clock(step=60)
        sky, _ = _sky([MISSING] * 100, clock)
        probe = MagicMock(return_value=True)
        with pytest.raises(WorkloadFailedException):
            await _poll(slurm_env, sky, clock, probe=probe)
        # The 900 s default grace at 60 s per poll.
        assert 15 <= sky.job_status.call_count <= 17
        probe.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", ["nan", "inf", float("nan"), "-5", "soon", True])
    async def test_a_bad_grace_falls_back_to_the_default(self, slurm_env, bad):
        """``nan`` would make every ``>=`` test False: a monitor that never
        gives up, with nothing else on SLURM to end it."""
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 100, clock)
        with pytest.raises(WorkloadFailedException):
            await _poll(slurm_env, sky, clock, poll_failure_grace_seconds=bad)
        assert 15 <= sky.job_status.call_count <= 17


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


# The name SkyPilot submitted the job under, as the launch records it.
JOB = "gb-train-l1-0a1b2c3d"


class TestLsfProbe:
    @pytest.fixture(autouse=True)
    def _lsf(self, lsf_env):
        lsf_env._lsf_clusters["l1"] = "bluevela"
        lsf_env._lsf_job_names["l1"] = JOB

    @pytest.mark.asyncio
    async def test_lsf_saying_gone_is_final_right_after_the_grace(self, lsf_env):
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 100, clock)
        probe = MagicMock(return_value=False)
        with pytest.raises(WorkloadFailedException):
            await _poll(
                lsf_env, sky, clock, probe=probe, poll_failure_grace_seconds=300
            )
        probe.assert_called_once_with("bluevela", JOB)

    @pytest.mark.asyncio
    async def test_lsf_saying_alive_keeps_the_cluster_until_the_ceiling(self, lsf_env):
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 500, clock)
        probe = MagicMock(return_value=True)
        kill = MagicMock(return_value=True)
        with pytest.raises(WorkloadFailedException):
            await _poll(
                lsf_env,
                sky,
                clock,
                probe=probe,
                kill=kill,
                poll_failure_grace_seconds=300,
                poll_failure_max_seconds=3600,
            )
        # Held for the whole hour, not the five-minute grace...
        assert 59 <= sky.job_status.call_count <= 61
        # ...and bjobs was asked at most every five minutes, not every poll.
        assert 10 <= probe.call_count <= 12
        # At the ceiling the job is bkill-ed: sky.down cannot reach a cluster
        # SkyPilot has no record of, so the retry would otherwise run beside it.
        kill.assert_called_once_with("bluevela", JOB)

    @pytest.mark.asyncio
    async def test_no_bkill_before_the_ceiling(self, lsf_env):
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 100, clock)
        kill = MagicMock(return_value=True)
        with pytest.raises(WorkloadFailedException):
            await _poll(
                lsf_env,
                sky,
                clock,
                probe=MagicMock(return_value=False),
                kill=kill,
                poll_failure_grace_seconds=300,
            )
        kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_bkill_fails_without_handing_off_to_the_retry(self, lsf_env):
        """With a RetryHandler deferring, a FAILED status would wait on
        stop_event for the handler to start a retry -- a second allocation
        next to the job LSF would not kill. It must raise instead."""
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 500, clock)
        with pytest.raises(WorkloadFailedException, match="without a retry"):
            await asyncio.wait_for(
                _poll(
                    lsf_env,
                    sky,
                    clock,
                    probe=MagicMock(return_value=True),
                    kill=MagicMock(return_value=False),
                    poll_failure_grace_seconds=300,
                    poll_failure_max_seconds=1800,
                    defer_terminal_failure=True,
                ),
                timeout=10,
            )

    @pytest.mark.asyncio
    async def test_failed_bkill_says_why_and_stops_the_log_stream(self, lsf_env):
        """The no-retry raise leaves through the same door as a terminal
        status: the live log stream is stopped, and a MESSAGE says what to do.
        No FAILED WORKLOAD_STATUS event, which is what a RetryHandler acts on."""
        clock = _Clock(step=60)
        running = MagicMock()
        running.is_terminal.return_value = False
        running.__str__ = lambda self: "JobStatus.RUNNING"
        sky, _ = _sky([running] + [SSH] * 500, clock)
        stream_task = asyncio.create_task(asyncio.sleep(60))
        event_q = asyncio.Queue()
        with (
            patch.object(
                lsf_env,
                "_start_log_stream_task",
                MagicMock(return_value=(stream_task, MagicMock())),
            ),
            pytest.raises(WorkloadFailedException, match="without a retry"),
        ):
            await _poll(
                lsf_env,
                sky,
                clock,
                probe=MagicMock(return_value=True),
                kill=MagicMock(return_value=False),
                poll_failure_grace_seconds=300,
                poll_failure_max_seconds=1800,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="b1"),
                event_configs=[
                    {"event_type": "MESSAGE_EVENT", "line_regex": "never matches"}
                ],
                log_retrieval={"mode": "stream"},
            )
        assert stream_task.cancelled()
        events = [event_q.get_nowait() for _ in range(event_q.qsize())]
        assert not [e for e in events if e.type is BuildEventType.WORKLOAD_STATUS_EVENT]
        assert "bkill it by hand" in events[-1].payload.msg

    @pytest.mark.asyncio
    async def test_failed_bkill_never_reaches_the_retry_handler(self, lsf_env):
        """End to end through monitor_skypilot_monitor and a real
        RetryHandler: the step fails, and retry_workload is never called."""
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 500, clock)
        lsf_env._release_monitors("l1")
        event_q = asyncio.Queue()
        retry = AsyncMock()
        with (
            patch.object(skypilot_mod, "sky", sky),
            patch.object(skypilot_mod, "HAS_SKYPILOT", True),
            patch.object(skypilot_mod, "_poll_failure_clock", clock),
            patch.object(skypilot_mod, "_lsf_job_alive", MagicMock(return_value=True)),
            patch.object(
                skypilot_mod, "_lsf_cancel_job", MagicMock(return_value=False)
            ),
            patch.object(lsf_env, "retry_workload", retry),
            pytest.raises(WorkloadFailedException, match="without a retry"),
        ):
            await asyncio.wait_for(
                lsf_env.monitor_skypilot_monitor(
                    launch_id="l1",
                    event_q=event_q,
                    entityrun_metadata=EntityRunMetadata(build_id="b1"),
                    poll_interval=0,
                    poll_failure_grace_seconds=300,
                    poll_failure_max_seconds=1800,
                ),
                timeout=10,
            )
        assert lsf_env._get_retry_strategies(), "no RetryHandler was created"
        retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_ceiling_below_the_grace_is_the_grace(self, lsf_env):
        """A ceiling set under the grace cannot cut the grace short."""
        clock = _Clock(step=60)
        sky, _ = _sky([SSH] * 100, clock)
        probe = MagicMock(return_value=True)
        kill = MagicMock(return_value=True)
        with pytest.raises(WorkloadFailedException):
            await _poll(
                lsf_env,
                sky,
                clock,
                probe=probe,
                kill=kill,
                poll_failure_grace_seconds=600,
                poll_failure_max_seconds=60,
            )
        assert 10 <= sky.job_status.call_count <= 12
        probe.assert_not_called()
        kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_hung_bjobs_reads_as_unknown(self, lsf_env):
        """bjobs can retry forever against an overloaded scheduler; the poll
        loop must not wait on it."""
        clock = _Clock(step=60)
        sky, ok = _sky([SSH] * 6 + [None], clock)
        sky.get.side_effect = lambda status: {1: ok}
        release = threading.Event()

        def hung(*_a):
            release.wait(5)
            return False

        try:
            with patch.object(skypilot_mod, "_LSF_COMMAND_TIMEOUT_SECONDS", 0.05):
                await asyncio.wait_for(
                    _poll(
                        lsf_env,
                        sky,
                        clock,
                        probe=hung,
                        poll_failure_grace_seconds=300,
                    ),
                    timeout=5,
                )
        finally:
            release.set()
        # "unknown" keeps the cluster, so the poll recovered and succeeded.
        assert sky.job_status.call_count == 7

    @pytest.mark.asyncio
    async def test_a_hung_login_node_does_not_starve_another_cluster(self):
        """Each timed-out call keeps its thread. If every LSF cluster shared one
        pool, a login node that hangs every call would fill it, and a healthy
        cluster's `bjobs` -- and its bkill at the ceiling -- would time out in
        the queue."""
        release = threading.Event()

        def hung(_cluster):
            release.wait(5)

        try:
            with (
                patch.object(skypilot_mod, "_LSF_COMMAND_TIMEOUT_SECONDS", 0.05),
                patch.dict(skypilot_mod._LSF_EXECUTORS, clear=True),
            ):
                for _ in range(8):  # more than one pool's worth of threads
                    with pytest.raises(asyncio.TimeoutError):
                        await skypilot_mod._run_lsf_call(hung, "hung-cluster")
                result = await skypilot_mod._run_lsf_call(
                    lambda cluster, name: (cluster, name), "healthy", JOB
                )
        finally:
            release.set()
        assert result == ("healthy", JOB)

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

    def _run(self, states=None, error=None, func=None):
        client = MagicMock()
        if error is not None:
            client.get_jobs_state_by_name.side_effect = error
            client.cancel_jobs_by_name.side_effect = error
        else:
            client.get_jobs_state_by_name.return_value = states
        ssh_cfg = MagicMock()
        ssh_cfg.lookup.return_value = {"hostname": "login4", "user": "u"}
        with (
            patch("sky.adaptors.lsf.LsfClient", return_value=client),
            patch("sky.provision.lsf.utils.get_lsf_ssh_config", return_value=ssh_cfg),
        ):
            result = (func or skypilot_mod._lsf_job_alive)("bluevela", JOB)
        return result, client

    def test_running_job_is_alive(self):
        result, client = self._run(["RUN"])
        assert result is True
        client.get_jobs_state_by_name.assert_called_once_with(JOB)

    def test_unrecorded_job_name_falls_back_to_skypilots_rewriting(self):
        """Only when a launch recorded no name: bjobs must then look up the
        name SkyPilot's LSF provisioner passes to ``bsub -J``, its
        cluster_name_on_cloud, built from a display name that exercises the
        rewriting (case, ``_``, ``.``)."""
        pytest.importorskip("sky")
        from sky.utils import common_utils

        display = "gb-My_Build.v2-train-3168aa02-123"
        tracker = skypilot_mod._PollFailureTracker(
            display, grace=0, ceiling=0, ssh_hpc=True, lsf_cluster="bluevela"
        )
        assert tracker.lsf_job_name == (
            f"gb-my-build-v2-train-3168aa02-123-{common_utils.get_user_hash()}"
        )

    def test_recorded_job_name_wins(self):
        tracker = skypilot_mod._PollFailureTracker(
            "gb-train-l1",
            grace=0,
            ceiling=0,
            ssh_hpc=True,
            lsf_cluster="bluevela",
            lsf_job_name=JOB,
        )
        assert tracker.lsf_job_name == JOB

    def test_bkill_targets_the_same_job(self):
        result, client = self._run(func=skypilot_mod._lsf_cancel_job)
        assert result is True
        client.cancel_jobs_by_name.assert_called_once_with(JOB)

    def test_bkill_ssh_failure_is_false(self):
        result, _ = self._run(
            error=RuntimeError("return code 255"), func=skypilot_mod._lsf_cancel_job
        )
        assert result is False

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


@pytest.mark.asyncio
async def test_cleanup_forgets_the_hpc_registration(lsf_env):
    """A relaunch re-registers from its own infra; a stale entry would give a
    k8s relaunch the LSF grace and `bjobs` check."""
    lsf_env._lsf_clusters["l1"] = "bluevela"
    lsf_env._lsf_job_names["l1"] = JOB
    with (
        patch.object(skypilot_mod, "HAS_SKYPILOT", True),
        patch.object(skypilot_mod, "sky", MagicMock()),
        patch.object(lsf_env, "_teardown", MagicMock(side_effect=_async_none)),
    ):
        await lsf_env.cleanup_skypilot(launch_id="l1")
    assert "l1" not in lsf_env._lsf_clusters
    assert "l1" not in lsf_env._lsf_job_names
    assert "l1" not in lsf_env._ssh_hpc_launches


async def _async_none(*_a, **_k):
    return None
