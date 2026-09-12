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

"""Tests for the resilient SSH-tunnel establishment path (issue #368).

Covers ``Lsf._ensure_ssh_tunnel`` (reuse/rebuild/failover/budget) and the
``SshTunnel.is_healthy`` predicate it relies on.
"""

import asyncio
from typing import List, Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.environment.lsf import Lsf
from gbserver.utils.ssh_tunnel import SshTunnel, SshTunnelError


def _make_lsf(login_nodes: List[str]) -> Lsf:
    """Minimal Lsf instance with just the state ``_ensure_ssh_tunnel`` touches."""
    with patch.object(Lsf, "__init__", lambda self, **kw: None):
        lsf = Lsf.__new__(Lsf)
    lsf.use_ssh = True
    lsf.login_nodes = list(login_nodes)
    lsf.login_node_idx = 0
    lsf.unreachable_ssh_nodes = []
    lsf.node_search_lock = asyncio.Lock()
    lsf._tunnel_lock = asyncio.Lock()
    lsf._retired_tunnel_tasks = set()
    lsf._ssh_tunnel = None
    lsf._key_file_path = "/tmp/fake_key"
    lsf.username = "tester"
    lsf.ssh_host_key_verification = False
    lsf.ssh_port = 22
    lsf.ssh_max_sessions = 10
    lsf.ssh_timeout = 5
    # Short budget + backoff so a failing test fails fast rather than hanging.
    lsf.ssh_connect_budget_s = 3600
    lsf.ssh_connect_base_backoff_s = 1
    lsf.ssh_connect_max_backoff_s = 4
    # Per-attempt banner/login bounds threaded into the tunnel + probe.
    lsf.ssh_connect_timeout_s = 20
    lsf.ssh_login_timeout_s = 90
    lsf.ssh_keepalive_interval_s = 10
    lsf.ssh_keepalive_count_max = 3
    lsf.ssh_command_timeout_s = 120
    return lsf


def _healthy_tunnel(host: str = "node") -> MagicMock:
    """A stand-in SshTunnel that is alive and whose open()/echo succeed."""
    t = MagicMock(spec=SshTunnel)
    t.host = host
    t.is_healthy.return_value = True
    t.open = AsyncMock()
    t.run_remote = AsyncMock(return_value=(0, "tunnel-ready", ""))
    t.close = AsyncMock()
    t.close_when_idle = AsyncMock()
    return t


class TestEnsureSshTunnel:
    """Behavior of the idempotent, self-healing establish loop."""

    @pytest.mark.asyncio
    async def test_reuses_healthy_tunnel_without_rebuilding(self: Self) -> None:
        lsf = _make_lsf(["a", "b"])
        existing = _healthy_tunnel("a")
        lsf._ssh_tunnel = existing

        with (
            patch("gbserver.environment.lsf.SshTunnel") as tunnel_cls,
            patch.object(lsf, "_get_reachable_ssh_node", new=AsyncMock()) as get_node,
        ):
            result = await lsf._ensure_ssh_tunnel()

        assert result is existing
        tunnel_cls.assert_not_called()  # no rebuild
        get_node.assert_not_awaited()  # no node search
        existing.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rebuilds_when_existing_tunnel_unhealthy(self: Self) -> None:
        lsf = _make_lsf(["a", "b"])
        stale = _healthy_tunnel("a")
        stale.is_healthy.return_value = False  # dead connection
        lsf._ssh_tunnel = stale
        fresh = _healthy_tunnel("b")

        with (
            patch("gbserver.environment.lsf.SshTunnel", return_value=fresh),
            patch.object(
                lsf, "_get_reachable_ssh_node", new=AsyncMock(return_value="b")
            ),
        ):
            result = await lsf._ensure_ssh_tunnel()
            # Let the background retirement GC task run.
            await asyncio.sleep(0)

        assert result is fresh
        assert lsf._ssh_tunnel is fresh
        # Stale tunnel is retired (drained then closed), not force-closed.
        stale.close_when_idle.assert_awaited()
        stale.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fails_over_when_open_fails_then_succeeds(self: Self) -> None:
        """First candidate's open() raises; the next sweep must succeed."""
        lsf = _make_lsf(["a", "b"])
        bad = _healthy_tunnel("a")
        bad.open = AsyncMock(side_effect=SshTunnelError("connect refused"))
        good = _healthy_tunnel("b")
        built: List[MagicMock] = [bad, good]

        with (
            patch("gbserver.environment.lsf.SshTunnel", side_effect=built),
            patch.object(
                lsf, "_get_reachable_ssh_node", new=AsyncMock(side_effect=["a", "b"])
            ),
            patch("gbserver.environment.lsf.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            result = await lsf._ensure_ssh_tunnel()

        assert result is good
        bad.close.assert_awaited()  # half-open attempt torn down
        sleep.assert_awaited()  # backed off between sweeps

    @pytest.mark.asyncio
    async def test_fails_over_when_open_ok_but_command_fails(self: Self) -> None:
        """A node that connects but can't execute must be rejected (echo probe)."""
        lsf = _make_lsf(["a", "b"])
        cannot_exec = _healthy_tunnel("a")
        cannot_exec.run_remote = AsyncMock(side_effect=Exception("no exec"))
        good = _healthy_tunnel("b")

        with (
            patch(
                "gbserver.environment.lsf.SshTunnel", side_effect=[cannot_exec, good]
            ),
            patch.object(
                lsf, "_get_reachable_ssh_node", new=AsyncMock(side_effect=["a", "b"])
            ),
            patch("gbserver.environment.lsf.asyncio.sleep", new=AsyncMock()),
        ):
            result = await lsf._ensure_ssh_tunnel()

        assert result is good
        cannot_exec.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_raises_only_after_budget_exhausted(self: Self) -> None:
        """No reachable node ever: keep retrying until the budget elapses, then raise."""
        lsf = _make_lsf(["a"])
        lsf.ssh_connect_budget_s = 100

        # Simulated clock advances by 40s per backoff sleep, so the budget is
        # exhausted after a few sweeps without any real waiting.
        clock = {"t": 0.0}

        def fake_monotonic() -> float:
            return clock["t"]

        async def fake_sleep(_delay: float) -> None:
            clock["t"] += 40.0

        with (
            patch("gbserver.environment.lsf.SshTunnel"),
            patch.object(
                lsf,
                "_get_reachable_ssh_node",
                new=AsyncMock(side_effect=RuntimeError("no reachable node")),
            ),
            patch(
                "gbserver.environment.lsf.time.monotonic", side_effect=fake_monotonic
            ),
            patch("gbserver.environment.lsf.asyncio.sleep", side_effect=fake_sleep),
        ):
            with pytest.raises(SshTunnelError) as excinfo:
                await lsf._ensure_ssh_tunnel()

        # Original cause preserved for debugging.
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_clears_unreachable_nodes_each_sweep(self: Self) -> None:
        """A node marked unreachable on a prior sweep is retried on the next."""
        lsf = _make_lsf(["a", "b"])
        lsf.unreachable_ssh_nodes = ["a", "b"]  # stale from a previous outage
        good = _healthy_tunnel("a")

        with (
            patch("gbserver.environment.lsf.SshTunnel", return_value=good),
            patch.object(
                lsf, "_get_reachable_ssh_node", new=AsyncMock(return_value="a")
            ),
        ):
            result = await lsf._ensure_ssh_tunnel()

        assert result is good
        assert lsf.unreachable_ssh_nodes == []  # reset before the sweep

    @pytest.mark.asyncio
    async def test_open_ssh_tunnel_delegates_and_sets_tunnel(self: Self) -> None:
        """setup path (_open_ssh_tunnel) still populates self._ssh_tunnel."""
        lsf = _make_lsf(["a"])
        good = _healthy_tunnel("a")

        with (
            patch("gbserver.environment.lsf.SshTunnel", return_value=good),
            patch.object(
                lsf, "_get_reachable_ssh_node", new=AsyncMock(return_value="a")
            ),
        ):
            await lsf._open_ssh_tunnel(setup_id="setup-1")

        assert lsf._ssh_tunnel is good

    @pytest.mark.asyncio
    async def test_builds_tunnel_with_banner_login_bounds(self: Self) -> None:
        """The tunnel is constructed with the connect/login/keepalive bounds so a
        slow-banner or wedged login node fails over instead of hanging."""
        lsf = _make_lsf(["a"])
        good = _healthy_tunnel("a")

        with (
            patch("gbserver.environment.lsf.SshTunnel", return_value=good) as cls,
            patch.object(
                lsf, "_get_reachable_ssh_node", new=AsyncMock(return_value="a")
            ),
        ):
            await lsf._ensure_ssh_tunnel()

        _, kwargs = cls.call_args
        assert kwargs["connect_timeout"] == lsf.ssh_connect_timeout_s
        assert kwargs["login_timeout"] == lsf.ssh_login_timeout_s
        assert kwargs["keepalive_interval"] == lsf.ssh_keepalive_interval_s
        assert kwargs["keepalive_count_max"] == lsf.ssh_keepalive_count_max
        # command_timeout bounds the slow server-side session/exec setup that is
        # the actual bluevela bottleneck.
        assert kwargs["command_timeout"] == lsf.ssh_command_timeout_s

    @pytest.mark.asyncio
    async def test_aborts_when_key_file_removed_by_teardown(self: Self) -> None:
        """A teardown mid-sweep (key file nulled) aborts promptly, rather than
        continuing to probe dead nodes for the rest of the budget."""
        lsf = _make_lsf(["a", "b"])
        lsf.ssh_connect_budget_s = 14400  # long budget; teardown must still win

        async def _node_then_teardown() -> str:
            # Simulate teardown_bsub racing in: the key file is removed while the
            # sweep is between attempts.
            lsf._key_file_path = None
            raise RuntimeError("no reachable node")

        get_node = AsyncMock(wraps=_node_then_teardown)
        with (
            patch("gbserver.environment.lsf.SshTunnel"),
            patch.object(lsf, "_get_reachable_ssh_node", new=get_node),
            patch("gbserver.environment.lsf.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            with pytest.raises(SshTunnelError, match="torn down / build"):
                await lsf._ensure_ssh_tunnel()
            # Second sweep's top-of-loop check bails before another node search.
            assert get_node.await_count == 1
            # No long budget wait — we aborted, not slept out 4h.
            sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_healthy_fast_path_skips_lock(self: Self) -> None:
        """A healthy tunnel is returned without ever taking _tunnel_lock."""
        lsf = _make_lsf(["a"])
        existing = _healthy_tunnel("a")
        lsf._ssh_tunnel = existing
        # A held lock would deadlock if the fast path tried to acquire it.
        await lsf._tunnel_lock.acquire()
        try:
            result = await asyncio.wait_for(lsf._ensure_ssh_tunnel(), timeout=1)
        finally:
            lsf._tunnel_lock.release()
        assert result is existing


class TestEnsureSshTunnelConcurrency:
    """The crux of the PR: rebuilds serialize and retire is transfer-safe."""

    @pytest.mark.asyncio
    async def test_concurrent_callers_rebuild_once(self: Self) -> None:
        """Under contention, exactly one rebuild happens; all callers share it."""
        lsf = _make_lsf(["a", "b"])
        fresh = _healthy_tunnel("a")
        builds = 0

        def _build(*_args: object, **_kwargs: object) -> MagicMock:
            nonlocal builds
            builds += 1
            return fresh

        async def _slow_node() -> str:
            await asyncio.sleep(0)  # yield so callers pile up on the lock
            return "a"

        with (
            patch("gbserver.environment.lsf.SshTunnel", side_effect=_build),
            patch.object(
                lsf, "_get_reachable_ssh_node", new=AsyncMock(wraps=_slow_node)
            ),
        ):
            results = await asyncio.gather(
                *(lsf._ensure_ssh_tunnel() for _ in range(5))
            )

        assert builds == 1  # lock serialized; re-check under lock reused the build
        assert all(r is fresh for r in results)

    @pytest.mark.asyncio
    async def test_ensure_and_use_holds_tunnel_across_retire(self: Self) -> None:
        """A concurrent rebuild retires the old tunnel but can't close it while a
        caller holds it via ensure_and_use (the retire-window fix)."""
        lsf = _make_lsf(["a", "b"])
        # Real SshTunnel so use()/close_when_idle refcounting is exercised.
        first = SshTunnel(host="a", username="u", key_file="/tmp/k")
        first._conn = MagicMock()
        first._conn.is_closed.return_value = False
        first.close = AsyncMock()  # type: ignore[method-assign]
        lsf._ssh_tunnel = first

        entered = asyncio.Event()
        release = asyncio.Event()

        async def _hold() -> None:
            async with lsf.ensure_and_use():
                entered.set()
                await release.wait()

        holder = asyncio.create_task(_hold())
        await entered.wait()  # caller now holds `first` in-use

        # Force a rebuild: mark first unhealthy, then ensure again.
        first._conn.is_closed.return_value = True
        second = _healthy_tunnel("b")
        with (
            patch("gbserver.environment.lsf.SshTunnel", return_value=second),
            patch.object(
                lsf, "_get_reachable_ssh_node", new=AsyncMock(return_value="b")
            ),
        ):
            rebuilt = await lsf._ensure_ssh_tunnel()
            await asyncio.sleep(0)  # let the retire GC task run

        assert rebuilt is second
        # first is retired but NOT yet closed — the holder still has it in-use.
        first.close.assert_not_awaited()

        release.set()
        await holder
        await asyncio.sleep(0)  # let close_when_idle drain and close
        first.close.assert_awaited()  # closed once the in-flight use drained


class TestReachabilityProbeBannerBound:
    """The pre-tunnel `ssh` probe gates tunnel establishment, so its ConnectTimeout
    reuses the tunnel's login_timeout — NOT the old 5s ssh_timeout, which cut off a
    slow-but-recoverable node before the tunnel could try it."""

    @pytest.mark.asyncio
    async def test_probe_uses_login_timeout_not_ssh_timeout(self: Self) -> None:
        lsf = _make_lsf(["a"])
        captured: dict = {}

        async def _capture(command_list, launch_id):  # noqa: ANN001
            captured["cmd"] = command_list
            return MagicMock(), "", ""

        with patch(
            "gbserver.environment.lsf.launch_command_and_raise_errors",
            new=AsyncMock(side_effect=_capture),
        ):
            ok = await lsf._Lsf__is_ssh_node_reachable(node="a", launch_id="lid")

        assert ok is True
        cmd = captured["cmd"]
        # Probe reuses the tunnel's patient login_timeout, never the rigid 5s.
        assert f"ConnectTimeout={lsf.ssh_login_timeout_s}" in cmd
        assert f"ConnectTimeout={lsf.ssh_timeout}" not in cmd


class TestSshTunnelIsHealthy:
    """The cheap, non-throwing liveness predicate."""

    def test_unopened_tunnel_is_not_healthy(self: Self) -> None:
        t = SshTunnel(host="h", username="u", key_file="/tmp/k")
        assert t.is_healthy() is False

    def test_open_connection_is_healthy(self: Self) -> None:
        t = SshTunnel(host="h", username="u", key_file="/tmp/k")
        conn = MagicMock()
        conn.is_closed.return_value = False
        t._conn = conn
        assert t.is_healthy() is True

    def test_closed_connection_is_not_healthy(self: Self) -> None:
        t = SshTunnel(host="h", username="u", key_file="/tmp/k")
        conn = MagicMock()
        conn.is_closed.return_value = True
        t._conn = conn
        assert t.is_healthy() is False

    def test_introspection_error_is_not_healthy(self: Self) -> None:
        t = SshTunnel(host="h", username="u", key_file="/tmp/k")
        conn = MagicMock()
        conn.is_closed.side_effect = RuntimeError("boom")
        t._conn = conn
        assert t.is_healthy() is False  # never raises

    def test_missing_port_forward_is_not_healthy(self: Self) -> None:
        """Live control connection but a lost port forward => not healthy."""
        t = SshTunnel(
            host="h", username="u", key_file="/tmp/k", port_forwards=[(0, "h", 22)]
        )
        conn = MagicMock()
        conn.is_closed.return_value = False
        t._conn = conn
        t._listeners = []  # forward never came up / was torn down
        assert t.is_healthy() is False


class TestSshTunnelRefcount:
    """use()/close_when_idle: a retired tunnel closes only once uses drain."""

    @pytest.mark.asyncio
    async def test_close_when_idle_waits_for_inflight_use(self: Self) -> None:
        t = SshTunnel(host="h", username="u", key_file="/tmp/k")
        t._conn = MagicMock()  # opened tunnel, so use() is valid
        t.close = AsyncMock()  # type: ignore[method-assign]

        release = asyncio.Event()

        async def _hold() -> None:
            async with t.use():
                await release.wait()

        holder = asyncio.create_task(_hold())
        await asyncio.sleep(0)  # let _hold enter the use() block

        gc = asyncio.create_task(t.close_when_idle())
        await asyncio.sleep(0)
        # Still in use — must not have closed yet.
        t.close.assert_not_awaited()

        release.set()
        await holder
        await gc
        t.close.assert_awaited_once()  # closed once the use drained

    @pytest.mark.asyncio
    async def test_use_rejects_closing_tunnel(self: Self) -> None:
        """use() on a tunnel already being retired fails fast (retire window)."""
        t = SshTunnel(host="h", username="u", key_file="/tmp/k")
        t._conn = MagicMock()  # otherwise use() rejects on conn is None
        t._closing = True
        with pytest.raises(SshTunnelError):
            async with t.use():
                pass  # pragma: no cover

    @pytest.mark.asyncio
    async def test_use_rejects_unopened_tunnel(self: Self) -> None:
        """use() before open() (no connection) fails fast rather than silently."""
        t = SshTunnel(host="h", username="u", key_file="/tmp/k")
        with pytest.raises(SshTunnelError):
            async with t.use():
                pass  # pragma: no cover

    def test_is_healthy_false_while_closing(self: Self) -> None:
        """A tunnel being retired must not be reused as healthy."""
        t = SshTunnel(host="h", username="u", key_file="/tmp/k")
        conn = MagicMock()
        conn.is_closed.return_value = False
        t._conn = conn
        t._closing = True
        assert t.is_healthy() is False

    @pytest.mark.asyncio
    async def test_close_when_idle_timeout_closes_anyway(self: Self) -> None:
        """A stuck in-flight use can't wedge close_when_idle past its timeout."""
        t = SshTunnel(host="h", username="u", key_file="/tmp/k")
        t._conn = MagicMock()  # opened tunnel, so use() is valid
        t.close = AsyncMock()  # type: ignore[method-assign]

        never = asyncio.Event()  # holder never releases

        async def _hold() -> None:
            async with t.use():
                await never.wait()

        holder = asyncio.create_task(_hold())
        await asyncio.sleep(0)  # enter the use() block

        # timeout=0 elapses immediately; close happens despite the in-flight use.
        await t.close_when_idle(timeout=0)
        t.close.assert_awaited_once()

        never.set()
        await holder
