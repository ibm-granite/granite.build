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

from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest

from gbserver.utils.ssh_tunnel import SshTunnel, SshTunnelError

pytestmark = pytest.mark.ibm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_conn(run_exit_status=0, run_stdout="", run_stderr=""):
    """Return a mock asyncssh SSHClientConnection."""
    conn = MagicMock(spec=asyncssh.SSHClientConnection)

    # run() returns a completed process mock
    result = MagicMock()
    result.exit_status = run_exit_status
    result.stdout = run_stdout
    result.stderr = run_stderr
    conn.run = AsyncMock(return_value=result)

    # close() is synchronous; wait_closed() is async
    conn.close = MagicMock()
    conn.wait_closed = AsyncMock()

    # forward_local_port() returns a listener mock
    listener = MagicMock()
    listener.close = MagicMock()
    listener.wait_closed = AsyncMock()
    conn.forward_local_port = AsyncMock(return_value=listener)

    return conn, listener


# ---------------------------------------------------------------------------
# open() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_connects_with_correct_kwargs():
    """open() should call asyncssh.connect with the right parameters."""
    mock_conn, _ = _make_mock_conn()

    with patch(
        "asyncssh.connect", new=AsyncMock(return_value=mock_conn)
    ) as mock_connect:
        tunnel = SshTunnel(host="myhost", username="alice", key_file="/tmp/id_rsa")
        await tunnel.open()

    mock_connect.assert_called_once_with(
        "myhost",
        username="alice",
        client_keys=["/tmp/id_rsa"],
    )


@pytest.mark.asyncio
async def test_open_disables_host_key_verification():
    """host_key_verification=False should pass known_hosts=None."""
    mock_conn, _ = _make_mock_conn()

    with patch(
        "asyncssh.connect", new=AsyncMock(return_value=mock_conn)
    ) as mock_connect:
        tunnel = SshTunnel(host="myhost", host_key_verification=False)
        await tunnel.open()

    _, kwargs = mock_connect.call_args
    assert kwargs.get("known_hosts") is None


@pytest.mark.asyncio
async def test_open_raises_ssh_tunnel_error_on_connection_failure():
    """open() should raise SshTunnelError if asyncssh.connect raises."""
    with patch(
        "asyncssh.connect", new=AsyncMock(side_effect=OSError("Connection refused"))
    ):
        tunnel = SshTunnel(host="myhost")
        with pytest.raises(SshTunnelError, match="Failed to connect"):
            await tunnel.open()


@pytest.mark.asyncio
async def test_open_sets_up_port_forwards():
    """open() should call forward_local_port for each port_forward entry."""
    mock_conn, mock_listener = _make_mock_conn()

    with patch("asyncssh.connect", new=AsyncMock(return_value=mock_conn)):
        tunnel = SshTunnel(
            host="myhost",
            port_forwards=[(8080, "remote-svc", 80)],
        )
        await tunnel.open()

    mock_conn.forward_local_port.assert_called_once_with("", 8080, "remote-svc", 80)


# ---------------------------------------------------------------------------
# run_remote() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_stdout_stderr_on_success():
    """run() should return (rc, stdout, stderr) from the connection."""
    mock_conn, _ = _make_mock_conn(
        run_exit_status=0, run_stdout="hello\n", run_stderr=""
    )

    with patch("asyncssh.connect", new=AsyncMock(return_value=mock_conn)):
        tunnel = SshTunnel(host="myhost")
        await tunnel.open()
        rc, stdout, stderr = await tunnel.run_remote("echo hello")

    assert rc == 0
    assert stdout == "hello\n"
    assert stderr == ""
    mock_conn.run.assert_called_once_with("echo hello", check=False)


@pytest.mark.asyncio
async def test_run_raises_on_nonzero_exit_by_default():
    """run() should raise ValueError on non-zero exit when raise_on_error=True."""
    mock_conn, _ = _make_mock_conn(run_exit_status=1, run_stderr="not found")

    with patch("asyncssh.connect", new=AsyncMock(return_value=mock_conn)):
        tunnel = SshTunnel(host="myhost")
        await tunnel.open()
        with pytest.raises(ValueError, match="failed with return code"):
            await tunnel.run_remote("bad-command", raise_on_error=True)


@pytest.mark.asyncio
async def test_run_returns_nonzero_without_raising_when_raise_on_error_false():
    """run() should return non-zero rc without raising when raise_on_error=False."""
    mock_conn, _ = _make_mock_conn(run_exit_status=2, run_stderr="oops")

    with patch("asyncssh.connect", new=AsyncMock(return_value=mock_conn)):
        tunnel = SshTunnel(host="myhost")
        await tunnel.open()
        rc, stdout, stderr = await tunnel.run_remote(
            "bad-command", raise_on_error=False
        )

    assert rc == 2
    assert stderr == "oops"


@pytest.mark.asyncio
async def test_run_before_open_raises_ssh_tunnel_error():
    """run() should raise SshTunnelError if tunnel is not open."""
    tunnel = SshTunnel(host="myhost")
    with pytest.raises(SshTunnelError, match="not open"):
        await tunnel.run_remote("hostname")


# ---------------------------------------------------------------------------
# close() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_closes_connection():
    """close() should call conn.close() and wait_closed()."""
    mock_conn, _ = _make_mock_conn()

    with patch("asyncssh.connect", new=AsyncMock(return_value=mock_conn)):
        tunnel = SshTunnel(host="myhost")
        await tunnel.open()
        await tunnel.close()

    mock_conn.close.assert_called_once()
    mock_conn.wait_closed.assert_called_once()
    assert tunnel._conn is None


@pytest.mark.asyncio
async def test_close_closes_port_forward_listeners():
    """close() should close all port forward listeners."""
    mock_conn, mock_listener = _make_mock_conn()

    with patch("asyncssh.connect", new=AsyncMock(return_value=mock_conn)):
        tunnel = SshTunnel(host="myhost", port_forwards=[(8080, "svc", 80)])
        await tunnel.open()
        await tunnel.close()

    mock_listener.close.assert_called_once()
    mock_listener.wait_closed.assert_called_once()


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_calls_open_and_close():
    """async with SshTunnel should call open() on enter and close() on exit."""
    tunnel = SshTunnel(host="myhost")
    tunnel.open = AsyncMock()
    tunnel.close = AsyncMock()

    async with tunnel:
        tunnel.open.assert_called_once()

    tunnel.close.assert_called_once()


@pytest.mark.asyncio
async def test_context_manager_calls_close_on_exception():
    """close() should be called even if the body raises."""
    tunnel = SshTunnel(host="myhost")
    tunnel.open = AsyncMock()
    tunnel.close = AsyncMock()

    with pytest.raises(RuntimeError):
        async with tunnel:
            raise RuntimeError("body error")

    tunnel.close.assert_called_once()
