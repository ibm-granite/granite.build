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

"""
General-purpose SSH tunnel using asyncssh.

Opens a persistent SSH connection and allows running commands on the remote
host through that connection. Optionally sets up local port forwarding.

Usage::

    tunnel = SshTunnel(host="login.hpc.example.com", username="alice", key_file="/tmp/id_rsa")
    await tunnel.open()
    rc, stdout, stderr = await tunnel.run("hostname")
    await tunnel.close()

    # Or as a context manager:
    async with SshTunnel(host="login.hpc.example.com", username="alice") as tunnel:
        rc, stdout, stderr = await tunnel.run("ls /scratch")
"""

import asyncio
from typing import List, Optional, Tuple

from gbserver.utils.optional_imports import HAS_ASYNCSSH

if HAS_ASYNCSSH:
    import asyncssh

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from gbserver.utils.launch import (
    launch_command_and_raise_errors,
    launch_command_and_retry_or_raise_errors,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class SshTunnelError(Exception):
    """Raised when the SSH tunnel fails to open or becomes unavailable."""


class SshTunnel:
    """
    Persistent SSH connection via asyncssh.

    ``open()`` establishes an SSH connection. ``run()`` sends commands over
    that open connection without spawning additional processes.

    Optional local port forwarding (equivalent to ``ssh -L``) can be set up
    at construction time and is activated during ``open()``.
    """

    def __init__(
        self,
        host: str,
        username: Optional[str] = None,
        key_file: Optional[str] = None,
        host_key_verification: bool = True,
        port_forwards: Optional[List[Tuple[int, str, int]]] = None,
        max_sessions: int = 10,  # 10 is the default MaxSessions value for sshd
    ) -> None:
        if not HAS_ASYNCSSH:
            raise ImportError(
                "asyncssh is required for SSH tunnel support. "
                "Install it with: pip install gbserver[ssh]"
            )
        self.host = host
        self.username = username
        self.key_file = key_file
        self.host_key_verification = host_key_verification
        self.port_forwards: List[Tuple[int, str, int]] = port_forwards or []

        self._conn: Optional[asyncssh.SSHClientConnection] = None
        self._listeners: List[asyncssh.SSHListener] = []
        self._actual_local_ports: List[int] = []
        self._semaphore = asyncio.Semaphore(max_sessions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Establish the SSH connection and activate any port forwards."""
        connect_kwargs: dict = {}
        if self.username:
            connect_kwargs["username"] = self.username
        if self.key_file:
            connect_kwargs["client_keys"] = [self.key_file]
        if not self.host_key_verification:
            connect_kwargs["known_hosts"] = None

        logger.info("[SshTunnel] Connecting to %s", self.host)
        try:
            self._conn = await asyncssh.connect(self.host, **connect_kwargs)
        except (asyncssh.Error, OSError) as e:
            raise SshTunnelError(
                f"[SshTunnel] Failed to connect to {self.host}: {e}"
            ) from e

        for local_port, remote_host, remote_port in self.port_forwards:
            logger.info(
                "[SshTunnel] Forwarding local port %d → %s:%d",
                local_port,
                remote_host,
                remote_port,
            )
            try:
                listener = await self._conn.forward_local_port(
                    "", local_port, remote_host, remote_port
                )
                self._listeners.append(listener)
                self._actual_local_ports.append(listener.get_port())
                logger.info(
                    "[SshTunnel] Local port %d forwarding to %s:%d",
                    listener.get_port(),
                    remote_host,
                    remote_port,
                )
            except (asyncssh.Error, OSError) as e:
                await self.close()
                raise SshTunnelError(
                    f"[SshTunnel] Failed to set up port forward "
                    f"{local_port}:{remote_host}:{remote_port}: {e}"
                ) from e

        logger.info("[SshTunnel] Connected to %s", self.host)

    async def close(self) -> None:
        """Close all port-forward listeners and the SSH connection."""
        for listener in self._listeners:
            listener.close()
            await listener.wait_closed()
        self._listeners.clear()
        self._actual_local_ports.clear()

        if self._conn is not None:
            self._conn.close()
            await self._conn.wait_closed()
            self._conn = None
            logger.info("[SshTunnel] Connection to %s closed", self.host)

    def __to_str(self, text: "Optional[asyncssh.BytesOrStr]") -> str:
        """Convert the given text to a str if not already one.  Handle None, and bytes."""
        if not text:
            return ""
        if isinstance(text, str):
            return str(text)
        assert isinstance(text, bytes)
        try:
            return text.decode("utf-8")
        except Exception as ee:
            logger.debug("failed to decode text: %s", ee)
            return ""

    async def run_remote(
        self,
        command: str,
        redacted_command: Optional[str] = None,
        raise_on_error: bool = True,
    ) -> Tuple[int, str, str]:
        """
        Run a shell command on the remote host via the open connection.

        Args:
            command: Shell command string to execute on the remote host.
            redacted_command: command to be logged instead of the given command, if provided.
            raise_on_error: If True (default), raise ValueError on non-zero exit.

        Returns:
            Tuple of (returncode, stdout, stderr).
        """
        logged_command = redacted_command if redacted_command else command
        logger.info("[SshTunnel] Running command: %s", logged_command)
        result = await self._run_with_semaphore(command, logged_command)

        rc: int = result.exit_status if result.exit_status is not None else -1
        stdout: str = self.__to_str(result.stdout)
        stderr: str = self.__to_str(result.stderr)
        if len(stdout) > 0:
            logger.info("[SshTunnel] stdout: %s", stdout)
        if len(stderr) > 0:
            logger.warning("[SshTunnel] stderr: %s", stderr)

        if raise_on_error and rc != 0:
            raise ValueError(
                f"[SshTunnel] Command '{logged_command}' failed with return code {rc}\n{stderr}"
            )

        return rc, stdout, stderr

    async def run_remote_with_retries(
        self,
        command: str,
        redacted_command: Optional[str] = None,
        raise_on_error: bool = True,
    ) -> Tuple[int, str, str]:
        """Same as ``run_remote`` but retries on transient channel/timeout errors."""

        @retry(
            stop=stop_after_attempt(10),
            wait=wait_random_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type((asyncssh.ChannelOpenError, TimeoutError)),
        )
        async def _inner():
            return await self.run_remote(
                command,
                redacted_command=redacted_command,
                raise_on_error=raise_on_error,
            )

        return await _inner()

    async def _run_with_semaphore(
        self, command: str, logged_command: str
    ) -> "asyncssh.SSHCompletedProcess":
        if self._conn is None:
            raise SshTunnelError("Tunnel is not open. Call open() first.")
        logger.info("[SshTunnel] Running command: %s", logged_command)
        async with self._semaphore:
            result = await self._conn.run(command, check=False)
        return result

    async def run_local(
        self,
        command: List[str],
        launch_id: str,
    ) -> Tuple[int, str, str]:
        """
        Run a local subprocess command throttled by this tunnel's semaphore.

        Use this for commands (e.g. rsync) that must run as a local process but
        connect to the remote host through this tunnel's port forward.

        Args:
            command: Argument list for the local subprocess.
            launch_id: Identifier used for logging by the launcher.

        Returns:
            Tuple of (returncode, stdout, stderr).
        """
        logger.info("[SshTunnel] Running local command: %s", command)
        async with self._semaphore:
            process, stdout, stderr = await launch_command_and_raise_errors(
                command_list=command,
                launch_id=launch_id,
            )
        rc: int = process.returncode if process.returncode is not None else -1
        return rc, self.__to_str(stdout), self.__to_str(stderr)

    async def run_local_with_retries(
        self,
        command: List[str],
        launch_id: str,
    ) -> Tuple[int, str, str]:
        """Same as ``run_local`` but retries on transient network errors."""
        logger.info("[SshTunnel] Running local command: %s", command)
        async with self._semaphore:
            process, stdout, stderr = await launch_command_and_retry_or_raise_errors(
                command_list=command,
                launch_id=launch_id,
            )
        rc: int = process.returncode if process.returncode is not None else -1
        return rc, self.__to_str(stdout), self.__to_str(stderr)

    async def start_sftp(self) -> "asyncssh.SFTPClient":
        """Start an SFTP subsystem on the open connection.

        Callers should close the returned client themselves (``await client.exit()``).
        Exposed so code outside this module doesn't need to reach into ``_conn``.
        """
        if self._conn is None:
            raise SshTunnelError("Tunnel is not open. Call open() first.")
        return await self._conn.start_sftp_client()

    def get_local_port(self, remote_host: str, remote_port: int) -> Optional[int]:
        """Return the actual local port forwarding to remote_host:remote_port, or None."""
        for i, (_, rh, rp) in enumerate(self.port_forwards):
            if (
                rh == remote_host
                and rp == remote_port
                and i < len(self._actual_local_ports)
            ):
                return self._actual_local_ports[i]
        return None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SshTunnel":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
