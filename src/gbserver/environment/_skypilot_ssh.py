"""Shared SSH utilities for SkyPilot environments (unmanaged and managed).

Provides host SSH info extraction and remote command execution used by
both skypilot.py and skypilot_managed.py for post-launch tasks (sidecars).
"""

import asyncio
import contextlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


@retry(
    retry=retry_if_exception(lambda e: isinstance(e, FileNotFoundError)),
    stop=stop_after_attempt(30),
    wait=wait_exponential(multiplier=1, max=10),
    reraise=True,
)
def extract_host_ssh_info(cluster_name: str) -> Tuple[str, str]:
    """Extract host IP and SSH key from SkyPilot's generated SSH config.

    SkyPilot writes SSH config to ~/.sky/generated/ssh/<cluster_name> after
    cluster provisioning. This function reads that file to extract:
    - HOST_IP: The target host IP (from ProxyCommand)
    - SSH_KEY_PATH: The path to the private key file (from IdentityFile)

    Retries with exponential backoff only on FileNotFoundError (config not
    yet written). Parse failures (RuntimeError) raise immediately.

    Args:
        cluster_name: The SkyPilot cluster name.

    Returns:
        Tuple of (host_ip, ssh_key_path).

    Raises:
        RuntimeError: If SSH config cannot be read or parsed.
        FileNotFoundError: If SSH config file does not exist after retries exhausted.
    """
    sky_dir = Path.home() / ".sky" / "generated" / "ssh" / cluster_name
    if not sky_dir.exists():
        raise FileNotFoundError(f"SkyPilot SSH config not found: {sky_dir}")

    try:
        with open(sky_dir, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        raise RuntimeError(f"Failed to read SkyPilot SSH config {sky_dir}: {e}") from e

    # Extract HOST_IP from ProxyCommand line
    # Format: ProxyCommand ssh -i /path/key -p 10022 -W %h:%p ubuntu@HOST_IP
    host_match = re.search(r"ProxyCommand.*?(\d+\.\d+\.\d+\.\d+)", content)
    if not host_match:
        raise RuntimeError(
            f"Could not extract host IP from SkyPilot SSH config {sky_dir}"
        )
    host_ip = host_match.group(1)

    # Extract SSH_KEY_PATH from IdentityFile line
    # Format: IdentityFile /path/to/private/key
    key_match = re.search(r"IdentityFile\s+(.+)", content)
    if not key_match:
        raise RuntimeError(
            f"Could not extract SSH key path from SkyPilot SSH config {sky_dir}"
        )
    ssh_key_path = key_match.group(1).strip()

    logger.info(
        "Extracted SkyPilot host info: host_ip=%s ssh_key=%s (cluster=%s)",
        host_ip,
        ssh_key_path,
        cluster_name,
    )
    return host_ip, ssh_key_path


def _host_ssh_base_cmd(ssh_key: str, host_ip: str, connect_timeout: int) -> List[str]:
    """Shared `ssh` prefix for the post-launch host connection."""
    return [
        "ssh",
        "-i",
        ssh_key,
        "-p",
        "22",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=3",
        f"ubuntu@{host_ip}",
    ]


# SSH failures that never succeed on retry — a rejected key stays rejected, and a
# missing binary/key stays missing. Mirrors _NON_TRANSIENT_PROVISION_SUBSTRINGS in
# skypilot.py (kept local: that one is matched against SkyPilot exception text).
_FATAL_SSH_SUBSTRINGS = (
    "permission denied",
    "too many authentication failures",
    "host key verification failed",
    "no such identity",
    "invalid privatekey",
    "unprotected private key file",
    "bad configuration option",
    "no such file or directory",
)


def _is_fatal_ssh_error(stderr_text: str) -> bool:
    """True if this SSH failure will not be fixed by retrying."""
    lowered = stderr_text.lower()
    return any(sub in lowered for sub in _FATAL_SSH_SUBSTRINGS)


async def _await_host_reachable(host_ip: str, ssh_key: str, login_timeout: int) -> None:
    """Wait until the host completes an SSH login, retrying the connect phase.

    ``ConnectTimeout`` bounds only the TCP leg, so a slow-banner host hangs past it;
    an `echo` under ``wait_for`` bounds banner+auth+session setup. Retries only this
    phase — never the caller's command, which may not be idempotent.
    """
    from gbserver.types.constants import GBSERVER_SKYPILOT_HOST_SSH_ATTEMPTS

    attempts = max(1, GBSERVER_SKYPILOT_HOST_SSH_ATTEMPTS)
    last = "no attempt made"
    for attempt in range(1, attempts + 1):
        cmds = _host_ssh_base_cmd(ssh_key, host_ip, login_timeout) + [
            "echo",
            "gbserver probe",
        ]
        fatal = False
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmds,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:  # noqa: BLE001 — falls through to warn + backoff
            last = f"could not spawn ssh: {e}"
        else:
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=login_timeout
                )
            except asyncio.CancelledError:
                await _kill_and_reap(proc)
                raise
            except Exception as e:  # noqa: BLE001 — timeout => host not ready
                # wait_for only cancels the await; kill so a hung ssh can't linger.
                await _kill_and_reap(proc)
                last = f"login did not complete within {login_timeout}s ({type(e).__name__})"
            else:
                if proc.returncode == 0:
                    if attempt > 1:
                        logger.info("host %s reachable on attempt %d", host_ip, attempt)
                    return
                last = (stderr or b"").decode("utf-8", errors="replace").strip() or (
                    f"ssh exited {proc.returncode}"
                )
                # A rejected key stays rejected: fail now instead of burning the
                # budget on identical retries (mirrors the non-transient set in
                # skypilot.py).
                fatal = _is_fatal_ssh_error(last)
        logger.warning(
            "host %s not ready for post-launch SSH (attempt %d/%d): %s",
            host_ip,
            attempt,
            attempts,
            last,
        )
        if fatal:
            break
        if attempt < attempts:
            await asyncio.sleep(min(2 ** (attempt - 1), 10))
    raise RuntimeError(
        f"Host {host_ip} did not accept an SSH login after {attempts} attempt(s): {last}"
    )


async def _kill_and_reap(proc) -> None:
    """Kill a subprocess and reap it, ignoring an already-exited child."""
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(Exception):
        await proc.wait()


async def execute_on_host_via_ssh(
    host_ip: str,
    ssh_key: str,
    commands: str,
    env_vars: Optional[Dict[str, str]] = None,
    timeout: int = 600,
) -> None:
    """Execute commands on the host VM via direct SSH.

    Establishes an SSH session to ubuntu@<host_ip>:22 (not container proxy
    on 10022) and runs the given commands with optional environment variables.

    Waits for a bounded login first (:func:`_await_host_reachable`), so a slow-banner
    host fails in ~login_timeout rather than hanging to ``timeout``. Only that phase
    retries; ``commands`` runs at most once (it may not be idempotent).

    Args:
        host_ip: The host VM IP address.
        ssh_key: The path to the SSH private key.
        commands: The bash commands to execute.
        env_vars: Optional dict of environment variables to inject.
        timeout: Max seconds to wait for command completion (default: 600).

    Raises:
        RuntimeError: If the host never accepts a login, or execution fails/times out.
    """
    # Build environment variable exports at the start of the command
    env_setup = ""
    if env_vars:
        for key, value in env_vars.items():
            # Escape single quotes in values by replacing ' with '\''
            escaped_value = value.replace("'", "'\\''")
            env_setup += f"export {key}='{escaped_value}'\n"

    # Build the full bash command with env vars injected
    full_command = f"{env_setup}{commands}"

    from gbserver.types.constants import (
        GBSERVER_SKYPILOT_HOST_SSH_LOGIN_TIMEOUT_S as _LOGIN_TIMEOUT,
    )

    # Bound banner+auth before sending the payload; otherwise a wedged host hangs
    # to `timeout` (600s default).
    await _await_host_reachable(host_ip, ssh_key, _LOGIN_TIMEOUT)

    ssh_cmd = _host_ssh_base_cmd(ssh_key, host_ip, _LOGIN_TIMEOUT) + ["bash"]

    logger.info(
        "Executing post-launch task on host %s via SSH (key=%s)",
        host_ip,
        ssh_key,
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to start post-launch task on {host_ip}: {e}") from e

    # Native async subprocess, not subprocess.run in a thread: a thread is
    # unreachable by cancellation, so a cancelled build orphaned the ssh child.
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(input=full_command.encode("utf-8")), timeout=timeout
        )
    except asyncio.CancelledError:
        await _kill_and_reap(proc)
        raise
    except (asyncio.TimeoutError, TimeoutError) as e:
        await _kill_and_reap(proc)
        raise RuntimeError(
            f"Post-launch task on {host_ip} timed out after {timeout}s"
        ) from e
    except Exception as e:  # noqa: BLE001 — IO/decode error, not a timeout
        # Reported distinctly: labelling an IO failure a timeout sends a debugger
        # down the wrong path.
        await _kill_and_reap(proc)
        raise RuntimeError(
            f"Post-launch task on {host_ip} failed while reading output: {e}"
        ) from e

    stdout_str = (stdout_b or b"").decode("utf-8", errors="replace")
    if proc.returncode != 0:
        stderr_str = (stderr_b or b"").decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Post-launch task failed on {host_ip} (exit code {proc.returncode}).\n"
            f"stderr: {stderr_str}\nstdout: {stdout_str}"
        )
    logger.info(
        "Post-launch task succeeded on host %s. Output:\n%s", host_ip, stdout_str
    )
