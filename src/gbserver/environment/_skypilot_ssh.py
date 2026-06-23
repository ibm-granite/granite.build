"""Shared SSH utilities for SkyPilot environments (unmanaged and managed).

Provides host SSH info extraction and remote command execution used by
both skypilot.py and skypilot_managed.py for post-launch tasks (sidecars).
"""

import asyncio
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

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

    Args:
        host_ip: The host VM IP address.
        ssh_key: The path to the SSH private key.
        commands: The bash commands to execute.
        env_vars: Optional dict of environment variables to inject.
        timeout: Max seconds to wait for command completion (default: 600).

    Raises:
        RuntimeError: If SSH execution fails or times out.
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

    # Build SSH command with connection timeouts to fail fast on
    # dead/unreachable hosts instead of hanging until the subprocess timeout.
    ssh_cmd = [
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
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=3",
        f"ubuntu@{host_ip}",
        "bash",
    ]

    logger.info(
        "Executing post-launch task on host %s via SSH (key=%s)",
        host_ip,
        ssh_key,
    )

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ssh_cmd,
            input=full_command.encode("utf-8"),
            timeout=timeout,
            capture_output=True,
        )
        if result.returncode != 0:
            stderr_str = result.stderr.decode("utf-8", errors="replace")
            stdout_str = result.stdout.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Post-launch task failed on {host_ip} (exit code {result.returncode}).\n"
                f"stderr: {stderr_str}\nstdout: {stdout_str}"
            )
        stdout_str = result.stdout.decode("utf-8", errors="replace")
        logger.info(
            "Post-launch task succeeded on host %s. Output:\n%s", host_ip, stdout_str
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Post-launch task on {host_ip} timed out after {timeout}s"
        ) from e
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to execute post-launch task on {host_ip}: {e}"
        ) from e
