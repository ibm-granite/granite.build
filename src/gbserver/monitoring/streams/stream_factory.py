#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""
This file contains a unified stream factory function
"""

from pathlib import Path
from typing import List, Optional

from gbserver.monitoring.streams.local_file_stream import LocalFileStream
from gbserver.monitoring.streams.log_stream_base import LogStreamSource
from gbserver.monitoring.streams.ssh_file_stream import SSHFileStream
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def make_stream(
    *,
    path: Optional[str | Path] = None,
    host: Optional[str] = None,
    user: Optional[str] = None,
    ssh_opts: Optional[List[str]] = None,
    use_ssh: bool = False,
    **retry_kwargs: int,
) -> LogStreamSource:
    """
    Create the appropriate LogStreamSource automatically based on parameters.

    Args:
        path: Path to a file (local or remote).
        host: host for remote SSH connection.
        user: SSH username.
        ssh_opts: Additional SSH command options.
        use_ssh: flag to select remote streams over ssh
        retry_kwargs: Reconnect/backoff parameters for SSH-based streams.

    Returns:
        An initialized LogStreamSource instance.
    """
    if use_ssh:
        logger.info("Creating SSHFileStream for %s:%s", host, path)
        return SSHFileStream(host or "", user, path or "", ssh_opts, **retry_kwargs)
    else:
        logger.info("Creating LocalFileStream for %s", path)
        return LocalFileStream(path or "")
