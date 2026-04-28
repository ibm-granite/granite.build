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
