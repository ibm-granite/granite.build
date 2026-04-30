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


"""Errors module."""

from typing import Union


class LogMonitoringFailedException(Exception):
    """Log Monitoring Failed Exception implementation."""

    def __init__(self, *args, build_id: str = ""):
        self.build_id = build_id
        super().__init__(*args)


class WorkloadFailedException(Exception):
    """Workload Failed Exception implementation."""

    def __init__(self, *args, build_id: str = ""):
        self.build_id = build_id
        super().__init__(*args)


ERR_CONNECTION_RESET_BY_PEER = "Connection reset by peer"


class ErrConnResetByPeer(Exception):
    """ssh connection failed with 'Connection reset by peer'"""

    def matches_error_str(s: Union[str, bytes]) -> bool:
        """Matches error str."""
        return (isinstance(s, str) and ERR_CONNECTION_RESET_BY_PEER in s) or (
            isinstance(s, bytes) and ERR_CONNECTION_RESET_BY_PEER.encode("utf-8") in s
        )


ERR_NETWORK_UNREACHABLE = "Network is unreachable"


class ErrNetworkUnreachable(Exception):
    """ssh connection failed with 'Network is unreachable'"""

    def matches_error_str(s: Union[str, bytes]) -> bool:
        """Matches error str."""
        return (isinstance(s, str) and ERR_NETWORK_UNREACHABLE in s) or (
            isinstance(s, bytes) and ERR_NETWORK_UNREACHABLE.encode("utf-8") in s
        )


ERR_SSH_CONNECTION_PATTERNS = ["connection", "ssh", "network", "timeout", "refused"]


class ErrSSHConnectionError(Exception):
    """SSH/connection error that should trigger a retry"""

    @staticmethod
    def matches_error_str(s: Union[str, bytes]) -> bool:
        """Matches error str."""
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="replace")
        s_lower = s.lower()
        return any(pattern in s_lower for pattern in ERR_SSH_CONNECTION_PATTERNS)


ERR_LSF_CANNOT_OPEN_JOB_FILE = "Cannot open your job file"


class ErrLSFCannotOpenJobFile(Exception):
    """LSF job failed with 'Cannot open your job file' - a transient error that should be retried"""

    def __init__(self, *args, job_id: str = "", launch_id: str = ""):
        self.job_id = job_id
        self.launch_id = launch_id
        super().__init__(*args)

    @staticmethod
    def matches_error_str(s: Union[str, bytes]) -> bool:
        """Matches error str."""
        return (isinstance(s, str) and ERR_LSF_CANNOT_OPEN_JOB_FILE in s) or (
            isinstance(s, bytes) and ERR_LSF_CANNOT_OPEN_JOB_FILE.encode("utf-8") in s
        )
