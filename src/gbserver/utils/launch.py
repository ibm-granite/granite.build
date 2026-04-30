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
Helpers for launching local subprocesses with optional retry logic.
"""

import asyncio
from asyncio.subprocess import Process
from typing import List, Optional, Tuple, Union

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from gbserver.types.errors import ErrConnResetByPeer, ErrNetworkUnreachable
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import cmd_safe_join

logger = get_logger(__name__)


async def launch_command_and_raise_errors(
    command_list: List[str],
    launch_id: str,
    start_new_session: bool = False,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
    raise_error: bool = True,
    redacted_command_str: str = "",
) -> Tuple[Process, Union[bytes, str], Union[bytes, str]]:
    """
    Launchers a command and raise an error if the return code is non-zero.

        Returns the launched process object.
    """
    if redacted_command_str:
        logger.info("running redacted_command_str: %s", redacted_command_str)
    else:
        logger.info("running command: %s", command_list)
    process = await asyncio.create_subprocess_exec(
        *command_list,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=start_new_session,
        cwd=cwd if cwd else ".",
        env=env,
    )
    stdout, stderr = await process.communicate()
    try:
        stdout = stdout.decode("utf-8")
    except Exception as ee:
        logger.debug("failed to decode stdout: %s", ee)
    try:
        stderr = stderr.decode("utf-8")
    except Exception as ee:
        logger.debug("failed to decode stderr: %s", ee)
    is_failed = process.returncode is None or process.returncode != 0
    command_str = redacted_command_str if redacted_command_str else cmd_safe_join(command_list)
    if len(stdout) > 0:
        logger.info(
            "launch_id %s, command: `%s` , stdout: %s",
            launch_id,
            command_str,
            stdout,
        )
    if len(stderr) > 0:
        if raise_error and is_failed:
            logger.error(
                "failed step %s, command: `%s` , stderr: %s",
                launch_id,
                command_str,
                stderr,
            )
        else:
            logger.warning(
                "step %s, command: `%s` , stderr: %s",
                launch_id,
                command_str,
                stderr,
            )
    if process.returncode is None:
        raise ValueError(f"failed to launch the process `{command_str}`")
    if process.returncode != 0:
        err_msg = (
            f"the process `{command_str}` failed with return code: {process.returncode}\n{stderr}"
        )
        if raise_error:
            if ErrConnResetByPeer.matches_error_str(stderr):
                raise ErrConnResetByPeer(err_msg)
            if ErrNetworkUnreachable.matches_error_str(stderr):
                raise ErrNetworkUnreachable(err_msg)
            raise ValueError(err_msg)
        logger.warning("%s", err_msg)
    return process, stdout, stderr


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, max=30),
    retry=retry_if_exception_type((ErrConnResetByPeer, ErrNetworkUnreachable)),
)
async def launch_command_and_retry_or_raise_errors(
    command_list: List[str],
    launch_id: str,
    start_new_session: bool = False,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
    raise_error: bool = True,
    redacted_command_str: str = "",
) -> Tuple[Process, Union[bytes, str], Union[bytes, str]]:
    """Same as launch_command_and_raise_errors but with retries."""
    return await launch_command_and_raise_errors(
        command_list=command_list,
        launch_id=launch_id,
        start_new_session=start_new_session,
        env=env,
        cwd=cwd,
        raise_error=raise_error,
        redacted_command_str=redacted_command_str,
    )
