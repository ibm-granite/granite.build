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

"""The Bash environment runs a step's commands directly on the gbserver host,
so it may only be constructed on a STANDALONE server (the user's own machine).
Anywhere else, a build targeting it would let any user run code on the server.
"""

import asyncio
from unittest.mock import patch

import pytest

from gbserver.environment import bash as bash_module
from gbserver.environment.bash import Bash, BashEnvironmentNotAllowed
from gbserver.environment.environment import Environment

BASH_ENV_URI = "file:configurations/assets/environments/bash"


def test_bash_rejected_outside_standalone():
    with patch.object(bash_module, "is_standalone", return_value=False):
        with pytest.raises(BashEnvironmentNotAllowed, match="STANDALONE"):
            Bash(event_q=asyncio.Queue())


def test_bash_allowed_in_standalone():
    with patch.object(bash_module, "is_standalone", return_value=True):
        assert isinstance(Bash(event_q=asyncio.Queue()), Bash)


def test_get_environment_rejects_bash_outside_standalone():
    """The path a build takes: environment.yaml `type: Bash` -> get_environment."""
    with patch.object(bash_module, "is_standalone", return_value=False):
        with pytest.raises(BashEnvironmentNotAllowed):
            Environment.get_environment(BASH_ENV_URI, event_q=asyncio.Queue())


def test_allow_bash_environment_fixture(allow_bash_environment):
    # The shared opt-in fixture used by tests that construct Bash directly.
    assert isinstance(Bash(event_q=asyncio.Queue()), Bash)
