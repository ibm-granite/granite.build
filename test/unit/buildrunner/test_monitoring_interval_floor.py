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

"""Regression tests for the monitoring_interval floor.

A sub-second interval (especially 0) turns the BuildRunner event loop and the
BuildWatcher poll loop into CPU/storage busy-loops.  The floor
(MIN_MONITORING_INTERVAL_SECONDS) is enforced in two independent places, each
covering a distinct consumer:

* AbstractBuildRunner (clamping property) — for every runner, set via __init__
  or post-construction assignment.
* BuildWatcherConfig (validator + validate_assignment) — for the watcher's own
  poll loop, set via construction or post-construction assignment.
"""

import pytest

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.types.buildwatcherconfig import BuildWatcherConfig
from gbserver.types.constants import MIN_MONITORING_INTERVAL_SECONDS

_FLOOR = MIN_MONITORING_INTERVAL_SECONDS


@pytest.mark.parametrize(
    "given, expected",
    [
        (0, _FLOOR),  # the busy-loop value
        (0.01, _FLOOR),  # any sub-second value
        (-5, _FLOOR),  # negative is nonsensical
        (_FLOOR, _FLOOR),  # at the floor
        (5, 5),  # the default, unchanged
    ],
)
def test_runner_monitoring_interval_is_floored(given, expected):
    """AbstractBuildRunner's setter floors via __init__ and later assignment."""
    runner = object.__new__(BuildRunner)  # skip __init__; exercise the setter
    runner.monitoring_interval = given
    assert runner.monitoring_interval == expected


def test_buildwatcher_config_floors_on_construction():
    assert BuildWatcherConfig(monitoring_interval=0).monitoring_interval == _FLOOR


def test_buildwatcher_config_floors_on_assignment():
    """The watcher mutates this field after construction, so assignment must clamp."""
    config = BuildWatcherConfig()
    config.monitoring_interval = 0
    assert config.monitoring_interval == _FLOOR
    config.monitoring_interval = 5
    assert config.monitoring_interval == 5
