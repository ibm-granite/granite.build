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

A 0 (or negative) interval turns the BuildWatcher poll loop and the dispatched
BuildRunner event loop into CPU/storage busy-loops. The floor is enforced with a
single declarative bound — ``Field(ge=MIN_MONITORING_INTERVAL_SECONDS)`` — on
both watcher configs (BuildWatcherConfig drives the watcher's own poll loop and
the runners it spawns; PrWatcherConfig drives the PR watcher's poll loop). The
build-runner CLI applies the same bound via ``click.IntRange``. An out-of-range
value is rejected at construction rather than silently clamped.
"""

import pytest
from pydantic import ValidationError

from gbserver.types.buildwatcherconfig import BuildWatcherConfig
from gbserver.types.constants import MIN_MONITORING_INTERVAL_SECONDS
from gbserver.types.prwatcherconfig import PrWatcherConfig

_FLOOR = MIN_MONITORING_INTERVAL_SECONDS


@pytest.mark.parametrize("config_cls", [BuildWatcherConfig, PrWatcherConfig])
@pytest.mark.parametrize("bad", [0, -1, -5])
def test_monitoring_interval_rejected_at_construction(config_cls, bad):
    """A below-floor interval raises rather than silently busy-looping."""
    with pytest.raises(ValidationError):
        config_cls(monitoring_interval=bad)


@pytest.mark.parametrize("config_cls", [BuildWatcherConfig, PrWatcherConfig])
def test_valid_monitoring_intervals_accepted(config_cls):
    assert config_cls().monitoring_interval == 5  # default unchanged
    assert config_cls(monitoring_interval=_FLOOR).monitoring_interval == _FLOOR
    assert config_cls(monitoring_interval=30).monitoring_interval == 30
