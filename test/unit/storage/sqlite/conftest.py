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

"""Isolate the SQLite storage tests from the developer's real home directory.

The standalone SQLite db lives under ``GB_HOME_DIR`` (default ~/.granite.build).
Without this fixture the storage tests would read/write the developer's real db;
point GB_HOME_DIR at a per-test temp dir instead.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_gb_home_dir(tmp_path, monkeypatch):
    home = tmp_path / "gb_home"
    home.mkdir()
    # sqlite_storage resolves the home dir via get_gb_home_dir() at call time, so
    # setting the env var is enough to redirect it to a per-test temp dir.
    monkeypatch.setenv("GB_HOME_DIR", str(home))
    yield
