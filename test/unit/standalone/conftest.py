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

"""Isolation for standalone-mode tests.

Several tests here flip GB_ENVIRONMENT to STANDALONE and reload
``gbserver.types.constants``. The is_standalone() block in that module writes
standalone defaults into ``os.environ`` via ``setdefault`` (e.g.
GBSERVER_LINEAGE_PROVIDER=none, GBSERVER_METADATA_STORAGE=sqlite). Those writes
are NOT tracked by monkeypatch, so without explicit cleanup they leak into the
process environment and poison later tests on the same xdist worker (e.g. the
lineage/artifact API tests that then get a NoopLineageStore).

This autouse fixture snapshots the relevant env vars and restores them — and
reloads constants back to the pre-test state — after every test in this package.
"""

import importlib
import os

import pytest

# Env vars the standalone setdefault block may introduce (see
# gbserver.types.constants is_standalone()).
_STANDALONE_LEAK_KEYS = (
    "GB_ENVIRONMENT",
    "GBSERVER_METADATA_STORAGE",
    "GBSERVER_DEFAULT_BUILDRUNNER_TYPE",
    "GBSERVER_PROCEED_WITHOUT_SECRETS",
    "GBSERVER_LINEAGE_PROVIDER",
    "GBSERVER_USER_SECRET_MANAGER",
)


@pytest.fixture(autouse=True)
def _restore_standalone_env():
    snapshot = {k: os.environ.get(k) for k in _STANDALONE_LEAK_KEYS}
    try:
        yield
    finally:
        changed = False
        for key, original in snapshot.items():
            if os.environ.get(key) != original:
                changed = True
                if original is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original
        if changed:
            # Re-evaluate constants under the restored environment so module-level
            # values captured at import time (e.g. GBSERVER_LINEAGE_PROVIDER) match.
            from gbserver.types import constants

            importlib.reload(constants)

        # Always reset the lineage-store singleton. A standalone test may have
        # cached a NoopLineageStore (whether via a leaked env var or a direct
        # monkeypatch of GBSERVER_LINEAGE_PROVIDER); that singleton would otherwise
        # persist process-wide and make later lineage/artifact tests on the same
        # worker see an empty (noop) store.
        from gbserver.lineage.jobstats import reset_lineage_store

        reset_lineage_store()
