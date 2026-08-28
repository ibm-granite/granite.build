"""Contains a real import-time side effect in the ``gbcli``/``gbserver`` stack.

Importing ``gbcli.utils.log_query`` (as ``test_gbcli_reader.py`` does, via
``monkeypatch.setattr("gbcli.utils.log_query.run_logquery", ...)``) transitively
imports ``gbserver.types.constants``, which calls ``load_dotenv(override=False)``
at module import time. That is a genuine third-party side effect, not code in
this repository: on a machine whose ``.env`` sets ``AUTOTUNEX_JOB_BACKEND=llmb``
(as this project's own does once the llmb backend is configured — see
``docs/superpowers/specs/2026-08-06-job-launch-runner-design.md``), the first
such import mutates the real process ``os.environ`` directly, bypassing
``pytest``'s ``monkeypatch`` undo entirely. Discovered while writing this
directory's tests: without this fixture, that one import leaked
``AUTOTUNEX_JOB_BACKEND=llmb`` into every later test in the same session that
builds an unpinned ``Settings()`` (``tests/conftest.py``'s ``make_settings``
does not pin ``job_backend``), breaking several dozen unrelated tests
elsewhere in the suite depending on run order.

Snapshotting and restoring ``os.environ`` around every test in this directory
contains the leak here rather than papering over it in ``tests/conftest.py``,
which this task does not otherwise touch.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _restore_environ() -> Iterator[None]:
    """Undo any process-``os.environ`` mutation a test in this directory causes."""
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)
