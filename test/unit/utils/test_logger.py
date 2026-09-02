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

"""Regression tests for gbserver.utils.logger stream binding.

Issue #315: a bare ``logging.StreamHandler()`` captures ``sys.stderr`` at
construction and holds it forever. The ``gbserver`` root group reconfigures
logging inside ``click.testing.CliRunner.invoke()``, where ``sys.stderr`` is an
in-memory buffer that CliRunner later closes / garbage-collects. A captured
reference lets a cross-boundary log write (or CliRunner's own ``getvalue()``)
land on a closed file, producing flaky ``ValueError: I/O operation on closed
file`` failures under large parallel test runs. The console handler must resolve
the *current* ``sys.stderr`` at emit time rather than pin a transient stream.
"""

import logging
import sys

import click
from click.testing import CliRunner

from gbserver.utils import logger as logger_mod


def _console_handler() -> logging.StreamHandler:
    """Return the root logger's console StreamHandler (not a FileHandler)."""
    handlers = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert handlers, "expected a console StreamHandler on the root logger"
    return handlers[0]


def test_console_handler_does_not_retain_clirunner_captured_stream():
    """Reconfiguring logging inside a CliRunner isolation must not pin its buffer.

    On the buggy implementation the handler captures ``sys.stderr`` (CliRunner's
    transient buffer) at construction and keeps pointing at it after the
    isolation exits and the buffer is abandoned. The fix makes the handler track
    the live ``sys.stderr``.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        runner = CliRunner()
        with runner.isolation():
            # Mimic the gbserver root group calling configure_logging() while
            # CliRunner has swapped sys.stderr for its in-memory buffer.
            logger_mod.configure_logging(skip_if_already_configured=False)
            captured_stream = sys.stderr
            handler = _console_handler()
            # While inside the isolation the handler should target the live
            # (captured) stream so log output is still capturable.
            assert handler.stream is captured_stream

        # After the isolation exits, CliRunner has restored sys.stderr and will
        # close/GC its buffer. The handler must no longer reference it.
        assert handler.stream is not captured_stream
        assert handler.stream is sys.stderr
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_log_output_is_captured_during_clirunner_invoke():
    """A record logged during invoke() must still land in ``result.output``.

    Guards against a naive "fix" that pins the handler to ``sys.__stderr__``:
    that would stop the flaky-close failure but silently break log capture,
    since CliRunner only redirects ``sys.stdout``/``sys.stderr``.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        logger_mod.configure_logging(level="INFO", skip_if_already_configured=False)
        log = logging.getLogger("gbserver.test.capture")

        @click.command()
        def emit():
            log.warning("captured-during-invoke")

        result = CliRunner().invoke(emit, [])

        assert result.exit_code == 0, result.output
        assert "captured-during-invoke" in result.output
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
