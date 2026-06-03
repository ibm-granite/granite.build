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

"""``gb space list --refresh`` is blocked in standalone mode.

In standalone mode spaces are always fetched fresh from the local gbserver and the local
cache/profile that ``--refresh`` repopulates is never used (writing it would also corrupt
~/.gbcli/config). The flag should warn and exit non-zero instead.
"""

import importlib

from click.testing import CliRunner

# Assert against result.output (combined stdout+stderr): CliRunner mixes the streams by
# default on Click 8.1.x, where accessing result.stderr raises.
WARNING_FRAGMENT = "'--refresh' is not supported in standalone mode"


def _space_cli():
    module = importlib.import_module("gbcli.commands.command_space")
    return module.cli


class TestSpaceRefreshStandalone:
    def test_refresh_blocked_in_standalone(self):
        """`space list --all --refresh` should warn and exit non-zero in standalone."""
        runner = CliRunner()
        result = runner.invoke(
            _space_cli(),
            ["list", "--all", "--refresh", "--skip-version-check"],
            env={"GB_ENVIRONMENT": "STANDALONE"},
        )

        assert result.exit_code != 0, (
            f"'space list --all --refresh' should exit non-zero in standalone mode, "
            f"got {result.exit_code}"
        )
        assert (
            WARNING_FRAGMENT in result.output
        ), f"expected --refresh standalone warning, got: {result.output!r}"

    def test_refresh_not_blocked_outside_standalone(self):
        """Outside standalone the --refresh guard must not fire (no warning)."""
        runner = CliRunner()
        # The command may fail later (no gbserver/credentials), but it must not emit the
        # standalone --refresh warning.
        result = runner.invoke(
            _space_cli(),
            ["list", "--all", "--refresh", "--skip-version-check"],
            env={"GB_ENVIRONMENT": "PROD"},
        )

        assert WARNING_FRAGMENT not in result.output, (
            f"--refresh standalone warning should not appear outside standalone mode, "
            f"got: {result.output!r}"
        )
