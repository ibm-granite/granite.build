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

"""Verify that cloud-only gbcli commands are guarded in standalone mode.

The ``admin`` and ``secret`` command groups depend on cloud-only backends and cannot
work when ``GB_ENVIRONMENT=STANDALONE``. Invoking any of their subcommands in standalone
mode should warn and exit non-zero *before* hitting an auth/network error, while plain
``--help`` should still work.
"""

import importlib

import pytest
from click.testing import CliRunner

STANDALONE_ENV = {"GB_ENVIRONMENT": "STANDALONE"}
WARNING_FRAGMENT = "not supported in standalone mode"

# (module path, group name, a representative subcommand invocation)
GUARDED_COMMANDS = [
    ("gbcli.commands.command_secret", "secret", ["list"]),
    ("gbcli.commands.command_admin", "admin", ["log", "gbserver-rest-server"]),
    ("gbcli.commands.command_template", "template", ["list"]),
    ("gbcli.commands.command_step", "step", ["list"]),
    ("gbcli.commands.command_artifact", "artifact", ["list"]),
    ("gbcli.commands.command_model", "model", ["list"]),
]


def _load_cli(module_path: str):
    """Import a command module fresh and return its ``cli`` group."""
    module = importlib.import_module(module_path)
    return module.cli


class TestGbcliStandaloneGuards:
    """The admin and secret groups must refuse to run subcommands in standalone mode."""

    @pytest.mark.parametrize("module_path,group_name,subcommand", GUARDED_COMMANDS)
    def test_subcommand_blocked_in_standalone(
        self, module_path: str, group_name: str, subcommand: list
    ):
        """A guarded subcommand should warn and exit non-zero in standalone mode."""
        cli = _load_cli(module_path)
        runner = CliRunner()
        result = runner.invoke(cli, subcommand, env=STANDALONE_ENV)

        assert result.exit_code != 0, (
            f"'{group_name} {' '.join(subcommand)}' should exit non-zero in standalone "
            f"mode, got {result.exit_code}"
        )
        # Assert against result.output (combined stdout+stderr) rather than
        # result.stderr: CliRunner mixes the streams by default on Click 8.1.x, where
        # accessing result.stderr raises "stderr not separately captured".
        assert WARNING_FRAGMENT in result.output, (
            f"expected standalone warning for '{group_name}', "
            f"got: {result.output!r}"
        )

    @pytest.mark.parametrize("module_path,group_name,subcommand", GUARDED_COMMANDS)
    def test_group_help_works_in_standalone(
        self, module_path: str, group_name: str, subcommand: list
    ):
        """Group-level --help must still work in standalone mode (no guard)."""
        cli = _load_cli(module_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"], env=STANDALONE_ENV)

        assert result.exit_code == 0, (
            f"'{group_name} --help' should succeed in standalone mode, "
            f"got {result.exit_code}: {result.output}"
        )
        assert WARNING_FRAGMENT not in result.output

    @pytest.mark.parametrize("module_path,group_name,subcommand", GUARDED_COMMANDS)
    def test_subcommand_not_blocked_outside_standalone(
        self, module_path: str, group_name: str, subcommand: list
    ):
        """Outside standalone mode the guard must not fire (no warning)."""
        cli = _load_cli(module_path)
        runner = CliRunner()
        # Force a non-standalone environment so the guard is a no-op. The command may
        # still fail later (no credentials), but it must not emit the standalone warning.
        result = runner.invoke(cli, subcommand, env={"GB_ENVIRONMENT": "PROD"})

        assert WARNING_FRAGMENT not in result.output, (
            f"standalone warning should not appear for '{group_name}' outside "
            f"standalone mode, got: {result.output!r}"
        )
