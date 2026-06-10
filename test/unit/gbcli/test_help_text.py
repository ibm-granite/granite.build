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

"""Help text and CLI documentation tests.

Verifies that all command groups and their subcommands respond correctly to
``--help`` with non-empty, useful output. ``--help`` exits before any API call, so
these need no credentials and no live server.

Ported from gbcli-upstream test/integration/test_help_text.py. The original drove
the installed ``llmb`` entry point via subprocess; here we exercise the same
``gbcli`` group in-process with Click's CliRunner so the tests run with zero
infrastructure. The runner pins ``GB_ENVIRONMENT=DEV`` so the standalone-mode guards
(which would short-circuit guarded subcommands like ``secret list`` before Click can
print help) do not interfere.
"""

import pytest
from click.testing import CliRunner

from gbcli.cli import gbcli

pytestmark = pytest.mark.standalone

# Pin a non-standalone environment so guarded subcommands still render --help.
_HELP_ENV = {"GB_ENVIRONMENT": "DEV"}

# All top-level command groups
_COMMAND_GROUPS = [
    "artifact",
    "build",
    "space",
    "tag",
    "template",
    "step",
    "auth",
    "secret",
    "admin",
]

# Subcommands to spot-check for help availability
_SUBCOMMANDS = [
    ("artifact", "list"),
    ("artifact", "describe"),
    ("artifact", "push"),
    ("artifact", "download"),
    ("artifact", "archive"),
    ("artifact", "unarchive"),
    ("artifact", "update"),
    ("artifact", "copy"),
    ("artifact", "checksum"),
    ("artifact", "register"),
    ("artifact", "lineage"),
    ("build", "list"),
    ("build", "describe"),
    ("build", "start"),
    ("build", "status"),
    ("build", "log"),
    ("build", "cancel"),
    ("build", "update"),
    ("build", "validate"),
    ("build", "diff"),
    ("build", "lineage"),
    ("build", "monitor"),
    ("build", "init"),
    ("space", "list"),
    ("space", "set"),
    ("tag", "list"),
    ("template", "list"),
    ("template", "describe"),
    ("step", "list"),
    ("step", "describe"),
    ("auth", "login"),
    ("secret", "list"),
    ("secret", "get"),
    ("secret", "create"),
    ("secret", "update"),
    ("secret", "delete"),
    ("admin", "log"),
]


class TestMainHelp:
    def test_main_help(self):
        """Test that the CLI --help succeeds and lists top-level commands."""
        runner = CliRunner()
        result = runner.invoke(gbcli, ["--help"], env=_HELP_ENV)

        assert result.exit_code == 0, f"--help failed: {result.output}"
        assert len(result.output) > 0, "Expected non-empty help output"
        # Verify core command groups appear in top-level help
        for group in ["artifact", "build", "space"]:
            assert (
                group in result.output.lower()
            ), f"Expected '{group}' in main help output. Got: {result.output}"


class TestCommandGroupHelp:
    @pytest.mark.parametrize("group", _COMMAND_GROUPS)
    def test_command_group_help(self, group):
        """Test that each command group responds to --help."""
        runner = CliRunner()
        result = runner.invoke(gbcli, [group, "--help"], env=_HELP_ENV)

        assert (
            result.exit_code == 0
        ), f"'{group} --help' failed with code {result.exit_code}: {result.output}"
        assert len(result.output) > 0, f"Expected non-empty help for '{group}'"


class TestSubcommandHelp:
    @pytest.mark.parametrize("group,subcommand", _SUBCOMMANDS)
    def test_subcommand_help(self, group, subcommand):
        """Test that each subcommand responds to --help with non-empty output."""
        runner = CliRunner()
        result = runner.invoke(gbcli, [group, subcommand, "--help"], env=_HELP_ENV)

        assert (
            result.exit_code == 0
        ), f"'{group} {subcommand} --help' failed: {result.output}"
        assert len(result.output) > 0, f"Empty help output for '{group} {subcommand}'"
        # Every subcommand help should mention its own name or usage
        combined = result.output.lower()
        assert (
            "usage" in combined or subcommand in combined
        ), f"Expected 'Usage' or '{subcommand}' in help output. Got: {result.output}"
