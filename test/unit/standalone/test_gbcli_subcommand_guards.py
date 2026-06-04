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

"""Per-subcommand / per-flag standalone guards in the ``gb`` CLI.

Unlike the whole-group guards (admin, secret, template, ...), these block individual
subcommands of otherwise-supported groups, or a specific flag:

* ``space set``                  — whole subcommand
* ``build lineage``              — whole subcommand
* ``build notification``         — whole subcommand
* ``build init --from-template`` — flag only (``--from-build`` stays allowed)

All assert against ``result.output`` (combined stdout+stderr): CliRunner mixes the
streams by default on Click 8.1.x, where accessing ``result.stderr`` raises.
"""

import importlib

import pytest
from click.testing import CliRunner

WARNING_FRAGMENT = "not supported in standalone mode"

# (module path, argv) for whole-subcommand blocks.
GUARDED_SUBCOMMANDS = [
    ("gbcli.commands.command_space", ["set", "myspace", "--skip-version-check"]),
    ("gbcli.commands.command_build", ["lineage", "someid", "--skip-version-check"]),
    ("gbcli.commands.command_build", ["notification", "on", "--skip-version-check"]),
]


def _cli(module_path: str):
    return importlib.import_module(module_path).cli


class TestSubcommandStandaloneGuards:
    @pytest.mark.parametrize("module_path,argv", GUARDED_SUBCOMMANDS)
    def test_blocked_in_standalone(self, module_path: str, argv: list):
        result = CliRunner().invoke(
            _cli(module_path), argv, env={"GB_ENVIRONMENT": "STANDALONE"}
        )
        assert result.exit_code != 0, f"{argv} should exit non-zero in standalone"
        assert (
            WARNING_FRAGMENT in result.output
        ), f"expected standalone warning for {argv}, got: {result.output!r}"

    @pytest.mark.parametrize("module_path,argv", GUARDED_SUBCOMMANDS)
    def test_not_blocked_outside_standalone(self, module_path: str, argv: list):
        result = CliRunner().invoke(
            _cli(module_path), argv, env={"GB_ENVIRONMENT": "PROD"}
        )
        assert WARNING_FRAGMENT not in result.output, (
            f"standalone warning should not appear for {argv} outside standalone, "
            f"got: {result.output!r}"
        )


class TestBuildInitFromTemplateGuard:
    """`build init --from-template` is blocked in standalone; --from-build is not."""

    def test_from_template_blocked_in_standalone(self):
        result = CliRunner().invoke(
            _cli("gbcli.commands.command_build"),
            ["init", "myb", "--from-template", "hello", "--skip-version-check"],
            env={"GB_ENVIRONMENT": "STANDALONE"},
        )
        assert result.exit_code != 0
        assert "build init --from-template" in result.output

    def test_from_build_not_blocked_in_standalone(self):
        """--from-build must NOT trigger the --from-template standalone guard."""
        result = CliRunner().invoke(
            _cli("gbcli.commands.command_build"),
            [
                "init",
                "myb",
                "--from-build",
                "00000000-0000-0000-0000-000000000000",
                "--skip-version-check",
            ],
            env={"GB_ENVIRONMENT": "STANDALONE"},
        )
        assert "currently not supported in standalone" not in result.output
