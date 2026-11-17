#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the gbcli CLI entry point.

Ported from gbcli-upstream test/unit_tests/test_cli.py. Fully mocked / in-process
via Click's CliRunner — no IBM infrastructure required.
"""

import pytest
from click.testing import CliRunner

from gbcli.cli import gbcli

pytestmark = pytest.mark.standalone


class TestCli:
    def test__cli_invoke_help(self):
        """Test CLI invokation with no option"""
        runner = CliRunner()
        result = runner.invoke(gbcli, ["--help"])
        assert result.exit_code == 0

    def test__cli_invoke_wrong_option(self):
        """Test CLI invokation with a wrong command"""
        runner = CliRunner()
        result = runner.invoke(gbcli, ["notavalidcommand"])
        assert result.exit_code == 2
