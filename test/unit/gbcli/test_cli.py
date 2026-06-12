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
