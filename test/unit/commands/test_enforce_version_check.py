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

"""Command-layer version enforcement.

``enforce_version_check`` is where the version policy turns into user-facing behavior:
it echoes to stderr and exits. The floor decision itself is computed by the click-free
``versionutil.evaluate_version_status`` (covered in ``test/unit/utils/test_versionutil.py``);
here we verify the echo/exit/skip behavior and the ``gb version --check-updates`` command.
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from gbcli.commands import common_options
from gbcli.commands.command_version import cli as version_cli
from gbcli.utils.versionutil import VersionCheckResult, VersionStatus


class _FakeCtx:
    """A stand-in for a Click context whose ``exit`` raises, like the real one."""

    def __init__(self):
        self.exited_with = None

    def exit(self, code=0):
        self.exited_with = code
        raise SystemExit(code)


class TestEnforceVersionCheck:
    def _result(self, status, message=""):
        return VersionCheckResult(status=status, message=message)

    def test_below_floor_blocks(self, capsys):
        ctx = _FakeCtx()
        with patch.object(
            common_options,
            "evaluate_version_status",
            return_value=self._result(VersionStatus.BELOW_FLOOR, "must upgrade now"),
        ):
            with pytest.raises(SystemExit):
                common_options.enforce_version_check(ctx, skip_version_check=False)
        assert ctx.exited_with == 1
        assert "must upgrade now" in capsys.readouterr().err

    def test_outdated_warns_but_proceeds(self, capsys):
        ctx = _FakeCtx()
        with patch.object(
            common_options,
            "evaluate_version_status",
            return_value=self._result(VersionStatus.OUTDATED_WARN, "newer available"),
        ):
            common_options.enforce_version_check(ctx, skip_version_check=False)
        assert ctx.exited_with is None  # did not exit
        assert "newer available" in capsys.readouterr().err

    def test_up_to_date_is_silent(self, capsys):
        ctx = _FakeCtx()
        with patch.object(
            common_options,
            "evaluate_version_status",
            return_value=self._result(VersionStatus.UP_TO_DATE),
        ):
            common_options.enforce_version_check(ctx, skip_version_check=False)
        assert ctx.exited_with is None
        assert capsys.readouterr().err == ""

    def test_unknown_never_blocks(self, capsys):
        """A failed/incomplete check (UNKNOWN) proceeds silently — best-effort."""
        ctx = _FakeCtx()
        with patch.object(
            common_options,
            "evaluate_version_status",
            return_value=self._result(VersionStatus.UNKNOWN),
        ):
            common_options.enforce_version_check(ctx, skip_version_check=False)
        assert ctx.exited_with is None
        assert capsys.readouterr().err == ""

    def test_skip_short_circuits(self):
        """With --skip-version-check the evaluator is never even called."""
        ctx = _FakeCtx()
        with patch.object(common_options, "evaluate_version_status") as mock_eval:
            common_options.enforce_version_check(ctx, skip_version_check=True)
        mock_eval.assert_not_called()
        assert ctx.exited_with is None


class TestVersionCheckUpdatesCommand:
    """`gb version --check-updates`: report status, exit 0 unless below floor."""

    def setup_method(self):
        self.runner = CliRunner()

    def _result(self, status, message="", current="1.0.0"):
        return VersionCheckResult(
            status=status, message=message, current_version=current
        )

    def test_below_floor_exits_nonzero(self):
        with patch(
            "gbcli.commands.command_version.versionutil.evaluate_version_status",
            return_value=self._result(
                VersionStatus.BELOW_FLOOR, "unsupported: upgrade"
            ),
        ):
            result = self.runner.invoke(version_cli, ["--check-updates"])
        assert result.exit_code == 1
        assert "unsupported: upgrade" in result.output

    def test_outdated_exits_zero_with_warning(self):
        with patch(
            "gbcli.commands.command_version.versionutil.evaluate_version_status",
            return_value=self._result(VersionStatus.OUTDATED_WARN, "newer available"),
        ):
            result = self.runner.invoke(version_cli, ["--check-updates"])
        assert result.exit_code == 0
        assert "newer available" in result.output

    def test_up_to_date_exits_zero(self):
        with (
            patch(
                "gbcli.commands.command_version.versionutil.evaluate_version_status",
                return_value=self._result(VersionStatus.UP_TO_DATE),
            ),
            patch(
                "gbcli.commands.command_version.get_current_version",
                return_value="1.0.0",
            ),
        ):
            result = self.runner.invoke(version_cli, ["--check-updates"])
        assert result.exit_code == 0
        assert "up to date" in result.output
