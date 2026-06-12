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

"""Unit tests to verify clean JSON output for --format json commands.

These tests ensure that when --format json is used, the output is pure JSON
without warnings, validation messages, or other text pollution.

Ported from gbcli-upstream test/unit_tests/commands/test_json_output_clean.py.
``click.echo`` is mocked — no IBM infrastructure required.
"""

import unittest
from unittest.mock import patch

import pytest

from gbcli.utils.click_utils import validation_formatting

pytestmark = pytest.mark.standalone


class TestValidationFormattingWithQuiet(unittest.TestCase):
    """Test validation_formatting function respects quiet parameter."""

    def test_validation_formatting_quiet_true_no_output(self):
        """Test that validation_formatting produces no output when quiet=True."""
        callback_args = {
            "build_path": "test.yaml",
            "validations": [
                {
                    "warning": True,
                    "type": "test",
                    "summary": "Test warning",
                    "status_display_text": "⚠️  WARNING #1 (test): Test warning",
                }
            ],
        }

        # Capture output
        with patch("click.echo") as mock_echo:
            validation_formatting(callback_args, verbose_validation=False, quiet=True)
            # Should not call click.echo when quiet=True
            mock_echo.assert_not_called()

    def test_validation_formatting_quiet_false_outputs_warning(self):
        """Test that validation_formatting outputs to stderr when quiet=False."""
        callback_args = {
            "build_path": "test.yaml",
            "validations": [
                {
                    "warning": True,
                    "type": "test",
                    "summary": "Test warning",
                    "status_display_text": "⚠️  WARNING #1 (test): Test warning",
                }
            ],
        }

        # Capture output
        with patch("click.echo") as mock_echo:
            validation_formatting(callback_args, verbose_validation=False, quiet=False)
            # Should call click.echo with err=True
            calls = mock_echo.call_args_list
            # Check that at least one call was made with err=True
            assert any(
                call.kwargs.get("err") is True for call in calls
            ), f"Expected err=True in calls, got: {calls}"

    def test_validation_formatting_with_errors_quiet_true(self):
        """Test that validation_formatting with errors outputs one message when quiet=True."""
        callback_args = {
            "build_path": "test.yaml",
            "validations": [
                {
                    "error": True,
                    "type": "test",
                    "summary": "Test error",
                    "status_display_text": "❌ ERROR #1 (test): Test error",
                }
            ],
        }

        # Capture output
        with patch("click.echo") as mock_echo:
            validation_formatting(callback_args, verbose_validation=False, quiet=True)
            # Should output error message to stderr even when quiet=True so user knows validation failed
            mock_echo.assert_called_once()
            # Verify it goes to stderr
            call_args = mock_echo.call_args
            assert call_args[1].get("err") is True, "Error message should go to stderr"


class TestValidationFormattingToStderr(unittest.TestCase):
    """Test that validation output goes to stderr."""

    def test_validation_formatting_to_stderr(self):
        """Test that validation output goes to stderr with err=True."""
        callback_args = {
            "build_path": "test.yaml",
            "validations": [
                {
                    "warning": True,
                    "type": "test",
                    "summary": "Test warning",
                    "status_display_text": "⚠️  WARNING #1 (test): Test warning",
                }
            ],
        }

        # When quiet=False, output should use err=True
        with patch("click.echo") as mock_echo:
            validation_formatting(callback_args, verbose_validation=False, quiet=False)

            # At least one call should have err=True
            found_err_true = False
            for call in mock_echo.call_args_list:
                if call.kwargs.get("err") is True:
                    found_err_true = True
                    break

            assert found_err_true, "Expected at least one click.echo call with err=True"

    def test_validation_formatting_verbose_to_stderr(self):
        """Test that verbose validation output goes to stderr with err=True."""
        callback_args = {
            "build_path": "test.yaml",
            "validations": [
                {
                    "warning": True,
                    "type": "test",
                    "summary": "Test warning",
                    "detail": "This is a detail",
                    "solution": "This is a solution",
                    "status_display_text": "⚠️  WARNING #1 (test): Test warning",
                }
            ],
        }

        # When verbose=True and quiet=False, output should use err=True
        with patch("click.echo") as mock_echo:
            validation_formatting(callback_args, verbose_validation=True, quiet=False)

            # All calls should have err=True
            for call in mock_echo.call_args_list:
                assert call.kwargs.get("err") is True, f"Expected err=True, got: {call}"


if __name__ == "__main__":
    unittest.main()
