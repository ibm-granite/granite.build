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

"""Verify all gbcli and gbserver entry points resolve correctly."""

import subprocess

import pytest

pytestmark = pytest.mark.g4os

GBCLI_ENTRY_POINTS = ["gb", "gbcli", "llmbuild", "llmb", "lamb"]


class TestGbcliEntryPoints:
    """Verify that all gbcli entry points are installed and functional."""

    @pytest.mark.parametrize("cmd", GBCLI_ENTRY_POINTS)
    def test_entry_point_help(self, cmd: str):
        """Each entry point should print help and exit 0."""
        result = subprocess.run(
            [cmd, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert (
            result.returncode == 0
        ), f"{cmd} --help failed with rc={result.returncode}: {result.stderr}"
        assert "usage:" in result.stdout.lower()

    def test_all_entry_points_same_commands(self):
        """All entry points should expose the same set of commands."""
        outputs = {}
        for cmd in GBCLI_ENTRY_POINTS:
            result = subprocess.run(
                [cmd, "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0
            # Normalize: strip the Usage line (contains the command name)
            lines = result.stdout.splitlines()
            body = "\n".join(line for line in lines if not line.startswith("Usage:"))
            outputs[cmd] = body

        reference = outputs[GBCLI_ENTRY_POINTS[0]]
        for cmd in GBCLI_ENTRY_POINTS[1:]:
            assert (
                outputs[cmd] == reference
            ), f"{cmd} --help output differs from {GBCLI_ENTRY_POINTS[0]}"

    def test_gbserver_entry_point(self):
        """gbserver entry point should also be functional."""
        result = subprocess.run(
            ["gbserver", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"gbserver --help failed: {result.stderr}"
        assert "usage:" in result.stdout.lower()
