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

"""Guard the gb CLI's startup cost.

Every ``gb`` invocation imports the command module it dispatches to (and
``gb --help`` imports all of them), so a heavy third-party import at module
level in anything they reach adds to every command. These packages are each
only needed by a few subcommands and must be imported lazily at their use site.

Checks module presence in a fresh interpreter rather than timing, so it is
deterministic in CI.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.standalone

# Packages that cost ~50-450ms each to import and that no command needs at startup.
HEAVY_PACKAGES = [
    "dateparser",
    "fastapi",
    "git",
    "numpy",
    "openai",
    "pandas",
    "portalocker",
    "redis",
    "starlette",
]

COMMANDS_DIR = Path(__file__).parents[3] / "src" / "gbcli" / "commands"

_PROBE = """
import importlib, json, sys
for name in sys.argv[2:]:
    importlib.import_module(name)
heavy = json.loads(sys.argv[1])
print(json.dumps(sorted(m for m in heavy if m in sys.modules)))
"""


def _command_modules() -> list[str]:
    return sorted(f"gbcli.commands.{p.stem}" for p in COMMANDS_DIR.glob("command_*.py"))


def test_command_modules_do_not_import_heavy_packages():
    modules = ["gbcli.cli", "gbcli.client", *_command_modules()]
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(HEAVY_PACKAGES), *modules],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        f"importing the gb command modules loaded {loaded}; import them lazily "
        "where they are used (run `python -X importtime -c 'import gbcli.commands."
        "command_build'` to find the importer)"
    )
