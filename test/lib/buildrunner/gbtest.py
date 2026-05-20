# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""``gbtest`` CLI: run a single buildtest.yaml via the pytest harness.

Installed as a console script via ``[project.scripts]`` in pyproject.toml::

    gbtest path/to/buildtest.yaml [extra pytest args...]

``main()`` invokes ``pytest.main(...)`` against the sibling
``gbtest_runner.py`` (which defines ``TestYamlRunnerCli``) with
``--buildtest-yaml`` set to the supplied path.  Any extra args after the
YAML path are forwarded to pytest (e.g. ``-k test_runner_cancellation`` to
pick a single test method, ``-vv`` for more verbose output).

This module is intentionally kept free of test-infrastructure imports so
that ``main()`` can run pytest in-process without prematurely loading
``lib.test_utils`` (which would freeze ``GBSERVER_GITHUB_TOKEN`` to its
pre-secrets value before ``pytest_sessionstart`` fires).  The heavy imports
live in ``gbtest_runner.py`` and are only triggered when pytest collects
that file — i.e. AFTER sessionstart.
"""

import sys
from pathlib import Path

import pytest


def main() -> int:
    """Entry point for the ``gbtest`` console script.

    Returns:
        Exit code from pytest (0 for success, non-zero for failure).
    """
    if len(sys.argv) < 2:
        sys.stderr.write(
            "Usage: gbtest path/to/buildtest.yaml [extra pytest args...]\n"
        )
        return 2
    yaml_path = Path(sys.argv[1]).resolve()
    if not yaml_path.is_file():
        sys.stderr.write(f"gbtest: not a file: {yaml_path}\n")
        return 1
    extra = sys.argv[2:]
    runner_module = Path(__file__).resolve().parent / "gbtest_runner.py"
    return pytest.main(
        ["-s", str(runner_module), f"--buildtest-yaml={yaml_path}", *extra]
    )


if __name__ == "__main__":
    sys.exit(main())
