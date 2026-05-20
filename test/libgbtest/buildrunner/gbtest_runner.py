# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pytest test class invoked by the ``gbtest`` CLI.

This module is loaded only when pytest collects it via the explicit path
passed by ``gbtest.py``'s ``main()``.  Keeping the heavy
``AbstractYamlBuildRunnerTest`` import (and its transitive load of
``lib.test_utils``) here — rather than in the CLI entry module — ensures
those imports happen AFTER ``pytest_sessionstart`` runs, so
``GBSERVER_GITHUB_TOKEN`` is captured fresh from the loaded secrets.

Pytest collects test classes from any file given by explicit path regardless
of the ``python_files`` pattern, so the unconventional filename is fine.
``test/libgbtest/`` is not in ``testpaths`` (see pyproject.toml), so a plain
``pytest`` run won't pick this class up as a side effect.
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import AbstractYamlBuildRunnerTest


class TestYamlRunnerCli(AbstractYamlBuildRunnerTest):
    """Generic YAML-driven test class invoked by the gbtest CLI.

    Without ``--buildtest-yaml=<path>``, both inherited test methods skip
    with a clear reason — so accidental collection (e.g. an explicit pytest
    invocation against this file without the flag) doesn't error out.
    """

    @pytest.fixture(autouse=True)
    def _capture_yaml_path(self, request):
        """Pull --buildtest-yaml from pytest config into self before the test runs."""
        value = request.config.getoption("--buildtest-yaml")
        if not value:
            pytest.skip(
                "requires --buildtest-yaml=<path>; use the gbtest CLI or "
                "pass the flag directly"
            )
        path = Path(value).resolve()
        assert path.is_file(), f"--buildtest-yaml not a file: {path}"
        self._yaml_path = path

    def _get_yaml_spec_dir(self) -> Path:
        return self._yaml_path.parent
