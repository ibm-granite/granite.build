# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Generic CLI-driven buildtest runner.

Invocation::

    pytest --buildtest-yaml=<path> test/integration/test_yaml_runner.py

The path's parent directory is used as the spec dir, so build.yaml is expected
alongside buildtest.yaml.  Without the flag, both test methods skip with a
clear reason — so this module can sit in normal collection without breaking
anything.

For ergonomic invocation, see scripts/run-buildtest.sh.
"""

from pathlib import Path

import pytest
from lib.buildwatcher.buildtest import AbstractYamlBuildRunnerTest

pytestmark = pytest.mark.ibm


class TestYamlRunnerCli(AbstractYamlBuildRunnerTest):

    @pytest.fixture(autouse=True)
    def _capture_yaml_path(self, request):
        """Pull --buildtest-yaml from pytest config into self before the test runs.

        Skips the test when the flag wasn't passed so that running this module
        without the flag (e.g. during full collection) doesn't error out.
        """
        value = request.config.getoption("--buildtest-yaml")
        if not value:
            pytest.skip(
                "requires --buildtest-yaml=<path>; "
                "use scripts/run-buildtest.sh or pass the flag directly"
            )
        path = Path(value).resolve()
        assert path.is_file(), f"--buildtest-yaml not a file: {path}"
        self._yaml_path = path

    def _get_yaml_spec_dir(self) -> Path:
        return self._yaml_path.parent
