# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""A bash build whose hf:// input names one file in the repo.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below. The step checks that it is bound at the downloaded
file itself.
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

# These run the Bash environment in-process, which refuses to start outside
# STANDALONE; the test suite runs as GB_ENVIRONMENT=DEV.
pytestmark = [
    pytest.mark.standalone,
    pytest.mark.usefixtures("allow_bash_environment"),
]


# Extended/nightly test: the fixture performs a REAL HuggingFace pull, so it
# can't run in the fast/mock suites; it runs under `make extended-tests`.
@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_local")
class TestBuildRunnerPathInRepo(AbstractYamlBuildRunnerTest):
    """Bash build bound at a single file pulled from an hf:// repo."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "path-in-repo"
