# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""YAML-driven equivalent of TestBuildRunner1StepLocal.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.standalone


# Extended/nightly test: the fixture performs a REAL HuggingFace pull + push
# (using the CI HF_TOKEN secret), so it can't run in the fast/mock suites — it is
# excluded from quick-tests via this marker and runs under `make extended-tests`.
@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_local")
class TestBuildRunner1StepLocal(AbstractYamlBuildRunnerTest):
    """Local build flow exercising env:// and hf:// input + output on Bash."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step"
