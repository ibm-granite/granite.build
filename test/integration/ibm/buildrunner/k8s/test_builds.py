# Copyright Granite.Build Authors
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

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only, requires_k8s_image_tags

pytestmark = pytest.mark.ibm


@extended_testing_only
@requires_k8s_image_tags
class TestDiGiT_SFTFull_FMEval(AbstractYamlBuildRunnerTest):

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "builds/DiGiT_SFTFull_FMEval"

    @pytest.mark.timeout(
        7200
    )  # 120 minutes, matches the test-defined timeout in buildtest.yaml
    def test_runner(self):
        super().test_runner()
