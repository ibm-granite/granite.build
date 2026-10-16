# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Two command-step targets passing a mem:// value between them on local Bash.

`first` runs the generic `command` step to emit `mem_output1_value` and register
it as the `state` of mem:// output `mem_output1`; `second` binds `first.mem_output1`
as an input, reads the verbatim state back, and its command ASSERTS the value
equals `mem_output1_value` (exiting non-zero — failing the target and this test —
if the value did not flow), then re-registers it as its own mem:// output
`mem_output2`. This exercises cross-target output -> input binding over the mem://
(in-memory) assetstore on the in-process Bash environment — the local_bash
equivalent of the BlueVela command-2target test, but for mem:// instead of env://.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)

pytestmark = pytest.mark.standalone


@pytest.mark.xdist_group(name="buildtest_local")
class TestBuildRunner2TargetMemLocal(AbstractYamlBuildRunnerTest):
    """Two command-step targets on local Bash; target 2 binds target 1's mem:// output."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "2target"
