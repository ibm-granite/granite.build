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

import os

import pytest
from libgbtest.mode import is_mock_mode

from gbserver.github.myghapi import MyGHApi
from gbserver.types.constants import DEFAULT_GH_API_ENDPOINT


@pytest.mark.ibm
def test_branch_exists():
    # This test calls the real github.ibm.com API. In mock mode GITHUB_TOKEN is a
    # placeholder, so the call can't authenticate; it requires a live connection.
    if is_mock_mode():
        pytest.skip(reason="requires a live GitHub connection and a real token")
    mygit = MyGHApi(
        token=os.getenv("GITHUB_TOKEN"), owner="granite-dot-build", repo="granite.build"
    )
    main_exists = mygit.is_branch_present("main")
    assert main_exists, "main branch should have been found to exist"
    foobar_exists = mygit.is_branch_present("foobar")
    assert not foobar_exists, "foobar branch should have been found NOT to exist"
