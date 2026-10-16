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
