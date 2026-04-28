import os

import pytest

from gbserver.github.myghapi import MyGHApi
from gbserver.types.constants import DEFAULT_GH_API_ENDPOINT


@pytest.mark.ibm
def test_branch_exists():
    mygit = MyGHApi(
        token=os.getenv("GITHUB_TOKEN"), owner="granite-dot-build", repo="gbserver"
    )
    main_exists = mygit.is_branch_present("main")
    assert main_exists, "main branch should have been found to exist"
    foobar_exists = mygit.is_branch_present("foobar")
    assert not foobar_exists, "foobar branch should have been found NOT to exist"
