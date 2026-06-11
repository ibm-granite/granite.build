import pytest
from libgbtest.utils import check_test_config

from gbcommon.types.constants import DEFAULT_GH_DOMAIN
from gbcommon.uri.git import GitURI
from gbserver.types.constants import SPACE_REPO_CONFIG_BRANCH_NAME


def test_space_config_uris_without_config_branch():
    """Cases that do NOT append an ``@<branch>`` suffix.

    These are deterministic regardless of GitHub access: with no token (e.g.
    STANDALONE) ``get_gb_space_config_uri`` never appends a branch, and with a
    token these branches are confirmed absent — so the expected output is the
    same either way.  No cloud config required (non-ibm).
    """
    # Branch name that does not exist -> no @branch suffix.
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/granite.build"
    cfg_uri = GitURI.get_gb_space_config_uri(uri=uri, config_branch_name="notexists")
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/granite.build.git"
    assert cfg_uri == expected

    # Repo without a gbspace-config branch, with a fragment preserved.
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/granite.build#subdirectory=./foo"
    cfg_uri = GitURI.get_gb_space_config_uri(uri=uri)
    expected = (
        f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/granite.build.git"
        "#subdirectory=./foo"
    )
    assert cfg_uri == expected


@pytest.mark.ibm
def test_space_config_uris_with_config_branch():
    """Cases that append ``@gbspace-config`` when the branch is detected.

    Needs a GitHub token + live API access: ``get_gb_space_config_uri`` only
    appends the branch when it can confirm the branch exists, so these run only
    with cloud config (ibm).
    """
    check_test_config()

    # gbspace-public has a gbspace-config branch -> @branch appended.
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gbspace-public"
    config_branch_name = "gbspace-config"
    cfg_uri = GitURI.get_gb_space_config_uri(
        uri=uri, config_branch_name=config_branch_name
    )
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gbspace-public.git@{config_branch_name}"
    assert cfg_uri == expected

    # gb-test has a gbspace-config branch -> @branch appended.
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test"
    config_branch_name = "gbspace-config"
    cfg_uri = GitURI.get_gb_space_config_uri(
        uri=uri, config_branch_name=config_branch_name
    )
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test.git@{config_branch_name}"
    assert cfg_uri == expected

    # With gbspace-config branch (default) and a fragment.
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test#subdirectory=./foo"
    cfg_uri = GitURI.get_gb_space_config_uri(uri=uri)
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test.git@{SPACE_REPO_CONFIG_BRANCH_NAME}#subdirectory=./foo"
    assert cfg_uri == expected

    # With gbspace-config branch (default) and an empty subdir fragment.
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test#subdirectory="
    cfg_uri = GitURI.get_gb_space_config_uri(uri=uri)
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test.git@{SPACE_REPO_CONFIG_BRANCH_NAME}#subdirectory="
    assert cfg_uri == expected


def test_custom_step_uri():
    uri = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test.git@branch"
    gen_URI = GitURI.get_uri(uri)
    gen_uri = GitURI.get_uristr(gen_URI)
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test.git@branch"
    assert gen_uri == expected

    uri = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test#subdirectory=./foo"
    gen_URI = GitURI.get_uri(uri)
    gen_uri = GitURI.get_uristr(gen_URI)
    expected = (
        f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test#subdirectory=./foo"
    )
    assert gen_uri == expected

    uri = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test#subdirectory="
    gen_URI = GitURI.get_uri(uri)
    gen_uri = GitURI.get_uristr(gen_URI)
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test#subdirectory="
    assert gen_uri == expected
