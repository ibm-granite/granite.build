import pytest
from libgbtest.utils import check_test_config

from gbcommon.uri.git import GitURI
from gbserver.types.constants import SPACE_REPO_CONFIG_BRANCH_NAME
from gbserver.types.constants_base import DEFAULT_GH_DOMAIN

pytestmark = pytest.mark.ibm


def test_space_config_uris():
    check_test_config()

    # Without gbspace-config branch
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/granite.build"
    config_branch_name = "notexists"
    cfg_uri = GitURI.get_gb_space_config_uri(
        uri=uri, config_branch_name=config_branch_name
    )
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/granite.build.git"
    assert cfg_uri == expected

    # With gbspace-config branch
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gbspace-public"
    config_branch_name = "gbspace-config"
    cfg_uri = GitURI.get_gb_space_config_uri(
        uri=uri, config_branch_name=config_branch_name
    )
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gbspace-public.git@{config_branch_name}"
    assert cfg_uri == expected

    # With gbspace-config branch
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test"
    config_branch_name = "gbspace-config"
    cfg_uri = GitURI.get_gb_space_config_uri(
        uri=uri, config_branch_name=config_branch_name
    )
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test.git@{config_branch_name}"
    assert cfg_uri == expected

    # Without  a branch and with a fragment (not expected, but just in case)
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/granite.build#subdirectory=./foo"
    cfg_uri = GitURI.get_gb_space_config_uri(uri=uri)
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/granite.build.git#subdirectory=./foo"
    assert cfg_uri == expected

    # With gbspace-config branch and with a fragment  (not expected, but just in case)
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test#subdirectory=./foo"
    cfg_uri = GitURI.get_gb_space_config_uri(uri=uri)
    # expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test.git@{SPACE_REPO_CONFIG_BRANCH_NAME}"
    expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test.git@{SPACE_REPO_CONFIG_BRANCH_NAME}#subdirectory=./foo"
    assert cfg_uri == expected

    # With gbspace-config branch and with an empty subdir fragment  (not expected, but just in case)
    uri = f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test#subdirectory="
    cfg_uri = GitURI.get_gb_space_config_uri(uri=uri)
    # expected = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test.git@{SPACE_REPO_CONFIG_BRANCH_NAME}"
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
