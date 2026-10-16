from unittest.mock import patch

import pytest
from git import Repo
from git.exc import GitCommandError

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


_GB_TEST_SSH = f"git+ssh://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test.git"


@pytest.mark.parametrize(
    "uri",
    [
        f"{_GB_TEST_SSH}@gbspace-config",
        f"{_GB_TEST_SSH}@gbspace-config#subdirectory=./foo",
    ],
)
def test_space_config_uri_leaves_git_ssh_with_branch_unmodified(uri):
    """A git+ssh:// URI that already names its branch is returned as is.

    Converting it again must not probe GitHub: it already has the branch, and
    before the repo name was parsed correctly the probe asked about repo
    ``gb-test.git@gbspace-config``, which 404s and invalidates the build.
    """
    with patch(
        "gbserver.github.myghapi.MyGHApi",
        side_effect=AssertionError("GitHub was queried for a git+ssh URI"),
    ):
        assert GitURI.get_gb_space_config_uri(uri=uri, token="some-token") == uri


# "live" only opts out of the conftest's mock-mode stub of GitURI.__get_config_branch,
# which would otherwise replace the very lookup under test; MyGHApi is patched below,
# so no GitHub call is made in any mode.
@pytest.mark.live("github")
@pytest.mark.parametrize(
    "branch_present, expected",
    [(True, f"{_GB_TEST_SSH}@gbspace-config"), (False, _GB_TEST_SSH)],
)
def test_space_config_uri_looks_up_branch_for_bare_git_ssh(branch_present, expected):
    """A git+ssh:// URI without a branch still gets the config branch looked up.

    The lookup must ask about the bare repo name (no ``.git``), and the branch is
    appended only when it exists.
    """
    with patch("gbserver.github.myghapi.MyGHApi") as myghapi:
        myghapi.return_value.is_branch_present.return_value = branch_present
        cfg_uri = GitURI.get_gb_space_config_uri(
            uri=_GB_TEST_SSH, token="some-token", config_branch_name="gbspace-config"
        )
    assert cfg_uri == expected
    assert myghapi.call_args.kwargs["owner"] == "granite-dot-build"
    assert myghapi.call_args.kwargs["repo"] == "gb-test"


@pytest.mark.ibm
def test_space_config_uris_with_config_branch():
    """Cases that append ``@gbspace-config`` when the branch is detected.

    Needs a GitHub token + live API access: ``get_gb_space_config_uri`` only
    appends the branch when it can confirm the branch exists.  The `ibm` marker
    + the conftest ``_check_test_env`` fixture enforce the cloud-config bundle.
    """
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


def test_clone_retry_removes_partial_clone(tmp_path):
    """A transient clone failure that leaves a partial directory is cleaned up
    before the retry, so the retry doesn't fail on a non-empty destination."""
    dest = tmp_path / "repo"
    attempts = []

    def fake_clone_from(url, path, **kwargs):
        attempts.append(path.exists())
        if len(attempts) == 1:
            path.mkdir()
            (path / "partial").write_text("x")
            raise GitCommandError("clone", 128, "fatal: connection reset by peer")
        return "repo"

    uri = GitURI.__new__(GitURI)
    with (
        patch.object(Repo, "clone_from", side_effect=fake_clone_from),
        patch("tenacity.nap.time.sleep"),
    ):
        assert uri._clone_with_retry("https://example.com/o/r.git", dest) == "repo"
    # The second attempt saw the partial directory already removed.
    assert attempts == [False, False]
