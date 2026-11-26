# SPDX-License-Identifier: Apache-2.0

"""SSH -> HTTPS fallback and per-host token selection for GitURI clones."""

import base64
import logging
from unittest.mock import patch

import pytest
from git import Repo
from git.exc import GitCommandError

from gbcommon.types.constants import DEFAULT_GH_DOMAIN
from gbcommon.uri.git import GitURI
from gbcommon.uri.space import SpaceURI
from gbserver.utils.git_auth import (
    ENV_VAR_GIT_TRANSPORT,
    GitTransport,
    get_git_transport,
    get_token_candidates,
    git_auth_env,
    host_slug,
)

SSH_REFUSED = (
    "Cloning into '/tmp/x'...\n"
    "ssh: connect to host git.example.com port 22: Connection refused\n"
    "fatal: Could not read from remote repository.\n\n"
    "Please make sure you have the correct access rights\n"
    "and the repository exists.\n"
)
HTTPS_NOT_FOUND = "remote: Repository not found.\nfatal: repository not found"
HTTPS_BRANCH_MISSING = "fatal: Remote branch nope not found in upstream origin"

TOKEN_ENV_NAMES = (
    "GBSERVER_GITHUB_TOKEN",
    "GITHUB_TOKEN",
    "GBSERVER_GIT_TRANSPORT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in TOKEN_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for host in ("github.com", DEFAULT_GH_DOMAIN, "git.example.com"):
        monkeypatch.delenv(f"GBSERVER_GIT_TOKEN_{host_slug(host)}", raising=False)
    monkeypatch.delattr(SpaceURI._thread_local, "space_secrets", raising=False)
    monkeypatch.delattr(GitURI._thread_local, "repo_cache", raising=False)
    # No backoff sleeps between retried attempts.
    with patch("tenacity.nap.time.sleep"):
        yield


def _auth_token(env):
    """The token a clone's env sends, or None for an anonymous clone."""
    if not env or "GIT_CONFIG_COUNT" not in env:
        return None
    n = int(env["GIT_CONFIG_COUNT"]) - 1
    header = env[f"GIT_CONFIG_VALUE_{n}"]
    return base64.b64decode(header.split()[-1]).decode().split(":", 1)[1]


class FakeClone:
    """Records clone attempts; ``outcomes`` maps (url, token) to an error."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def __call__(self, url, path, env=None, **kwargs):
        token = _auth_token(env)
        self.calls.append((url, token))
        stderr = self.outcomes.get((url, token))
        if stderr is not None:
            raise GitCommandError(["git", "clone", url], 128, stderr)
        path.mkdir(parents=True)
        (path / "README").write_text("ok")
        return "repo"


def _ssh_uri(host=DEFAULT_GH_DOMAIN, secrets=None):
    uri = f"git+ssh://{host}/org/assets.git@dev#subdirectory=steps"
    return GitURI.get_uri(uri, secrets=secrets)


def test_host_slug():
    assert host_slug("git.example.com") == "GIT_EXAMPLE_COM"
    assert host_slug("GitHub.com") == "GITHUB_COM"


def test_token_candidates_are_per_host(monkeypatch):
    monkeypatch.setenv("GBSERVER_GITHUB_TOKEN", "enterprise-service")
    monkeypatch.setenv("GBSERVER_GIT_TOKEN_GITHUB_COM", "public-service")
    secrets = {
        "GIT_TOKEN_GIT_EXAMPLE_COM": "example-build",
        "GIT_TOKEN_GITHUB_COM": "public-build",
    }
    assert [c.token for c in get_token_candidates(DEFAULT_GH_DOMAIN, secrets)] == [
        "enterprise-service"
    ]
    assert [c.token for c in get_token_candidates("github.com", secrets)] == [
        "public-service",
        "public-build",
    ]
    assert [c.label for c in get_token_candidates("git.example.com", secrets)] == [
        "secret:GIT_TOKEN_GIT_EXAMPLE_COM"
    ]


def test_host_specific_service_token_beats_legacy_name(monkeypatch):
    monkeypatch.setenv("GBSERVER_GITHUB_TOKEN", "legacy")
    monkeypatch.setenv(f"GBSERVER_GIT_TOKEN_{host_slug(DEFAULT_GH_DOMAIN)}", "new")
    assert [c.token for c in get_token_candidates(DEFAULT_GH_DOMAIN)] == ["new"]


def test_token_repr_hides_value():
    (candidate,) = get_token_candidates(
        "github.com", {"GIT_TOKEN_GITHUB_COM": "s3cr3t"}
    )
    assert "s3cr3t" not in repr(candidate)


def test_auth_env_is_scoped_to_host_and_appends():
    env = git_auth_env("github.com", "tok", base_env={"GIT_CONFIG_COUNT": "2"})
    assert env["GIT_CONFIG_COUNT"] == "3"
    assert env["GIT_CONFIG_KEY_2"] == "http.https://github.com/.extraHeader"
    assert _auth_token(env) == "tok"
    assert git_auth_env("github.com", None) == {"GIT_TERMINAL_PROMPT": "0"}


def test_ssh_failure_falls_back_to_https_with_service_token(monkeypatch, caplog):
    """The host refuses ssh, so clone over https and warn."""
    monkeypatch.setenv("GBSERVER_GITHUB_TOKEN", "svc")
    ssh_url = f"git@{DEFAULT_GH_DOMAIN}:org/assets"
    https_url = f"https://{DEFAULT_GH_DOMAIN}/org/assets"
    fake = FakeClone({(ssh_url, None): SSH_REFUSED})
    with (
        patch.object(Repo, "clone_from", side_effect=fake),
        caplog.at_level(logging.WARNING),
    ):
        path = _ssh_uri().get_path_in_repo_from_cache()
    assert path is None  # the fake clone has no "steps" subdirectory
    assert fake.calls == [(ssh_url, None), (https_url, "svc")]
    warning = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "over https (token: service:GBSERVER_GITHUB_TOKEN)" in m
        and "Connection refused" in m
        for m in warning
    )
    assert not any("svc" in m for m in warning)


def test_https_mode_skips_ssh(monkeypatch):
    monkeypatch.setenv("GBSERVER_GIT_TRANSPORT", "https")
    fake = FakeClone({})
    with patch.object(Repo, "clone_from", side_effect=fake):
        _ssh_uri("github.com").get_repo_from_cache()
    assert fake.calls == [("https://github.com/org/assets", None)]


def test_ssh_mode_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("GBSERVER_GIT_TRANSPORT", "ssh")
    ssh_url = f"git@{DEFAULT_GH_DOMAIN}:org/assets"
    fake = FakeClone({(ssh_url, None): SSH_REFUSED})
    with (
        patch.object(Repo, "clone_from", side_effect=fake),
        pytest.raises(GitCommandError),
    ):
        _ssh_uri().get_repo_from_cache()
    assert fake.calls == [(ssh_url, None)]


def test_next_token_on_access_error(monkeypatch):
    """The service token can't see the repo; the build's (user/space) token can."""
    monkeypatch.setenv("GBSERVER_GIT_TRANSPORT", "https")
    monkeypatch.setenv("GBSERVER_GITHUB_TOKEN", "svc")
    secrets = {f"GIT_TOKEN_{host_slug(DEFAULT_GH_DOMAIN)}": "usr"}
    https_url = f"https://{DEFAULT_GH_DOMAIN}/org/assets"
    fake = FakeClone({(https_url, "svc"): HTTPS_NOT_FOUND})
    with patch.object(Repo, "clone_from", side_effect=fake):
        _ssh_uri(secrets=secrets).get_repo_from_cache()
    assert fake.calls == [(https_url, "svc"), (https_url, "usr")]


def test_thread_local_secrets_are_not_used(monkeypatch):
    """A secret left on the thread (e.g. by an earlier request) is never sent."""
    monkeypatch.setenv("GBSERVER_GIT_TRANSPORT", "https")
    SpaceURI.set_baseuris(
        base_uris=[], space_secrets={f"GIT_TOKEN_{host_slug(DEFAULT_GH_DOMAIN)}": "old"}
    )
    fake = FakeClone({})
    with patch.object(Repo, "clone_from", side_effect=fake):
        _ssh_uri().get_repo_from_cache()
    assert fake.calls == [(f"https://{DEFAULT_GH_DOMAIN}/org/assets", None)]


def test_public_repo_without_token_clones_anonymously():
    fake = FakeClone({("git@github.com:org/assets", None): SSH_REFUSED})
    with patch.object(Repo, "clone_from", side_effect=fake):
        _ssh_uri("github.com").get_repo_from_cache()
    assert fake.calls[-1] == ("https://github.com/org/assets", None)


def test_other_errors_do_not_try_more_tokens(monkeypatch):
    monkeypatch.setenv("GBSERVER_GIT_TRANSPORT", "https")
    monkeypatch.setenv("GBSERVER_GITHUB_TOKEN", "svc")
    https_url = f"https://{DEFAULT_GH_DOMAIN}/org/assets"
    fake = FakeClone({(https_url, "svc"): HTTPS_BRANCH_MISSING})
    with (
        patch.object(Repo, "clone_from", side_effect=fake),
        pytest.raises(GitCommandError) as err,
    ):
        _ssh_uri().get_repo_from_cache()
    # Retried as an unclassified git error, but never with another token.
    assert set(fake.calls) == {(https_url, "svc")}
    assert "service:GBSERVER_GITHUB_TOKEN" in str(err.value)


def test_error_names_every_attempt(monkeypatch):
    monkeypatch.setenv("GBSERVER_GITHUB_TOKEN", "svc")
    ssh_url = f"git@{DEFAULT_GH_DOMAIN}:org/assets"
    https_url = f"https://{DEFAULT_GH_DOMAIN}/org/assets"
    fake = FakeClone(
        {
            (ssh_url, None): SSH_REFUSED,
            (https_url, "svc"): HTTPS_NOT_FOUND,
            (https_url, None): HTTPS_NOT_FOUND,
        }
    )
    with (
        patch.object(Repo, "clone_from", side_effect=fake),
        pytest.raises(GitCommandError) as err,
    ):
        _ssh_uri().get_repo_from_cache()
    message = str(err.value)
    assert "ssh failed: ssh: connect to host" in message
    assert "https (service:GBSERVER_GITHUB_TOKEN)" in message
    assert "https (anonymous)" in message
    assert "svc" not in message.replace("service:", "")


def test_ssh_and_https_spellings_share_the_cache(monkeypatch):
    monkeypatch.setenv("GBSERVER_GIT_TRANSPORT", "https")
    fake = FakeClone({})
    with patch.object(Repo, "clone_from", side_effect=fake):
        first = _ssh_uri("github.com").get_repo_from_cache()
        second = GitURI.get_uri(
            "git+https://github.com/org/assets.git@dev"
        ).get_repo_from_cache()
    assert first == second
    assert len(fake.calls) == 1


def test_repo_name_with_dots():
    uri = GitURI.get_uri("git+ssh://github.com/org/granite.build.git@main")
    assert uri.get_repo_url() == "git@github.com:org/granite.build"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, GitTransport.SSH_THEN_HTTPS),
        ("", GitTransport.SSH_THEN_HTTPS),
        ("  ", GitTransport.SSH_THEN_HTTPS),
        ("bogus", GitTransport.SSH_THEN_HTTPS),
        ("HTTPS", GitTransport.HTTPS),
        ("ssh", GitTransport.SSH),
    ],
)
def test_transport_unset_or_empty_is_default(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(ENV_VAR_GIT_TRANSPORT, value)
    assert get_git_transport() == expected


def test_clone_with_a_build_token_is_not_shared(monkeypatch):
    """A clone made with one build's token is not reused by a build without it."""
    monkeypatch.setenv("GBSERVER_GIT_TRANSPORT", "https")
    url = "https://github.com/org/assets"
    fake = FakeClone({(url, None): HTTPS_NOT_FOUND})
    with patch.object(Repo, "clone_from", side_effect=fake):
        _ssh_uri(
            "github.com", secrets={"GIT_TOKEN_GITHUB_COM": "usr"}
        ).get_repo_from_cache()
        with pytest.raises(GitCommandError):
            _ssh_uri("github.com").get_repo_from_cache()
    assert fake.calls == [(url, "usr"), (url, None)]


def test_ssh_is_tried_once_before_falling_back():
    """Even a transient-looking ssh error goes straight to https."""
    ssh_url = "git@github.com:org/assets"
    fake = FakeClone({(ssh_url, None): "fatal: early EOF"})
    with patch.object(Repo, "clone_from", side_effect=fake):
        _ssh_uri("github.com").get_repo_from_cache()
    assert fake.calls == [(ssh_url, None), ("https://github.com/org/assets", None)]


def test_refused_credentials_are_not_retried(monkeypatch):
    monkeypatch.setenv("GBSERVER_GIT_TRANSPORT", "https")
    monkeypatch.setenv("GBSERVER_GITHUB_TOKEN", "svc")
    url = f"https://{DEFAULT_GH_DOMAIN}/org/assets"
    fake = FakeClone(
        {
            (url, "svc"): (
                f"fatal: unable to access '{url}/': "
                "The requested URL returned error: 403"
            ),
            (url, None): (
                "fatal: could not read Username for 'https://x': "
                "terminal prompts disabled"
            ),
        }
    )
    with (
        patch.object(Repo, "clone_from", side_effect=fake),
        pytest.raises(GitCommandError),
    ):
        _ssh_uri().get_repo_from_cache()
    assert fake.calls == [(url, "svc"), (url, None)]
