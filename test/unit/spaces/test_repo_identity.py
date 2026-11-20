#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""A repo is the same repo however it is accessed (https, git+ssh, ssh, scp)."""

import pytest

from gbserver.spaces.repo_identity import (
    is_scp_uri,
    normalize_repo_uri,
    same_repo,
    to_config_uri,
)

FORMS = [
    "https://github.example.com/Org/gbspace-x",
    "https://github.example.com/Org/gbspace-x/",
    "git+ssh://github.example.com/Org/gbspace-x.git@gbspace-config",
    "git+ssh://git@github.example.com/org/gbspace-x.git@gbspace-config#subdirectory=",
    "ssh://git@github.example.com:22/Org/gbspace-x.git",
    "git@github.example.com:Org/gbspace-x.git",
]


@pytest.mark.parametrize("uri", FORMS)
def test_all_forms_are_the_same_repo(uri):
    assert same_repo(uri, FORMS[0])


@pytest.mark.parametrize("uri", FORMS)
def test_all_forms_normalize_to_https(uri):
    assert normalize_repo_uri(uri).lower() == "https://github.example.com/org/gbspace-x"


def test_normalize_keeps_owner_repo_case():
    assert (
        normalize_repo_uri("git@github.example.com:Org/Repo.git")
        == "https://github.example.com/Org/Repo"
    )


def test_different_repos_differ():
    assert not same_repo(
        "https://github.example.com/org/gbspace-x",
        "https://github.example.com/org/assets",
    )


def test_normalize_rejects_non_repo():
    with pytest.raises(ValueError):
        normalize_repo_uri("https://github.example.com/only-owner")


@pytest.mark.parametrize("uri", FORMS)
def test_config_uri_matches_git_backed_form(uri):
    # What GitURI.get_gb_space_config_uri gives an https space repo with a
    # config branch, so IBM secret-group regexes see the same string.
    assert (
        to_config_uri(uri, "gbspace-config").lower()
        == "git+ssh://github.example.com/org/gbspace-x.git@gbspace-config"
    )


def test_config_uri_local_source(tmp_path):
    assert to_config_uri(tmp_path.as_uri(), "main") == f"file://{tmp_path}"


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("git@github.example.com:org/r.git", True),
        ("github.example.com:org/r", True),
        ("https://github.example.com/org/r", False),
        ("git+ssh://git@github.example.com/org/r.git", False),
        ("/abs/path", False),
        ("relative/path", False),
        ("c:/windows/path", False),  # a '/' right after ':'
    ],
)
def test_is_scp_uri(uri, expected):
    assert is_scp_uri(uri) is expected
