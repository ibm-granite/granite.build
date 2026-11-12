#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""A repo is the same repo however it is accessed (https, git+ssh, ssh, scp)."""

import pytest

from gbserver.spaces.repo_identity import (
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
