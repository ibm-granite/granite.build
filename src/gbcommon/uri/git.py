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

"""
URI for git repos.
"""

import shutil
import tempfile
import threading
import urllib.parse
from pathlib import Path
from typing import List, Optional, Self, Type

from git import Repo

from gbcommon.uri.uri import URI
from gbserver.github.myghapi import MyGHApi
from gbserver.types.constants import (
    GBSERVER_GITHUB_TOKEN,
    SPACE_REPO_CONFIG_BRANCH_NAME,
)
from gbserver.utils.filesystem import sync_or_copy
from gbserver.utils.git_retry import git_clone_retry
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import short_alphanumeric_lower_hash

logger = get_logger(__name__)


class GitURI(URI):
    """URI that deals with git repos (especially remote repos)."""

    _thread_local = threading.local()

    def __init__(
        self: Self,
        uri: Optional[urllib.parse.ParseResult] = None,
        context: Optional[str] = None,
        secrets: Optional[dict] = None,
        config: Optional[dict] = None,
        **kwargs: dict,
    ) -> None:
        self.secrets = secrets or {}
        self.config = config or {}
        super().__init__(uri, context, secrets, **kwargs)

    @property
    def scheme(self: Self) -> str:
        """The scheme part of the URI or empty string if the URI is empty"""
        return self.uri.scheme if self.uri else ""

    @property
    def hostname(self: Self) -> str:
        """The host part of the URI or empty string if the URI is empty"""
        return (self.uri.hostname or "") if self.uri else ""

    @property
    def path(self: Self) -> str:
        """The path part of the URI or empty string if the URI is empty"""
        return self.uri.path if self.uri else ""

    @staticmethod
    def get_supported_schemes() -> List[str]:
        """Return supported uri schemes as list"""
        return ["git+https", "git+git", "git+ssh"]

    @staticmethod
    def parse(uri_str: str) -> "GitURI":
        """Factory method from a raw string URI"""
        return GitURI(urllib.parse.urlparse(uri_str))

    @staticmethod
    def __parse_repo_components(uri: str) -> tuple[str, str, str]:
        uriobj: urllib.parse.ParseResult = urllib.parse.urlparse(uri)
        path = uriobj.path
        parts = path.split("/")
        # Leading slash causes parts[0] == ''
        owner = parts[1]
        repo = parts[2]
        if uriobj.hostname is None:
            raise ValueError(f"Did not find host name in uri {uri}")
        return uriobj.hostname, owner, repo

    @staticmethod
    def __get_config_branch(token: str, uri: str, config_branch_name: str) -> Optional[str]:
        host, owner, repo = GitURI.__parse_repo_components(uri)
        myapi = MyGHApi(token=token, owner=owner, repo=repo, domain=host)
        config_branch_exists = myapi.is_branch_present(config_branch_name)
        return config_branch_name if config_branch_exists else None

    @staticmethod
    def get_gb_space_config_uri(
        uri: str,
        token: str = GBSERVER_GITHUB_TOKEN,
        config_branch_name: str = SPACE_REPO_CONFIG_BRANCH_NAME,
    ) -> str:
        """
        Convert a bare https git repo uri for a space repo,
        to the format as expected by the build system.
        For the following uris:
        file:// return unmodified
        git+ssh:// return unmodified
        All others..
        Assume the form of https://{gitdomain}/{repoowner}/{reponame} and do the following
            1. change from https:// to git+ssh://
            4. Append .git
            2. If the repo has a gbspace-config branch in it, then append @gbspace-config
            3. Append #subdirectory=

        For example,
            https://github.ibm.com/granite-dot-build/gbspace-public becomes
            git+ssh://github.ibm.com/granite-dot-build/gbspace-public.git@gbspace-config#subdirectory=
        Args:
            token (): github token used to read the repo
            uri (str): https:// uri pointing to a git space repo.
        """
        if uri == "":
            return ""
        if uri.startswith("file://"):
            return uri  # Assume they are pointing to a local repository and know what their doing.
        uri = uri.rstrip("/")  # To allow appending ".git" below.
        uriobj = urllib.parse.urlparse(uri)
        if not uriobj.path.endswith(".git"):
            uriobj = uriobj._replace(path=uriobj.path + ".git")
        cfg_branch = (
            None
            if token == ""
            else GitURI.__get_config_branch(
                token=token, uri=uri, config_branch_name=config_branch_name
            )
        )
        if cfg_branch:
            logger.info("cfg_branch: %s", cfg_branch)
        else:
            logger.warning(
                "the 'cfg_branch' is empty '%s' , you might want to set the GITHUB_TOKEN env var",
                cfg_branch,
            )
        path = (
            uriobj.path
            if cfg_branch is None or "@" in uriobj.path
            else f"{uriobj.path}@{cfg_branch}"
        )
        uriobj = uriobj._replace(scheme="git+ssh", path=path)
        return uriobj.geturl()

    def exists(self: Self, force: bool = False) -> bool:
        path = self.get_path_in_repo_from_cache(force=force)
        return path is not None

    def is_accessible(self: Self) -> bool:
        return self.exists()

    def get_repo_url(self: Self) -> str:
        """Returns the actual URI that can clone the repo."""
        assert self.uri is not None, "Git uri is None"
        uri_parts = self.uri.path.split("/")
        org = uri_parts[1]
        repo_name = uri_parts[2].split(".")[0]
        if self.uri.scheme == "git+https":
            base_url = f"https://{self.uri.hostname}/{org}/{repo_name}"
            token_key = (
                self.config.get("token_secretname")
                if isinstance(self.config, dict)
                else "GITHUB_PAT_TUNING"
            )
            # check if a token is provided via secrets
            token = None
            if self.secrets is not None:
                token = self.secrets.get(token_key, "")
            if token:
                return f"https://{token}:x-oauth-basic@{self.uri.hostname}/{org}/{repo_name}"
            return base_url
        if self.uri.scheme == "git+ssh":
            return f"git@{self.uri.hostname}:{org}/{repo_name}"
        logger.error("Invalid protocol in Git URI.")
        return ""

    def get_repo_from_cache(self: Self, force: bool = False) -> Optional[Path]:
        """
        If the repo has already been cloned then get the cached path.
        Otherwise we clone the repo and cache the location.
        If force is True then this always clones the repo.
        """
        if not hasattr(self._thread_local, "repo_cache"):
            self._thread_local.repo_cache = Path(tempfile.mkdtemp())
        repo_url = self.get_repo_url()
        if repo_url == "":
            logger.error("Invalid Git Repo URI (%s).", self.uri)
            return None
        logger.info("repo_url: %s", repo_url)
        assert self.uri is not None, "self.uri is None"
        splits = self.uri.path.split("@")
        ref = None
        if len(splits) > 1:
            ref = splits[1]
        logger.info("repo ref: %s", ref)
        th_repo_cache = self._thread_local.repo_cache
        assert isinstance(th_repo_cache, Path)
        repo_cache_path = th_repo_cache / short_alphanumeric_lower_hash(
            repo_url + ("#" + ref if ref else "")
        )
        if repo_cache_path.is_dir():
            if not force:
                logger.info("force is False, reusing repo at '%s'", repo_cache_path)
                return repo_cache_path
            logger.info("force is True, deleting repo at '%s' and recloning", repo_cache_path)
            shutil.rmtree(repo_cache_path, ignore_errors=True)
        if ref:
            logger.info(
                "cloning repo '%s' and branch '%s' to '%s'",
                repo_url,
                ref,
                repo_cache_path,
            )
            self._clone_with_retry(
                repo_url, repo_cache_path, branch=ref, single_branch=True, depth=1
            )
        else:
            logger.info("cloning repo '%s' to '%s'", repo_url, repo_cache_path)
            self._clone_with_retry(repo_url, repo_cache_path, depth=1)
        return repo_cache_path

    @git_clone_retry
    def _clone_with_retry(self, repo_url: str, path: Path, **kwargs) -> Repo:
        """Clone repository with retry logic."""
        return Repo.clone_from(repo_url, path, **kwargs)

    def get_path_in_repo_from_cache(self: Self, force: bool = False) -> Optional[Path]:
        """
        Get the exact sub directory in the repo that this URI is referring to.
        If possible returns the path to an existing clone from the cache.
        Otherwise we clone the repo and cache the location.
        If force is True then this always clones the repo.
        """
        assert self.uri is not None, "self.uri is None"
        subdirectory = (
            urllib.parse.parse_qs(self.uri.fragment).get("subdirectory", [None])[0]
            if self.uri.fragment
            else None
        )
        repo_cache_path = self.get_repo_from_cache(force=force)
        if repo_cache_path is None:
            return None
        final_path = repo_cache_path
        if subdirectory:
            final_path = repo_cache_path / subdirectory
            if not final_path.exists():
                logger.debug(
                    "Subdirectory '%s' not found in repository '%s'",
                    subdirectory,
                    self.uri.geturl(),
                )
                return None
        return final_path

    @classmethod
    def clear_repo_cache(cls: Type[Self]) -> None:
        """
        Delete any existing clone of the repo.
        TODO: Invoke this appropriately to clean up.
        """
        if hasattr(cls._thread_local, "repo_cache"):
            shutil.rmtree(cls._thread_local.repo_cache)

    def append_path(self: Self, path: str) -> None:
        assert self.uri is not None, "self.uri is None"
        fragment_dict = urllib.parse.parse_qs(self.uri.fragment)
        subdirectory = fragment_dict.get("subdirectory", [None])[0] if self.uri.fragment else None
        if subdirectory:
            fragment_dict["subdirectory"] = [subdirectory + "/" + path.lstrip("/")]
        else:
            fragment_dict["subdirectory"] = [path]
        new_fragment = urllib.parse.urlencode(fragment_dict, doseq=True)
        self.uri = self.uri._replace(fragment=new_fragment)

    def delete(self: Self) -> bool:
        raise NotImplementedError("GitURI delete is not implemented")

    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        try:
            assert self.uri is not None, "the URI is empty"
            path_in_repo = self.get_path_in_repo_from_cache(force=force)
            if path_in_repo is None:
                return False
            if sync_or_copy(path_in_repo, dest):
                return True
            return False
        except ValueError as e:
            logger.error("Error while pulling: %s", e)
            return False
        except Exception as e:
            logger.error("An unexpected error occurred: %s", e)
            return False
