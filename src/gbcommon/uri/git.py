#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""
URI for git repos.
"""

import shutil
import tempfile
import threading
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Self, Tuple, Type

if TYPE_CHECKING:
    from git import Repo
    from git.exc import GitCommandError

from gbcommon.types.constants import get_gh_api_base
from gbcommon.uri.uri import URI
from gbserver.types.constants import SPACE_REPO_CONFIG_BRANCH_NAME
from gbserver.utils.filesystem import sync_or_copy
from gbserver.utils.git_auth import (
    ENV_VAR_GIT_TRANSPORT,
    GitToken,
    GitTransport,
    get_git_transport,
    get_service_token,
    get_token_candidates,
    git_auth_env,
)
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import short_alphanumeric_lower_hash

logger = get_logger(__name__)

# Errors after which a different token might succeed (bad token, or a repo
# the token can't see: GitHub reports both as "not found").
_ACCESS_ERROR_PATTERNS = (
    "authentication failed",
    "invalid username or password",
    "could not read username",
    "terminal prompts disabled",
    "repository not found",
    "returned error: 401",
    "returned error: 403",
    "returned error: 404",
)

_NOISE_LINE_PREFIXES = (
    "cloning into",
    "please make sure",
    "and the repository exists",
)


def _is_access_error(error: "GitCommandError") -> bool:
    text = f"{error.stderr}".lower()
    return any(p in text for p in _ACCESS_ERROR_PATTERNS)


def _git_error_summary(error: "GitCommandError") -> str:
    """The informative lines of a failed git command's stderr, on one line."""
    lines = []
    for raw in f"{error.stderr}".splitlines():
        line = raw.strip()
        if line.lower().startswith("stderr:"):
            line = line[len("stderr:") :].strip()
        line = line.strip("'").strip()
        if line and not line.lower().startswith(_NOISE_LINE_PREFIXES):
            lines.append(line)
    return "; ".join(lines) or f"exit code {error.status}"


def split_repo_path(path: str) -> Tuple[str, str]:
    """Extract ``(owner, repo)`` from a git URL path.

    Shared by :func:`get_uri_parts` and the build-validation space-repo comparison
    so the owner/repo parsing lives in one place. Strips a pip-style ``@<ref>``
    (e.g. ``repo.git@branch``) and a trailing ``.git`` (case-insensitively);
    preserves case otherwise. Missing segments yield ``""`` (no ``IndexError`` on a
    short or empty path).

    Args:
        path: The path portion of a git URL (``/owner/repo(.git)``) or the
            ``owner/repo`` tail of an scp-form URL.

    Returns:
        ``(owner, repo)`` with ``@<ref>`` and ``.git`` removed from ``repo``.
    """
    segments = [s for s in path.split("/") if s]
    owner = segments[0] if len(segments) >= 1 else ""
    repo = segments[1] if len(segments) >= 2 else ""
    repo = repo.split("@", 1)[0]
    if repo.lower().endswith(".git"):
        repo = repo[: -len(".git")]
    return owner, repo


def get_uri_parts(uri: str) -> Tuple[str, str, str, str, str]:
    """
    >>> urlparse('git+ssh://mysourcecontrol.com/granite-dot-build/assets.git#subdirectory=steps/hello-helm-data-upload')
    ParseResult(
        scheme='git+ssh',
        netloc='mysourcecontrol.com',
        path='/granite-dot-build/assets.git',
        params='',
        query='',
        fragment='subdirectory=steps/hello-helm-data-upload',
    )
    >>> Path(u.path).parts
    ('/', 'granite-dot-build', 'assets.git')

    Example: ('git+ssh', 'mysourcecontrol.com', 'granite-dot-build', 'assets', 'steps/hello-helm-data-upload')

    Returns (scheme, domain, owner, repo, sub_directory)
    """
    u = urllib.parse.urlparse(uri)
    scheme = u.scheme
    domain = u.netloc
    owner, repo = split_repo_path(u.path)
    subdirprefix = "subdirectory="
    subdir = (
        u.fragment.removeprefix(subdirprefix)
        if u.fragment.startswith(subdirprefix)
        else ""
    )
    return (scheme, domain, owner, repo, subdir)


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
        # The bare repo name: drop any "@<branch>" and ".git" (as in a git+ssh config
        # uri) so the GitHub API is asked about the repo itself.
        repo = parts[2].split("@", 1)[0].removesuffix(".git")
        if uriobj.hostname is None:
            raise ValueError(f"Did not find host name in uri {uri}")
        return uriobj.hostname, owner, repo

    @staticmethod
    def __get_config_branch(
        token: str, uri: str, config_branch_name: str
    ) -> Optional[str]:
        host, owner, repo = GitURI.__parse_repo_components(uri)
        from gbserver.github.myghapi import MyGHApi

        myapi = MyGHApi(
            token=token, owner=owner, repo=repo, gh_api_endpoint=get_gh_api_base(host)
        )
        config_branch_exists = myapi.is_branch_present(config_branch_name)
        return config_branch_name if config_branch_exists else None

    @staticmethod
    def get_gb_space_config_uri(
        uri: str,
        token: Optional[str] = None,
        config_branch_name: str = SPACE_REPO_CONFIG_BRANCH_NAME,
    ) -> str:
        """
        Convert a bare https git repo uri for a space repo,
        to the format as expected by the build system.
        For the following uris:
        file:// return unmodified
        git+ssh:// that already names a branch (.../{reponame}.git@{branch}) return unmodified
        All others (https://, or git+ssh:// without a branch)..
        Assume the form of https://{gitdomain}/{repoowner}/{reponame} and do the following
            1. change from https:// to git+ssh://
            4. Append .git
            2. If the repo has a gbspace-config branch in it, then append @gbspace-config
            3. Append #subdirectory=

        For example,
            https://github.ibm.com/granite-dot-build/gbspace-public becomes
            git+ssh://github.ibm.com/granite-dot-build/gbspace-public.git@gbspace-config#subdirectory=
        Args:
            token (): github token used to read the repo; None (default) uses
                the service token for the repo's host, "" skips the lookup.
            uri (str): https:// uri pointing to a git space repo.
        """
        if uri == "":
            return ""
        if uri.startswith("file://"):
            return uri  # Assume they are pointing to a local repository and know what their doing.
        if uri.startswith("git+ssh://") and "@" in urllib.parse.urlparse(uri).path:
            # Already names its branch (e.g. a prior result of this function), so
            # there is nothing to look up; the result would be unchanged anyway.
            return uri
        uri = uri.rstrip("/")  # To allow appending ".git" below.
        uriobj = urllib.parse.urlparse(uri)
        if token is None:
            token = get_service_token(uriobj.hostname or "")
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

    def _repo_parts(self: Self) -> Tuple[str, str, str]:
        """``(host, owner, repo)``, without ``.git`` or ``@ref``."""
        assert self.uri is not None, "Git uri is None"
        owner, repo = split_repo_path(self.uri.path)
        return self.hostname, owner, repo

    def get_repo_url(self: Self) -> str:
        """The URL that clones the repo over the URI's own scheme (no credentials)."""
        host, owner, repo = self._repo_parts()
        if self.scheme == "git+https":
            return f"https://{host}/{owner}/{repo}"
        if self.scheme == "git+ssh":
            return f"git@{host}:{owner}/{repo}"
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
        # Keyed on the repo, not the clone URL: an ssh and an https clone of
        # the same repo share an entry.
        host, owner, repo = self._repo_parts()
        repo_cache_path = th_repo_cache / short_alphanumeric_lower_hash(
            f"{host}/{owner}/{repo}".lower() + ("#" + ref if ref else "")
        )
        if repo_cache_path.is_dir():
            if not force:
                logger.info("force is False, reusing repo at '%s'", repo_cache_path)
                return repo_cache_path
            logger.info(
                "force is True, deleting repo at '%s' and recloning", repo_cache_path
            )
            shutil.rmtree(repo_cache_path, ignore_errors=True)
        if ref:
            logger.info(
                "cloning repo '%s' and branch '%s' to '%s'",
                repo_url,
                ref,
                repo_cache_path,
            )
            self._clone(repo_cache_path, branch=ref, single_branch=True, depth=1)
        else:
            logger.info("cloning repo '%s' to '%s'", repo_url, repo_cache_path)
            self._clone(repo_cache_path, depth=1)
        return repo_cache_path

    def _clone(self: Self, path: Path, **kwargs) -> "Repo":
        """Clone per ``GBSERVER_GIT_TRANSPORT``: a ``git+ssh`` URI may go over HTTPS."""
        from git.exc import GitCommandError

        if self.scheme != "git+ssh":
            return self._clone_https(path, fallback_reason=None, **kwargs)
        transport = get_git_transport()
        if transport == GitTransport.HTTPS:
            reason = f"{ENV_VAR_GIT_TRANSPORT}={transport.value}"
        else:
            try:
                return self._clone_with_retry(self.get_repo_url(), path, **kwargs)
            except GitCommandError as e:
                if transport == GitTransport.SSH:
                    raise
                shutil.rmtree(path, ignore_errors=True)
                reason = f"ssh failed: {_git_error_summary(e)}"
        return self._clone_https(path, fallback_reason=reason, **kwargs)

    def _clone_https(
        self: Self, path: Path, fallback_reason: Optional[str], **kwargs
    ) -> "Repo":
        """Clone over HTTPS with each token for the host, then anonymously.

        Only an access error moves on to the next token; any other error is
        raised right away.
        """
        from git.exc import GitCommandError

        host, owner, repo = self._repo_parts()
        url = f"https://{host}/{owner}/{repo}"
        # Only secrets handed to this URI: a thread-local could still hold
        # another build's (e.g. on a reused server thread).
        candidates: List[Optional[GitToken]] = [
            *get_token_candidates(host, self.secrets),
            None,
        ]
        failures: List[Tuple[str, GitCommandError]] = []
        for candidate in candidates:
            label = candidate.label if candidate else "anonymous"
            env = git_auth_env(host, candidate.token if candidate else None)
            try:
                cloned = self._clone_with_retry(url, path, env=env, **kwargs)
            except GitCommandError as e:
                shutil.rmtree(path, ignore_errors=True)
                failures.append((label, e))
                if not _is_access_error(e):
                    break
                continue
            if fallback_reason:
                logger.warning(
                    "cloned %s over https (token: %s) instead of ssh (%s);"
                    " use git+https:// in this URI to skip ssh",
                    str(self),
                    label,
                    fallback_reason,
                )
            elif failures:
                logger.info("cloned %s with token %s", url, label)
            return cloned
        attempts = [fallback_reason] if fallback_reason else []
        attempts += [f"https ({lbl}): {_git_error_summary(e)}" for lbl, e in failures]
        raise GitCommandError(
            ["git", "clone", url], failures[-1][1].status, "; ".join(attempts)
        )

    def _clone_with_retry(self, repo_url: str, path: Path, **kwargs) -> "Repo":
        """Clone repository with retry logic."""
        # Imported here, not at module level: git_retry and myghapi pull in
        # gitpython, which runs `git --version` on import, and the gb CLI loads
        # this module at startup through the URI handler scan.
        from gbserver.utils.git_retry import git_clone_retry

        @git_clone_retry
        def clone(repo_url: str, path: Path, **kwargs) -> "Repo":
            from git import Repo

            return Repo.clone_from(repo_url, path, **kwargs)

        # Keyword-only call: git_retry's cleanup_failed_clone takes args[1] as the
        # clone path when there are 2+ positional args (here that would be the URL)
        # and otherwise kwargs["path"], which it removes before each retry.
        return clone(repo_url=repo_url, path=path, **kwargs)

    def get_path_in_repo_from_cache(self: Self, force: bool = False) -> Optional[Path]:
        """
        Get the exact sub directory in the repo that this URI is referring to.
        If possible returns the path to an existing clone from the cache.
        Otherwise we clone the repo and cache the location.
        If force is True then this always clones the repo.

        Returns None if the repo or the ``#subdirectory=`` does not exist.

        Raises:
            ValueError: If the ``#subdirectory=`` (after percent-decoding and
                resolving symlinks) points outside the cloned repository.
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
            # `repo / "/abs"` is `/abs`, parse_qs has already decoded `%2E%2E` to
            # `..`, and the repo may hold a symlink pointing anywhere -- so check
            # the fully resolved path, or a step/environment URI could copy
            # server files (e.g. ~/.kube) into the step and on to compute.
            if not final_path.resolve().is_relative_to(repo_cache_path.resolve()):
                raise ValueError(
                    f"subdirectory {subdirectory!r} resolves outside the repository"
                    f" of {self.uri.geturl()}"
                )
            if not final_path.exists():
                logger.debug(
                    "Subdirectory '%s' not found in repository '%s'",
                    subdirectory,
                    str(self),
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
        subdirectory = (
            fragment_dict.get("subdirectory", [None])[0] if self.uri.fragment else None
        )
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
            logger.error("Error while pulling %s: %s", self, e)
            return False
        except Exception as e:
            logger.error("Failed to pull %s: %s", self, e)
            return False
