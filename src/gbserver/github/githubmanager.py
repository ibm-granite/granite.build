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
The github monitor.
"""

import functools
import shutil
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from random import randint
from typing import List, Optional, Self

import yaml
from requests import Timeout
from tqdm import tqdm

from gbserver.buildwatcher.buildlogger import (
    BuildEventMessageLogger,
    BuildMultiMessageLogger,
    BuildPRLogger,
)
from gbserver.buildwatcher.validators.check_build_artifacts import (
    check_build_input_artifacts_registered,
)
from gbserver.github.myghapi import MyGHApi
from gbserver.github.utils import post_validated_build_message
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.buildconfig import (
    BUILD_FILENAME,
    BUILD_RUN_YAML_FILENAME,
    BuildConfig,
    BuildRunConfig,
)
from gbserver.types.constants import (
    BUILD_YAML_BASE_KEYS,
    CURRENT_BUILD_YAML_VERSION,
    CURRENT_BUILD_YAML_VERSION_KEY,
    DEFAULT_BUILDWATCHER_COMMITTER_NAME,
    DEFAULT_REPO_DIR_TO_WATCH,
    DEFAULT_ROOT_PRWATCHER_WORKSPACE_DIR,
    GBSERVER_FUNCTIONAL_IDS,
    WORKSPACE_PRS_DIR,
)
from gbserver.types.prwatcherconfig import (
    PrWatcherConfig,
    get_uri_parts,
)
from gbserver.types.pullrequest import PullRequest
from gbserver.types.status import Status
from gbserver.utils.filesystem import create_temp_subdir
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import (
    get_common_ancestor,
    normalize_to_filename,
)

logger = get_logger(__name__)
_BUILD_EVENT_SOURCE_NAME = "pr-watcher"


def get_progress_callback():
    """For use with git library during clone and push for indicating the progress."""
    pbar = tqdm()

    # pylint: disable=unused-argument
    def progress_callback(op_code, cur_count, max_count=None, message=""):
        pbar.total = max_count
        pbar.n = cur_count
        pbar.refresh()

    return progress_callback


class GitHubManager:
    """Manager for GitHub pull requests, comments, etc."""

    token: str
    config_path: Optional[Path]
    watch_for_config_changes: bool
    config: PrWatcherConfig
    # pr.html_url -> pull request
    past_open_pr_urls: dict[
        str, PullRequest
    ]  # TODO: I think this can be just a set of urls, to save memory.
    # list of pr.html_url that have been stored to gb_builds table
    completed_pr_urls: set[str]
    stop_event: Optional[threading.Event] = None
    build_workspace_dir: Path
    storage: SingletonAdminStorage
    # Only fetch PRs created after this timestamp
    created_after: Optional[datetime] = None

    # for testing
    _skip_all_but_prs: list[str]

    def __init__(
        self: Self,
        token: str,
        config_path: Optional[Path] = None,
        watch_for_config_changes: bool = True,
    ):
        assert token != "", "The GitHub token is empty"
        self._skip_all_but_prs = []
        self.token = token
        self.config_path = config_path
        self.watch_for_config_changes = watch_for_config_changes
        self.storage = get_admin_storage()
        self.past_open_pr_urls = {}
        self.completed_pr_urls = set()
        build_workspace_dir = create_temp_subdir(DEFAULT_ROOT_PRWATCHER_WORKSPACE_DIR)
        self.build_workspace_dir = build_workspace_dir.resolve()
        self.config = PrWatcherConfig()
        self.__reload_config_file(first=True)
        logger.debug("using the config: %s", self.config)

    def __reload_config_file(self: Self, first: bool = False):
        """If the config file is provided, reload it"""
        if self.config_path is None:
            if first:
                self.config.initialize(spaces_storage=self.storage.space_storage)
            return
        logger.info("loading config from %s", self.config_path)
        prev_config = self.config
        self.config = PrWatcherConfig.from_yaml(self.config_path)
        self.config.initialize(
            spaces_storage=self.storage.space_storage, prev_config=prev_config
        )

    def start_and_wait(self: Self) -> None:
        """Starts monitoring Github for pull requests."""
        logger.debug("GitHubManager.start_and_wait start")
        self.stop_event = threading.Event()
        self.__start_monitoring_github()
        logger.debug("GitHubManager.start_and_wait end")

    def stop(self: Self) -> None:
        """Stop the monitoring thread."""
        logger.debug("GitHubManager.stop start")
        if self.stop_event is None:
            logger.error("the stop_event is None")
            return
        self.stop_event.set()
        logger.debug("GitHubManager.stop end")

    def __populate_builds_cache(self: Self) -> None:
        """Populate the pending builds cache from Lakehouse."""
        logger.debug("GitHubManager.populate_builds_cache start")
        logger.info("populating the build cache from Lakehouse")
        # TODO: load only PRs for the space(s) being watched to avoid loading the whole table.
        pending_builds = self.storage.build_storage.get_by_where({})
        logger.info("found pending builds: %d", len(pending_builds))
        for b in pending_builds:
            assert isinstance(b, StoredBuild)
            self.__record_pr_as_processed(b.source_uri)
        logger.debug("GitHubManager.populate_builds_cache end")

    def __start_monitoring_github(self: Self) -> None:
        logger.debug("GitHubManager.start_monitoring_github start")
        self.__populate_builds_cache()
        assert self.stop_event is not None, "the stop_event is None"
        while not self.stop_event.is_set():
            if self.watch_for_config_changes:
                self.__reload_config_file()
            for space_name, space in self.config._spaces.items():
                logger.info("space_name: %s space: %s", space_name, space)
                repo_uri = space.git_repo_uri
                try:
                    logger.info("looking for open pull requests for repo %s", repo_uri)
                    self.__merge_open_prs(repo_uri=repo_uri)
                except Exception as e:
                    logger.error(
                        "failed to process newly opened pull requests, error: %s",
                        e,
                    )
                    logger.debug("%s", traceback.format_exc())
                new_merged_prs = []
                try:
                    logger.info("fetching pull requests from repo %s", repo_uri)
                    new_merged_prs = self.__get_new_merged_prs(repo_uri=repo_uri)
                except Exception as e:
                    logger.error(
                        "failed to fetch newly merged pull requests, error: %s", e
                    )
                    logger.debug("%s", traceback.format_exc())

                # to_print = [(x.my_pr_id, x.title) for x in new_merged_prs]
                # logger.info("newly merged prs %d : %s", len(to_print), to_print)
                for pr in new_merged_prs:
                    try_count = 0
                    while try_count < self.config.lh_max_retries:
                        if self.stop_event.is_set():
                            break
                        try:
                            try_count += 1
                            self.__validate_and_store_pr(pr=pr, space_name=space_name)
                            break
                        except Timeout as te:
                            logger.error(
                                "timeout during validation and storing of the PR %s, error: %s",
                                pr.my_pr_id,
                                te,
                            )
                            if try_count >= self.config.lh_max_retries:
                                logger.error(
                                    "max retries exceeded, try_count: %s", try_count
                                )
                                break
                            logger.warning("trying again, try_count: %s", try_count)
                        except Exception as e:
                            logger.error(
                                "failed to store the PR %s, error: %s", pr.my_pr_id, e
                            )
                            logger.error("%s", traceback.format_exc())
                            break
            logger.info("sleeping...")
            time.sleep(self.config.monitoring_interval)
        logger.debug("GitHubManager.start_monitoring_github end")

    def __merge_open_prs(self: Self, repo_uri: str) -> None:
        logger.debug("GitHubManager.process_open_prs start")
        open_prs = []
        try:
            # open_prs = self.get_prs(repo_uri=repo_uri, state="open")
            # Include already seen open PRs to workaround GitHub API issues (HTTP 405 on merge)
            open_prs = self.__get_new_open_prs(repo_uri=repo_uri, include_old=True)
        except Exception as e:
            logger.error(
                "failed to fetch open PRs for the repo %s error: %s", repo_uri, e
            )
            return
        myapi = self.__get_my_gh_api(uri=repo_uri)
        merged_prs = []
        for pr in open_prs:
            try:
                if pr.title.lower().startswith("ignore"):
                    logger.warning("ignoring PR: %s %s", pr.my_pr_id, pr.title)
                    continue
                if (pr.user.login == DEFAULT_BUILDWATCHER_COMMITTER_NAME) or (
                    pr.user.login in GBSERVER_FUNCTIONAL_IDS
                ):
                    logger.warning(
                        "ignoring PR made by the functional ID %s : %s %s",
                        pr.user.login,
                        pr.my_pr_id,
                        pr.title,
                    )
                    continue
                pr_files = self.__get_pr_files_list(pr=pr)
                if len(pr_files) == 0:
                    continue
                logger.info("merging the open PR %s %s", pr.my_pr_id, pr.title)
                merged_pr = myapi.merge_pr(pr_id=pr.my_pr_id)
                if not merged_pr.merged:
                    logger.warning("failed to merge the PR: %s", pr.url)
                    continue
                merged_prs.append(pr.my_pr_id)
            except Exception as e:
                logger.error("failed to merge the PR %s , error: %s", pr.url, e)
                logger.debug("%s", traceback.format_exc())
        logger.info(
            "merged the following open PRs %d : %s", len(merged_prs), merged_prs
        )
        logger.debug("GitHubManager.process_open_prs end")

    def __get_new_merged_prs(self: Self, repo_uri: str) -> List[PullRequest]:
        """Get a list of pull requests that have been newly merged."""
        logger.debug("GitHubManager.get_new_merged_prs start")
        closed_prs = self.__get_prs(
            repo_uri=repo_uri,
            state="closed",
            direction="desc",
            created_after=self.created_after,
        )
        logger.debug("closed_prs: %d", len(closed_prs))
        if len(closed_prs) == 0:
            return []
        _new_merged_prs: List[PullRequest] = []
        for pr in closed_prs:
            # logger.debug("pr: %s pr.merged_at: %s", pr.my_pr_id, pr.merged_at)
            if self.created_after is None:
                self.created_after = pr.created_at
            self.created_after = max(self.created_after, pr.created_at)
            if pr.merged_at is not None and not self.__is_pr_processed(pr.html_url):
                logger.debug("PR %s identified as newly merged", pr.html_url)
                _new_merged_prs.append(pr)

        # Now do some additional filtering...
        new_merged_prs: List[PullRequest] = []
        for pr in _new_merged_prs:
            if len(self._skip_all_but_prs) > 0:
                if pr.html_url not in self._skip_all_but_prs:
                    logger.warning(
                        "skipping pr %s which is not in _skip_all_but_prs",
                        pr.html_url,
                    )
                    continue
            if pr.title.lower().startswith("ignore"):
                logger.warning("ignoring PR: %s %s", pr.my_pr_id, pr.title)
                continue
            if pr.user.login == DEFAULT_BUILDWATCHER_COMMITTER_NAME:
                logger.warning(
                    "ignoring PR made by the functional ID %s : %s %s",
                    DEFAULT_BUILDWATCHER_COMMITTER_NAME,
                    pr.my_pr_id,
                    pr.title,
                )
                continue
            new_merged_prs.append(pr)

        logger.debug("new_merged_prs: %d", len(new_merged_prs))
        logger.debug("GitHubManager.get_new_merged_prs end")
        return new_merged_prs

    def __get_new_open_prs(
        self: Self, repo_uri: str, include_old: bool = False
    ) -> List[PullRequest]:
        """Get a list of pull requests that have been newly opened."""
        logger.debug("GitHubManager.get_new_open_prs start")
        open_prs = self.__get_prs(repo_uri=repo_uri, state="open")
        logger.debug("open_prs: %d", len(open_prs))
        if len(open_prs) == 0:
            return []
        new_open_prs = []
        for pr in open_prs:
            # logger.debug("pr: %s pr.updated_at: %s", pr.my_pr_id, pr.updated_at)
            old_pr = self.past_open_pr_urls.get(pr.html_url, None)
            self.past_open_pr_urls[pr.html_url] = pr
            if pr.merged_at is not None:
                # logger.debug("already merged")
                continue
            if old_pr is None or include_old:
                logger.debug("pr: %s pr.updated_at: %s", pr.my_pr_id, pr.updated_at)
                if old_pr is None:
                    logger.debug("not seen before")
                else:
                    logger.debug("have seen before, but include_old is True")
                new_open_prs.append(pr)
                continue
        logger.debug("new_open_prs: %d", len(new_open_prs))
        logger.debug("GitHubManager.get_new_open_prs end")
        return new_open_prs

    def __get_my_gh_api(self: Self, uri: str) -> MyGHApi:
        """Get a GitHub API client from a PR url or a repo URL."""
        _, domain, owner, repo, _ = get_uri_parts(uri=uri)
        myapi = MyGHApi(
            token=self.token,
            owner=owner,
            repo=repo,
            domain=domain,
        )
        return myapi

    def __get_prs(
        self: Self,
        repo_uri: str,
        state: str = "all",
        sort: str = "created",
        direction: str = "asc",
        created_after: Optional[datetime] = None,
    ) -> List[PullRequest]:
        myapi = self.__get_my_gh_api(uri=repo_uri)
        prs = myapi.get_prs(
            state=state,
            sort=sort,
            direction=direction,
            created_after=created_after,
        )
        # msg = ""
        # for pr in prs:
        #     msg += f"pr: {pr.url}\n"
        # logger.info(f"PRs:\n{msg}")
        return prs

    def __is_pr_processed(self: Self, pr_url: str) -> bool:
        return pr_url in self.completed_pr_urls

    def __record_pr_as_processed(self: Self, pr_url: str) -> None:
        self.completed_pr_urls.add(pr_url)
        if pr_url in self.past_open_pr_urls:
            del self.past_open_pr_urls[pr_url]  # Garbage collection

    def __validate_and_store_pr(self: Self, pr: PullRequest, space_name: str) -> None:
        logger.debug("GitHubManager.validate_and_store_pr start")
        pr_id = pr.my_pr_id
        logger.info("processing pr %s %s", pr_id, pr.title)
        pr_rel_paths = self.__get_pr_files_list(pr=pr)
        logger.info("pr_file_paths: %d %s", len(pr_rel_paths), pr_rel_paths)
        rel_build_yamls = [x for x in pr_rel_paths if x.name == BUILD_FILENAME]
        logger.info("rel_build_yamls: %s", rel_build_yamls)
        if len(rel_build_yamls) == 0:
            logger.info(
                "no %s were added in the relevant directory, skipping", BUILD_FILENAME
            )
            self.__record_pr_as_processed(pr.html_url)
            return
        download_root_path, pr_downloaded_paths = self.__fetch_pr_files(pr=pr)
        logger.info(
            "download_root_path: %s pr_downloaded_paths: %s",
            download_root_path,
            pr_downloaded_paths,
        )
        logger.debug("get the downloaded build.yaml path")
        build_yaml_paths = [x for x in pr_downloaded_paths if x.name == BUILD_FILENAME]
        logger.info("build_yaml_paths: %s", build_yaml_paths)
        assert len(build_yaml_paths) > 0, f"expected at least one {BUILD_FILENAME}"
        build_yaml_path = build_yaml_paths[0]
        logger.info("build_yaml_path: %s", build_yaml_path)
        assert build_yaml_path.is_file(), f"expected {build_yaml_path} to be a file"
        logger.info("PR %s validing file at path %s", pr_id, build_yaml_path)
        build_err = ""
        try:
            with open(build_yaml_path, "r", encoding="utf-8") as f:
                build_yaml_dict = yaml.safe_load(f)
            logger.debug("check the version first")
            assert isinstance(build_yaml_dict, dict)
            build_yaml_dict_inner = None
            for key in BUILD_YAML_BASE_KEYS:
                if key in build_yaml_dict:
                    build_yaml_dict_inner = build_yaml_dict[key]
                    break
            if build_yaml_dict_inner is None:
                raise ValueError(
                    f"failed to find the base keys {BUILD_YAML_BASE_KEYS} in {build_yaml_dict}"
                )
            assert isinstance(build_yaml_dict_inner, dict)
            version = build_yaml_dict_inner.get(CURRENT_BUILD_YAML_VERSION_KEY, "")
            if version not in ("", CURRENT_BUILD_YAML_VERSION):
                raise ValueError(
                    f"the build.yaml version doesn't match, actual {version} expected {CURRENT_BUILD_YAML_VERSION}"
                )
            logger.debug("the version matches so validate the schema")
            build_config = BuildConfig.from_yaml(path=build_yaml_path)
            logger.debug("build_config: %s", build_config)
            if self.config.validate_inputs_are_registered:
                curr_errors = check_build_input_artifacts_registered(
                    artifact_registry=self.storage.artifact_registry,
                    build_config=build_config,
                    space_name=space_name,
                )
                if not curr_errors.is_valid():
                    logger.error("artifacts are registered: %s", curr_errors)
        except Exception as e:
            build_err = (
                f"the {BUILD_FILENAME} at {build_yaml_path} is invalid, error:\n{e}"
            )
            logger.error("%s", build_err)
            self.__record_pr_as_processed(pr.html_url)
        is_valid_build = build_err == ""
        logger.debug("is_valid_build: %s build_err: %s", is_valid_build, build_err)
        # TODO: once the existence check becomes less costly (by moving away from LH-based admin table) we should make this check first so that we can reduce the number of github API calls.
        logger.debug("skip if the PR exists in Lakehouse already")
        source_uri = pr.html_url
        result = self.storage.build_storage.get_by_where({"source_uri": source_uri})
        if len(result) > 0:
            logger.info("the PR %s already exists in Lakehouse, skipping", pr_id)
            if len(result) > 1:
                logger.warning("found multiple builds with the same URI in Lakehouse")
            stored_build = result[0]
            assert isinstance(stored_build, StoredBuild)
            self.__record_pr_as_processed(stored_build.source_uri)
            shutil.rmtree(
                download_root_path, ignore_errors=True
            )  # Clean up the filesystem.
            return
        logger.debug("PR does NOT exist in Lakehouse, store it")
        stored_build = self.__compress_and_store_pr(
            pr=pr, build_yaml_path=build_yaml_path, is_valid=is_valid_build
        )
        # Clean up the filesystem.
        shutil.rmtree(download_root_path, ignore_errors=True)
        # -----------------------------------------
        build_message_logger = self.__get_message_logger(build_id=stored_build.uuid)
        post_validated_build_message(
            build_message_logger=build_message_logger,
            stored_build=stored_build,
            build_err=build_err,
        )
        # -----------------------------------------
        logger.debug("GitHubManager.validate_and_store_pr end")

    def __compress_and_store_pr(
        self: Self,
        pr: PullRequest,
        build_yaml_path: Path,
        is_valid: bool = True,
    ) -> StoredBuild:
        logger.debug("GitHubManager.compress_and_store_pr start")
        assert build_yaml_path.is_file(), f"expected {build_yaml_path} to be a file"
        run_yaml_path = build_yaml_path.parent / BUILD_RUN_YAML_FILENAME
        targets: Optional[list[str]] = None
        if run_yaml_path.is_file():
            logger.info("found some build run config at %s", run_yaml_path)
            build_run_config = BuildRunConfig.from_yaml(path=run_yaml_path)
            targets = list(build_run_config.targets_to_run.keys())
            logger.info("build_run_config: %s targets: %s", build_run_config, targets)
        pr_id = pr.my_pr_id
        source_uri = pr.html_url
        stored_build_name = f"build-for-pr-{pr_id}"
        username = pr.user.login
        stored_space = self.config.get_space_from_pr_url(pr.html_url)
        assert (
            stored_space is not None
        ), f"failed to find a space for the PR {pr.html_url}"
        status = Status.PENDING if is_valid else Status.INVALID
        stored_build = StoredBuild.create(
            name=stored_build_name,
            space_name=stored_space.name,
            source_uri=source_uri,
            username=username,
            build_yaml_path=build_yaml_path,
            status=status,
            targets=targets,
        )
        assert build_yaml_path.is_file(), f"expected {build_yaml_path} to be a file"
        build_dir = build_yaml_path.parent
        stored_build.save_to_build_archive_from_dir(build_dir=build_dir)
        logger.info("build_archive b64 length: %d", len(stored_build.build_archive))
        logger.info(
            "storing build in cache and Lakehouse: build_id %s", stored_build.uuid
        )  # Blows travis log limit if info() because of archive
        logger.debug(
            "storing build in cache and Lakehouse: %s", stored_build
        )  # Blows travis log limit if info() because of archive
        try_count = 0
        while try_count < self.config.lh_max_retries:
            try:
                try_count += 1
                result = self.storage.build_storage.add(stored_build)
                logger.info("stored PR with id: %s", result)
                break
            except Exception as e:
                logger.error("failed to store the PR, error: %s", e)
                if try_count >= self.config.lh_max_retries:
                    logger.error("max retries %d exceeded", self.config.lh_max_retries)
                    break
                logger.info("try_count: %d sleeping...", try_count)
                time.sleep(1)
        self.__record_pr_as_processed(stored_build.source_uri)
        logger.debug("GitHubManager.compress_and_store_pr end")
        return stored_build

    def __get_pr_files_list(self: Self, pr: PullRequest) -> List[Path]:
        """Get the list of the files in the appropriate folder in the PR."""
        # pr = self.pr_cache[pr_html_url]
        pr_html_url = pr.html_url
        repo_url = self.config.get_repo_from_pr_url(pr_html_url=pr_html_url)
        assert repo_url is not None, f"failed to find a repo for the PR {pr_html_url}"
        pr_html_url_parts = get_uri_parts(uri=repo_url)
        _repo_dir_to_watch = pr_html_url_parts[4]
        if _repo_dir_to_watch == "":
            _repo_dir_to_watch = DEFAULT_REPO_DIR_TO_WATCH
        repo_dir_to_watch = Path(_repo_dir_to_watch)
        logger.info(
            "pr_html_url %s repo_dir_to_watch: %s", pr_html_url, repo_dir_to_watch
        )
        if pr.files is None:
            pr.files = self.__get_files_added_in_pr(pr=pr)
        pr.files = [f for f in pr.files if repo_dir_to_watch in f.parents]
        return pr.files

    def __get_files_added_in_pr(self: Self, pr: PullRequest) -> List[Path]:
        pr_html_url = pr.html_url
        pr_id = pr.my_pr_id
        myapi = self.__get_my_gh_api(uri=pr_html_url)
        pr_files = myapi.get_pr_files(pr_id=pr_id)
        paths = [Path(f.filename) for f in pr_files if f.status == "added"]
        logger.debug("pr_id: %s paths: %d", pr_id, len(paths))
        return paths

    def __fetch_pr_files(self: Self, pr: PullRequest) -> tuple[Path, List[Path]]:
        """Clone the PR branch and get the actual path to the files in the PR."""
        files = self.__get_pr_files_list(pr=pr)
        if len(files) == 0:
            raise ValueError(f"PR {pr.my_pr_id} has no files to fetch")
        build_yaml_paths = [f for f in files if f.name == BUILD_FILENAME]
        repo_path = files[0]
        if len(build_yaml_paths) == 0:
            logger.info("0 build.yamls")
            common_parent = get_common_ancestor(files)
            if common_parent == Path("."):
                logger.warning(
                    "PR %s failed to find the common parent: %s", pr.my_pr_id, files
                )
            else:
                repo_path = common_parent
        else:
            build_yaml_path = build_yaml_paths[0]
            if len(build_yaml_paths) > 1:
                logger.info("more than 1 build.yamls, using %s", build_yaml_path)
            else:
                logger.info("exactly 1 build.yaml %s", build_yaml_path)
            if len(files) == 1:
                logger.info("only the build.yaml was added in the PR, fetch the file")
                repo_path = build_yaml_path
            else:
                logger.info("multiple files were added in the PR, fetch parent dir")
                repo_path = build_yaml_path.parent
        pr_html_url = pr.html_url
        pr_html_url_parts = get_uri_parts(uri=pr.html_url)
        pr_owner = pr_html_url_parts[2]
        pr_repo = pr_html_url_parts[3]
        pr_clone_path = (
            self.build_workspace_dir
            / WORKSPACE_PRS_DIR
            / normalize_to_filename(pr_owner)
            / normalize_to_filename(pr_repo)
            / pr.my_pr_id
        )
        myapi = self.__get_my_gh_api(uri=pr_html_url)
        sleep_time = randint(1, 3)
        logger.info("sleeping for %s seconds to mitigate GH rate limit", sleep_time)
        time.sleep(sleep_time)
        myapi.fetch_repo_contents(repo_path=repo_path, output_dir=pr_clone_path)
        downloaded_files = [(pr_clone_path / f) for f in files]
        return pr_clone_path, downloaded_files

    # A cache may not be necessary since we only log 1 message per build. So use a small cache?
    @functools.lru_cache(maxsize=4)
    def __get_message_logger(self: Self, build_id: str) -> BuildMultiMessageLogger:
        stored_build = self.storage.build_storage.get_by_uuid(build_id)
        assert isinstance(stored_build, StoredBuild)
        pr_logger = BuildPRLogger(stored_build=stored_build)
        message_event_logger = BuildEventMessageLogger(
            event_source=_BUILD_EVENT_SOURCE_NAME, stored_build=stored_build
        )
        build_logger = BuildMultiMessageLogger(
            stored_build=stored_build, loggers=[pr_logger, message_event_logger]
        )
        return build_logger

    # def clone_repo_for_pr(self: Self, pr: PullRequest) -> Path:
    #     """Clone the repo fork and branch used in the PR."""
    #     pr_id = pr.my_pr_id
    #     logger.debug("GitHubManager.clone_repo_for_pr start pr_id: %s", pr_id)
    #     if pr.head.repo is None:
    #         raise ValueError(
    #             "the PR was made from a repo that has been deleted: %s", pr.html_url
    #         )
    #     pr_clone_url = pr.head.repo.clone_url
    #     pr_head_ref = pr.head.ref
    #     checkout_sha = pr.head.sha
    #     # https://stackoverflow.com/questions/22331524/get-pull-request-merge-commit-sha-from-pull-request-number-using-github-api
    #     # https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28#get-a-pull-request
    #     if pr.merged_at is not None:
    #         logger.debug("the pr %s was merged at %s", pr_id, pr.merged_at)
    #         assert (
    #             pr.base.repo is not None
    #         ), f"this PR's target repo doesn't exist: {pr}"
    #         pr_clone_url = pr.base.repo.clone_url
    #         pr_head_ref = pr.base.ref
    #         assert pr.merge_commit_sha is not None
    #         checkout_sha = pr.merge_commit_sha
    #     logger.debug("pr_clone_url: %s", pr_clone_url)
    #     logger.debug("pr_head_ref: %s", pr_head_ref)
    #     logger.debug("checkout_sha: %s", checkout_sha)
    #     pr_html_url_parts = get_uri_parts(uri=pr.html_url)
    #     pr_owner = pr_html_url_parts[2]
    #     pr_repo = pr_html_url_parts[3]
    #     pr_clone_path = (
    #         self.build_workspace_dir
    #         / WORKSPACE_REPOS_DIR
    #         / normalize_to_filename(pr_owner)
    #         / normalize_to_filename(pr_repo)
    #         / checkout_sha
    #     )
    #     logger.debug("pr_clone_path: %s", pr_clone_path)
    #     pr_clone_url_no_prefix = pr_clone_url.removeprefix("https://")
    #     clone_url_with_creds = "https://" + self.token + "@" + pr_clone_url_no_prefix
    #     clone_url_with_creds_redacted = "https://****@" + pr_clone_url_no_prefix
    #     logger.info("clone url: %s", clone_url_with_creds_redacted)
    #     shutil.rmtree(pr_clone_path, ignore_errors=True)
    #     pr_clone_path.mkdir(mode=DEFAULT_DIR_PERMS, parents=True, exist_ok=False)
    #     clone_kwargs = {
    #         "url": clone_url_with_creds,
    #         "to_path": pr_clone_path,
    #         "progress": get_progress_callback(),
    #     }
    #     if pr.merge_commit_sha is None:
    #         clone_kwargs["branch"] = pr_head_ref
    #         clone_kwargs["depth"] = 1
    #     logger.debug("cloning the pull request commit from the repo")
    #     repo = Repo.clone_from(**clone_kwargs)
    #     logger.info("cloned pull request repo %s", repo)
    #     repo.git.checkout(checkout_sha)
    #     logger.debug("GitHubManager.clone_repo_for_pr end %s", pr_id)
    #     return pr_clone_path
