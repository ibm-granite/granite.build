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

# from .artifact import ArtifactStoreType, ArtifactType
# from .resources import ResourceSpec, ResourceType

"""
Logging about the build to a PR/Issue.
"""

import functools
from abc import abstractmethod
from collections import deque
from typing import List, Optional, Self, Tuple

from gbserver.github.myghapi import MyGHApi
from gbserver.storage import singleton_storage
from gbserver.storage.event_storage import IStoredEventStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_event import StoredEvent
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    BuildLogLevel,
    EntityRunMetadata,
    create_message_event,
)
from gbserver.types.constants import DEFAULT_GH_API_ENDPOINT, GBSERVER_GITHUB_TOKEN
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class AbstractBuildLogger:
    """Abstract Build Logger implementation."""

    stored_build: StoredBuild

    def __init__(self: Self, stored_build: StoredBuild) -> None:
        self.stored_build = stored_build

    @abstractmethod
    def _log(
        self: Self,
        level: BuildLogLevel,
        markdown: str,
        triggering_event: Optional[BuildEvent] = None,
    ) -> None:
        raise NotImplementedError("_AbstractBuildLogger._log is not implemented")

    def info(self: Self, markdown: str, triggering_event: Optional[BuildEvent] = None) -> None:
        """Print with info log level"""
        self._log(
            level=BuildLogLevel.INFO,
            markdown=markdown,
            triggering_event=triggering_event,
        )

    def warning(self: Self, markdown: str, triggering_event: Optional[BuildEvent] = None) -> None:
        """Print with warning log level"""
        self._log(
            level=BuildLogLevel.WARNING,
            markdown=markdown,
            triggering_event=triggering_event,
        )

    def error(self: Self, markdown: str, triggering_event: Optional[BuildEvent] = None) -> None:
        """Print with error log level"""
        self._log(
            level=BuildLogLevel.ERROR,
            markdown=markdown,
            triggering_event=triggering_event,
        )


class BuildPRLogger(AbstractBuildLogger):
    """Log about the build to the PR/Issue."""

    gh_token: str = GBSERVER_GITHUB_TOKEN
    gh_api_endpoint: str = DEFAULT_GH_API_ENDPOINT
    pr_id: str = ""
    github_api: Optional[MyGHApi] = None
    # Stores the pending messages since the PR/Issue may not exist yet.
    messages: deque[Tuple[BuildLogLevel, str, Optional[BuildEvent]]]

    def __init__(
        self: Self,
        stored_build: StoredBuild,
        gh_token: str = GBSERVER_GITHUB_TOKEN,
        gh_api_endpoint: str = DEFAULT_GH_API_ENDPOINT,
    ) -> None:
        super().__init__(stored_build=stored_build)
        self.gh_token = gh_token
        self.gh_api_endpoint = gh_api_endpoint
        self.messages = deque()
        # TODO: gh_api_endpoint domain should come from PR/Issue URI
        if self.gh_token == "":
            return
        owner, repo, pr_id = self.stored_build.get_pr_info()
        logger.info("owner: %s repo: %s pr_id: %s", owner, repo, pr_id)
        if pr_id == "":
            logger.warning("no corresponding PR for build: %s", self.stored_build.uuid)
            return
        # Logging to the PR is enabled by setting these
        self.github_api = MyGHApi(
            token=self.gh_token,
            owner=owner,
            repo=repo,
            gh_api_endpoint=self.gh_api_endpoint,
        )
        self.pr_id = pr_id

    def _log_helper(
        self: Self,
        level: BuildLogLevel,
        markdown: str,
        triggering_event: Optional[BuildEvent] = None,
    ) -> None:
        # Prefix with a log-level indication
        truncated_body = markdown[:20] + "..." if len(markdown) > 20 else markdown
        if level == BuildLogLevel.INFO:
            logger.info("posting to PR %s, body: %s", self.pr_id, truncated_body)
            markdown = "## ℹ️  INFO\n\n" + str(markdown)
        elif level == BuildLogLevel.WARNING:
            logger.warning("posting to PR %s, body: %s", self.pr_id, truncated_body)
            markdown = "## ⚠️  WARNING!\n\n" + str(markdown)
        elif level == BuildLogLevel.ERROR:
            logger.error("posting to PR %s, body: %s", self.pr_id, truncated_body)
            markdown = "## ❌  ERROR!\n\n" + str(markdown)
        # Create a logger message.
        # Post to git PR
        if self.github_api is not None or self.pr_id is not None:
            assert self.github_api is not None
            assert self.pr_id is not None
            assert isinstance(self.github_api, MyGHApi)
            try:
                self.github_api.update_issue_comment(body=markdown, pr_id=self.pr_id)
            except Exception as e:
                logger.error(
                    "failed to update pr %s in repo %s as %s: %s",
                    self.pr_id,
                    self.github_api.repo,
                    self.github_api.owner,
                    e,
                )

    def _log(
        self: Self,
        level: BuildLogLevel,
        markdown: str,
        triggering_event: Optional[BuildEvent] = None,
    ) -> None:
        self.messages.append((level, markdown, triggering_event))
        logger.info("messages to be send: %d", len(self.messages))
        while len(self.messages) > 0:
            try:
                l, m, event = self.messages[0]
                self._log_helper(level=l, markdown=m, triggering_event=event)
                self.messages.popleft()
            except Exception as e:
                logger.error(
                    "failed to send message, skipping for now, left to send: %d . error: %s",
                    len(self.messages),
                    e,
                )
                break


class BuildEventMessageLogger(AbstractBuildLogger):
    """Log about the build to the events table."""

    event_source: str
    event_storage: Optional[IStoredEventStorage] = None

    def __init__(self: Self, event_source: str, stored_build: StoredBuild) -> None:
        super().__init__(stored_build=stored_build)
        self.event_source = event_source
        self.event_storage = singleton_storage.get_admin_storage().event_storage

    def _log(
        self: Self,
        level: BuildLogLevel,
        markdown: str,
        triggering_event: Optional[BuildEvent] = None,
    ) -> None:

        # Create a logger message.
        truncated_body = markdown[:20] + "..." if len(markdown) > 20 else markdown
        logger.info("posting a %s log event: %s", level.name, truncated_body)

        # Post to gb_events table
        assert (
            triggering_event is None
            or triggering_event.run_metadata.build_id == self.stored_build.uuid
        )
        if triggering_event is None:
            # Create a BuildEvent so that the username from the build gets stored in the table
            run_metadata = EntityRunMetadata()
            run_metadata.build_id = self.stored_build.uuid
            run_metadata.username = self.stored_build.username
            triggering_event = BuildEvent(
                run_metadata=run_metadata,
                type=BuildEventType.MESSAGE_EVENT,
                payload=None,
                source=self.event_source,
            )
        message_event = create_message_event(
            source=self.event_source,
            build_id=self.stored_build.uuid,
            level=level,
            message=markdown,
            tiggering_event=triggering_event,
        )
        stored_event = StoredEvent(build_event=message_event)
        assert self.event_storage is not None
        self.event_storage.add(stored_event)


class BuildMultiMessageLogger(AbstractBuildLogger):
    """Simply aggregates 1 or more loggers and call _log() on all of them."""

    def __init__(self: Self, stored_build: StoredBuild, loggers: List[AbstractBuildLogger]) -> None:
        super().__init__(stored_build=stored_build)
        assert len(loggers) > 0, "No loggers provided"
        for curr_logger in loggers:
            assert isinstance(curr_logger, AbstractBuildLogger), f"invalid logger: {curr_logger}"
        self._loggers = loggers

    def _log(
        self: Self,
        level: BuildLogLevel,
        markdown: str,
        triggering_event: Optional[BuildEvent] = None,
    ) -> None:
        for curr_logger in self._loggers:
            assert isinstance(curr_logger, AbstractBuildLogger), f"invalid logger: {curr_logger}"
            curr_logger._log(level=level, markdown=markdown, triggering_event=triggering_event)


# A cache may not be necessary since we only log 1 message per build, but that may change. So use a small cache for now?
@functools.lru_cache(maxsize=4)
def get_message_logger(stored_build: StoredBuild, event_source: str) -> AbstractBuildLogger:
    """Create the message logger for this build.  This will always logs as
    events, but if the give build has a PR associated, it will also log to the PR.
    """
    # TODO: when PRs go away the build_message_logger becomes a BuildEventMessageLogger instead.
    message_event_logger = BuildEventMessageLogger(
        event_source=event_source, stored_build=stored_build
    )
    if not stored_build.source_uri:
        logger = message_event_logger
    else:
        pr_logger = BuildPRLogger(stored_build=stored_build)
        logger = BuildMultiMessageLogger(  # type: ignore[assignment]
            stored_build=stored_build, loggers=[pr_logger, message_event_logger]
        )
    return logger
