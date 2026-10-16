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
Storing submitted builds in persistent storage.
"""

import datetime
import shutil
import tempfile
from base64 import b64decode, b64encode
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Self, Tuple, Type
from urllib.parse import urlparse

from pydantic import Field

from gbserver.storage.storage import BaseStoredItem, TaggedItem

if TYPE_CHECKING:
    # Imported for typing only — build_storage imports StoredBuild, so a runtime
    # import here would be circular.
    from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.types.buildconfig import BUILD_FILENAME, BuildConfig
from gbserver.types.status import Status
from gbserver.utils import archive
from gbserver.utils.archive import extract_archive
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_sha256sum, get_utc_time

logger = get_logger(__name__)


class StoredBuild(BaseStoredItem, TaggedItem):
    """
    This class is used to store the full details of a submitted build.
    Each instance of this class becomes a row in the gb_builds table in persistent storage.
    """

    name: str
    space_name: str
    source_uri: str
    username: str
    build_archive: str = ""  # PR folder archived and encoded as a base64 string
    description: str = ""
    status: Status = Status.PENDING
    failure_reason: str = ""  # Always set this when the build fails if possible
    targets: Optional[List[str]] = None
    # The name of this field must match that defined in storage.CREATED_TIME_FIELD_NAME
    created_time: datetime.datetime = Field(
        default_factory=get_utc_time, description="Time at which it was created"
    )
    # The name of this field must match that defined in storage.UPDATED_TIME_FIELD_NAME
    updated_time: datetime.datetime = Field(
        default_factory=get_utc_time, description="Time at which it last updated"
    )
    build_config_cache: Dict[str, BuildConfig] = Field(default_factory=dict)
    retry_count: int = (
        0  # Number of automatic retries that have been submitted for this build
    )

    @classmethod
    def create(
        cls: Type[Self],
        name: str,
        space_name: str,
        source_uri: str,
        username: str,
        **kwargs,
    ) -> Self:
        """_summary_

        Args:
            name (str): a name for the build
            space_name (str): name of the space where the build should run
            source_uri (str): e.g. a pull request URL
            username (str): username of who requested the build
            kwargs:
                build_yaml_path(Union[str,Path]): path to a build.yaml file
                targets[Optional[list[str]]]
                description[Optional[str]]
                tags[Optional[list[str]]]
        """
        build_archive = kwargs.get("build_archive", "")
        assert isinstance(build_archive, str), f"invalid build_archive: {build_archive}"
        status = kwargs.get("status", Status.PENDING)
        assert isinstance(status, Status), f"invalid status: {status}"
        targets = kwargs.get("targets", None)
        assert targets is None or isinstance(
            targets, list
        ), f"invalid targets: {targets}"
        tags = kwargs.get("tags", None)
        assert tags is None or isinstance(tags, list), f"invalid tags: {tags}"
        description = kwargs.get("description", "")
        if description is None:
            description = ""
        assert isinstance(description, str), f"invalid description: {description}"
        stored_build = cls(
            name=name,
            space_name=space_name,
            source_uri=source_uri,
            username=username,
            build_archive=build_archive,
            status=status,
            targets=targets,
            description=description,
            tags=tags,
        )
        if build_archive == "":
            build_yaml_path = kwargs.get("build_yaml_path")
            assert build_yaml_path is not None, "build_yaml_path must be provided"
            if isinstance(build_yaml_path, str):
                build_yaml_path = Path(build_yaml_path)
            assert isinstance(build_yaml_path, Path)
            assert (
                build_yaml_path.is_file()
            ), f"Build yaml does not exist: {build_yaml_path}"
            build_dir = build_yaml_path.parent
            # logger.debug(f"build_dir={build_dir}")
            assert build_dir.is_dir(), f"{build_dir} is not a valid directory"
            if build_yaml_path.name == BUILD_FILENAME:
                stored_build.save_to_build_archive_from_dir(build_dir)
            else:
                # yaml is not named build.yaml — copy it into a temp dir under the correct name
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_build_dir = Path(tmp_dir)
                    shutil.copy2(build_yaml_path, tmp_build_dir / BUILD_FILENAME)
                    stored_build.save_to_build_archive_from_dir(tmp_build_dir)
        return stored_build

    def load_from_build_archive(self: Self) -> bytes:
        """Decode the base64 string to bytes in zip archive format and return it."""
        return b64decode(self.build_archive)

    def save_to_build_archive(self: Self, data: bytes) -> None:
        """Encode the bytes to a base64 string and save it."""
        self.build_archive = b64encode(data).decode("utf-8")

    def save_to_build_archive_from_dir(self: Self, build_dir: Path) -> None:
        """Compress the directory as a zip archive and save it."""
        assert build_dir.is_dir(), f"expected {build_dir} to be a directory"
        data = archive.create_archive_bytes(dir=build_dir, format="zip")
        self.save_to_build_archive(data=data)

    def get_pr_info(self: Self) -> Tuple[str, str, str]:
        """
        Gathers PR info from source_uri
        Example: https://github.ibm.com/granite-dot-build/gb-test/pull/11
        will return ("granite-dot-build", "gb-test", "11")

        Returns (owner, repo, pr_id)
        """
        if self.source_uri == "":
            return ("", "", "")
        stored_build_url = urlparse(self.source_uri)
        owner, repo, pull, pr_id = Path(stored_build_url.path).parts[-4:]
        assert pull == "pull", f"invalid source_uri: {self.source_uri}"
        pr_num = int(pr_id, base=10)
        assert pr_num > 0, f"invalid source_uri: {self.source_uri}"
        return owner, repo, pr_id

    def get_build_config(self: Self, validate: bool = False) -> BuildConfig:
        """Get the corresponding build config"""
        build_archive_bytes = self.load_from_build_archive()
        checksum = get_sha256sum(build_archive_bytes)
        if checksum in self.build_config_cache:
            return self.build_config_cache[checksum]
        build_dir = Path(tempfile.mkdtemp()).resolve()
        extract_archive(build_archive_bytes, build_dir)
        build_yaml_path = build_dir / BUILD_FILENAME
        assert build_yaml_path.is_file(), f"expected '{build_yaml_path}' to be a file"
        build_config = BuildConfig.from_yaml(path=build_yaml_path, validate=validate)
        shutil.rmtree(build_dir, ignore_errors=True)
        # remove old stuff in case build_archive was modified
        self.build_config_cache = {}
        # cache only the latest
        self.build_config_cache[checksum] = build_config
        return build_config

    def has_retries_remaining(self: Self) -> bool:
        """Return True if this build is still eligible for another automatic retry.

        True when the build config specifies a positive ``max_retries`` and fewer
        retries have been attempted so far than that limit. This is independent of
        the build's current status — callers combine it with a status check as
        needed (e.g. the retry loop only retries a FAILED build). Reading the build
        config can fail (e.g. a missing/corrupt archive); any such failure is
        treated conservatively as "no retries remaining".

        Returns:
            bool: whether another retry attempt is permitted.
        """
        try:
            build_config = self.get_build_config()
        except Exception as e:
            logger.warning(
                "Could not read build config for build %s, treating as no retries "
                "remaining: %s",
                self.uuid,
                e,
            )
            return False
        if build_config.retries.max_retries <= 0:
            return False
        return self.retry_count < build_config.retries.max_retries

    def __hash__(self) -> int:
        """Make StoredBuild hashable using all fields."""
        return hash(
            (
                self.uuid,
                self.name,
                self.space_name,
                self.source_uri,
                self.username,
                self.build_archive,
                self.description,
                self.status,
                tuple(self.targets) if self.targets else None,
                tuple(self.tags) if self.tags else None,
                self.created_time,
                self.updated_time,
            )
        )

    def __eq__(self, other: object) -> bool:
        """Two StoredBuilds are equal if all their fields match."""
        if not isinstance(other, StoredBuild):
            return NotImplemented
        return (
            self.uuid == other.uuid
            and self.name == other.name
            and self.space_name == other.space_name
            and self.source_uri == other.source_uri
            and self.username == other.username
            and self.build_archive == other.build_archive
            and self.description == other.description
            and self.status == other.status
            and self.targets == other.targets
            and self.tags == other.tags
            and self.created_time == other.created_time
            and self.updated_time == other.updated_time
        )


def reopen_finished_build(
    build_storage: "IStoredBuildStorage", build: StoredBuild
) -> Optional[StoredBuild]:
    """Re-open a finished, non-SUCCESS build in place so it can be restarted.

    A restart reuses the **same** build id: the finished build is flipped
    back to ``SUBMITTED`` so the BuildWatcher re-dispatches it through the ordinary
    submission path onto a fresh runner (the watcher's status-scoped, garbage-
    collected seen-set treats the re-``SUBMITTED`` id as newly unseen). The retry
    budget is reset (``retry_count -> 0``) so the build.yaml ``max_retries`` is
    counted fresh for the restart, and any prior ``failure_reason`` is cleared.

    A build that finished with status ``SUCCESS`` is **not** re-openable: every
    target already succeeded, so a restart would reuse them all and do no work.
    Only a finished build that did not fully succeed (``FAILED``, ``INVALID``, or
    ``CANCELLED``) can be restarted.

    Target reuse is driven by the build's existing SUCCESS target runs (see
    ``BuildRunner.__is_target_already_run``), not by ``retry_count``, so targets
    that already succeeded are skipped and only the failed/unfinished ones re-run.

    The flip is atomically guarded on the build still being finished-and-not-
    SUCCESS, so a restart racing an already-live runner, a second concurrent
    restart, or a run that just succeeded is rejected rather than attaching a
    fresh runner or re-opening a completed build.

    Args:
        build_storage: storage used to atomically update the build.
        build: the finished build to restart (its status must satisfy
            ``is_finished()`` and must not be ``SUCCESS``).

    Returns:
        The re-opened ``StoredBuild``, or ``None`` if the guard rejected the flip
        because the build was no longer finished, or had become ``SUCCESS`` (a
        concurrent writer won the race).
    """
    return build_storage.update_fields(
        build.uuid,
        {"status": Status.SUBMITTED, "retry_count": 0, "failure_reason": ""},
        should_update=lambda item: item.status.is_finished()
        and item.status != Status.SUCCESS,
    )
