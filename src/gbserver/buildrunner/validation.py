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

"""Validate the build."""

import asyncio
import json
import os
import tempfile
import traceback
import urllib.parse
from base64 import b64decode
from pathlib import Path

import yaml
from git import List, Optional

from gbcommon.uri.git import GitURI, split_repo_path
from gbserver.build.build import Build
from gbserver.build.buildrun import BuildRun
from gbserver.build.space import Space
from gbserver.metrics.metrics_client import push_metrics
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.api.builds import BuildValidateRequestType
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    BuildEventValidationDataPayload,
)
from gbserver.types.constants import truncate
from gbserver.types.metrics import (
    Metric,
    MetricMetadata,
    MetricName,
)
from gbserver.types.validation import (
    GBValidationErrors,
    GBValidationErrorsException,
    GBValidationWarningType,
    gather_val_errors_from_exception,
)
from gbserver.utils.archive import extract_archive
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_utc_time, get_uuid

logger = get_logger(__name__)


def _same_space_repo(uri_a: str, uri_b: str) -> bool:
    """Return whether two space URIs refer to the same repository/location.

    Identity is computed per scheme so a bare stored ``git_repo_uri`` and the
    runner's resolved ``space.uristr`` for the same space compare equal without a
    GitHub token:

    * **Remote git** (``https`` / ``git+ssh`` / ``git``, and the scp form
      ``[user@]host:owner/repo`` that git emits): ``(host, owner, repo)`` compared
      case-insensitively, ignoring the scheme, SSH userinfo and port (uses the URL
      *hostname*), a ``.git`` suffix, a pip-style ``@<ref>`` branch suffix, and a
      ``#subdirectory=`` fragment. So ``git+ssh://git@github.ibm.com/o/repo.git@branch``,
      ``https://github.ibm.com/o/repo``, and ``git@github.ibm.com:o/repo.git`` all
      match.
    * **Local** (``file://`` or a bare path): the normalized filesystem path
      (netloc + path, as :func:`gbcommon.uri.space._file_uri_to_path` composes it).
      Two distinct local spaces never collapse to the same identity, and paths
      stay case-sensitive.

    Parsing is total: short or empty URIs yield empty components rather than
    raising, so a genuine mismatch returns ``False`` instead of crashing.

    Args:
        uri_a: A space URI (e.g. a stored space's ``git_repo_uri``).
        uri_b: A space URI (e.g. the runner's ``space.uristr``).

    Returns:
        True when both URIs identify the same repository/location.
    """

    def git_identity(host: str, path: str) -> tuple[str, str, str, str]:
        # owner/repo (+ @<ref> and .git stripping) come from the shared
        # split_repo_path so this doesn't parse git paths in parallel with
        # get_uri_parts. Comparison-specific normalization (case-insensitive host/
        # owner/repo) is applied here.
        owner, repo = split_repo_path(path)
        return ("git", host.lower(), owner.lower(), repo.lower())

    def identity(uri: str) -> tuple[str, ...]:
        # scp-like git URL ([user@]host:owner/repo(.git)) — the form git emits
        # (git@host:org/repo.git). It has no "scheme://", so urlparse would mis-read
        # it as a file path; detect it explicitly (a ':' before any '/', with a
        # non-empty relative tail) and strip an optional "user@".
        if "://" not in uri:
            head, sep, tail = uri.partition(":")
            if sep and "/" not in head and tail and not tail.startswith("/"):
                return git_identity(head.rsplit("@", 1)[-1], tail)
        parsed = urllib.parse.urlparse(uri, scheme="file")
        if parsed.scheme in ("file", ""):
            # Local space: identity is the whole normalized path (no owner/repo
            # structure). netloc + path mirrors _file_uri_to_path; keep it
            # case-sensitive (POSIX paths are).
            path = (parsed.netloc or "") + (parsed.path or "")
            return ("file", os.path.normpath(path) if path else "")
        # Remote git URL: hostname strips any user@ and :port.
        return git_identity(parsed.hostname or "", parsed.path)

    return identity(uri_a) == identity(uri_b)


class BuildValidation:
    """Perform some validation on the build."""

    @staticmethod
    def __create_space_and_build(
        build_archive: str,
        space: Space,
        username: str,
        targets: Optional[List[str]] = None,
        force_fetch: bool = False,
        dry_run: bool = False,
    ) -> Build:
        """Create a space and a build."""
        build_archive_bytes = b64decode(build_archive)
        build_id = get_uuid()
        build_dir = Path(tempfile.mkdtemp()) / build_id
        extract_archive(build_archive_bytes, build_dir)
        event_q: asyncio.Queue[asyncio.Event] = asyncio.Queue()
        build = Build(
            build_dir=build_dir,
            build_id=build_id,
            username=username,
            space=space,
            # workspace_dir=None,
            event_q=event_q,
            targets=targets,
            force_fetch=force_fetch,
        )
        if not dry_run:
            return build
        logger.warning("a validation dry run was requested, running...")
        cancel_on_error = True
        build_run = BuildRun(
            build=build,
            event_q=event_q,
            cancel_on_error=cancel_on_error,
            dry_run=dry_run,
        )

        errors = GBValidationErrors()

        async def run_build_and_wait_on_queue() -> None:
            """Run the build and wait for events forever."""
            build_task = build_run.async_run()
            await build_task
            json_patches = []
            while not event_q.empty():
                event = await event_q.get()
                assert isinstance(event, BuildEvent), f"invalid event: {event}"
                event_str = truncate(str(event))
                logger.info(
                    "\x1b[0;35mGot a new event: %s : %s\x1b[0m", event.type, event_str
                )
                if event.type is not BuildEventType.VALIDATION_DATA_EVENT:
                    continue
                event_payload = event.payload
                if event_payload is None:
                    continue
                assert isinstance(event_payload, BuildEventValidationDataPayload)
                val_data = event_payload.data
                if not isinstance(val_data, str):
                    continue
                # action:replace_config_section:tuning_data_config:
                if not val_data.startswith("action:"):
                    continue
                val_data = val_data.removeprefix("action:")
                if not val_data.startswith("replace_config_section:"):
                    continue
                val_data = val_data.removeprefix("replace_config_section:")
                config_section_name, config_b64 = val_data.split(":")
                try:
                    config_section_str = b64decode(config_b64).decode(encoding="utf-8")
                    config_section = yaml.safe_load(config_section_str)
                    # could be llm.build
                    top_level_key = build.config.matched_base_key or "granite.build"
                    target_name = event.run_metadata.target_name or "placeholder_target"
                    target_step_index = event.run_metadata.target_step_index or 0
                    build_yaml_path = "/" + "/".join(
                        [
                            top_level_key,
                            "targets",
                            target_name,
                            "steps",
                            str(target_step_index),
                            "config",
                            config_section_name,
                        ]
                    )
                    json_patches.append(
                        {
                            "op": "replace",
                            "path": build_yaml_path,
                            "value": config_section,
                        }
                    )
                except Exception as e:
                    logger.warning("skipping, failed to decode recommendation %s", e)
            if json_patches:
                warning = "These are recommendations from the validator"
                solution = json.dumps({"json_patches": json_patches})
                errors.add_warning(
                    warning=warning,
                    type=GBValidationWarningType.RECOMMENDATION,
                    solution=solution,
                )

        asyncio.run(run_build_and_wait_on_queue())
        errors.raise_if_invalid(check_warnings=True)
        return build

    @staticmethod
    def validate_build_archive(
        build_archive: str,
        username: str,
        targets: Optional[List[str]] = None,
        space_or_name: Optional[str | Space] = None,
        space_uri: str = "",
        validation_type: BuildValidateRequestType = BuildValidateRequestType.STATIC,
    ) -> GBValidationErrors:
        """Determine errors, if any, in the build archive (build.yaml).  Check errors.is_valid() for status.

        Args:
            build_archive (str): _description_
            username (str): _description_
            targets (Optional[List[str]], optional): _description_. Defaults to None.
            space_name (str, optional): _description_. Defaults to "".
            space_uri (str, optional): _description_. Defaults to "".

        Raises:
            ValueError: _description_

        Returns:
            GBValidationErrors: _description_
        """
        dry_run = validation_type is BuildValidateRequestType.DYNAMIC
        errors = GBValidationErrors()
        try:
            if space_or_name:
                assert (
                    space_uri == ""
                ), "Only one of 'space_or_name' or 'space_uri' can be specified"
            if isinstance(space_or_name, Space):
                space = space_or_name
            else:
                if (
                    space_or_name
                ):  # A space name at this point, so get the space URI from the db.
                    space_storage: IStoredSpaceStorage = (
                        get_admin_storage().space_storage
                    )

                    stored_space = space_storage.get_by_name(space_or_name)
                    if stored_space is None:
                        raise ValueError(
                            f"Space '{space_or_name}' not found in space storage"
                        )
                    space_uri = GitURI.get_gb_space_config_uri(
                        uri=stored_space.git_repo_uri
                    )
                space = Space(uri=space_uri, username=username)
            logger.info("using Space with uri: %s", space.uristr)
            BuildValidation.__create_space_and_build(
                build_archive=build_archive,
                space=space,
                username=username,
                targets=targets,
                force_fetch=True,
                dry_run=dry_run,
            )
        except GBValidationErrorsException as gbe:
            gbe_errors = gbe.errors
            assert isinstance(gbe_errors, GBValidationErrors)
            errors.add(err=gbe_errors)
        except Exception as e:
            curr_val_err = gather_val_errors_from_exception(e)
            if curr_val_err is None:
                logger.error("%s", traceback.format_exc())
                logger.error("failed to validate the build, error: %s", e)
                errors.add(err=e)
            else:
                errors.add(err=curr_val_err)
        return errors

    @staticmethod
    def validate_stored_build(
        stored_build: StoredBuild, space: Optional[Space] = None
    ) -> GBValidationErrors:
        """Determine errors, if any, in the stored build.  Check errors.is_valid() for status.

        Args:
            stored_build (StoredBuild): _description_
            space(Space): space to use. if not provided, determine the space from the given build's spacename.
        Returns:
            GBValidationErrors: _description_
        """
        validation_start = get_utc_time()
        errors = GBValidationErrors()
        build_id = stored_build.uuid

        if space:
            # Make sure the space the runner loaded matches the build's declared
            # space (by name). Compare the git repo identity (host/owner/repo)
            # against the already-resolved space.uristr rather than re-deriving the
            # full config URI via get_gb_space_config_uri: that call only appends
            # the space-config branch when a GitHub token is available, and the
            # module-level token default is frozen empty when git.py is imported
            # before the token env var is set (e.g. under pytest). Comparing repo
            # identity is token-free and ignores the derived @<branch> /
            # #subdirectory suffix, so a bare stored git_repo_uri and the runner's
            # git+ssh space.uristr for the same repo compare equal.
            space_storage: IStoredSpaceStorage = get_admin_storage().space_storage
            stored_space = space_storage.get_by_name(stored_build.space_name)
            if stored_space is None:
                errors.add(
                    err=f"Could not find space '{stored_build.space_name}' of build."
                )
                return errors
            if not _same_space_repo(stored_space.git_repo_uri, space.uristr):
                errors.add(
                    err=(
                        f"Build space {stored_space.git_repo_uri!r} does not match "
                        f"the runner's space {space.uristr!r}."
                    )
                )
                return errors
        validation_time = -1

        try:
            errors = BuildValidation.validate_build_archive(
                build_archive=stored_build.build_archive,
                username=stored_build.username,
                targets=stored_build.targets,
                space_or_name=space if space else stored_build.space_name,
            )
            errors.raise_if_invalid()
            validation_end = get_utc_time()
            validation_time = (validation_end - validation_start).total_seconds()  # type: ignore[assignment]
            logger.info(
                "the build '%s' is valid (validation took %s seconds)",
                build_id,
                validation_time,
            )
        except Exception as e:
            validation_end = get_utc_time()
            validation_time = (validation_end - validation_start).total_seconds()  # type: ignore[assignment]
            logger.error(
                "the build '%s' is invalid (validation took %s seconds), error: %s",
                build_id,
                validation_time,
                e,
            )
        finally:
            if validation_time >= 0:
                push_metrics(
                    metrics=[
                        Metric(
                            name=MetricName.VALIDATION_TIME,
                            value=validation_time,
                            metadata=MetricMetadata(build_id=build_id),
                        ),
                    ]
                )
        return errors
