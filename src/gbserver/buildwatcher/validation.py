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
import tempfile
import traceback
from base64 import b64decode
from pathlib import Path

import yaml
from git import List, Optional

from gbcommon.uri.git import GitURI
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
            # Make sure the space matches the build's space
            space_storage: IStoredSpaceStorage = get_admin_storage().space_storage
            stored_space = space_storage.get_by_name(stored_build.space_name)
            assert (
                stored_space
            ), f"Could not find space {stored_build.space_name} of build."
            space_uri = GitURI.get_gb_space_config_uri(uri=stored_space.git_repo_uri)
            assert (
                space_uri == space.uristr
            ), f"Derived build space uri {space_uri} from space {stored_space}, does not match the given space {space.uristr}"
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
            validation_time = (validation_end - validation_start).total_seconds()
            logger.info(
                "the build '%s' is valid (validation took %s seconds)",
                build_id,
                validation_time,
            )
        except Exception as e:
            validation_end = get_utc_time()
            validation_time = (validation_end - validation_start).total_seconds()
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
