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
Utils for posting comments to PRs
"""

from gbserver.buildrunner.buildlogger import (
    AbstractBuildLogger,
)
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import (
    DASHBOARD_LINK_MESSAGE_FOR_BUILD,
    HELP_INSTRUCTIONS_FOR_BUILD,
    LINEAGE_LINK_MESSAGE_FOR_BUILD,
)
from gbserver.utils.mermaid_graph import get_mermaid_graph_str
from gbserver.utils.utils import get_build_status_link, get_dashboard_link


def post_validated_build_message(
    build_message_logger: AbstractBuildLogger,
    stored_build: StoredBuild,
    build_err: str = "",
) -> None:
    """
    Post some useful messages (as message events):
    - CLI commands to get logs or validation error
    - Lineage link
    - build.yaml mermaid graph
    """
    # Log the build has been posted
    # TODO also post a ValidationEvent to event storage using BuildEventLogger.
    build_id = stored_build.uuid
    # logger.debug("the version matches so validate the schema")
    build_config = stored_build.get_build_config()
    # logger.debug("build_config: %s", build_config)
    graph_str = get_mermaid_graph_str(build_config=build_config)
    is_valid_build = build_err == ""
    if not is_valid_build:
        body = f"The build is invalid.\n```\n{build_err}\n```\n"
        build_message_logger.error(body)
    else:
        body = "👍  Validation succeeded!\n\n" + HELP_INSTRUCTIONS_FOR_BUILD.format(
            build_id=build_id
        )
        build_message_logger.info(body)
    # pr_id = "<pr number>"
    # logger.info("posting a comment to the pr_id: %s body: %s", pr_id, body)
    if is_valid_build:
        body = LINEAGE_LINK_MESSAGE_FOR_BUILD.format(
            build_status_link=get_build_status_link(build_id)
        )
        body += DASHBOARD_LINK_MESSAGE_FOR_BUILD.format(
            dashboard_link=get_dashboard_link(build_id)
        )
        build_message_logger.info(body)
    if graph_str != "":
        body = f"## Build Graph\n\n```mermaid\n{graph_str}\n```\n"
        build_message_logger.info(body)
