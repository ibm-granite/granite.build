import logging
import time
from typing import Optional

from gbcli.utils.gbconstants import (
    BUILD_LOGALL_PAGE_SIZE,
    BUILD_LOG_DEFAULT_QUERY_RANGE,
    BUILD_LOG_FOLLOW_SLEEP_TIME,
    GBSERVER_BUILD_API,
    PROJECT_NAME,
    gb_environment_config,
)
from gbcli.utils.gbserver import get_builds, make_gbserver_call
from gbcli.utils.log_query import run_logquery
from gbcli.utils.spaceutil import resolve_space
from gbcli.utils.utils import (
    change_timestamp_by_days,
    check_current_timestamp,
    convert_milliseconds_to_seconds,
    get_current_epoch,
)

logger = logging.getLogger(__name__)


def server_log(
    github_token: str,
    module: str,
    id_format: str,
    start_epoch: Optional[int] = None,
    end_epoch: Optional[int] = None,
    page_size: Optional[int] = None,
    page_index: Optional[int] = None,
    stream: Optional[str] = None,
    text: Optional[str] = None,
    sort: Optional[str] = None,
    build_id: Optional[str] = None,
    build_step_id: Optional[str] = None,
    build_step_name: Optional[str] = None,
    follow: Optional[bool] = False,
    all: Optional[bool] = False,
    callback=None,
):
    global_space = resolve_space(github_token, "public", callback=callback)
    if global_space is None or not global_space["is_admin"]:
        raise Exception(
            f"Error: Server logs are available to {PROJECT_NAME} admin users only."
        )

    if build_id:
        if id_format == "url":
            build_id_from_url = get_build_id_from_url(github_token, build_id, callback)
            build_id = build_id_from_url[0]["uuid"]

    current_epoch = get_current_epoch()
    if start_epoch == None:
        start_epoch = change_timestamp_by_days(
            current_epoch, BUILD_LOG_DEFAULT_QUERY_RANGE
        )
    if end_epoch == None:
        end_epoch = current_epoch
    if callback:
        callback(
            callback_event="querying_log_range",
            callback_args={
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
            },
        )
    if all:
        page_size = BUILD_LOGALL_PAGE_SIZE
    elif page_size == None:
        page_size = 50

    if all or follow:
        page_index = 0
        displayed_logs_ids = []
        next_timestamp = start_epoch
        continue_logquery = True
        is_current_timestamp = False
    else:
        if page_index == None:
            page_index = 0

    if callback:
        callback(
            callback_event="querying_log",
            callback_args={
                "start_epoch": start_epoch,
                "end_epoch": end_epoch if end_epoch else round(time.time()),
            },
        )

    application_name = gb_environment_config()["server_log_application_name"]

    if not all or follow:
        if all:
            sort = "asc"
        elif sort == None:
            sort = "desc"
        response = run_logquery(
            github_token,
            start_epoch,
            end_epoch,
            page_size,
            page_index,
            application_name,
            stream,
            text,
            sort,
            build_id,
            build_step_id,
            build_step_name,
            True,
            callback,
            module,
            is_admin=True,
        )

        logs = response["logs"]

        if follow:
            if logs != None:
                # query is successful
                if len(logs) > 0:
                    next_timestamp = convert_milliseconds_to_seconds(
                        logs[len(logs) - 1]["timestamp"]
                    )
                    if sort == "desc":
                        logs.reverse()
                    if callback:
                        callback(
                            callback_event="display_logs",
                            callback_args={"logs": logs},
                        )
                    displayed_logs_ids = [log["logId"] for log in logs]
        else:
            return logs

    if all or follow:
        sort = "asc"
        while continue_logquery or follow:
            if follow:
                time.sleep(BUILD_LOG_FOLLOW_SLEEP_TIME)

            end_epoch, is_current_timestamp = check_current_timestamp(
                change_timestamp_by_days(
                    next_timestamp, BUILD_LOG_DEFAULT_QUERY_RANGE, True
                )
            )

            response = run_logquery(
                github_token,
                next_timestamp,
                end_epoch,
                page_size,
                page_index,
                application_name,
                stream,
                text,
                sort,
                build_id,
                build_step_id,
                build_step_name,
                True,
                callback,
                module,
                is_admin=True,
            )

            if response["logs"]:
                logs = [
                    log
                    for log in response["logs"]
                    if (not displayed_logs_ids)
                    or (displayed_logs_ids and (log["logId"] not in displayed_logs_ids))
                ]
                if logs and len(logs) > 0:
                    timestamps = []
                    for log in logs:
                        timestamp = convert_milliseconds_to_seconds(log["timestamp"])
                        if timestamp not in timestamps:
                            timestamps.append(timestamp)
                    next_timestamp = timestamps[len(timestamps) - 2]
                    displayed_logs_ids = displayed_logs_ids + [
                        log["logId"] for log in logs
                    ]
                    if callback:
                        callback(
                            callback_event="display_logs",
                            callback_args={"logs": logs},
                        )
            if not response["logs"] or not logs or len(logs) == 0:
                if not follow and is_current_timestamp:
                    continue_logquery = False
                else:
                    next_timestamp, is_current_timestamp = check_current_timestamp(
                        change_timestamp_by_days(
                            next_timestamp, BUILD_LOG_DEFAULT_QUERY_RANGE, True
                        ),
                        True,
                    )

        return displayed_logs_ids


# TODO remove duplicate from service_build.py
def get_build_id_from_url(user_token: str, build_url: str, callback=None) -> list:
    if callback is not None:
        callback(
            callback_event="fetching_build_id",
            callback_args={"steps": 1, "source_uri": build_url},
        )

    build_from_url = make_gbserver_call(
        lambda: get_builds(user_token, GBSERVER_BUILD_API, source_uri=build_url)[
            "builds"
        ],
        callback,
    )

    logger.debug(f"Found {len(build_from_url)} builds with 'source_uri = {build_url}'.")
    if len(build_from_url) == 0:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": f"No builds were found with URL {build_url}."},
            )
        return None

    if callback is not None:
        build_id = build_from_url[0]["uuid"]
        callback(
            callback_event="fetched_build_id",
            callback_args={"steps": 100, "build_id": build_id, "source_uri": build_url},
        )

    return build_from_url
