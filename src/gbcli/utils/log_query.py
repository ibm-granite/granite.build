import logging
from typing import Optional

import requests
from requests.exceptions import ConnectionError

from gbcli.utils.gbconstants import (
    GBSERVER_LOGS_API,
    USER_NOT_LOGGED_IN_ERROR_MESSAGE,
    VPN_CONNECTION_ERROR_MESSAGE,
)
from gbcli.utils.gbserver import gbserver_post
from gbcli.utils.gh_auth import get_user
from gbcli.utils.utils import convert_seconds_to_milliseconds
from gbcommon.types.gbenvconfig import is_standalone

logger = logging.getLogger(__name__)


# TODO refactor below


def build_query_def(
    start_epoch_in_s: int,
    end_epoch_in_s: int,
    page_size: Optional[int] = None,
    page_index: Optional[int] = None,
    application_name: Optional[str] = None,
    stream: Optional[str] = None,
    text: Optional[str] = None,
    sort: Optional[str] = None,
    build_id: Optional[str] = None,
    build_step_id: Optional[str] = None,
    build_step_name: Optional[str] = None,
    module: str = None,
):
    ## queryDef
    queryDef = {}

    ## queryDef -> date, page, etc
    queryDef["startDate"] = convert_seconds_to_milliseconds(start_epoch_in_s)
    queryDef["endDate"] = convert_seconds_to_milliseconds(end_epoch_in_s)

    if page_size != None:
        queryDef["pageSize"] = page_size
    if page_index != None:
        queryDef["pageIndex"] = page_index

    queryDef["type"] = "freeText"

    ## queryDef -> queryParmas
    queryParams = {}

    if text != None:
        # Lucene text filter
        queryParams["query"] = {"text": text, "type": "exact", "syntax": "Lucene"}

    metadata = {}
    if application_name != None:
        metadata["applicationName"] = [f"{application_name}"]
    if module != None:
        metadata["subsystemName"] = [f"{module}"]
    queryParams["metadata"] = metadata

    jsonObject = {}
    if build_id != None:
        jsonObject["kubernetes.labels.granite-dot-build/build-id"] = [f"{build_id}"]
    if build_step_id != None:
        jsonObject["kubernetes.labels.granite-dot-build/build-step-id"] = [
            f"{build_step_id}"
        ]
    if build_step_name != None:
        jsonObject["kubernetes.labels.granite-dot-build/build-step-name"] = [
            f"{build_step_name}"
        ]
    if stream != None:
        jsonObject["stream"] = [f"{stream}"]
    queryParams["jsonObject"] = jsonObject

    queryDef["queryParams"] = queryParams

    ## queryDef -> sortModel
    if sort != None:
        # Since Cloud Logs can contain log lines with completely idential timestamp, insert an additional sort order to make the order deterministic
        sort0 = {"field": "timestamp", "ordering": f"{sort}", "missing": "_last"}
        # sort1 = {"field": "textObject.time", "ordering": f"{sort}", "missing": "_last"}
        sort1 = {
            "field": "textObject.log",
            "ordering": "asc",
            "initial": False,
            "missing": "_last",
        }
        sort2 = {"field": "logId", "ordering": "asc", "missing": "_last"}
        sortModel = [sort0, sort1, sort2]
        queryDef["sortModel"] = sortModel

    return queryDef


def run_logquery(
    github_token: str,
    start_epoch_in_s: int,
    end_epoch_in_s: int,
    page_size: Optional[int] = None,
    page_index: Optional[int] = None,
    application_name: Optional[str] = None,
    stream: Optional[str] = None,
    text: Optional[str] = None,
    sort: Optional[str] = None,
    build_id: Optional[str] = None,
    build_step_id: Optional[str] = None,
    build_step_name: Optional[str] = None,
    query_server: Optional[bool] = None,
    callback=None,
    module: str = None,
    is_admin: Optional[bool] = None,
):
    username = get_user(github_token).login
    # In standalone mode an empty token is legitimate (the local gbserver allows
    # localhost access when no GBSERVER_API_KEY is configured), so only require a
    # non-empty token outside standalone mode.
    if not username or (not github_token and not is_standalone()):
        raise Exception(USER_NOT_LOGGED_IN_ERROR_MESSAGE)

    url = f"{GBSERVER_LOGS_API}logquery"
    if query_server:
        url += "/server/admin" if is_admin else "/server/" + build_id

    ## queryDef
    queryDef = build_query_def(
        start_epoch_in_s,
        end_epoch_in_s,
        page_size,
        page_index,
        application_name,
        stream,
        text,
        sort,
        build_id,
        build_step_id,
        build_step_name,
        module,
    )

    payload = {"queryDef": queryDef}

    try:
        logger.info(f"Calling {url} with {payload}")
        response = gbserver_post(github_token, url, payload)

        if response.get("Error") != None:
            callback(
                callback_event="error",
                callback_args={"reason": f"Logs server returns '{response['Error']}'"},
            )
            return None

        if response["status"] != 200:
            callback(
                callback_event="error",
                callback_args={"reason": f"Query fails from log server"},
            )
            return None

        return response

    except ConnectionError:
        callback(
            callback_event="error",
            callback_args={
                "reason": f"Logs server connection error. {VPN_CONNECTION_ERROR_MESSAGE}"
            },
        )
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying log server: {e}")
        callback(
            callback_event="error",
            callback_args={"reason": f"No response from log server"},
        )
        return None
