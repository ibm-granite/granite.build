#!/usr/bin/env python3

# Copyright Granite.Logs Authors
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

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from gbserver.api.utils import get_lh_token_if_needed
from gbserver.spaces.user_spaces_list import build_id_access_check, space_admin_check
from gbserver.types.logs import Item, LogqueryResponse
from gbserver.utils.cloud_logquery import get_log_manager
from gbserver.utils.cloud_logquery_server import get_log_server_manager
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


logs_api = FastAPI()


@logs_api.post("/logquery")
def logquery(query: Item) -> LogqueryResponse:
    log_manager = None
    try:
        log_manager = get_log_manager()
    except Exception as e:
        logger.error("failed to get the log_manager, error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the logs manager was not configured!",
        )
    try:
        logs = log_manager.query_cloud_logquery(query)
        return logs
    except Exception as e:
        logger.error("failed to query the cloud logs API: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Logquery not found!"
        )


@logs_api.post("/logquery/server/admin")
def logquery(request: Request, query: Item) -> LogqueryResponse:
    log_server_manager = None
    try:
        log_server_manager = get_log_server_manager()
    except Exception as e:
        logger.error("failed to get the log_server_manager, error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the logs manager was not configured for server/admin logs!",
        )

    username = request.state.data["user"].email
    lh_token = get_lh_token_if_needed(request)

    is_admin = space_admin_check(username, lh_token=lh_token)
    if not is_admin:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "detail": "Unauthorized: Server logs are available to LLM.Build admin users only!",
            },
        )

    try:
        logs = log_server_manager.query_cloud_logquery(query)
        return logs
    except Exception as e:
        logger.error("failed to query the cloud logs API: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Logquery not found!"
        )


@logs_api.post("/logquery/server/{build_id}")
def logquery(request: Request, build_id: str, query: Item) -> LogqueryResponse:
    log_server_manager = None
    try:
        log_server_manager = get_log_server_manager()
    except Exception as e:
        logger.error("failed to get the log_server_manager, error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the logs manager was not configured for server/admin logs!",
        )

    username = request.state.data["user"].email
    lh_token = get_lh_token_if_needed(request)

    """ Set query model name to 'gbserver-build-runner' """
    query.queryDef.queryParams.metadata["subsystemName"] = ["gbserver-build-runner"]

    has_access = build_id_access_check(username, build_id, lh_token=lh_token)

    if not isinstance(has_access, bool):
        return has_access
    elif not has_access:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "detail": "Unauthorized: Server logs are available to LLM.Build users only!",
            },
        )

    try:
        logs = log_server_manager.query_cloud_logquery(query)
        return logs
    except Exception as e:
        logger.error("failed to query the cloud logs API: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Logquery not found!"
        )
