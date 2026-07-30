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

from gbserver.spaces.user_spaces_list import build_id_access_check, space_admin_check
from gbserver.types.logs import Item, LogqueryResponse
from gbserver.utils.cloud_logquery import get_log_manager
from gbserver.utils.cloud_logquery_server import get_log_server_manager
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Cloud Logs identifies a build's logs by this k8s label key, set by the
# client for build-scoped queries (see Item.get_logs_for_build). Defined once
# here rather than duplicated per literal, since both handlers below key off
# it for authorization, not just filtering.
BUILD_ID_LABEL_KEY = "kubernetes.labels.granite-dot-build/build-id"

# FREE_TEXT_SCOPE_ASSUMPTION: both routes below enforce the build-id
# authorization check only against queryDef.queryParams.jsonObject, then send
# the whole Item (jsonObject + queryParams.query free-text search + metadata)
# to Cloud Logs in one request. This assumes Cloud Logs intersects (ANDs)
# every field of a single query rather than treating queryParams.query as an
# independent/OR'd match path — i.e. that the jsonObject build-id constraint
# can only narrow what queryParams.query matches, never be bypassed by it.
#
# Accepted as of this change: one combined request body (not a "should"/
# "must" split) is the standard shape for conjunctive multi-field query APIs,
# and queryParams.query can't simply be stripped — it backs the documented
# `gb build log --text` CLI flag (gbcli/utils/log_query.py's build_query_def
# sets it alongside the build-id filter for a real, supported feature). Not
# verified against IBM Cloud Logs' real behavior (no accessible docs, no test
# backend for it in this repo) — revisit if that's ever obtained, or if this
# endpoint's actual behavior in production ever contradicts it.


logs_api = FastAPI()


def _require_dict_json_object(query: Item) -> dict:
    """Validate/normalize queryDef.queryParams.jsonObject to a dict.

    jsonObject is typed Any and forwarded to Cloud Logs essentially verbatim.
    A non-dict value (or one with a non-list build-id entry) would silently
    bypass the per-key authorization checks below rather than fail closed, so
    reject it outright instead of skipping the check.
    """
    query_params = query.queryDef.queryParams
    if query_params is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="queryParams is required"
        )
    json_object = query_params.jsonObject
    if json_object is None:
        json_object = {}
        query_params.jsonObject = json_object
    if not isinstance(json_object, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="queryParams.jsonObject must be a JSON object",
        )
    return json_object


@logs_api.post("/logquery")
def logquery(request: Request, query: Item) -> LogqueryResponse:
    log_manager = None
    try:
        log_manager = get_log_manager()
    except Exception as e:
        logger.error("failed to get the log_manager, error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the logs manager was not configured!",
        )

    # This endpoint has no build_id path param. Cloud Logs supports filtering
    # by free text, other labels, or metadata, none of which are checked here
    # — so rather than only checking when a build-id filter happens to be
    # present (which a caller can simply omit to read any tenant's logs by
    # some other dimension), every query must be scoped to a build id the
    # caller can access.
    json_object = _require_dict_json_object(query)
    build_ids = json_object.get(BUILD_ID_LABEL_KEY, [])
    if not isinstance(build_ids, list) or not build_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"queryParams.jsonObject must filter by {BUILD_ID_LABEL_KEY}",
        )
    # Force the query mode server-side rather than trusting the caller's
    # value — the authorized-build-id constraint above assumes Cloud Logs
    # intersects jsonObject with the rest of the query; an unpinned type
    # could plausibly select a different query mode with different (unknown)
    # combination semantics. This does not by itself prove queryParams.query
    # (free text, kept — see FREE_TEXT_SCOPE_ASSUMPTION) can't independently
    # surface unscoped results; that assumption is documented, not verified.
    query.queryDef.type = "freeText"

    username = request.state.data["user"].email
    for build_id in build_ids:
        has_access = build_id_access_check(username, build_id)
        if not isinstance(has_access, bool):
            return has_access
        elif not has_access:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": f"Unauthorized: no access to build {build_id}!",
                },
            )

    try:
        logs = log_manager.query_cloud_logquery(query)
        return logs
    except Exception as e:
        logger.error("failed to query the cloud logs API: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Logquery not found!"
        )


@logs_api.post("/logquery/server/admin")  # type: ignore[no-redef]
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

    is_admin = space_admin_check(username)
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


@logs_api.post("/logquery/server/{build_id}")  # type: ignore[no-redef]
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

    # Validate before use: jsonObject is attacker-controlled request-body
    # content, and a non-dict value would previously skip the build-id
    # override below entirely, re-opening the path/body smuggling bypass.
    json_object = _require_dict_json_object(query)

    """ Set query model name to 'gbserver-build-runner' """
    query.queryDef.queryParams.metadata["subsystemName"] = ["gbserver-build-runner"]

    has_access = build_id_access_check(username, build_id)

    if not isinstance(has_access, bool):
        return has_access
    elif not has_access:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "detail": "Unauthorized: Server logs are available to LLM.Build users only!",
            },
        )

    # The build_id above (from the path) is what was actually authorized.
    # A caller could otherwise pass their own build_id in the path
    # (authorized) while smuggling a different, unauthorized build's id into
    # the body filter and have that filter forwarded unmodified. Force it to
    # the authorized build, preserving any other filter keys (step id/name,
    # stream) the caller set.
    json_object[BUILD_ID_LABEL_KEY] = [build_id]
    # See FREE_TEXT_SCOPE_ASSUMPTION above the module-level constants.
    query.queryDef.type = "freeText"

    try:
        logs = log_server_manager.query_cloud_logquery(query)
        return logs
    except Exception as e:
        logger.error("failed to query the cloud logs API: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Logquery not found!"
        )
