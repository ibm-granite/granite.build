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

# Usage: please see main as example
# export INGESTION_API_KEY=`ibmcloud iam api-key-create logs-ingestion --output json | jq -r '.apikey'`
# export IAM_TOKEN=`ibmcloud iam oauth-tokens --output json | jq -r '.iam_token'`

import os
import time
from typing import Optional, Self

import requests

from gbserver.types.constants import (
    FETCH_CLOUD_LOGS_MAX_RETRIES,
    FETCH_CLOUD_LOGS_PR_MAX_CHARS,
    FETCH_CLOUD_LOGS_RETRY_INTERVAL,
    GBSERVER_IBM_CLOUD_LOGS_API_KEY,
    GBSERVER_IBM_CLOUD_LOGS_API_URL,
)
from gbserver.types.logs import Item, LogqueryResponse
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class IBMCloudLogQueryAPI:
    """
    Logquery Manager API client.
    """

    api_endpoint: str
    api_key: str = ""
    token: str = ""
    token_expiration: int = 0

    def __init__(self, api_endpoint: str, api_key: str = ""):
        """
        Args:
            api_endpoint (str): _description_
            api_key (str, optional): IAM api Key,
            token (str, optional): Auth token, only one of api_key or token is required
        """
        logger.info("IBMCloudLogQueryAPI.__init__ api_endpoint %s", api_endpoint)
        assert api_endpoint != "", "must specify an API endpoint"
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        if self.api_key == "":
            assert self.token != "", "must specify either an API key or an IAM token"

    def get_new_token(self: Self) -> None:
        """
        Get a new token.
        https://cloud.ibm.com/docs/account?topic=account-iamtoken_from_apikey
        """
        logger.debug("get_new_token start")
        assert self.api_key != "", "cannot request a token, no API key specified"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = (
            f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={self.api_key}"
        )
        response = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers=headers,
            data=data,
            timeout=10 * 60,
        )
        response.raise_for_status()
        data = response.json()
        assert isinstance(data, dict)
        self.token = data["access_token"]
        self.token_expiration = data["expiration"]
        logger.debug("get_new_token end")

    def query_cloud_logquery(self: Self, query: Item) -> LogqueryResponse:
        logger.debug("query_cloud_logquery start")
        if self.token == "":
            self.get_new_token()
        else:
            current_time = time.time()
            if current_time > self.token_expiration:
                self.get_new_token()
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        url = f"{self.api_endpoint}/api/v1/logquery"
        try:
            # Make a POST request to the external API with headers
            # turn on stream flag
            data = query.model_dump(exclude_none=True)
            logger.debug(
                "url, headers, json, %s %s %s",
                url,
                headers,
                data,
            )
            response = requests.post(url, headers=headers, json=data, stream=True)
            response.raise_for_status()
            t1 = response.json()
            t2 = LogqueryResponse.model_validate(t1)
            return t2
        except Exception as e:
            # Handle exceptions, e.g., connection errors
            return LogqueryResponse(status=400, error=str(e))

    def get_build_logs(
        self: Self,
        build_id: str,
        empty_ok: bool = False,
        max_retries: int = FETCH_CLOUD_LOGS_MAX_RETRIES,
        retry_interval: int = FETCH_CLOUD_LOGS_RETRY_INTERVAL,
        max_size: int = FETCH_CLOUD_LOGS_PR_MAX_CHARS,
    ) -> str:
        """
        Get the logs of a build.
        Will retry if empty_ok=True and the logs are empty string.
        """
        retry_count = 0
        logs_str = ""
        query = Item.get_logs_for_build(build_id=build_id)
        while (logs_str.strip() == "") and (retry_count < max_retries):
            try:
                logger.info("get_build_logs retry_count: %d", retry_count)
                retry_count += 1
                logs = self.query_cloud_logquery(query)
                logs_str = logs.output_format_plain(
                    reverse=True,  # sort old -> new
                    max_size=max_size,
                )
                logger.info("get_build_logs logs length: %d", len(logs_str))
                if logs_str.strip() == "":
                    logger.info("get_build_logs got empty logs")
                    if empty_ok:
                        break
                    logger.info("get_build_logs sleep for %d seconds", retry_interval)
                    time.sleep(retry_interval)
            except Exception as e:
                logger.error(
                    "failed to fetch the logs for the build %s : %s",
                    build_id,
                    e,
                )
        logger.info("get_build_logs end logs length: %d", len(logs_str))
        return logs_str


_LOG_MANAGER = None


def get_log_manager():
    global _LOG_MANAGER
    if _LOG_MANAGER is not None:
        return _LOG_MANAGER
    if os.getenv("GB_ENVIRONMENT", "").upper() == "STANDALONE":
        from gbserver.utils.local_logquery import LocalLogQueryAPI

        _LOG_MANAGER = LocalLogQueryAPI()
        logger.info("created a local log query API client (standalone mode)")
    else:
        assert (
            GBSERVER_IBM_CLOUD_LOGS_API_URL != ""
        ), "GBSERVER_IBM_CLOUD_LOGS_API_URL has not been set"
        assert (
            GBSERVER_IBM_CLOUD_LOGS_API_KEY != ""
        ), "GBSERVER_IBM_CLOUD_LOGS_API_KEY has not been set"
        _LOG_MANAGER = IBMCloudLogQueryAPI(
            api_endpoint=GBSERVER_IBM_CLOUD_LOGS_API_URL,
            api_key=GBSERVER_IBM_CLOUD_LOGS_API_KEY,
        )
        logger.info("created an IBM Cloud Logs API client %s", _LOG_MANAGER)
    return _LOG_MANAGER
