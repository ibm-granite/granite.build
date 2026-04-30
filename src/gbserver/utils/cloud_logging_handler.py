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

"""Cloud logging handler module."""

import logging
import os

# import Alexei's secret manager API
import sys
from typing import Optional

import requests

sys.path.append("../secretsmanager")
from sm import SecretManager

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class IBMCloudLogger(logging.StreamHandler):
    """I B M Cloud Logger implementation."""

    def __init__(
        self,
        endpoint: str,
        buildid: str,
        stream: str = "stdout",
        buildstepid: Optional[str] = None,
        buildstepname: Optional[str] = None,
    ):
        super().__init__()
        self.endpoint = endpoint
        self.buildid = buildid
        self.stream = stream
        if buildstepid:
            self.buildstepid = buildstepid
        else:
            self.buildstepid = "NOT SET"
        if buildstepname:
            self.buildstepname = buildstepname
        else:
            self.buildstepname = "NOT SET"
        self.get_iam_token()

    def get_iam_token(self):
        """Get the iam token."""
        cloud_api_key = os.environ.get("IBM_CLOUD_API_KEY")
        secret_manager_url = os.environ.get("IBM_CLOUD_SECRETS_MANAGER_SERVICE_URL")
        if (
            secret_manager_url is None
            or secret_manager_url == ""
            or cloud_api_key is None
            or cloud_api_key == ""
        ):
            print(
                f"Fatal error: environment variable IBM_CLOUD_API_KEY or "
                f"IBM_CLOUD_SECRETS_MANAGER_SERVICE_URL are not found"
            )
            sys.exit(1)

        sm = SecretManager(serviceurl=secret_manager_url)
        log_api_key = sm.get_secret(
            "VELA_LOGGING_ACCESS_KEY", secret_group_name="vela-logging-REST-access"
        )
        token_url = "https://iam.cloud.ibm.com/identity/token"
        ingestion_api_key = log_api_key["payload"]
        token_headers = {"Content-Type": "application/x-www-form-urlencoded"}
        token_data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={ingestion_api_key}"
        try:
            response = requests.post(token_url, headers=token_headers, data=token_data)
            response.raise_for_status()  # Raise an exception for error HTTP statuses
            self.iam_token = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Fatal error to acquire IAM_TOKEN to submit logs: {e}")
            sys.exit(1)

    def emit(self, record: logging.LogRecord) -> None:
        severity = {
            logging.CRITICAL: 5,
            logging.ERROR: 4,
            logging.WARNING: 3,
            logging.INFO: 2,
            logging.DEBUG: 1,
        }
        log_msg_body = {
            "stream": self.stream,
            "log": record.msg,
            "granite-dot-build/build-id": self.buildid,
            "granite-dot-build/build-step-id": self.buildstepid,
            "granite-dot-build/build-step-name": self.buildstepname,
        }
        cloud_log_json = {
            "applicationName": "granite-build",
            "text": log_msg_body,
            "severity": severity[record.levelno],
        }

        try:
            headers = {
                "Authorization": self.iam_token["access_token"],
                "Content-Type": "application/json",  # If sending JSON data
            }
            response = requests.post(self.endpoint, headers=headers, json=cloud_log_json)
            response.raise_for_status()  # Raise an exception for error HTTP statuses
        except requests.exceptions.RequestException as e:
            print(f"Error sending log message: {e}")


def add_cloud_log_handler(
    default_logger: logging.Logger,
    endpoint: str,
    buildid: str,
    stream: str = "stdout",
    buildstepid: Optional[str] = None,
    buildstepname: Optional[str] = None,
) -> None:
    """Add cloud log handler."""
    cloud_log_handler = IBMCloudLogger(endpoint, buildid, stream, buildstepid, buildstepname)
    default_logger.addHandler(cloud_log_handler)


if __name__ == "__main__":
    print(
        f"Dependent on environment variables IBM_CLOUD_API_KEY and IBM_CLOUD_SECRETS_MANAGER_SERVICE_URL"
    )
    IBMCLoudLogInstance_endpoint = "https://597e0cf2-90e2-47e8-8085-8d74afabeb14.ingress.us-east.logs.cloud.ibm.com/logs/v1/singles"
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    add_cloud_log_handler(
        logger,
        endpoint=IBMCLoudLogInstance_endpoint,
        buildid="582c795a-d7cf-4297-8462-b21ac85112a3",
        stream="stdout",
        buildstepid="c95bf125-078f-47f9-a495-0fdafb4c9246",
        buildstepname="Test",
    )
    import datetime
    import time

    logger.debug(f'This is a DEBUG message {datetime.datetime.now().strftime("%H:%M:%S")}')
    time.sleep(3)
    logger.info(f'This is an INFO message {datetime.datetime.now().strftime("%H:%M:%S")}')
    time.sleep(3)
    logger.warning(f'This is a WARNING message {datetime.datetime.now().strftime("%H:%M:%S")}')
    time.sleep(3)
    logger.error(f'This is an ERROR message {datetime.datetime.now().strftime("%H:%M:%S")}')
    time.sleep(3)
    logger.critical(f'This is a CRITICAL message {datetime.datetime.now().strftime("%H:%M:%S")}')
