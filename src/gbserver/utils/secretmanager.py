#!/usr/bin/env python3

"""
A client for the IBM Cloud Secrets Manager API.
"""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Self

import requests

from gbserver.types.secret import MySecret, TokenResponse
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class MySecretsManagerAPI:
    """
    Secrets Manager API client.

    https://cloud.ibm.com/apidocs/secrets-manager/secrets-manager-v2#pagination
    https://cloud.ibm.com/apidocs/secrets-manager/secrets-manager-v2#rate-limits
    "As a rule of thumb it is recommended to keep the rate below 20 requests per second."
    """

    api_endpoint: str
    api_key: str = ""
    token: str = ""
    token_resp: Optional[TokenResponse] = None

    def __init__(self, api_endpoint: str, api_key: str = "", token: str = ""):
        """_summary_

        Args:
            api_endpoint (str): _description_
            api_key (str, optional): IAM api Key,
            token (str, optional): Auth token, only one of api_key or token is required
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.token = token
        if self.api_key == "":
            assert self.token != "", "must specify either an API key or an IAM token"

    async def get_new_token(self: Self) -> None:
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
        self.token_resp = TokenResponse.model_validate(data)
        self.token = self.token_resp.access_token
        logger.debug("get_new_token end")

    async def get_secret(self: Self, sec_id: str) -> MySecret:
        """
        Get a secret.
        https://cloud.ibm.com/apidocs/secrets-manager/secrets-manager-v2#get-secret
        "To view only the details of a specified secret without retrieving its value,
        use the Get secret metadata operation."
        """
        logger.debug("get_secret start")
        if self.token == "":
            await self.get_new_token()
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        api_url = f"{self.api_endpoint}/api/v2/secrets/{sec_id}"
        f1 = functools.partial(
            requests.get,
            api_url,
            headers=headers,
        )
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(executor=executor, func=f1)
            if response.status_code == 401:
                logger.debug("get a new token and try again")
                await self.get_new_token()
                headers = {
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/json",
                }
                f2 = functools.partial(
                    requests.get,
                    api_url,
                    headers=headers,
                )
                response = await loop.run_in_executor(executor=executor, func=f2)
        response.raise_for_status()
        data = response.json()
        secret = MySecret.model_validate(data)
        logger.debug("get_secret end")
        return secret
