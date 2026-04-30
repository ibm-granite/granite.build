"""Lakehouse token generator module."""

import os
from enum import IntEnum

import requests

from gbserver.types.constants import LAKEHOUSE_ENVIRONMENT

LAKEHOUSE_TOKEN_EXPIRATION_MINUTES = 720


def generate_lakehouse_key_for_artifact(space):
    """Generate a temporary key for CLI to perform `artifact push`"""
    lakehouse_token = os.getenv("LAKEHOUSE_TOKEN_ARTIFACTS")
    return generate_lakehouse_key_from_key(lakehouse_token, LAKEHOUSE_TOKEN_EXPIRATION_MINUTES)


def generate_lakehouse_key_from_key(source_token, duration):
    """Generate a new key (typically short-lived) from another key (typically long-lived)."""
    from lakehouse.api import ConfigMap

    from gbserver.utils.lakehouse_utils import create_lakehouse_iceberg

    lh = create_lakehouse_iceberg(
        config="map",
        conf_map=ConfigMap(
            token=source_token,
            environment=LAKEHOUSE_ENVIRONMENT,
        ),
    )

    class MyDuration(IntEnum):
        """My Duration implementation."""

        time = duration

    new_token = lh.generate_new_token(MyDuration.time)

    return new_token


def generate_lakehouse_key_from_user_token(user_token):
    """Generate a new key (typically short-lived) from the user token."""
    from lakehouse import Environment

    # Token from PROD Lakehouse is valid across all Lakehouse environments.
    url = f"{Environment.build_from('PROD').value}/token/new_from_gh_token"
    headers = {
        "Content-Type": "application/json",
        "accept": "*/*",
    }
    json_data = {
        "jwtExpirationInMinutes": LAKEHOUSE_TOKEN_EXPIRATION_MINUTES,
        "ibmGitHubToken": user_token,
    }

    response = requests.post(
        url,
        headers=headers,
        json=json_data,
    )

    return response.json()


def generate_lakehouse_key_from_ibmid_token(user_token):
    """Generate a Lakehouse key from an IBMid id_token."""
    from lakehouse import Environment

    # Token from PROD Lakehouse is valid across all Lakehouse environments.
    url = f"{Environment.build_from('PROD').value}/token/new"
    headers = {
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    json_data = {"jwtExpirationInMinutes": LAKEHOUSE_TOKEN_EXPIRATION_MINUTES}

    response = requests.post(
        url,
        headers=headers,
        json=json_data,
    )

    return response.json()
