import os

import pytest

pytestmark = pytest.mark.ibm

from gbserver.utils.lakehouse_token_generator import (
    generate_lakehouse_key_from_ibmid_token,
    generate_lakehouse_key_from_user_token,
)


class TestLHToken:
    def test_generate_lakehouse_key_from_user_token(self):
        # Test that the generated new key from the user github token is valid

        token = os.environ.get("GITHUB_TOKEN")

        lakehouse_token = generate_lakehouse_key_from_user_token(token)

        assert (
            "token" in lakehouse_token
            and "expiration" in lakehouse_token
            and "email" in lakehouse_token
        )

    @pytest.mark.skipif(
        not os.environ.get("IBMID_ID_TOKEN"),
        reason="requires IBMid id_token from interactive login",
    )
    def test_generate_lakehouse_key_from_ibmid_token(self):
        token = os.environ.get("IBMID_ID_TOKEN")

        lakehouse_token = generate_lakehouse_key_from_ibmid_token(token)

        assert (
            "token" in lakehouse_token
            and "expiration" in lakehouse_token
            and "email" in lakehouse_token
        )
