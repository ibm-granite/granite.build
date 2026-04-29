import os

from fastapi.testclient import TestClient
from gbserver_test.test_utils import AbstractSingletonStorageUsingPreloadedSpaceTest

from gbserver.api.auth import get_gh_user
from gbserver.api.builds import BuildStatusResponse2
from gbserver.api.root_api import root_api
from gbserver.types.auth import User
from gbserver.types.constants import GBSERVER_GITHUB_TOKEN


class AbstractAPITest(AbstractSingletonStorageUsingPreloadedSpaceTest):

    def get_gh_username(self, token: str = GBSERVER_GITHUB_TOKEN) -> str:
        """Get the username/login associated with the given token, generally uesd to make client requests."""
        __tracebackhide__ = True  # Hide token during stack traces
        user, _ = get_gh_user(token)
        assert user is not None and isinstance(
            user, User
        ), "Could not get username/login from git token"
        username = user.login
        return username

    @staticmethod
    def get_test_client(token: str = GBSERVER_GITHUB_TOKEN) -> TestClient:
        """_summary_
        Get a TestClient that is configured with auth to talk to the server

        Returns:
            TestClient: _description_
        """
        __tracebackhide__ = True  # Hide token during stack traces
        if token == None or token == "":
            token = os.environ.get("GITHUB_TOKEN")
            assert (
                token != None
            ), "GBSERVER_GITHUB_TOKEN or GITHUB_TOKEN env var must be set to enable server authentication"
        client = TestClient(root_api, headers={"authorization": "Bearer " + token})
        return client


if __name__ == "__main__":
    client = AbstractAPITest.get_test_client()
    id = "39bbdc33-cfb2-4113-accc-c180aa3cd483"
    url = f"api/v1/builds/{id}/status2"
    resp = client.get(url)
    print(f"\nurl={url}")
    resp_json = resp.json()
    # print(f"\nresp.content={resp.content}")
    print(f"\njson resp={resp_json}")
    resp: BuildStatusResponse2 = BuildStatusResponse2.model_validate(resp_json)
    print(f"\n\nbuild status={resp}")
