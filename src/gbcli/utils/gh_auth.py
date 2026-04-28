# import time is required here, even the it is not referenced in the code explicitly
import time
from typing import List, Optional, Tuple
from urllib.parse import parse_qs

import requests
from pydantic import BaseModel

# https://github.com/cli/cli/blob/8288011149e71d5658b80ebef393522ba2d0e7cc/internal/authflow/flow.go#L24-L27
# The "GitHub CLI" OAuth app
oauthClientID = "178c6fc778ccc68e1d6a"
# This value is safe to be embedded in version control (public GitHub CLI OAuth app secret)
oauthClientSecret = "34ddeff2b558a23d38fba8a6de74f086ede1cc0b"  # gitleaks:allow
ContentTypeAppFormUrlEncoded = "application/x-www-form-urlencoded"
DeviceGrantType = "urn:ietf:params:oauth:grant-type:device_code"
MyBaseURL = "https://github.ibm.com"
DeviceCodeURL = MyBaseURL + "/login/device/code"
AuthorizeURL = MyBaseURL + "/login/oauth/authorize"
TokenURL = MyBaseURL + "/login/oauth/access_token"
UserInfoURL = "https://api.github.ibm.com/user"
UserReposURL = "https://api.github.ibm.com/users/{username}/repos"


class DeviceCodeURLResponse(BaseModel):
    device_code: List[str]
    expires_in: List[int]
    interval: List[int]
    user_code: List[str]
    verification_uri: List[str]

    def model_post_init(self, _):
        assert len(self.device_code) == 1
        assert len(self.user_code) == 1
        assert len(self.verification_uri) == 1


class TokenURLResponse(BaseModel):
    access_token: List[str]
    token_type: List[str]
    scope: Optional[List[str]] = None

    def model_post_init(self, _):
        assert len(self.access_token) == 1
        assert len(self.token_type) == 1


class TokenCodeObject(BaseModel):
    device_code: str
    verification_uri: str
    user_code: str


class UserInfoResponse(BaseModel):
    """User info from GitHub"""

    login: str
    id: int
    url: str
    html_url: str
    name: str
    email: str


def get_device_code() -> DeviceCodeURLResponse:
    minimumScopes = ["repo", "read:org", "notifications"]
    data = {"client_id": oauthClientID, "scope": " ".join(minimumScopes)}
    headers = {"Content-Type": ContentTypeAppFormUrlEncoded}
    response = requests.post(DeviceCodeURL, headers=headers, data=data)
    response.raise_for_status()
    data = response.text
    data_obj = parse_qs(data)
    data_obj_parsed = DeviceCodeURLResponse.model_validate(data_obj)
    return data_obj_parsed


def get_token_using_device_code(
    device_code: str,
) -> Tuple[Optional[TokenURLResponse], bool]:
    data = {
        "client_id": oauthClientID,
        "client_secret": oauthClientSecret,
        "device_code": device_code,
        "grant_type": DeviceGrantType,
    }
    headers = {"Content-Type": ContentTypeAppFormUrlEncoded}
    response = requests.post(TokenURL, headers=headers, data=data)
    response.raise_for_status()
    data = response.text
    data_obj = parse_qs(data)
    if "error" in data_obj:
        assert len(data_obj["error"]) > 0, "invalid error"
        error_type = data_obj["error"][0]
        error_description = error_type
        if "error_description" in data_obj:
            error_description = data_obj["error_description"][0]
        if error_type == "authorization_pending":
            # waiting for the user to authorize
            return None, False
        if error_type == "slow_down":
            # polling too fast, need to slow down
            return None, True
        if error_type == "access_denied":
            raise Exception("the user denied access")
        if error_type == "expired_token":
            raise Exception("the code has expired")
        raise Exception(f"auth error: {error_description}")
    data_obj_parsed = TokenURLResponse.model_validate(data_obj)
    return data_obj_parsed, False


def get_user(token: str) -> UserInfoResponse:
    import os

    from gbcli.utils.gbconstants import is_standalone
    from gbcli.utils.gbcredentials import GBCredentials

    if is_standalone():
        creds = GBCredentials()
        login = creds.get("login", section="user.gbserver") or os.environ.get(
            "GBSERVER_API_USER", "standalone"
        )
        return UserInfoResponse(
            login=login, id=0, url="", html_url="", name=login, email=""
        )

    creds = GBCredentials()
    default_provider = creds.get("default_provider", section="user")

    if default_provider == "ibmid":
        login = creds.get("login", section="user.ibmid") or ""
        name = creds.get("name", section="user.ibmid") or login
        email = creds.get("email", section="user.ibmid") or ""
        return UserInfoResponse(
            login=login, id=0, url="", html_url="", name=name, email=email
        )

    if default_provider == "apikey":
        login = creds.get("login", section="user.gbserver") or os.environ.get(
            "GBSERVER_API_USER", ""
        )
        return UserInfoResponse(
            login=login, id=0, url="", html_url="", name=login, email=""
        )

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.get(UserInfoURL, headers=headers)
    response.raise_for_status()
    data_obj = response.json()
    data_obj_parsed = UserInfoResponse.model_validate(data_obj)
    return data_obj_parsed


def get_token() -> TokenCodeObject:
    device_code_obj = get_device_code()
    verification_uri = device_code_obj.verification_uri[0]
    user_code = device_code_obj.user_code[0]

    return TokenCodeObject.model_validate(
        {
            "device_code": device_code_obj.device_code[0],
            "verification_uri": verification_uri,
            "user_code": user_code,
        }
    )

    # print("Open this link in the browser:", verification_uri)
    # print("and enter this code:", user_code)
    # answer = input("Open the browser? [y/n]: ")

    # if answer == "" or answer.lower() == "y" or answer.lower() == "yes":
    #     # https://docs.python.org/3/library/webbrowser.html
    #     webbrowser.open(verification_uri)

    # device_code = device_code_obj.device_code[0]
    # # ------------------
    # interval = 5
    # while True:
    #     time.sleep(interval)
    #     try:
    #         token, slow = get_token_using_device_code(device_code)
    #         if token is not None:
    #             return token
    #         if slow:
    #             # https://datatracker.ietf.org/doc/html/rfc8628#section-3.5
    #             # increase polling interval by 5 seconds
    #             interval += 5
    #     except Exception as e:
    #         raise Exception("Login failed") from e
    # # ------------------
