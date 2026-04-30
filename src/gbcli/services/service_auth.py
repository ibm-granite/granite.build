"""Service auth module."""

import os

from requests.exceptions import ConnectionError

from gbcli.utils.gbconstants import (
    GBSERVER_SECRETS_API,
    RITS_LIST_URL,
    VPN_TUNNELALL_CONNECTION_ERROR_MESSAGE,
)
from gbcli.utils.gbcredentials import GBCredentials
from gbcli.utils.gbserver import gbserver_get, make_gbserver_call
from gbcli.utils.gh_auth import *
from gbcli.utils.spaceutil import resolve_space


def gh_token():
    """Gh token."""
    return get_token()


def gh_token_verify(device_code: str) -> TokenURLResponse:
    """Gh token verify."""
    interval = 5
    while True:
        time.sleep(interval)
        try:
            token, slow = get_token_using_device_code(device_code)
            if token is not None:
                return token
            if slow:
                # https://datatracker.ietf.org/doc/html/rfc8628#section-3.5
                # increase polling interval by 5 seconds
                interval += 5
        except Exception as e:
            raise Exception("Login failed") from e


def gh_login(gh_access_token: str):
    """Gh login."""
    user_obj = get_user(token=gh_access_token)
    credentials = GBCredentials()
    credentials.set("token", gh_access_token, section="user.github")
    credentials.set("login", user_obj.login, section="user.github")
    credentials.set("email", user_obj.email, section="user.github")
    credentials.set("default_provider", "github", section="user")
    credentials.save()

    return user_obj.login


def lh_artifact_token(github_token: str, callback=None) -> str:
    """Lh artifact token."""
    url = f"{GBSERVER_SECRETS_API}lakehouse/artifact_token"
    token = make_gbserver_call(
        lambda: gbserver_get(github_token, url),
        callback,
    )

    if not token or not token["lakehouse_token"] or not token["lakehouse_token"]["token"]:
        raise Exception("Error getting Lakehouse Token.")
    return token["lakehouse_token"]["token"]


def lh_user_token(github_token: str, callback=None) -> str:
    """Lh user token."""
    url = f"{GBSERVER_SECRETS_API}lakehouse/user_token"
    token = make_gbserver_call(
        lambda: gbserver_get(github_token, url),
        callback,
    )

    if not token or not token["lakehouse_token"] or not token["lakehouse_token"]["token"]:
        raise Exception("Error getting Lakehouse Token.")
    return token["lakehouse_token"]["token"]


def lakehouse_token_for_space(github_token: str, space: str = None, callback=None) -> str:
    """Lakehouse token for space."""
    global_space = resolve_space(github_token, space, callback=callback)
    namespace = global_space.get("lakehouse_namespace")
    public = global_space.get("name") == "public"

    if public:
        return lh_artifact_token(github_token, callback)
    else:
        return lh_user_token(github_token, callback)


def gbserver_login(api_user: str, api_key: str) -> str:
    """Store gbserver API credentials under [user.gbserver] in credentials file."""
    credentials = GBCredentials()
    credentials.set("api_key", api_key, section="user.gbserver")
    credentials.set("login", api_user, section="user.gbserver")
    credentials.set("default_provider", "apikey", section="user")
    credentials.save()
    return api_user


def ibmid_login(open_browser=None) -> str:
    """Authenticate via IBMid OIDC through gbserver proxy and store credentials."""
    from gbcli.utils.ibmid_auth import IBMidOIDCClient

    client = IBMidOIDCClient()
    result = client.start_auth_code_flow(open_browser=open_browser)

    credentials = GBCredentials()
    credentials.set("access_token", result.access_token, section="user.ibmid")
    credentials.set("id_token", result.id_token, section="user.ibmid")
    if result.refresh_token:
        credentials.set("refresh_token", result.refresh_token, section="user.ibmid")
    if result.expires_in:
        import time as _time

        credentials.set("expires_at", int(_time.time()) + result.expires_in, section="user.ibmid")
    login_name = (
        result.user_info.preferred_username or result.user_info.email or result.user_info.sub
    )
    credentials.set("login", login_name, section="user.ibmid")
    credentials.set("email", result.user_info.email, section="user.ibmid")
    credentials.set("name", result.user_info.name, section="user.ibmid")
    credentials.set("default_provider", "ibmid", section="user")
    credentials.save()

    return login_name


def rits_user_api_key():
    """Rits user api key."""
    return os.environ.get("RITS_API_KEY", None)


def verify_rits_api_key(rits_api_key, callback=None):
    """Verify rits api key."""
    try:
        response = requests.get(
            RITS_LIST_URL,
            headers={"RITS_API_KEY": rits_api_key},
            timeout=10,
        )
        if response.status_code == 200:
            return True
        else:
            return False
    except ConnectionError:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"RITS connection error. {VPN_TUNNELALL_CONNECTION_ERROR_MESSAGE}"
                },
            )
        return None
