import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

from gbcli.utils.gbconstants import (
    GBCLI_REPO_URL,
    PROJECT_NAME,
    USER_NOT_LOGGED_IN_ERROR_MESSAGE,
)
from gbcli.utils.gbcredentials import GBCredentials
from gbcli.utils.gh_clone import get_repo_tags, run_github_command
from gbcommon.types.constants import get_gh_credentials_section


def get_latest_version(user_token: str, repo_org: str, repo_name: str) -> str:
    tags = run_github_command(lambda: get_repo_tags(user_token, repo_org, repo_name))

    latest_tag = "0.0.0"
    for tag in tags:
        tag_version = str(tag["ref"]).split("/")[-1].replace("v", "")
        if Version(tag_version) > Version(latest_tag):
            latest_tag = tag_version

    return latest_tag


def get_current_version(package_name: str) -> str:
    try:
        return str(version(package_name))
    except PackageNotFoundError:
        return "unknown"


def check_current_and_latest_versions() -> str:
    credentials = GBCredentials()
    if not credentials.check_values():
        raise Exception(USER_NOT_LOGGED_IN_ERROR_MESSAGE)

    user_token = credentials.get("token", section=get_gh_credentials_section())
    repo_org, repo_name = GBCLI_REPO_URL.split("/")[3:]
    latest_version = get_latest_version(user_token, repo_org, repo_name)
    current_version = get_current_version("granite.build")

    if Version(current_version) < Version(latest_version):
        return (
            f"A new version of {PROJECT_NAME} CLI ({latest_version}) is available. "
            f"You are currently running version {current_version}. "
            "Run `pip install --upgrade granite.build` or a command suitable to your environment to upgrade."
        )
    else:
        return ""
