import logging
from importlib.metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version

from gbcli.utils.gbconstants import PROJECT_NAME
from gbcli.utils.gh_clone import get_public_repo_tags, run_github_command
from gbcommon.types.constants import GB_PUBLIC_REPO_NAME, GB_PUBLIC_REPO_ORG

logger = logging.getLogger(__name__)


def get_latest_version(repo_org: str, repo_name: str) -> str:
    logger.debug(
        "Checking latest CLI version from public repo %s/%s", repo_org, repo_name
    )
    tags = run_github_command(lambda: get_public_repo_tags(repo_org, repo_name))

    versions = []
    for tag in tags:
        raw = str(tag["ref"]).split("/")[-1].lstrip("v")
        try:
            versions.append(Version(raw))
        except InvalidVersion:
            logger.debug("Skipping non-PEP440 tag: %s", tag.get("ref"))
            continue  # skip non-PEP440 tags rather than failing the whole check

    latest = str(max(versions)) if versions else "0.0.0"
    logger.debug("Latest CLI version resolved to %s", latest)
    return latest


def get_current_version(package_name: str) -> str:
    try:
        return str(version(package_name))
    except PackageNotFoundError:
        return "unknown"


def check_current_and_latest_versions() -> str:
    # The version check queries the public granite.build repo over unauthenticated
    # HTTPS, so it needs no GitHub credentials, SSH keys, or login. It therefore works
    # everywhere, including standalone mode.
    #
    # This is a best-effort notice, not a gate: if the check can't complete (offline,
    # rate-limited, or the installed version can't be parsed — e.g. "unknown" from a
    # non-pip-installed source checkout) we silently skip it rather than blocking the
    # command the user actually ran. The whole comparison runs inside the try so a
    # parse failure on either side never escapes.
    try:
        latest_version = get_latest_version(GB_PUBLIC_REPO_ORG, GB_PUBLIC_REPO_NAME)
        current_version = get_current_version("granite.build")
        is_outdated = Version(current_version) < Version(latest_version)
    except Exception as e:
        logger.debug("Skipping version check: %s", e)
        return ""

    if is_outdated:
        return (
            f"A new version of {PROJECT_NAME} CLI ({latest_version}) is available. "
            f"You are currently running version {current_version}. "
            "Run `pip install --upgrade granite.build` or a command suitable to your environment to upgrade."
        )
    else:
        return ""
