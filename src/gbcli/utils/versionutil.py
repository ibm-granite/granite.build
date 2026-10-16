import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version

from gbcli.utils.cli_config import get_local_gb_config
from gbcli.utils.gbconstants import PROJECT_NAME
from gbcli.utils.gh_clone import get_public_repo_tags, run_github_command
from gbcommon.types.constants import GB_PUBLIC_REPO_NAME, GB_PUBLIC_REPO_ORG

logger = logging.getLogger(__name__)

# The moving tag marking the oldest client we still support. A client at or above this
# floor is allowed to run (with a warning if a newer version exists); a client below it
# is blocked. It shares a commit with the vX.Y.Z tag it points at, so we resolve the
# floor version by matching the commit SHAs reported by the /repos/.../tags endpoint.
MIN_SUPPORTED_TAG = "min-supported"

# Shown to users who need to upgrade. Pins the rolling `stable` tag rather than the
# `granite.build` PyPI name (the project isn't published to PyPI).
_UPGRADE_CMD = (
    "pip install --upgrade "
    "'git+https://github.com/ibm-granite/granite.build.git@stable'"
)


def _env_seconds(name: str, default: int) -> int:
    # Parsed at import by every command, so a bad value must not raise.
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        logger.warning("Ignoring invalid %s; using %ds", name, default)
        return default


# The per-command version check caches the (latest, floor) it resolved from the public
# repo tags, so most commands skip the GitHub round-trip (and its 60/hour anonymous
# rate limit). Status is still recomputed against the installed version every time.
VERSION_CHECK_CACHE_FILE = "version_check.json"
VERSION_CHECK_CACHE_TTL_S = _env_seconds("GBCLI_VERSION_CHECK_TTL", 6 * 3600)
# A failed lookup (offline, or rate-limited with 403) is remembered briefly too, so
# each command doesn't re-pay the timeout or keep the anonymous quota exhausted.
VERSION_CHECK_FAILURE_TTL_S = 15 * 60


class VersionStatus(Enum):
    UP_TO_DATE = "up_to_date"
    OUTDATED_WARN = "outdated_warn"  # at/above floor, newer available -> warn, proceed
    BELOW_FLOOR = "below_floor"  # below min-supported -> hard block
    UNKNOWN = "unknown"  # check couldn't complete -> proceed silently


@dataclass
class VersionCheckResult:
    """Outcome of the version check. Pure data — the command layer decides how to act.

    ``message`` holds the fully-formed stderr text for OUTDATED_WARN / BELOW_FLOOR and is
    empty otherwise, so every user-facing string stays in this click-free module.
    """

    status: VersionStatus
    current_version: str = ""
    latest_version: str = ""
    floor_version: str = ""
    message: str = ""


def get_latest_version(repo_org: str, repo_name: str) -> str:
    logger.debug(
        "Checking latest CLI version from public repo %s/%s", repo_org, repo_name
    )
    tags = run_github_command(lambda: get_public_repo_tags(repo_org, repo_name))
    latest, _ = _resolve_versions_from_tags(tags)
    logger.debug("Latest CLI version resolved to %s", latest)
    return latest


def _resolve_versions_from_tags(tags) -> tuple[str, str]:
    """From a ``/repos/.../tags`` response return ``(latest_version, floor_version)``.

    Each tag is expected as ``{"name": ..., "commit": {"sha": ...}}`` — the list
    endpoint's shape, where ``commit.sha`` is already peeled to the underlying commit for
    both lightweight and annotated tags. The floor is resolved by finding the ``vX.Y.Z``
    tag sharing ``min-supported``'s commit SHA.

    latest: ``str(max)`` of the vX.Y.Z PEP 440 tags, or "0.0.0" if none are valid.
    floor:  the vX.Y.Z tag sharing ``min-supported``'s commit SHA, or "" when
            ``min-supported`` is absent or its commit matches no version tag. An
            unresolved floor means "no known floor" — the caller then mandates the
            upgrade for an outdated client rather than silently softening the block.
    """
    versions = []
    sha_to_version: dict[str, Version] = {}
    min_supported_sha = None

    for tag in tags:
        name = str(tag["name"])
        sha = (tag.get("commit") or {}).get("sha")
        if name == MIN_SUPPORTED_TAG:
            min_supported_sha = sha
            continue
        try:
            parsed = Version(name.lstrip("v"))
        except InvalidVersion:
            logger.debug("Skipping non-PEP440 tag: %s", name)
            continue  # skip non-PEP440 tags rather than failing the whole check
        versions.append(parsed)
        if sha:
            # If two version tags share a commit (e.g. a re-tag), last one wins. Which
            # one is irrelevant: the floor only needs *a* version at that commit, and
            # `latest` comes from max(versions), not this map.
            sha_to_version[sha] = parsed

    latest = str(max(versions)) if versions else "0.0.0"
    floor = ""
    if min_supported_sha and min_supported_sha in sha_to_version:
        floor = str(sha_to_version[min_supported_sha])
    return latest, floor


def get_current_version(package_name: str) -> str:
    try:
        return str(version(package_name))
    except PackageNotFoundError:
        return "unknown"


def warn_message(current: str, latest: str) -> str:
    return (
        f"A new version of {PROJECT_NAME} CLI ({latest}) is available. "
        f"You are currently running version {current}. "
        f"Run `{_UPGRADE_CMD}` or a command suitable to your environment to upgrade."
    )


def below_floor_message(current: str, floor: str) -> str:
    return (
        f"Your {PROJECT_NAME} CLI version ({current}) is no longer supported. "
        f"The minimum supported version is {floor}. You must upgrade to continue. "
        f"Run `{_UPGRADE_CMD}` or a command suitable to your environment to upgrade."
    )


def mandatory_upgrade_message(current: str, latest: str) -> str:
    """Block message used when a newer version exists and no floor is known.

    Distinct from ``warn_message`` (which only notifies): this says the command was
    refused, so the user isn't left wondering why an "upgrade available" notice aborted
    their command.
    """
    return (
        f"A new version of {PROJECT_NAME} CLI ({latest}) is available and an upgrade is "
        f"required to continue (you are running {current}). "
        f"Run `{_UPGRADE_CMD}` or a command suitable to your environment to upgrade."
    )


def _fetch_latest_and_floor() -> tuple[str, str]:
    tags = run_github_command(
        lambda: get_public_repo_tags(GB_PUBLIC_REPO_ORG, GB_PUBLIC_REPO_NAME)
    )
    return _resolve_versions_from_tags(tags)


def _cached_latest_and_floor() -> tuple[str, str]:
    """``_fetch_latest_and_floor`` behind a TTL file cache in the gb config dir.

    Cache I/O problems never fail the check; they just fall through to a live fetch.
    """
    try:
        path = get_local_gb_config() / VERSION_CHECK_CACHE_FILE
    except Exception:
        return _fetch_latest_and_floor()
    try:
        cached = json.loads(path.read_text())
        age = time.time() - float(cached["checked_at"])
    except Exception:
        cached, age = None, -1.0  # missing or unreadable -> refresh
    if cached is not None and age >= 0:
        if cached.get("failed"):
            if age < VERSION_CHECK_FAILURE_TTL_S:
                raise RuntimeError("version lookup failed recently; backing off")
        elif age < VERSION_CHECK_CACHE_TTL_S:
            return str(cached["latest"]), str(cached["floor"])

    try:
        latest, floor = _fetch_latest_and_floor()
    except Exception:
        _write_version_cache(path, {"failed": True})
        raise
    _write_version_cache(path, {"latest": latest, "floor": floor})
    return latest, floor


def _write_version_cache(path, entry: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({**entry, "checked_at": time.time()}))
        os.replace(tmp, path)
    except OSError as e:
        logger.debug("Could not write version check cache: %s", e)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def evaluate_version_status(
    package_name: str = "granite.build", use_cache: bool = False
) -> VersionCheckResult:
    """Resolve the CLI's version status against the public repo. Click-free, best-effort.

    The check queries the public granite.build repo over unauthenticated HTTPS, so it
    needs no GitHub credentials, SSH keys, or login and works everywhere (including
    standalone mode).

    It is a best-effort notice, not a hard gate: any failure (offline, rate-limited, an
    unparseable installed version such as "unknown" from a non-pip source checkout)
    yields ``UNKNOWN`` so the caller proceeds.

    The floor loosens the check only when it is known. When ``min-supported`` resolves to
    a floor version, a client at or above it is merely warned about a newer release
    (OUTDATED_WARN) and only a client below it is blocked (BELOW_FLOOR). When the floor is
    unknown (no ``min-supported`` tag, or it can't be resolved) we fall back to the
    pre-floor behavior and mandate the upgrade for any outdated client, so a missing tag
    never silently downgrades a block to a warning.

    ``use_cache`` reuses tags resolved within ``VERSION_CHECK_CACHE_TTL_S`` (the
    per-command gate); an explicit ``gb version --check-updates`` always goes live.
    """
    try:
        latest, floor = (
            _cached_latest_and_floor() if use_cache else _fetch_latest_and_floor()
        )
        current = get_current_version(package_name)
        current_v = Version(current)
        latest_v = Version(latest)
    except Exception as e:
        logger.debug("Skipping version check: %s", e)
        return VersionCheckResult(VersionStatus.UNKNOWN)

    # Whether we know a floor to compare against. An unparseable floor is treated as
    # unknown so it can't accidentally suppress the mandatory-upgrade fallback below.
    floor_v = None
    if floor:
        try:
            floor_v = Version(floor)
        except InvalidVersion:
            floor = ""  # unresolvable floor -> treat as no floor

    if current_v >= latest_v:
        return VersionCheckResult(VersionStatus.UP_TO_DATE, current, latest, floor)

    # Outdated. Warn only when we know a floor and the client is at/above it; otherwise
    # (below a known floor, or no floor known at all) block with a mandatory upgrade.
    if floor_v is not None and current_v >= floor_v:
        return VersionCheckResult(
            VersionStatus.OUTDATED_WARN,
            current,
            latest,
            floor,
            message=warn_message(current, latest),
        )

    message = (
        below_floor_message(current, floor)
        if floor
        else mandatory_upgrade_message(current, latest)
    )
    return VersionCheckResult(
        VersionStatus.BELOW_FLOOR, current, latest, floor, message=message
    )
