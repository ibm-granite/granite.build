import logging
from datetime import datetime, timedelta

from gbcli.utils.gbconstants import (
    SPACE_TIMESTAMP_DELTA_HOURS,
    gb_environment,
)
from gbcli.utils.gbcredentials import (
    ConfigLockException,
    GBConfig,
)
from gbcli.utils.gbserver import get_remote_spaces
from gbcommon.types.gbenvconfig import gb_environment_config, is_standalone

logger = logging.getLogger(__name__)


def check_and_set_spaces_profile(remote_spaces, config_section):
    """
    utility that generates profile section
    create new profiles section if it does not exist

    remove spaces in profile if they no longer exist in remote spaces
    """
    format_profile_check()
    profile_space = "profiles"
    gbconfig = GBConfig()
    existing_space_profiles = gbconfig.get(profile_space, config_section)
    if existing_space_profiles is None:
        # create new profile section. In this version, there's always only one profile "default".
        default_profile = {"name": "default", "profile": {"default": "public"}}
        gbconfig.set(profile_space, [default_profile], config_section)
        gbconfig.save()
        # save_profile("default", "public") # Since the default profile is initialized with the hard-coded value above, no need to call save_profile() again
    else:
        # remove profile dictionary items that no longer exist in remote spaces

        new_space_profiles = []
        for existing_profile in existing_space_profiles:
            new_profile_config = {"name": existing_profile.get("name")}
            profile = {}
            for key, value in existing_profile["profile"].items():
                # lookup value to see if it exists in remote_spaces
                for space in remote_spaces:
                    if space.get("name") == value and key != value:
                        profile[key] = value

            default_space = profile.get("default", None)
            if remote_spaces and default_space is None:
                if profile.get("public", None):
                    profile["default"] = "public"
                else:
                    profile["default"] = remote_spaces[0].get("name")

            new_profile_config["profile"] = profile
            if profile:
                new_space_profiles.append(new_profile_config)
        gbconfig.set(profile_space, new_space_profiles, config_section)
        gbconfig.save()


def resolve_space(github_token: str, space=None, callback=None):
    """
    -called by service code to get back all data for a space_name

    accept a space_name (if None, use default space)
    use provided space_name to look up in profile
    then return space from gb.spaces array with the corresponding name from profile
    """

    updated_spaces = get_spaces(github_token, callback)

    # In standalone mode, get_spaces() returns remote spaces directly.
    # Resolve space by name without going through the profile layer.
    if is_standalone():
        space_name = (
            gb_environment_config().default_space
            if (not space or space == "default")
            else space
        )
        found_space = next(
            (s for s in updated_spaces if s.get("name") == space_name), None
        )
        if found_space is None and callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"Space '{space_name}' not found on gbserver.",
                },
            )
        return found_space

    config_profile_name = gb_environment_config().config_profile
    if config_profile_name:
        gbconfig = GBConfig()
        config_profile = gbconfig.get_section(config_profile_name)
    else:
        config_profile = None

    if not config_profile:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": "Space name resolution failed. Please run space list --all --refresh first.",
                },
            )
        return None

    profile = next(
        (profile for profile in config_profile if profile.get("name") == "default"),
        None,
    )
    if profile:
        curr_profile = profile["profile"]
    else:
        curr_profile = {}

    if space:
        space_name = curr_profile.get(space) or space
    else:
        space_name = curr_profile.get("default")

    if not space_name:
        # if no specified space or default space, try public
        space_name = curr_profile.get("public")

    if not space_name:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"Space '{space if space else 'default'} not found in profile",
                },
            )
        return None

    found_space = next((s for s in updated_spaces if s.get("name") == space_name), None)

    if found_space is None:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"Space {space if space else 'default'} is not available. Please run space list --all --refresh to see all available spaces",
                },
            )
        return None

    return found_space


def save_new_spaces(remote_spaces, callback=None):
    """
    saves spaces into credentials (to be removed) and config
    """
    if not remote_spaces:
        return

    expiration_timestamp = (
        datetime.now() + timedelta(hours=SPACE_TIMESTAMP_DELTA_HOURS)
    ).timestamp()

    # Saving the remote space list in the config file
    gbconfig = GBConfig()

    space_section = gb_environment_config().config_spaces
    gbconfig.set(
        "expiration_timestamp",
        expiration_timestamp,
        space_section,
    )
    gbconfig.set("spaces", remote_spaces, space_section)
    try:
        gbconfig.save()
    except ConfigLockException as e:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": f"{str(e)}.\nPlease try again.",
                },
            )
        return None

    # Initialize the profile in the config file
    check_and_set_spaces_profile(remote_spaces, space_section)

    if callback:
        callback(callback_event="done_fetching_spaces", callback_args={"steps": 50})


def get_spaces(github_token: str, callback=None):
    """
    makes a call to gbserver if new spaces are needed from remote
    """
    if is_standalone():
        # In standalone mode, skip caching and always fetch directly from gbserver.
        logger.info("fetching user spaces from GBSERVER (standalone)")
        return get_remote_spaces(github_token, callback)

    try:
        gbconfig = GBConfig()
        config_spaces_name = gb_environment_config().config_spaces
        spaces = gbconfig.get("spaces", config_spaces_name)
        expiration_timestamp = gbconfig.get("expiration_timestamp", config_spaces_name)
        if not spaces:
            raise Exception

        # compare timestamp to see if we need to refresh from server
        if not expiration_timestamp or datetime.now() > datetime.fromtimestamp(
            expiration_timestamp
        ):
            raise Exception

        return spaces
    except Exception:
        # need to refresh user spaces cache
        logger.info("fetching user spaces from GBSERVER")
        remote_spaces = get_remote_spaces(github_token, callback)
        # save to config
        save_new_spaces(remote_spaces, callback)
        return remote_spaces


def get_profile(profile_name="default"):
    """
    returns the profile dictionary (right now just using default)
    """
    try:
        format_profile_check()
        gbconfig = GBConfig()
        config_profile_name = gb_environment_config().config_profile
        profile_arr = gbconfig.get_section(config_profile_name)
        profile = next(
            (profile for profile in profile_arr if profile.get("name") == profile_name),
            None,
        )
        if profile:
            return profile["profile"]
        else:
            return {}
    except Exception:
        return {}


def save_profile(space_key: str, space_name: str, profile_name="default"):
    """Save profile section of config"""
    format_profile_check()
    profile_space = "profiles"
    gbconfig = GBConfig()
    config_spaces = gb_environment_config().config_spaces
    space_profiles = gbconfig.get(profile_space, config_spaces)
    for profile in space_profiles:
        if profile.get("name") == profile_name:
            profile["profile"][space_key] = space_name

    gbconfig.set(profile_space, space_profiles, config_spaces)
    gbconfig.save()


# handle legacy profile format that is not an array
def format_profile_check():
    profile_space = "profiles"
    gbconfig = GBConfig()
    config_section = gb_environment_config().config_spaces
    space_profiles = gbconfig.get(profile_space, config_section)

    # confirm if existing_space_profiles is an array or does not exist
    if type(space_profiles) is list or space_profiles is None:
        return

    if type(space_profiles) is dict:
        # format to array and save
        new_space_profiles = [{"name": "default", "profile": space_profiles}]
        gbconfig.set(profile_space, new_space_profiles, config_section)
        gbconfig.save()
        return


def user_is_space_admin(github_token: str, space: str, callback=None) -> bool:
    space_info = resolve_space(github_token, space, callback)
    if space_info:
        return space_info["is_admin"]
    else:
        return False
