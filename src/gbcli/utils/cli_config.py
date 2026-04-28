import os
from pathlib import Path


def configureGBWorkingEnv():
    if "GB_CONFIG" not in os.environ:
        os.environ["GB_CONFIG"] = os.path.join(os.path.expanduser("~"), ".gbcli")

    if "DMF_CACHE" not in os.environ:
        os.environ["DMF_CACHE"] = os.path.join(os.environ["GB_CONFIG"], "dmf_cache")

    if "GB_CACHE" not in os.environ:
        os.environ["GB_CACHE"] = os.path.join(os.environ["GB_CONFIG"], "workdir")


def get_local_gb_config():
    return Path(os.path.abspath(os.environ["GB_CONFIG"]))


def get_local_build_cache():
    return Path(os.path.abspath(os.environ["GB_CACHE"]))
