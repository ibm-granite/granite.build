"""Service secret module."""

import base64
import itertools
import threading
import time
from typing import Any, Optional, Tuple

from gbcli.utils.gbconstants import (
    GBSERVER_SECRETS_API,
    SECRET_SPACE_ADMIN_ERROR,
)
from gbcli.utils.gbserver import (
    create_space_secret,
    delete_space_secret,
    get_secrets,
    make_gbserver_call,
    update_space_secret,
)
from gbcli.utils.gh_auth import get_user
from gbcli.utils.spaceutil import resolve_space
from gbcli.utils.utils import remove_suffix


def list_secrets(
    github_token: str,
    personal: bool,
    space: Optional[str] = None,
    callback=None,
) -> Any:
    """List secrets."""
    username = get_user(github_token).login

    space_default_name = None
    callback_space_name = None
    if not personal:
        s = resolve_space(github_token, space, callback)
        space_default = s["git_repo_uri"] if s is not None else None
        space_default_name = s["name"] if s is not None else None
        if not space_default:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"Space {space} not found in available spaces."},
                )
            return None

        space_org, space_name = space_default.split("/")[-2:]
        space_name = remove_suffix(space_name, ".git")
        callback_space_name = f"{space_org}/{space_name}"

        if not s["is_admin"]:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": SECRET_SPACE_ADMIN_ERROR},
                )
            return None

    stop_event = threading.Event()
    # Start spinner in a separate thread
    spinner_thread = threading.Thread(
        target=spinner_running,
        args=(
            stop_event,
            space_default_name,
            callback_space_name,
            username,
            callback,
            "listing_secrets_spinner",
        ),
    )
    spinner_thread.start()

    def stop_spinner():
        stop_event.set()  # Stop spinner thread
        spinner_thread.join()  # Ensure spinner stops

    secrets = make_gbserver_call(
        lambda: get_secrets(github_token, GBSERVER_SECRETS_API, personal, space_default_name),
        callback,
        stop_spinner,
    )
    secrets["secrets"].sort()

    if callback is not None:
        callback(
            callback_event="listed_secrets",
            callback_args={
                "steps": 100,
                "user": username,
                "space": space_default_name,
                "space_name": callback_space_name,
            },
        )

    return secrets


def get_secret(
    github_token: str,
    secret_name: str,
    personal: bool,
    space: Optional[str] = None,
    callback=None,
) -> Any:
    """Get the secret."""
    username = get_user(github_token).login

    space_default_name = None
    callback_space_name = None
    if not personal:
        s = resolve_space(github_token, space, callback)
        space_default = s["git_repo_uri"] if s is not None else None
        space_default_name = s["name"] if s is not None else None
        if not space_default:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"Space {space} not found in available spaces."},
                )
            return None

        space_org, space_name = space_default.split("/")[-2:]
        space_name = remove_suffix(space_name, ".git")
        callback_space_name = f"{space_org}/{space_name}"

        if not s["is_admin"]:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": SECRET_SPACE_ADMIN_ERROR},
                )
            return None

    stop_event = threading.Event()
    # Start spinner in a separate thread
    spinner_thread = threading.Thread(
        target=spinner_running,
        args=(
            stop_event,
            space_default_name,
            callback_space_name,
            username,
            callback,
            "obtaining_secret_spinner",
        ),
    )
    spinner_thread.start()

    def stop_spinner():
        stop_event.set()  # Stop spinner thread
        spinner_thread.join()  # Ensure spinner stops

    secret = make_gbserver_call(
        lambda: get_secrets(
            github_token,
            GBSERVER_SECRETS_API,
            personal,
            space_default_name,
            secret_name,
        ),
        callback,
        stop_spinner,
    )

    if callback is not None:
        callback(
            callback_event="obtained_secret",
            callback_args={
                "steps": 100,
                "user": username,
                "space": space_default_name,
                "space_name": callback_space_name,
            },
        )

    return secret


def create_secret(
    github_token: str,
    secret_name: str,
    personal: bool,
    secret_value: Optional[str] = None,
    space: Optional[str] = None,
    from_file: Optional[str] = None,
    callback=None,
) -> Tuple[Any, str, str]:
    """Create secret."""
    username = get_user(github_token).login

    space_default_name = None
    callback_space_name = None
    if not personal:
        s = resolve_space(github_token, space, callback)
        space_default = s["git_repo_uri"] if s is not None else None
        space_default_name = s["name"] if s is not None else None
        if not space_default:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"Space {space} not found in available spaces."},
                )
            return None

        space_org, space_name = space_default.split("/")[-2:]
        space_name = remove_suffix(space_name, ".git")
        callback_space_name = f"{space_org}/{space_name}"

        if not s["is_admin"]:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": SECRET_SPACE_ADMIN_ERROR},
                )
            return None

    if callback is not None:
        callback(callback_event="encoding_secret", callback_args={})

    if not secret_value and from_file:
        try:
            with open(from_file, "r", encoding="utf-8") as f:
                secret_value = f.read()
        except OSError as e:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"Error reading {from_file}: {str(e)}."},
                )
            return None

    encoded_value = base64.b64encode(secret_value.encode("utf-8")).decode("ascii")

    stop_event = threading.Event()
    # Start spinner in a separate thread
    spinner_thread = threading.Thread(
        target=spinner_running,
        args=(
            stop_event,
            space_default_name,
            callback_space_name,
            username,
            callback,
            "creating_secret_spinner",
        ),
    )
    spinner_thread.start()

    def stop_spinner():
        stop_event.set()  # Stop spinner thread
        spinner_thread.join()  # Ensure spinner stops

    secret = make_gbserver_call(
        lambda: create_space_secret(
            github_token,
            GBSERVER_SECRETS_API,
            personal,
            space_default_name,
            secret_name,
            encoded_value,
        ),
        callback,
        stop_spinner,
    )

    return secret, space_default_name, username


def update_secret(
    github_token: str,
    secret_name: str,
    personal: bool,
    secret_value: Optional[str] = None,
    space: Optional[str] = None,
    from_file: Optional[str] = None,
    callback=None,
) -> Tuple[Any, str, str]:
    """Update secret."""
    username = get_user(github_token).login

    space_default_name = None
    callback_space_name = None
    if not personal:
        s = resolve_space(github_token, space, callback)
        space_default = s["git_repo_uri"] if s is not None else None
        space_default_name = s["name"] if s is not None else None
        if not space_default:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"Space {space} not found in available spaces."},
                )
            return None

        space_org, space_name = space_default.split("/")[-2:]
        space_name = remove_suffix(space_name, ".git")
        callback_space_name = f"{space_org}/{space_name}"

        if not s["is_admin"]:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": SECRET_SPACE_ADMIN_ERROR},
                )
            return None

    if callback is not None:
        callback(callback_event="encoding_update_secret", callback_args={})

    if not secret_value and from_file:
        try:
            with open(from_file, "r", encoding="utf-8") as f:
                secret_value = f.read()
        except OSError as e:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"Error reading {from_file}: {str(e)}."},
                )
            return None

    encoded_value = base64.b64encode(secret_value.encode("utf-8")).decode("ascii")

    stop_event = threading.Event()
    # Start spinner in a separate thread
    spinner_thread = threading.Thread(
        target=spinner_running,
        args=(
            stop_event,
            space_default_name,
            callback_space_name,
            username,
            callback,
            "updating_secret_spinner",
        ),
    )
    spinner_thread.start()

    def stop_spinner():
        stop_event.set()  # Stop spinner thread
        spinner_thread.join()  # Ensure spinner stops

    secret = make_gbserver_call(
        lambda: update_space_secret(
            github_token,
            GBSERVER_SECRETS_API,
            personal,
            space_default_name,
            secret_name,
            encoded_value,
        ),
        callback,
        stop_spinner,
    )

    return secret, space_default_name, username


def delete_secret(
    github_token: str,
    secret_name: str,
    personal: bool,
    space: Optional[str] = None,
    callback=None,
) -> Tuple[Any, str, str]:
    """Remove secret."""
    username = get_user(github_token).login

    space_default_name = None
    callback_space_name = None
    if not personal:
        s = resolve_space(github_token, space, callback)
        space_default = s["git_repo_uri"] if s is not None else None
        space_default_name = s["name"] if s is not None else None
        if not space_default:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": f"Space {space} not found in available spaces."},
                )
            return None

        space_org, space_name = space_default.split("/")[-2:]
        space_name = remove_suffix(space_name, ".git")
        callback_space_name = f"{space_org}/{space_name}"

        if not s["is_admin"]:
            if callback is not None:
                callback(
                    callback_event="error",
                    callback_args={"reason": SECRET_SPACE_ADMIN_ERROR},
                )
            return None

    stop_event = threading.Event()
    # Start spinner in a separate thread
    spinner_thread = threading.Thread(
        target=spinner_running,
        args=(
            stop_event,
            space_default_name,
            callback_space_name,
            username,
            callback,
            "deleting_secret_spinner",
        ),
    )
    spinner_thread.start()

    def stop_spinner():
        stop_event.set()  # Stop spinner thread
        spinner_thread.join()  # Ensure spinner stops

    secret = make_gbserver_call(
        lambda: delete_space_secret(
            github_token,
            GBSERVER_SECRETS_API,
            personal,
            space_default_name,
            secret_name,
        ),
        callback,
        stop_spinner,
    )

    return secret, space_default_name, username


def spinner_running(
    stop_event,
    space: str,
    space_name: str,
    user: str,
    callback=None,
    callback_event=None,
):
    """Displays a spinner until `stop_event` is set."""
    spinner = itertools.cycle(["-", "\\", "|", "/"])  # Spinner characters
    while not stop_event.is_set():
        if callback and callback_event:
            callback(
                callback_event=callback_event,
                callback_args={
                    "spinner": next(spinner),
                    "space": space,
                    "space_name": space_name,
                    "user": user,
                },
            )
            time.sleep(0.1)
