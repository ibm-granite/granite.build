from gbcli.utils.gbconstants import GBSERVER_INSTANCE, USER_NOT_LOGGED_IN_ERROR_MESSAGE
from gbcli.utils.gbserver import get_server_version, make_gbserver_call
from gbcommon.types.gbenvconfig import is_standalone


def get_gbserver_version(github_token: str, quiet: bool, callback=None) -> str:
    # In standalone mode an empty token is legitimate (the local gbserver allows
    # localhost access when no GBSERVER_API_KEY is configured), so only require a
    # non-empty token outside standalone mode.
    if not github_token and not is_standalone():
        raise Exception(USER_NOT_LOGGED_IN_ERROR_MESSAGE)

    if callback and not quiet:
        callback(callback_event="fetching_server_version", callback_args={})

    version_commit = make_gbserver_call(
        lambda: get_server_version(github_token, GBSERVER_INSTANCE)["git_commit"],
        callback,
    )

    return version_commit
