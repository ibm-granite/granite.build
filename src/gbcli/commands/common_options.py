import sys
from functools import wraps

import click

from gbcommon.types.gbenvconfig import is_standalone


def exit_if_standalone(command_name: str) -> None:
    """Warn and exit non-zero if an unsupported command is run in standalone mode.

    Commands that depend on cloud-only services (Lakehouse, GitHub Enterprise, the
    gbserver secret/admin backends) cannot work when GB_ENVIRONMENT=STANDALONE. Calling
    this at the start of such a command surfaces a clear warning instead of letting it
    fail later with a confusing auth/network error.
    """
    if is_standalone():
        click.echo(
            f"⚠️  Warning: '{command_name}' is not supported in standalone mode "
            f"(GB_ENVIRONMENT=STANDALONE).",
            err=True,
        )
        sys.exit(1)


def common_options(f):
    @wraps(f)
    @click.option(
        "--skip-version-check",
        is_flag=True,
        default=False,
        help="Skip current version check.",
    )
    @click.option(
        "--quiet",
        "-q",
        is_flag=True,
        default=False,
        help="Enables quiet mode.",
    )
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper
