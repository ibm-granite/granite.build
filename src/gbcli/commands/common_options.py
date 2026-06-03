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


def unsupported_in_standalone(command_name: str):
    """Decorator that guards a Click command or group against standalone mode.

    Apply below ``@click.group(...)`` / ``@cli.command(...)`` to warn and exit non-zero
    when the command is invoked in standalone mode (see :func:`exit_if_standalone`).

    Works on both groups and leaf commands:

    * On a ``@click.group`` callback the guard only fires when a subcommand is actually
      being invoked, so ``<group> --help`` (and bare ``<group>``) still work.
    * On a leaf command the guard fires whenever the command runs; Click handles
      ``--help`` before the callback, so help is unaffected.

    Example::

        @cli.command()
        @unsupported_in_standalone("model list")
        def list(...):
            ...
    """

    def decorator(f):
        @wraps(f)
        @click.pass_context
        def wrapper(ctx: click.Context, *args, **kwargs):
            # For a group, ctx.invoked_subcommand is set when a subcommand is being
            # dispatched and None for bare/`--help` invocations. For a leaf command it
            # is always None, so the guard fires as expected.
            is_group = isinstance(ctx.command, click.Group)
            if not is_group or ctx.invoked_subcommand is not None:
                exit_if_standalone(command_name)
            return ctx.invoke(f, *args, **kwargs)

        return wrapper

    return decorator


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
