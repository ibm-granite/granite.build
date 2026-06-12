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
            f"❌ Error: '{command_name}' is currently not supported in standalone mode "
            f"(GB_ENVIRONMENT=STANDALONE).",
            err=True,
        )
        sys.exit(1)


def _command_name_from_context(ctx: click.Context) -> str:
    """Return the qualified command name (e.g. ``"artifact register"``) for a context.

    Click already tracks the full invocation as ``ctx.command_path`` (e.g.
    ``"gb artifact register"``); we just drop the leading program name so the standalone
    warning reads ``'artifact register'`` rather than the full ``'gb artifact register'``,
    without callers having to spell the name out.
    """
    _prog, _, qualified = ctx.command_path.partition(" ")
    return qualified or ctx.command_path


def reject_standalone(f):
    """Decorator that guards a single *leaf* command against standalone mode.

    Unlike :func:`pass_context_and_reject_standalone`, this does *not* inject the Click
    context -- it reads the active context via ``click.get_current_context`` and leaves the
    wrapped callback's signature untouched, so it composes with a command's own
    ``@click.pass_context`` (and other option decorators) without consuming an argument.
    Use it as a bare decorator to gate the cloud-only subcommands of a group whose other
    subcommands work in standalone mode, so the group itself stays unguarded::

        @cli.command()
        @click.pass_context
        @reject_standalone
        def register(ctx, ...):
            ...

    Click handles ``--help`` before the callback, so help is unaffected. The name shown in
    the warning is derived from the active context (e.g. ``"artifact register"``), so no
    argument is needed.

    The guard fires at callback time, i.e. *after* Click validates required args/options.
    So ``gb artifact register`` with no args reports a missing-option error rather than the
    standalone message. We accept this: the standalone guard is expected to be temporary
    (these commands will eventually work standalone), so it is not worth the extra
    complexity of an eager pre-parse hook to reorder the messages.

    This is leaf-only by design: applied to a ``click.Group`` callback it would fire for
    *every* subcommand once one is dispatched, silently re-introducing the blanket guard we
    are trying to avoid. Guard against that misuse with a clear programming error; use
    :func:`pass_context_and_reject_standalone` (which has the group/leaf handling) on a
    group instead.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        ctx = click.get_current_context()
        if isinstance(ctx.command, click.Group):
            raise RuntimeError(
                "@reject_standalone is leaf-only; it would block every subcommand if "
                "applied to a group. Use @pass_context_and_reject_standalone on groups."
            )
        exit_if_standalone(_command_name_from_context(ctx))
        return f(*args, **kwargs)

    return wrapper


def pass_context_and_reject_standalone(command_name=None):
    """Decorator that passes the Click context and guards against standalone mode.

    Combines ``@click.pass_context`` with a standalone-mode guard: apply it below
    ``@click.group(...)`` / ``@cli.command(...)`` (in place of ``@click.pass_context``)
    to warn and exit non-zero when the command is invoked in standalone mode (see
    :func:`exit_if_standalone`). The wrapped callback still receives ``ctx`` as its first
    argument, so callbacks that need the context keep their ``ctx`` parameter and do not
    declare their own ``@click.pass_context``.

    The command name shown in the warning defaults to ``ctx.info_name`` (the Click
    command name as invoked), so group-level guards need no argument::

        @click.group("secret")
        @pass_context_and_reject_standalone
        def cli(ctx):
            ...

    Pass an explicit name when the default would lose context -- e.g. a leaf subcommand
    whose ``info_name`` is just ``"set"`` but should read ``"space set"``::

        @cli.command()
        @pass_context_and_reject_standalone("space set")
        def set(ctx, ...):
            ...

    Works on both groups and leaf commands:

    * On a ``@click.group`` callback the guard only fires when a subcommand is actually
      being invoked, so ``<group> --help`` (and bare ``<group>``) still work.
    * On a leaf command the guard fires whenever the command runs; Click handles
      ``--help`` before the callback, so help is unaffected.
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
                exit_if_standalone(command_name or ctx.info_name)
            # ``f`` is the bare callback (its own ``@click.pass_context`` was replaced
            # by this decorator), so ``ctx.invoke`` will not auto-inject the context --
            # pass it explicitly as the first positional argument.
            return ctx.invoke(f, ctx, *args, **kwargs)

        return wrapper

    # Allow usage as a bare decorator (@pass_context_and_reject_standalone) in addition
    # to the called forms (@pass_context_and_reject_standalone() / (...("name")).
    if callable(command_name):
        f, command_name = command_name, None
        return decorator(f)

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
