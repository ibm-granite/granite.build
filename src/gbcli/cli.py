import os
import sys
from typing import Any, Callable, Dict, Union

import click

from gbcli.utils.gbconstants import GB_ENVIRONMENT_DEFAULT, gb_environment
from gbserver.utils.logger import configure_logging

CONTEXT_SETTINGS = dict(auto_envvar_prefix="GBCLI")

# A registry value: either a lazy thunk that imports and returns an in-tree
# command, or a plugin's already-loaded click command object.
CommandSource = Union[Callable[[], click.BaseCommand], click.BaseCommand]


class Environment:
    def __init__(self):
        self.verbose = False
        self.home = os.getcwd()

    def log(self, msg, *args):
        """Logs a message to stderr."""
        if args:
            msg %= args
        click.echo(msg, file=sys.stderr)

    def vlog(self, msg, *args):
        """Logs a message to stderr only if verbose is enabled."""
        if self.verbose:
            self.log(msg, *args)


pass_environment = click.make_pass_decorator(Environment, ensure=True)
command_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "commands"))
hidden_commands = ["command_dataset.py"]
hidden_names = [
    command.split("_")[1].removesuffix(".py") for command in hidden_commands
]


def _intree_command_loader(name: str) -> Callable[[], click.BaseCommand]:
    """Build a thunk that imports the in-tree ``command_<name>`` module's ``cli``.

    Registered as the registry *value* so building the registry only records the
    command name — the (potentially heavy) command module is imported lazily, on
    first resolution in ``get_command``, exactly as before.
    """

    def load() -> click.BaseCommand:
        return __import__(f"gbcli.commands.command_{name}", None, None, ["cli"]).cli

    return load


def _resolve_command(value: "CommandSource") -> click.BaseCommand:
    """Return the click command for a registry value.

    In-tree values are lazy thunks (``() -> command``) that import their module
    on demand; plugin values are the already-loaded command object. A command is
    itself callable, so discriminate on type rather than callability.
    """
    if isinstance(value, click.BaseCommand):
        return value
    return value()


class GraniteBuildCLI(click.Group):
    # Subcommand name -> its command *source*. Both the in-tree
    # ``command_<name>.py`` scan and any ``gbcli.plugins`` entry points are filed
    # here through the shared PluginRegistrar (core-wins), like the other
    # pluggable subsystems. An in-tree source is a lazy thunk (so building the
    # registry never imports a command module); a plugin source is the loaded
    # ``click`` command object. ``_resolve_command`` normalizes both to a command.
    command_types: Dict[str, "CommandSource"] = {}

    def __init__(
        self,
        **attrs: Any,
    ):
        super().__init__(**attrs)
        self._set_configs()

    def _set_configs(self):
        from gbcli.utils.cli_config import configureGBWorkingEnv

        configureGBWorkingEnv()
        configure_logging(level="WARNING")

    @classmethod
    def _load_commands(cls) -> None:
        """(Re)build ``command_types`` from the in-tree modules and any plugins.

        Keys are discovered by scanning the ``commands`` directory (so dropping
        in a ``command_<name>.py`` still adds a command with no wiring) and by
        enumerating the ``gbcli.plugins`` entry-point group. Uses the shared
        reset-and-rebuild contract; a plugin can only *add* a command, never
        shadow a built-in (core-wins).
        """
        from gbcommon.plugins import (
            GROUP_CLI_PLUGINS,
            PluginRegistrar,
            keys_by_name,
            rebuild_registry,
        )

        registrar = PluginRegistrar(cls.command_types, "CLI command", keys_by_name)

        def populate() -> None:
            for filename in os.listdir(command_folder):
                if (
                    filename.endswith(".py")
                    and filename.startswith("command_")
                    and filename not in hidden_commands
                ):
                    name = filename[8:-3]
                    registrar.add(_intree_command_loader(name), name)
            # Plugin commands resolve directly to a click command object; wrap
            # each in a thunk so every registry value is uniformly ``() -> cmd``.
            registrar.discover_objects(GROUP_CLI_PLUGINS)

        rebuild_registry(cls.command_types, populate)

    def list_commands(self, ctx):
        self._load_commands()
        # A plugin could file a name that collides with a hidden built-in; keep
        # list_commands and get_command consistent by hiding it in both.
        return sorted(n for n in self.command_types if n not in hidden_names)

    def get_command(self, ctx, name):
        self._load_commands()
        if name in hidden_names:
            return None
        loader = self.command_types.get(name)
        if loader is None:
            return None
        env = gb_environment()
        if env != GB_ENVIRONMENT_DEFAULT:
            click.echo(f"Warning: GB_ENVIRONMENT is set to {env}", err=True)
        try:
            return _resolve_command(loader)
        except ImportError as e:
            message = (
                f"❌ Some dependencies required by the command '{name}' may be missing."
                + f"\nPlease reinstall the 'gb' CLI.\nDetails: {e}"
            )
            click.echo(message=message, err=True)
            sys.exit(1)


@click.command(cls=GraniteBuildCLI, context_settings=CONTEXT_SETTINGS)
@click.option(
    "--loglevel",
    default=None,
    help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.pass_context
def gbcli(ctx, loglevel):
    """LLM.build command line interface."""
    ctx.ensure_object(dict)
    if loglevel is not None:
        configure_logging(level=loglevel, skip_if_already_configured=False)
