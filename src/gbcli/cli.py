"""Cli module."""

import os
import sys
from typing import Any

import click

from gbcli.utils.gbconstants import GB_ENVIRONMENT_DEFAULT, gb_environment
from gbserver.utils.logger import configure_logging

CONTEXT_SETTINGS = dict(auto_envvar_prefix="GBCLI")


class Environment:
    """Environment implementation."""

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
hidden_names = [command.split("_")[1].removesuffix(".py") for command in hidden_commands]


class GraniteBuildCLI(click.Group):
    """Granite Build C L I implementation."""

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

    def list_commands(self, ctx):
        rv = []
        for filename in os.listdir(command_folder):
            if (
                filename.endswith(".py")
                and filename.startswith("command_")
                and filename not in hidden_commands
            ):
                rv.append(filename[8:-3])
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        try:
            env = gb_environment()
            if env != GB_ENVIRONMENT_DEFAULT:
                click.echo(f"Warning: GB_ENVIRONMENT is set to {env}", err=True)
            if name in hidden_names:
                return
            mod = __import__(f"gbcli.commands.command_{name}", None, None, ["cli"])
        except ImportError as e:
            invalid_command = str(e).startswith("No module named 'gbcli.commands.command_")
            if invalid_command:
                return
            message = (
                f"❌ Some dependencies required by the command '{name}' may be missing."
                + f"\nPlease reinstall the 'gb' CLI.\nDetails: {e}"
            )
            click.echo(message=message, err=True)
            sys.exit(1)
        return mod.cli


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
