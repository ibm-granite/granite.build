#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0


import os

import click

from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


@click.command()
@pass_environment
def cli(
    ctx: CliEnvironment,
):
    """REST API server worker psuedo-command. It just initializes the process and gracefully exits."""
    try:
        logger.info("Worker process initialized: pid %d", os.getpid())
    finally:
        pass
