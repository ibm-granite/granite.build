#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Optional

import click


class CliEnvironment:
    """Contains information that is useful for CLI commands."""

    def __init__(self):
        self.home = os.getcwd()
        self.log_level = "info"
        self.log_path: Optional[Path] = None


pass_environment = click.make_pass_decorator(CliEnvironment, ensure=True)
