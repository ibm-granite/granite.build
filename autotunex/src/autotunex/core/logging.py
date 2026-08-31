# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Logging configuration."""

from __future__ import annotations

import logging
from typing import Final

_FORMAT: Final = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Install a single stderr handler at ``level``.

    Idempotent: calling this twice does not duplicate handlers.
    """
    logging.basicConfig(level=level, format=_FORMAT, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for ``name`` (pass ``__name__``)."""
    return logging.getLogger(name)
