# (C) Copyright IBM Corp. 2024.
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
#
import logging
import os
from pathlib import Path
from typing import Optional, Self

from gbserver.types.constants import DEFAULT_LOG_FORMAT, DEFAULT_LOG_LEVEL

__LOGGER_CONFIGURED = False


# def get_log_level(x: str) -> logging._Level:
def get_log_level(x: str) -> int:
    """Return the corresponding logging level."""
    d = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    return d[x.lower()]


# def get_log_level_str(x logging._Level) -> str:
def get_log_level_str(x: int) -> str:
    """Return the corresponding logging level as a string."""
    d = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "critical",
    }
    return d[x]


class CustomFormatter(logging.Formatter):
    """
    Print different log levels in different colors.

    ANSI color codes:
    https://en.wikipedia.org/wiki/ANSI_escape_code
    https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797#ansi-escape-sequences
    https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797#colors--graphics-mode
    https://jvns.ca/blog/2025/03/07/escape-code-standards/#ecma-48
    """

    ESC = "\x1b"
    ESCC = ESC + "["
    DELIM = ";"
    END = "m"

    RESET = "0"
    BOLD = "1"
    DIM = "2"
    BLACK = "30"
    RED = "31"
    GREEN = "32"
    YELLOW = "33"
    BLUE = "34"
    MAGENTA = "35"
    CYAN = "36"
    WHITE = "37"

    DO_RESET = ESCC + RESET + END
    DO_DEBUG_COLOR = ESCC + DIM + DELIM + WHITE + END
    DO_INFO_COLOR = ESCC + RESET + DELIM + WHITE + END
    DO_WARNING_COLOR = ESCC + RESET + DELIM + YELLOW + END
    DO_ERROR_COLOR = ESCC + RESET + DELIM + RED + END
    DO_CRITICAL_COLOR = ESCC + BOLD + DELIM + RED + END

    FORMATS = {
        logging.DEBUG: DO_DEBUG_COLOR + DEFAULT_LOG_FORMAT + DO_RESET,
        logging.INFO: DO_INFO_COLOR + DEFAULT_LOG_FORMAT + DO_RESET,
        logging.WARNING: DO_WARNING_COLOR + DEFAULT_LOG_FORMAT + DO_RESET,
        logging.ERROR: DO_ERROR_COLOR + DEFAULT_LOG_FORMAT + DO_RESET,
        logging.CRITICAL: DO_CRITICAL_COLOR + DEFAULT_LOG_FORMAT + DO_RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def configure_logging(
    level: str = DEFAULT_LOG_LEVEL,
    format: Optional[str] = None,
    log_file: Optional[str] = os.getenv("GBSERVER_LOG_FILE", None),
    skip_if_already_configured: bool = False,
):
    """Configure the basic logger."""
    global __LOGGER_CONFIGURED
    if skip_if_already_configured and __LOGGER_CONFIGURED:
        return
    if format is None:
        handler: logging.Handler = logging.StreamHandler()
        if log_file is not None:
            handler = logging.FileHandler(filename=log_file, encoding="utf-8", mode="w")
        handler.setFormatter(CustomFormatter())
        logging.basicConfig(
            handlers=[handler],
            level=get_log_level(level),
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
    else:
        logging.basicConfig(
            format=format,
            level=get_log_level(level),
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=log_file,
            filemode="w",
        )
    __LOGGER_CONFIGURED = True
    logger = logging.getLogger(__name__)
    logger.info("logging level set to %s", level)
    # logger.debug("THIS IS A DEBUG TEST!")
    # logger.info("THIS IS A INFO TEST!")
    # logger.warning("THIS IS A WARNING TEST!")
    # logger.error("THIS IS A ERROR TEST!")
    # logger.critical("THIS IS A CRITICAL TEST!")
    if log_file is not None:
        logger.info("saving logs to file at %s", log_file)


def get_logger(name: str) -> logging.Logger:
    configure_logging(skip_if_already_configured=True)
    logger = logging.getLogger(name)
    return logger


class LoggingUtility:
    """
    Enables easier logging at different levels and optional msg prefix
    """

    def __init__(self, logger: logging.Logger, msg_prefix: Optional[str] = None):
        self.logger = logger
        self.msg_prefix = msg_prefix

    def info(self: Self, msg: str):
        self._log(logging.INFO, msg, 2)

    def warn(self: Self, msg: str):
        self._log(logging.WARNING, msg, 2)

    def warning(self: Self, msg: str):
        self._log(logging.WARNING, msg, 2)

    def error(self: Self, msg: str):
        self._log(logging.ERROR, msg, 2)

    def debug(self: Self, msg: str):
        self._log(logging.DEBUG, msg, 2)

    def _log(self: Self, level: int, msg: str, stacklevel=2):
        if self.msg_prefix != None:
            msg = f"{self.msg_prefix} - {msg}"
        self.logger.log(level=level, msg=msg, stacklevel=stacklevel + 1)
