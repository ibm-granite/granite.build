"""
Utility for expanding $VAR or ${VAR} placeholders in text.

Usage
-----
from utils.env_expand import expand_env

expanded_text = expand_env(raw_yaml)               # strict mode (default)
expanded_text = expand_env(raw_yaml, strict=False) # ignore missing vars
"""

import os
import re

__all__ = ["expand_env"]

_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _lookup(var: str, strict: bool) -> str:
    if var in os.environ:
        return os.environ[var]
    if strict:
        raise KeyError(f"Environment variable '{var}' is not set")
    return ""  # empty string when not strict


def expand_env(text: str, *, strict: bool = True) -> str:
    """
    Replace $VAR / ${VAR} placeholders with values from the current env.

    Parameters
    ----------
    text   : str
        Original text containing placeholders.
    strict : bool, default True
        • True  → raise KeyError if a variable is missing
        • False → substitute an empty string instead.

    Returns
    -------
    str
        The text with environment variables expanded.
    """
    return _PATTERN.sub(lambda m: _lookup(m.group(1) or m.group(2), strict), text)
