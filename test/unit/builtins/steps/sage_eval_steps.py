"""Helpers shared by the sage-eval step tests.

Find a script in a step.yaml by its content, and read its shell options.
"""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml


def strings(node: Any) -> Iterator[str]:
    """Yield every string in a parsed YAML node, at any depth.

    Args:
        node: A parsed YAML node (dict, list or scalar).

    Yields:
        Each string value, depth first.
    """
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from strings(value)


def one_script(path: Path, needle: str | re.Pattern[str]) -> str:
    """Return the one script in a step.yaml that contains ``needle``.

    The lookup goes by content, not by the step's layout, so a change to the
    environment key or the launcher name does not break it.

    Args:
        path: A step.yaml file.
        needle: A substring, or a compiled pattern searched in each string.

    Returns:
        The script text.

    Raises:
        AssertionError: If no string, or more than one, contains ``needle``.
    """
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(needle, re.Pattern):
        found = [s for s in strings(cfg) if needle.search(s)]
    else:
        found = [s for s in strings(cfg) if needle in s]
    assert len(found) == 1, f"{path}: {len(found)} scripts contain {needle!r}"
    return found[0]


def shell_options(script: str) -> str:
    """Return the ``set`` lines at the top of a script, before any other command.

    Args:
        script: A script from :func:`one_script`.

    Returns:
        Those lines, so a block cut from the script runs with the step's own
        options and no others.
    """
    options = []
    for line in script.splitlines():
        if re.match(r"\s*set\s+[-+]", line):
            options.append(line.strip())
        elif line.strip() and not line.strip().startswith("#"):
            break
    return "\n".join(options)
