#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Render a framework step's step-template.yaml into its step.yaml.

common.mk's render-step-template runs this for `make space` and `make
publish-step`. It makes two passes and nothing else:

1. Each line ``{#- gb-include <path> -#}`` (trailing whitespace and a CRLF
   ending are ignored) is replaced by a ``{#- included from steps/<file> -#}``
   comment and then the lines of the file it names, each indented like the
   marker. ``<path>`` is relative to the file holding the marker and must
   resolve, after symlinks, to a file under the steps/ tree. Included files
   cannot include others. Any other line with a ``{#`` comment naming
   ``gb-include`` is refused, so a typo cannot pass as a plain Jinja comment.
   Files must be UTF-8; an error names the file.
2. The literal ``${IMAGE_REF}`` is replaced by the image reference.

Runtime Jinja (``{{ ... }}``) and shell expansions pass through untouched. Standard
library only (Python 3.9+), so rendering needs no install.

Usage: render_step_template.py TEMPLATE OUT [--image-ref REF]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

STEPS_ROOT = Path(__file__).resolve().parent.parent
IMAGE_REF_TOKEN = "${IMAGE_REF}"
_MARKER = re.compile(r"^(?P<indent> *)\{#- gb-include (?P<path>\S+) -#\}$")
_ATTEMPT = re.compile(r"\{#[-+]*\s*gb-include\b")


class IncludeError(Exception):
    """A gb-include line that cannot be expanded."""


def _read(path: Path) -> str:
    """Return a template or include file's text.

    Args:
        path: the file to read.

    Returns:
        Its text, decoded as UTF-8.

    Raises:
        IncludeError: if the file is not valid UTF-8. The message is the reason
            only; the caller adds which file and where it was included.
        OSError: if the file cannot be read.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise IncludeError(
            f"not valid UTF-8 ({exc.reason} at byte {exc.start})"
        ) from None


def _target(path: str, including: Path, root: Path) -> Path:
    """Resolve an include path against the including file; it must stay in ``root``.

    Args:
        path: the path written in the marker.
        including: the file that holds the marker; ``path`` is relative to it.
        root: the tree the resolved file must be in.

    Returns:
        The resolved path, after symlinks.

    Raises:
        IncludeError: if the path resolves outside ``root``, or to no file.
    """
    target = (including.parent / path).resolve()
    if not target.is_relative_to(root):
        raise IncludeError(f"{path} resolves outside {root} ({target})")
    if not target.is_file():
        raise IncludeError(f"{path}: no such file ({target})")
    return target


def expand(template: Path, root: Path = STEPS_ROOT) -> str:
    """Return ``template``'s text with its gb-include lines expanded.

    Args:
        template: the step-template.yaml to expand.
        root: the tree included files must resolve into (default: steps/).

    Returns:
        The expanded text.

    Raises:
        IncludeError: on a malformed marker, a path outside ``root``, a missing
            file, an included file with a gb-include line of its own, or a file
            that is not valid UTF-8.
        OSError: if a file cannot be read.
    """
    template = Path(template)
    root = Path(root).resolve()
    out: List[str] = []
    try:
        text = _read(template)
    except IncludeError as exc:
        raise IncludeError(f"{template}: {exc}") from None
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        if not _ATTEMPT.search(line):
            out.append(line)
            continue
        where = f"{template}:{number}"
        # Trailing whitespace and a CRLF ending are invisible; leading tabs are not
        # stripped, so they are still refused.
        marker = _MARKER.match(line.rstrip())
        if marker is None:
            written = line.rstrip("\r\n")
            raise IncludeError(
                f"{where}: malformed gb-include line {written!r}; write it as "
                "`{#- gb-include <path> -#}` on a line of its own, indented with spaces"
            )
        try:
            target = _target(marker["path"], template, root)
        except IncludeError as exc:
            raise IncludeError(f"{where}: {exc}") from None
        try:
            body = _read(target).splitlines()
        except IncludeError as exc:
            raise IncludeError(f"{where}: {marker['path']} is {exc}") from None
        if any(_ATTEMPT.search(inner) for inner in body):
            raise IncludeError(
                f"{where}: {marker['path']} has a gb-include line; includes do not nest"
            )
        indent = marker["indent"]
        # Names the source in the generated step.yaml, where an included file's own
        # comments would otherwise not say which file they belong to.
        source = f"{root.name}/{target.relative_to(root).as_posix()}"
        out.append(f"{indent}{{#- included from {source} -#}}\n")
        out.extend(f"{indent}{inner}\n" if inner else "\n" for inner in body)
    return "".join(out)


def render(template: Path, image_ref: str, root: Path = STEPS_ROOT) -> str:
    """Return ``template`` expanded, with ``${IMAGE_REF}`` replaced by ``image_ref``.

    Args:
        template: the step-template.yaml to render.
        image_ref: the image reference that replaces ``${IMAGE_REF}``.
        root: the tree included files must resolve into (default: steps/).

    Returns:
        The rendered step.yaml text.

    Raises:
        IncludeError: as :func:`expand`.
        OSError: if a file cannot be read.
    """
    return expand(template, root).replace(IMAGE_REF_TOKEN, image_ref)


def main(argv: Optional[List[str]] = None) -> int:
    """Render TEMPLATE into OUT, writing OUT only when the render succeeds.

    Args:
        argv: the command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        The exit status: 0 on success, 1 when the render fails (the reason goes
        to stderr).
    """
    parser = argparse.ArgumentParser(
        description="Render step-template.yaml into step.yaml."
    )
    parser.add_argument("template", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--image-ref", default="", help="replaces ${IMAGE_REF}")
    args = parser.parse_args(argv)
    try:
        text = render(args.template, args.image_ref)
    except (IncludeError, OSError) as exc:
        print(f"render_step_template: {exc}", file=sys.stderr)
        return 1
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
