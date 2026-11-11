#!/usr/bin/env python3
# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render a framework step's step-template.yaml into its step.yaml.

common.mk's render-step-template runs this for `make space` and `make
publish-step`. It makes two passes and nothing else:

1. Each line ``{#- gb-include <path> -#}`` is replaced by the lines of the file
   it names, each indented like the marker. ``<path>`` is relative to the file
   holding the marker and must resolve, after symlinks, to a file under the
   steps/ tree. Included files cannot include others. Any other line that
   starts with ``{#`` and ``gb-include`` is refused, so a typo cannot pass as
   a plain Jinja comment.
2. The literal ``${IMAGE_REF}`` is replaced by the image reference.

Runtime Jinja (``{{ ... }}``) and shell expansions pass through untouched. Standard
library only, so rendering needs no install.

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
_ATTEMPT = re.compile(r"^\s*\{#-?\s*gb-include\b")


class IncludeError(Exception):
    """A gb-include line that cannot be expanded."""


def _target(path: str, including: Path, root: Path) -> Path:
    """Resolve an include path against the including file; it must stay in ``root``."""
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
            file, or an included file with a gb-include line of its own.
    """
    template = Path(template)
    root = Path(root).resolve()
    out: List[str] = []
    lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    for number, line in enumerate(lines, start=1):
        if not _ATTEMPT.match(line):
            out.append(line)
            continue
        where = f"{template}:{number}"
        marker = _MARKER.match(line.rstrip("\n"))
        if marker is None:
            raise IncludeError(
                f"{where}: malformed gb-include line {line.strip()!r}; write it as "
                "`{#- gb-include <path> -#}` on a line of its own, indented with spaces"
            )
        try:
            target = _target(marker["path"], template, root)
        except IncludeError as exc:
            raise IncludeError(f"{where}: {exc}") from None
        body = target.read_text(encoding="utf-8").splitlines()
        if any(_ATTEMPT.match(inner) for inner in body):
            raise IncludeError(
                f"{where}: {marker['path']} has a gb-include line; includes do not nest"
            )
        indent = marker["indent"]
        out.extend(f"{indent}{inner}\n" if inner else "\n" for inner in body)
    return "".join(out)


def render(template: Path, image_ref: str, root: Path = STEPS_ROOT) -> str:
    """Return ``template`` expanded, with ``${IMAGE_REF}`` replaced by ``image_ref``."""
    return expand(template, root).replace(IMAGE_REF_TOKEN, image_ref)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point; returns the exit status."""
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
