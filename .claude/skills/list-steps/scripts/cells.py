"""Format-neutral table model shared by the steps inventory renderers.

A table is (category, rows); a row is a list of cells, one per COLUMNS entry; a cell is a
list of parts, each a plain str, a Code span, or a Link to a repo-relative path. Renderers
(Markdown, xlsx, pptx) decide how each part looks.
"""

from typing import NamedTuple


class Link(NamedTuple):
    """A hyperlink part.

    Attributes:
        label: Display text.
        path: Repo-relative path; renderers prefix it with the link base.
    """

    label: str
    path: str


class Code(NamedTuple):
    """A literal (monospace) text part, e.g. an image reference.

    Attributes:
        text: The literal text.
    """

    text: str


COLUMNS = ["Step", "Environments", "Tested Environments", "README", "Image", "Sample build"]

FOOTNOTES = [
    ["‡ Not published to ", Code("configurations/"), "; links the README under ", Code("steps/"), "."],
    ["Image: blank = custom image built from the step's Dockerfile."],
]


def part_text(part):
    """Return the visible text of one part.

    Args:
        part: str, Code, or Link.

    Returns:
        The text a reader sees.
    """
    if isinstance(part, Link):
        return part.label
    if isinstance(part, Code):
        return part.text
    return part


def cell_text(cell):
    """Return a cell's visible text with all formatting dropped.

    Args:
        cell: List of parts.

    Returns:
        Concatenated text.
    """
    return "".join(part_text(p) for p in cell)


def cell_links(cell):
    """Return the Link parts of a cell, in order.

    Args:
        cell: List of parts.

    Returns:
        List of Link.
    """
    return [p for p in cell if isinstance(p, Link)]


def link_url(path, link_base):
    """Resolve a repo-relative path against the link base.

    Args:
        path: Repo-relative path.
        link_base: URL prefix ending in '/', or '' to keep the path relative.

    Returns:
        The URL (or the relative path when link_base is empty).
    """
    return f"{link_base}{path}"
