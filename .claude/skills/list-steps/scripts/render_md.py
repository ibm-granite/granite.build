# Copyright Granite.Build Authors
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

"""Render the steps inventory tables as Markdown."""

from cells import COLUMNS, FOOTNOTES, Code, Link, link_url


def md_part(part, link_base):
    """Render one cell part as Markdown.

    Args:
        part: str, Code, or Link.
        link_base: Prefix for Link paths ('' keeps them repo-relative).

    Returns:
        Markdown text.
    """
    if isinstance(part, Link):
        return f"[{part.label}]({link_url(part.path, link_base)})"
    if isinstance(part, Code):
        return f"`{part.text}`"
    return part


def md_cells(cells, link_base):
    """Render a list of cells as one Markdown table row.

    Args:
        cells: List of cells (each a list of parts).
        link_base: See md_part.

    Returns:
        A '| ... |' row.
    """
    return (
        "| "
        + " | ".join("".join(md_part(p, link_base) for p in c) for c in cells)
        + " |"
    )


def render(tables, link_base=""):
    """Render all category tables, each under a '## <Category>' heading, plus footnotes.

    Args:
        tables: List of (category, rows) from steps_table.build_tables.
        link_base: See md_part.

    Returns:
        The Markdown document as one string.
    """
    lines = []
    for category, rows in tables:
        lines += [
            f"## {category}",
            "",
            "| " + " | ".join(COLUMNS) + " |",
            "|" + "---|" * len(COLUMNS),
        ]
        lines += [md_cells(row, link_base) for row in rows]
        lines.append("")
    lines += ["".join(md_part(p, link_base) for p in note) for note in FOOTNOTES]
    return "\n".join(lines) + "\n"
