"""Render the steps inventory tables as an Excel workbook: one sheet per category.

Requires XlsxWriter (BSD-2-Clause): `pip install -r requirements.txt` next to this file.
A cell holding exactly one link becomes a clickable hyperlink; a cell with several links
(e.g. tune/fm-tune's bash + skypilot READMEs) shows their labels and lists the URLs in a
cell comment, since Excel allows only one hyperlink per cell.
"""

import xlsxwriter

from cells import COLUMNS, FOOTNOTES, Code, cell_links, cell_text, link_url

# Approximate column widths in characters; Excel has no auto-fit on write.
WIDTHS = [26, 16, 30, 24, 52, 24]
MAX_SHEET_NAME = 31  # Excel's limit


def add_formats(book):
    """Create the cell formats the sheets share.

    Args:
        book: The xlsxwriter Workbook.

    Returns:
        Dict of name -> Format: header, text, code, link.
    """
    return {
        "header": book.add_format({"bold": True, "bg_color": "#DDE3EA", "border": 1, "valign": "top"}),
        "text": book.add_format({"text_wrap": True, "valign": "top", "border": 1}),
        "code": book.add_format({"text_wrap": True, "valign": "top", "border": 1, "font_name": "Menlo"}),
        "link": book.add_format({"text_wrap": True, "valign": "top", "border": 1,
                                 "font_color": "blue", "underline": 1}),
    }


def write_cell(sheet, row, col, cell, formats, link_base):
    """Write one cell, as a hyperlink when it holds exactly one link.

    Args:
        sheet: The xlsxwriter Worksheet.
        row: Zero-based row index.
        col: Zero-based column index.
        cell: List of parts.
        formats: Output of add_formats.
        link_base: Prefix for link paths.
    """
    text = cell_text(cell)
    links = cell_links(cell)
    if len(links) == 1:
        sheet.write_url(row, col, link_url(links[0].path, link_base), formats["link"], string=text)
        return
    fmt = formats["code"] if any(isinstance(p, Code) for p in cell) else formats["text"]
    sheet.write_string(row, col, text, fmt)
    if links:
        sheet.write_comment(row, col, "\n".join(f"{ln.label}: {link_url(ln.path, link_base)}"
                                                for ln in links), {"x_scale": 3})


def write_sheet(book, category, rows, formats, link_base):
    """Add one category's table as a worksheet with a frozen header and filters.

    Args:
        book: The xlsxwriter Workbook.
        category: Sheet name.
        rows: List of rows (each a list of cells).
        formats: Output of add_formats.
        link_base: Prefix for link paths.
    """
    sheet = book.add_worksheet(category[:MAX_SHEET_NAME])
    for col, (name, width) in enumerate(zip(COLUMNS, WIDTHS)):
        sheet.set_column(col, col, width)
        sheet.write_string(0, col, name, formats["header"])
    for r, row in enumerate(rows, start=1):
        for col, cell in enumerate(row):
            write_cell(sheet, r, col, cell, formats, link_base)
    sheet.freeze_panes(1, 0)
    sheet.autofilter(0, 0, len(rows), len(COLUMNS) - 1)
    for i, note in enumerate(FOOTNOTES):
        sheet.write_string(len(rows) + 2 + i, 0, cell_text(note))


def write(tables, path, link_base):
    """Write all category tables to an .xlsx file.

    Args:
        tables: List of (category, rows) from steps_table.build_tables.
        path: Output .xlsx path (overwritten).
        link_base: Prefix for link paths; use an absolute URL so links work outside the repo.

    Raises:
        xlsxwriter.exceptions.FileCreateError: If the file cannot be written.
    """
    book = xlsxwriter.Workbook(path)
    formats = add_formats(book)
    for category, rows in tables:
        write_sheet(book, category, rows, formats, link_base)
    book.close()
