# pylint: disable=line-too-long  # Office Open XML literals read best unbroken.
"""Render the steps inventory tables as a PowerPoint deck: one table slide per category.

Written directly as Office Open XML with the standard library (zipfile), so it needs no
third-party package. The deck is 16:9 with one blank layout; a category with more than
what fits (estimated from wrapped line counts) continues on further slides titled
'<Category> (cont.)'. Link parts
become clickable hyperlinks, so pass an absolute link base for links to work.
"""

import zipfile
from xml.sax.saxutils import escape, quoteattr

from cells import COLUMNS, FOOTNOTES, Code, Link, cell_text, link_url

SLIDE_W, SLIDE_H = 12192000, 6858000  # EMU, 13.333in x 7.5in
MARGIN = 365760  # 0.4in
EMU_PER_PT = 12700
# Pagination budget: the table area holds about this many wrapped text lines of FONT_PT.
LINES_PER_SLIDE = 22
COL_WEIGHTS = [18, 17, 15, 15, 22, 13]  # relative column widths, one per cells.COLUMNS
FONT_PT, TITLE_PT, NOTE_PT = 10, 28, 9
HEADER_FILL, BORDER, LINK_COLOR = "DDE3EA", "BFBFBF", "0563C1"
CODE_FONT = "Menlo"

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS = f'xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}"'
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_PML = "application/vnd.openxmlformats-officedocument.presentationml"
XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

EMPTY_TREE = (
    '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/>'
    "</p:nvGrpSpPr><p:grpSpPr/>{}</p:spTree>"
)

THEME = f"""<a:theme xmlns:a="{NS_A}" name="Office"><a:themeElements>
<a:clrScheme name="Office"><a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
<a:dk2><a:srgbClr val="44546A"/></a:dk2><a:lt2><a:srgbClr val="E7E6E6"/></a:lt2>
<a:accent1><a:srgbClr val="4472C4"/></a:accent1><a:accent2><a:srgbClr val="ED7D31"/></a:accent2>
<a:accent3><a:srgbClr val="A5A5A5"/></a:accent3><a:accent4><a:srgbClr val="FFC000"/></a:accent4>
<a:accent5><a:srgbClr val="5B9BD5"/></a:accent5><a:accent6><a:srgbClr val="70AD47"/></a:accent6>
<a:hlink><a:srgbClr val="{LINK_COLOR}"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme>
<a:fontScheme name="Office"><a:majorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>
<a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>
<a:fmtScheme name="Office"><a:fillStyleLst>{'<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>' * 3}</a:fillStyleLst>
<a:lnStyleLst>{'<a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>' * 3}</a:lnStyleLst>
<a:effectStyleLst>{'<a:effectStyle><a:effectLst/></a:effectStyle>' * 3}</a:effectStyleLst>
<a:bgFillStyleLst>{'<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>' * 3}</a:bgFillStyleLst></a:fmtScheme>
</a:themeElements></a:theme>"""

CLR_MAP = (
    'bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
    'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" '
    'hlink="hlink" folHlink="folHlink"'
)


def rels_xml(rels):
    """Build a .rels part.

    Args:
        rels: List of (rId, type suffix, target, external).

    Returns:
        XML string.
    """
    items = "".join(
        f'<Relationship Id="{rid}" Type="{REL}/{kind}" Target={quoteattr(target)}'
        f'{" TargetMode=" + quoteattr("External") if external else ""}/>'
        for rid, kind, target, external in rels
    )
    return f'{XML_DECL}<Relationships xmlns="{PKG_REL}">{items}</Relationships>'


def run_xml(part, size_pt, rels, link_base, bold=False):
    """Build one text run, registering a hyperlink relationship for Link parts.

    Args:
        part: str, Code, or Link.
        size_pt: Font size in points.
        rels: The slide's relationship list; Link parts append to it.
        link_base: Prefix for link paths.
        bold: Whether the run is bold.

    Returns:
        An <a:r> XML string.
    """
    attrs = f'lang="en-US" sz="{size_pt * 100}"{" b=" + quoteattr("1") if bold else ""}'
    inner = ""
    if isinstance(part, Code):
        inner = f'<a:latin typeface="{CODE_FONT}"/>'
    elif isinstance(part, Link):
        rid = f"rId{len(rels) + 1}"
        rels.append((rid, "hyperlink", link_url(part.path, link_base), True))
        inner = f'<a:solidFill><a:srgbClr val="{LINK_COLOR}"/></a:solidFill><a:hlinkClick r:id="{rid}"/>'
        attrs += ' u="sng"'
    text = escape(cell_text([part]))
    return f"<a:r><a:rPr {attrs}>{inner}</a:rPr><a:t>{text}</a:t></a:r>"


def para_xml(parts, size_pt, rels, link_base, bold=False):
    """Build one paragraph from parts (an empty paragraph keeps the font size).

    Args:
        parts: List of parts.
        size_pt: Font size in points.
        rels: See run_xml.
        link_base: See run_xml.
        bold: Whether runs are bold.

    Returns:
        An <a:p> XML string.
    """
    runs = "".join(
        run_xml(p, size_pt, rels, link_base, bold) for p in parts if cell_text([p])
    )
    return f'<a:p>{runs}<a:endParaRPr lang="en-US" sz="{size_pt * 100}"/></a:p>'


def cell_xml(parts, rels, link_base, header=False):
    """Build one table cell with thin borders (and a fill for header cells).

    Args:
        parts: List of parts.
        rels: See run_xml.
        link_base: See run_xml.
        header: Whether this is a header cell.

    Returns:
        An <a:tc> XML string.
    """
    line = f'<a:solidFill><a:srgbClr val="{BORDER}"/></a:solidFill>'
    borders = "".join(
        f'<a:{side} w="6350">{line}</a:{side}>' for side in ("lnL", "lnR", "lnT", "lnB")
    )
    fill = (
        f'<a:solidFill><a:srgbClr val="{HEADER_FILL}"/></a:solidFill>' if header else ""
    )
    body = para_xml(parts, FONT_PT, rels, link_base, bold=header)
    return (
        f"<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>{body}</a:txBody>"
        f'<a:tcPr marL="45720" marR="45720" marT="27432" marB="27432">{borders}{fill}</a:tcPr></a:tc>'
    )


def table_xml(rows, rels, link_base, top):
    """Build the table graphic frame for one slide's rows.

    Args:
        rows: Rows (lists of cells) for this slide.
        rels: See run_xml.
        link_base: See run_xml.
        top: Y offset of the table in EMU.

    Returns:
        A <p:graphicFrame> XML string.
    """
    width = SLIDE_W - 2 * MARGIN
    total = sum(COL_WEIGHTS)
    grid = "".join(f'<a:gridCol w="{width * w // total}"/>' for w in COL_WEIGHTS)
    row_h = FONT_PT * EMU_PER_PT * 2
    header = "".join(cell_xml([name], rels, link_base, header=True) for name in COLUMNS)
    body = "".join(
        f'<a:tr h="{row_h}">'
        + "".join(cell_xml(c, rels, link_base) for c in row)
        + "</a:tr>"
        for row in rows
    )
    return (
        f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="3" name="Table"/><p:cNvGraphicFramePr>'
        f'<a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>'
        f'<p:xfrm><a:off x="{MARGIN}" y="{top}"/><a:ext cx="{width}" cy="{row_h * (len(rows) + 1)}"/></p:xfrm>'
        f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
        f'<a:tbl><a:tblPr firstRow="1"/><a:tblGrid>{grid}</a:tblGrid><a:tr h="{row_h}">{header}</a:tr>'
        f"{body}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>"
    )


def textbox_xml(shape_id, name, paras, top, height):
    """Build a full-width text box.

    Args:
        shape_id: Unique shape id on the slide.
        name: Shape name.
        paras: Concatenated <a:p> XML.
        top: Y offset in EMU.
        height: Height in EMU.

    Returns:
        A <p:sp> XML string.
    """
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{name}"/><p:cNvSpPr txBox="1"/><p:nvPr/>'
        f'</p:nvSpPr><p:spPr><a:xfrm><a:off x="{MARGIN}" y="{top}"/>'
        f'<a:ext cx="{SLIDE_W - 2 * MARGIN}" cy="{height}"/></a:xfrm><a:prstGeom prst="rect">'
        f'<a:avLst/></a:prstGeom></p:spPr><p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>'
        f"{paras}</p:txBody></p:sp>"
    )


def slide_xml(title, rows, link_base):
    """Build one slide: title, table, footnotes.

    Args:
        title: Slide title.
        rows: Rows for the table on this slide.
        link_base: See run_xml.

    Returns:
        (slide XML, slide .rels XML).
    """
    rels = [("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml", False)]
    title_box = textbox_xml(
        2,
        "Title",
        para_xml([title], TITLE_PT, rels, link_base, bold=True),
        MARGIN // 2,
        TITLE_PT * EMU_PER_PT * 2,
    )
    table = table_xml(
        rows, rels, link_base, MARGIN // 2 + TITLE_PT * EMU_PER_PT * 2 + MARGIN // 2
    )
    notes = "".join(para_xml(n, NOTE_PT, rels, link_base) for n in FOOTNOTES)
    note_h = NOTE_PT * EMU_PER_PT * 2 * len(FOOTNOTES)
    note_box = textbox_xml(4, "Notes", notes, SLIDE_H - MARGIN - note_h, note_h)
    tree = EMPTY_TREE.format(title_box + table + note_box)
    xml = f"{XML_DECL}<p:sld {NS}><p:cSld>{tree}</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
    return xml, rels_xml(rels)


def col_chars():
    """Estimate how many characters fit on one line of each column.

    Returns:
        List of per-column character capacities (a 10pt glyph averages ~0.55em wide).
    """
    width_pt = (SLIDE_W - 2 * MARGIN) / EMU_PER_PT
    total = sum(COL_WEIGHTS)
    return [
        max(1, int(width_pt * w / total / (FONT_PT * 0.55)) - 1) for w in COL_WEIGHTS
    ]


def row_lines(row, capacities):
    """Estimate how many wrapped lines a row's tallest cell takes.

    Args:
        row: List of cells.
        capacities: Output of col_chars.

    Returns:
        Line count (at least 1).
    """
    return max(
        max(1, -(-len(cell_text(cell)) // cap)) for cell, cap in zip(row, capacities)
    )


def chunk_rows(rows, capacities):
    """Split rows into chunks whose estimated wrapped height fits one slide.

    Args:
        rows: List of rows.
        capacities: Output of col_chars.

    Returns:
        List of row lists; at least one (possibly empty) chunk.
    """
    chunks, chunk, used = [], [], 1  # 1 = header row
    for row in rows:
        lines = row_lines(row, capacities)
        if chunk and used + lines > LINES_PER_SLIDE:
            chunks.append(chunk)
            chunk, used = [], 1
        chunk.append(row)
        used += lines
    return chunks + [chunk]


def paginate(tables):
    """Split each table into slide-sized chunks; continuation slides get ' (cont.)'.

    Args:
        tables: List of (title, rows).

    Returns:
        List of (slide title, rows).
    """
    capacities = col_chars()
    slides = []
    for title, rows in tables:
        for n, chunk in enumerate(chunk_rows(rows, capacities)):
            slides.append((title if n == 0 else f"{title} (cont.)", chunk))
    return slides


def content_types_xml(n_slides):
    """Build [Content_Types].xml, declaring every part in the package.

    Args:
        n_slides: Number of slides.

    Returns:
        XML string.
    """
    overrides = "".join(
        f'<Override PartName="/ppt/slides/slide{i + 1}.xml" ContentType="{CT_PML}.slide+xml"/>'
        for i in range(n_slides)
    )
    return (
        f'{XML_DECL}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/ppt/presentation.xml" ContentType="{CT_PML}.presentation.main+xml"/>'
        f'<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="{CT_PML}.slideMaster+xml"/>'
        f'<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="{CT_PML}.slideLayout+xml"/>'
        f'<Override PartName="/ppt/presProps.xml" ContentType="{CT_PML}.presProps+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
        f"{overrides}</Types>"
    )


def presentation_parts(n_slides):
    """Build the presentation part and its relationships.

    Args:
        n_slides: Number of slides.

    Returns:
        Dict of part name -> XML string.
    """
    slide_ids = "".join(
        f'<p:sldId id="{256 + i}" r:id="rId{10 + i}"/>' for i in range(n_slides)
    )
    slide_rels = [
        (f"rId{10 + i}", "slide", f"slides/slide{i + 1}.xml", False)
        for i in range(n_slides)
    ]
    return {
        "ppt/presentation.xml": (
            f'{XML_DECL}<p:presentation {NS}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/>'
            f"</p:sldMasterIdLst><p:sldIdLst>{slide_ids}</p:sldIdLst>"
            f'<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
        ),
        "ppt/_rels/presentation.xml.rels": rels_xml(
            [
                ("rId1", "slideMaster", "slideMasters/slideMaster1.xml", False),
                ("rId2", "theme", "theme/theme1.xml", False),
                ("rId3", "presProps", "presProps.xml", False),
            ]
            + slide_rels
        ),
        "ppt/presProps.xml": f"{XML_DECL}<p:presentationPr {NS}/>",
    }


def master_parts():
    """Build the slide master (white background), its one blank layout, and the theme.

    Returns:
        Dict of part name -> XML string.
    """
    return {
        "ppt/slideMasters/slideMaster1.xml": (
            f'{XML_DECL}<p:sldMaster {NS}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/>'
            f'</a:solidFill><a:effectLst/></p:bgPr></p:bg>{EMPTY_TREE.format("")}</p:cSld><p:clrMap {CLR_MAP}/>'
            '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>'
        ),
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": rels_xml(
            [
                ("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml", False),
                ("rId2", "theme", "../theme/theme1.xml", False),
            ]
        ),
        "ppt/slideLayouts/slideLayout1.xml": (
            f'{XML_DECL}<p:sldLayout {NS} type="blank" preserve="1"><p:cSld name="Blank">'
            f'{EMPTY_TREE.format("")}</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
        ),
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": rels_xml(
            [("rId1", "slideMaster", "../slideMasters/slideMaster1.xml", False)]
        ),
        "ppt/theme/theme1.xml": XML_DECL + THEME,
    }


def package_parts(n_slides):
    """Build every fixed (non-slide) package part.

    Args:
        n_slides: Number of slides (slide{i}.xml for i in 1..n_slides).

    Returns:
        Dict of part name -> XML string, [Content_Types].xml first.
    """
    return {
        "[Content_Types].xml": content_types_xml(n_slides),
        "_rels/.rels": rels_xml(
            [("rId1", "officeDocument", "ppt/presentation.xml", False)]
        ),
        **presentation_parts(n_slides),
        **master_parts(),
    }


def write(tables, path, link_base):
    """Write all category tables to a .pptx file.

    Args:
        tables: List of (category, rows) from steps_table.build_tables.
        path: Output .pptx path (overwritten).
        link_base: Prefix for link paths; use an absolute URL so links work outside the repo.

    Raises:
        OSError: If the file cannot be written.
    """
    slides = paginate(tables)
    parts = package_parts(len(slides))
    for i, (title, rows) in enumerate(slides, start=1):
        (
            parts[f"ppt/slides/slide{i}.xml"],
            parts[f"ppt/slides/_rels/slide{i}.xml.rels"],
        ) = slide_xml(title, rows, link_base)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", parts.pop("[Content_Types].xml"))
        for name, xml in parts.items():
            zf.writestr(name, xml)
