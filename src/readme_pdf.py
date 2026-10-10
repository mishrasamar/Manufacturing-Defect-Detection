"""Markdown -> PDF converter used to build README.pdf from README.md.

The converter is intentionally small (ReportLab only) and supports the
Markdown subset used by this project's README:

* ATX headings (# .. ####)
* paragraphs with **bold**, *italic*, ``inline code`` and [links](url)
* bullet and numbered lists (one nesting level via indentation)
* fenced code blocks (```python / ```bash / ```text / ```yaml) with
  lightweight syntax highlighting
* pipe tables with per-column width balancing
* standalone images ![alt](relative/path)
* horizontal rules (---) and blockquotes (>)

It renders a professional cover page, a clickable table of contents with
page numbers, DejaVu typography (full Unicode, so the folder-structure
diagram keeps its box-drawing characters), and a footer with the project
name plus page numbers on every page. The output contains real,
searchable text (no bitmaps of text).
"""
from __future__ import annotations

import html
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

_FONT_CACHE: Dict[str, str] = {}


def _register_fonts() -> Tuple[str, str]:
    """Register DejaVu fonts (bundled with matplotlib); fall back to core fonts.

    Returns ``(body_font, mono_font)``. DejaVu provides full Unicode
    support (box-drawing characters in the folder-structure diagram,
    bullets, arrows), which the PDF core fonts lack.
    """
    if _FONT_CACHE:
        return _FONT_CACHE["body"], _FONT_CACHE["mono"]
    body, mono = "Helvetica", "Courier"
    try:
        from matplotlib import font_manager

        font_dir = (
            Path(font_manager.__file__).resolve().parent
            / "mpl-data"
            / "fonts"
            / "ttf"
        )
        candidates = {
            "DejaVu": "DejaVuSans.ttf",
            "DejaVu-Bold": "DejaVuSans-Bold.ttf",
            "DejaVu-Oblique": "DejaVuSans-Oblique.ttf",
            "DejaVuMono": "DejaVuSansMono.ttf",
            "DejaVuMono-Bold": "DejaVuSansMono-Bold.ttf",
        }
        available = {
            name: font_dir / filename
            for name, filename in candidates.items()
            if (font_dir / filename).exists()
        }
        if "DejaVu" in available:
            for name, path in available.items():
                pdfmetrics.registerFont(TTFont(name, str(path)))
            pdfmetrics.registerFontFamily(
                "DejaVu",
                normal="DejaVu",
                bold="DejaVu-Bold" if "DejaVu-Bold" in available else "DejaVu",
                italic="DejaVu-Oblique" if "DejaVu-Oblique" in available else "DejaVu",
                boldItalic="DejaVu-Bold" if "DejaVu-Bold" in available else "DejaVu",
            )
            body = "DejaVu"
            mono = "DejaVuMono" if "DejaVuMono" in available else "Courier"
    except Exception:
        body, mono = "Helvetica", "Courier"
    _FONT_CACHE["body"] = body
    _FONT_CACHE["mono"] = mono
    return body, mono


class _Styles:
    """Paragraph / table styles for the generated document."""

    def __init__(self, body_font: str, mono_font: str, doc_width: float):
        self.body_font = body_font
        self.mono = mono_font
        self.code_max_chars = max(60, int(doc_width / 4.8))
        navy = colors.HexColor("#1F3864")
        ink = colors.HexColor("#20242B")
        muted = colors.HexColor("#5A6472")
        self.cover_title = ParagraphStyle(
            "CoverTitle", fontName=body_font, fontSize=24, leading=30,
            alignment=TA_CENTER, textColor=navy, spaceAfter=10,
        )
        self.cover_sub = ParagraphStyle(
            "CoverSub", fontName=body_font, fontSize=11.5, leading=16,
            alignment=TA_CENTER, textColor=muted, spaceAfter=6,
        )
        self.toc_title = ParagraphStyle(
            "TocTitle", fontName=body_font, fontSize=16, leading=20,
            textColor=navy, spaceAfter=10,
        )
        self.h1 = ParagraphStyle(
            "DocH1", fontName=body_font, fontSize=16, leading=20,
            textColor=navy, spaceBefore=6, spaceAfter=8,
        )
        self.h2 = ParagraphStyle(
            "DocH2", fontName=body_font, fontSize=13, leading=17,
            textColor=navy, spaceBefore=4, spaceAfter=6,
        )
        self.h3 = ParagraphStyle(
            "DocH3", fontName=body_font, fontSize=11, leading=14,
            textColor=ink, spaceBefore=3, spaceAfter=4,
        )
        self.body = ParagraphStyle(
            "DocBody", fontName=body_font, fontSize=9.5, leading=13.8,
            alignment=TA_JUSTIFY, textColor=ink, spaceAfter=4,
        )
        self.small = ParagraphStyle(
            "DocSmall", fontName=body_font, fontSize=8, leading=11, textColor=muted
        )
        self.list0 = ParagraphStyle(
            "DocList0", parent=self.body, leftIndent=16, bulletIndent=4, spaceAfter=2
        )
        self.list1 = ParagraphStyle(
            "DocList1", parent=self.body, leftIndent=30, bulletIndent=18, spaceAfter=2
        )
        self.quote = ParagraphStyle(
            "DocQuote", parent=self.body, leftIndent=14,
            textColor=colors.HexColor("#45505C"),
        )
        self.th = ParagraphStyle(
            "DocTh", fontName=body_font, fontSize=8.6, leading=11.5, textColor=colors.white
        )
        self.td = ParagraphStyle(
            "DocTd", fontName=body_font, fontSize=8.6, leading=11.5, textColor=ink
        )
        self.code = ParagraphStyle(
            "DocCode", fontName=mono_font, fontSize=7.6, leading=9.8,
            textColor=colors.HexColor("#24292E"),
        )
        self.caption = ParagraphStyle(
            "DocCaption", parent=self.small, alignment=TA_CENTER, spaceBefore=2
        )


# -------------------------------------------------------------------------- #
# Markdown parsing
# -------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_HR_RE = re.compile(r"^\s*(?:-\s*){3,}$|^\s*(?:\*\s*){3,}$|^\s*_{3,}\s*$")
_IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")
_FENCE_RE = re.compile(r"^```(\w*)\s*$")
_QUOTE_RE = re.compile(r"^>\s?(.*)$")


def _parse_table_rows(table_lines: List[str]) -> List[List[str]]:
    """Split pipe-table lines into cells; drop the separator row."""
    rows: List[List[str]] = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell or "-") for cell in cells):
            continue  # separator row
        rows.append(cells)
    if not rows:
        return rows
    width = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (width - len(row)))
    return rows


def parse_markdown(md_text: str) -> List[Dict[str, Any]]:
    """Parse the supported Markdown subset into a block list."""
    lines = md_text.splitlines()
    blocks: List[Dict[str, Any]] = []
    paragraph: List[str] = []
    index, total = 0, len(lines)

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append({"kind": "p", "text": " ".join(s.strip() for s in paragraph)})
            paragraph.clear()

    while index < total:
        line = lines[index]
        stripped = line.strip()

        fence = _FENCE_RE.match(stripped)
        if fence:
            flush_paragraph()
            language = fence.group(1).lower()
            code_lines: List[str] = []
            index += 1
            while index < total and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1  # skip the closing fence
            blocks.append({"kind": "code", "lang": language or "text", "code": "\n".join(code_lines)})
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            flush_paragraph()
            blocks.append(
                {"kind": "heading", "level": len(heading.group(1)), "text": heading.group(2).strip()}
            )
            index += 1
            continue

        image = _IMAGE_RE.match(stripped)
        if image:
            flush_paragraph()
            blocks.append({"kind": "image", "alt": image.group(1), "path": image.group(2)})
            index += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            table_lines: List[str] = []
            while index < total and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            blocks.append({"kind": "table", "rows": _parse_table_rows(table_lines)})
            continue

        if _HR_RE.match(stripped):
            flush_paragraph()
            blocks.append({"kind": "hr"})
            index += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote_lines: List[str] = []
            while index < total and lines[index].strip().startswith(">"):
                quote_lines.append(_QUOTE_RE.match(lines[index].strip()).group(1))
                index += 1
            blocks.append({"kind": "quote", "text": " ".join(quote_lines)})
            continue

        list_match = _LIST_RE.match(line)
        if list_match:
            flush_paragraph()
            items: List[Dict[str, Any]] = []
            while index < total:
                matched = _LIST_RE.match(lines[index])
                if not matched:
                    break
                indent = len(matched.group(1).replace("\t", "    "))
                items.append(
                    {
                        "level": 1 if indent >= 2 else 0,
                        "ordered": matched.group(2)[0].isdigit(),
                        "text": matched.group(3).strip(),
                    }
                )
                index += 1
            blocks.append({"kind": "list", "items": items})
            continue

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        paragraph.append(stripped)
        index += 1
    flush_paragraph()
    return blocks


# -------------------------------------------------------------------------- #
# Lightweight syntax highlighting for fenced code blocks
# -------------------------------------------------------------------------- #
_PY_KEYWORDS = {
    "def", "class", "return", "if", "elif", "else", "for", "while", "with", "as",
    "try", "except", "finally", "raise", "import", "from", "pass", "break",
    "continue", "and", "or", "not", "in", "is", "None", "True", "False",
    "lambda", "yield", "global", "nonlocal", "assert", "del", "async", "await",
    "self",
}

_TOKEN_RE = re.compile(
    r'(?P<triple>"""|\'\'\')'
    r"|(?P<comment>\#.*$)"
    r'|(?P<string>f?"(?:\\.|[^"\\])*"|f?\'(?:\\.|[^\'\\])*\')'
    r"|(?P<number>\b\d+(?:\.\d+)?\b)"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)

_COMMENT_LANGS = {"bash", "sh", "shell", "powershell", "yaml", "yml", "ini", "conf", "toml"}


def _esc(text: str) -> str:
    """XML-escape text without touching quotes."""
    return html.escape(text, quote=False)


def _span(text: str, color: str) -> str:
    return f'<font color="{color}">{_esc(text)}</font>'


def _highlight_python(code: str) -> List[str]:
    """Very small Python highlighter returning XML-safe lines."""
    out: List[str] = []
    in_triple: Optional[str] = None
    for line in code.split("\n"):
        pieces: List[str] = []
        position = 0
        if in_triple:
            end = line.find(in_triple)
            if end == -1:
                pieces.append(_span(line, "#032F62"))
                out.append("".join(pieces))
                continue
            pieces.append(_span(line[: end + 3], "#032F62"))
            position = end + 3
            in_triple = None
        while position < len(line):
            match = _TOKEN_RE.search(line, position)
            if not match:
                pieces.append(_esc(line[position:]))
                break
            if match.start() > position:
                pieces.append(_esc(line[position : match.start()]))
            token = match.group(0)
            kind = match.lastgroup
            if kind == "triple":
                closer = token
                end = line.find(closer, match.end())
                if end == -1:
                    pieces.append(_span(line[match.start():], "#032F62"))
                    in_triple = closer
                    break
                pieces.append(_span(line[match.start() : end + 3], "#032F62"))
                position = end + 3
            elif kind == "comment":
                pieces.append(_span(line[match.start():], "#6A737D"))
                break
            elif kind == "string":
                pieces.append(_span(token, "#032F62"))
                position = match.end()
            elif kind == "number":
                pieces.append(_span(token, "#005CC5"))
                position = match.end()
            else:
                if token in _PY_KEYWORDS:
                    pieces.append(_span(token, "#D73A49"))
                elif token[0].isupper():
                    pieces.append(_span(token, "#6F42C1"))
                else:
                    pieces.append(_esc(token))
                position = match.end()
        out.append("".join(pieces))
    return out


def _highlight_simple(code: str, comment_char: Optional[str]) -> List[str]:
    """Escape each line; dim comments found outside simple quote spans."""
    out: List[str] = []
    for line in code.split("\n"):
        escaped = _esc(line)
        if comment_char:
            quote = None
            cut = None
            for position, char in enumerate(line):
                if char in "\"'" and quote is None:
                    quote = char
                elif quote and char == quote:
                    quote = None
                elif char == comment_char and quote is None:
                    cut = position
                    break
            if cut is not None:
                escaped = _esc(line[:cut]) + _span(line[cut:], "#6A737D")
        out.append(escaped)
    return out


def highlight_code(code: str, language: str) -> List[str]:
    """Highlight fenced code; returns one XML-safe html string per line."""
    if language in ("python", "py"):
        return _highlight_python(code)
    comment_char = "#" if language in _COMMENT_LANGS else None
    return _highlight_simple(code, comment_char)


def _wrap_code(code: str, max_chars: int) -> str:
    """Hard-wrap long code lines (continuations are indented)."""
    wrapped: List[str] = []
    for line in code.split("\n"):
        if len(line) <= max_chars:
            wrapped.append(line)
        else:
            pieces = textwrap.wrap(
                line,
                width=max_chars,
                subsequent_indent="    ",
                break_long_words=True,
                break_on_hyphens=False,
            )
            wrapped.extend(pieces or [""])
    return "\n".join(wrapped)


# -------------------------------------------------------------------------- #
# Inline formatting and block renderers
# -------------------------------------------------------------------------- #
def _inline(text: str, mono_font: str) -> str:
    """Convert inline Markdown (bold/italic/code/links) to ReportLab XML."""
    _AMP = "&" + "amp;"
    _LT = "&" + "lt;"
    _GT = "&" + "gt;"
    text = text.replace("&", _AMP).replace("<", _LT).replace(">", _GT)
    code_spans: List[str] = []

    def stash(match: "re.Match[str]") -> str:
        code_spans.append(match.group(1))
        return f"\x00{len(code_spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<link href="\2" color="#1F4E9C">\1</link>',
        text,
    )
    for position, code in enumerate(code_spans):
        text = text.replace(
            f"\x00{position}\x00",
            f'<font face="{mono_font}" color="#9A1B1B">{code}</font>',
        )
    return text


def _nbsp_outside_tags(line_html: str) -> str:
    """Replace spaces with &nbsp; only in text, never inside <tag ...> parts."""
    parts = re.split(r"(<[^>]+>)", line_html)
    return "".join(
        part if part.startswith("<") else part.replace(" ", "&nbsp;")
        for part in parts
    )


def _code_table(code: str, language: str, styles: "_Styles", doc_width: float) -> Table:
    """Render a code block as one table row per line (so it can split)."""
    prepared = _wrap_code(code, styles.code_max_chars)
    rows = []
    for line_html in highlight_code(prepared, language):
        if not line_html:
            line_html = "&nbsp;"
        rows.append([Paragraph(_nbsp_outside_tags(line_html), styles.code)])
    table = Table(rows, colWidths=[doc_width], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#D0D7DE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 0.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0.4),
            ]
        )
    )
    return table


def _table_flowable(rows: List[List[str]], styles: "_Styles", doc_width: float) -> Table:
    """Render a pipe table with content-based column widths."""
    if not rows:
        return Spacer(0, 0)
    header = rows[0]
    column_count = len(header)
    lengths = [
        max([len(row[index]) for row in rows if index < len(row)] + [3])
        for index in range(column_count)
    ]
    weights = [min(max(length, 6), 60) for length in lengths]
    total = float(sum(weights))
    col_widths = [doc_width * weight / total for weight in weights]
    data = [
        [Paragraph(f"<b>{_inline(cell, styles.mono)}</b>", styles.th) for cell in header]
    ]
    for row in rows[1:]:
        data.append([Paragraph(_inline(cell, styles.mono), styles.td) for cell in row])
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB4C0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    for position in range(2, len(data), 2):
        style.append(
            ("BACKGROUND", (0, position), (-1, position), colors.HexColor("#EEF3F9"))
        )
    table.setStyle(TableStyle(style))
    return table


def _image_flowable(
    path_text: str,
    alt: str,
    styles: "_Styles",
    doc_width: float,
    doc_height: float,
    base_dir: Path,
) -> List[Any]:
    """Embed a standalone image with its caption, centered."""
    image_path = Path(path_text)
    if not image_path.is_absolute():
        image_path = base_dir / image_path
    if not image_path.exists():
        return [Paragraph(f"[image not found: {path_text}]", styles.small)]
    try:
        with PILImage.open(image_path) as pil_image:
            width, height = pil_image.size
        scale = min(doc_width / width, doc_height * 0.48 / height, 1.0)
        image = Image(str(image_path), width=width * scale, height=height * scale)
        image.hAlign = "CENTER"
    except Exception as exc:
        return [Paragraph(f"[image could not be embedded: {exc}]", styles.small)]
    flow: List[Any] = [image]
    if alt:
        flow += [Spacer(1, 2), Paragraph(_inline(alt, styles.mono), styles.caption)]
    flow.append(Spacer(1, 6))
    return [KeepTogether(flow)]


def _blocks_to_story(
    blocks: List[Dict[str, Any]],
    styles: "_Styles",
    doc_width: float,
    doc_height: float,
    base_dir: Path,
) -> List[Any]:
    """Convert parsed Markdown blocks into a ReportLab story."""
    story: List[Any] = []
    heading_seen = False
    for block in blocks:
        kind = block["kind"]
        if kind == "heading":
            level = block["level"]
            if level <= 2 and heading_seen:
                story.append(PageBreak())
            heading_seen = True
            style = {1: styles.h1, 2: styles.h2}.get(level, styles.h3)
            story.append(Paragraph(_inline(block["text"], styles.mono), style))
            story.append(Spacer(1, 1))
        elif kind == "p":
            story.append(Paragraph(_inline(block["text"], styles.mono), styles.body))
            story.append(Spacer(1, 4))
        elif kind == "hr":
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.7,
                    color=colors.HexColor("#C9D2DC"),
                    spaceBefore=6,
                    spaceAfter=8,
                )
            )
        elif kind == "code":
            story.append(_code_table(block["code"], block["lang"], styles, doc_width))
            story.append(Spacer(1, 8))
        elif kind == "table":
            story.append(_table_flowable(block["rows"], styles, doc_width))
            story.append(Spacer(1, 8))
        elif kind == "image":
            story.extend(
                _image_flowable(
                    block["path"], block["alt"], styles, doc_width, doc_height, base_dir
                )
            )
        elif kind == "quote":
            story.append(Paragraph(_inline(block["text"], styles.mono), styles.quote))
            story.append(Spacer(1, 4))
        elif kind == "list":
            order_number = 0
            for item in block["items"]:
                text = _inline(item["text"], styles.mono)
                if item["ordered"] and item["level"] == 0:
                    order_number += 1
                    bullet = f"{order_number}."
                elif item["ordered"]:
                    bullet = "-"
                else:
                    bullet = "\u2022"
                style = styles.list1 if item["level"] else styles.list0
                story.append(Paragraph(text, style, bulletText=bullet))
            story.append(Spacer(1, 4))
    return story


# -------------------------------------------------------------------------- #
# Cover, TOC and document assembly
# -------------------------------------------------------------------------- #
class _DocsTemplate(BaseDocTemplate):
    """DocTemplate that feeds headings into the TOC and PDF outline."""

    def __init__(self, filename: str, **kwargs: Any) -> None:
        super().__init__(filename, **kwargs)
        self._toc_sequence = 0

    def beforeDocument(self) -> None:
        # multiBuild runs several passes; keys must be identical in every
        # pass or the TOC never converges.
        self._toc_sequence = 0

    def afterFlowable(self, flowable: Any) -> None:
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name not in ("DocH1", "DocH2"):
            return
        level = 0 if style_name == "DocH1" else 1
        text = flowable.getPlainText()
        self._toc_sequence += 1
        key = f"toc-{self._toc_sequence}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
        self.notify("TOCEntry", (level, text, self.page, key))


def _make_footer(footer_name: str, body_font: str):
    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B7BFC9"))
        canvas.setLineWidth(0.6)
        canvas.line(doc.leftMargin, 40, A4[0] - doc.rightMargin, 40)
        canvas.setFont(body_font, 8)
        canvas.setFillColor(colors.HexColor("#5A6472"))
        canvas.drawString(doc.leftMargin, 28, footer_name)
        canvas.drawRightString(A4[0] - doc.rightMargin, 28, f"Page {doc.page}")
        canvas.restoreState()

    return footer


def _cover_story(project_name, description, version, generated, styles, doc_width) -> List[Any]:
    meta_rows = [
        ("Documentation version", version),
        ("Generated on", generated),
        ("Source document", "README.md (converted automatically)"),
        ("Generator", "src/readme_pdf.py (ReportLab)"),
    ]
    meta_table = Table(
        [
            [Paragraph(f"<b>{_inline(label, styles.mono)}</b>", styles.td),
             Paragraph(_inline(value, styles.mono), styles.td)]
            for label, value in meta_rows
        ],
        colWidths=[doc_width * 0.34, doc_width * 0.66],
        hAlign="CENTER",
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AAB4C0")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3F9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return [
        Spacer(1, 110),
        Paragraph(_inline(project_name, styles.mono), styles.cover_title),
        Spacer(1, 8),
        Paragraph(_inline(description, styles.mono), styles.cover_sub),
        Spacer(1, 46),
        meta_table,
        PageBreak(),
    ]


def build_readme_pdf(
    readme_path: Path,
    out_path: Path,
    *,
    project_name: str,
    description: str,
    version: str,
    generated: str,
    footer_name: str,
) -> Path:
    """Convert a Markdown README into a styled, searchable PDF document."""
    readme_path = Path(readme_path)
    out_path = Path(out_path)
    md_text = readme_path.read_text(encoding="utf-8")

    body_font, mono_font = _register_fonts()
    doc = _DocsTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=52,
        rightMargin=52,
        topMargin=56,
        bottomMargin=58,
        title=project_name,
        author="Automated Manufacturing Defect Detection System",
        subject=f"{project_name} - project documentation",
    )
    styles = _Styles(body_font, mono_font, doc.width)
    blocks = parse_markdown(md_text)

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCLevel0", fontName=body_font, fontSize=10.5, leading=14.5,
            leftIndent=6, firstLineIndent=-6, spaceBefore=4,
            textColor=colors.HexColor("#1F3864"),
        ),
        ParagraphStyle(
            "TOCLevel1", fontName=body_font, fontSize=9, leading=12.5,
            leftIndent=22, firstLineIndent=-6,
            textColor=colors.HexColor("#333B45"),
        ),
    ]
    toc.dotsMinLevel = 0

    story: List[Any] = _cover_story(project_name, description, version, generated, styles, doc.width)
    story.append(Paragraph("Table of Contents", styles.toc_title))
    story.append(toc)
    story.append(PageBreak())
    story += _blocks_to_story(blocks, styles, doc.width, doc.height, readme_path.parent)

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(
        [PageTemplate(id="main", frames=[frame], onPage=_make_footer(footer_name, body_font))]
    )
    doc.multiBuild(story)
    return out_path






