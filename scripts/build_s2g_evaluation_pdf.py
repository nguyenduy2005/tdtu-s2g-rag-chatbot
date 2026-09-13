"""Build one polished PDF from the S2G pilot results and evaluation protocol."""

from __future__ import annotations

import html
import re
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/s2g-pilot-evaluation-results-vi.md"
PROTOCOL = ROOT / "docs/evaluation-protocol-vi.md"
OUTPUT = ROOT / "output/pdf/s2g-pilot-evaluation-report.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2368A2")
CYAN = colors.HexColor("#DCEFF7")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5F6C7B")
RULE = colors.HexColor("#C9D4DF")
WARN = colors.HexColor("#FFF3CD")


def register_fonts() -> None:
    root = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("Arial", root / "Arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", root / "Arial Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Italic", root / "Arial Italic.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-BoldItalic", root / "Arial Bold Italic.ttf"))
    pdfmetrics.registerFontFamily(
        "Arial",
        normal="Arial",
        bold="Arial-Bold",
        italic="Arial-Italic",
        boldItalic="Arial-BoldItalic",
    )


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=9.4,
            leading=13.4,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=5.5,
            allowWidows=0,
            allowOrphans=0,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=9.2,
            leading=13.0,
            textColor=INK,
            leftIndent=13,
            firstLineIndent=-8,
            bulletIndent=3,
            spaceAfter=3.5,
        ),
        "h1": ParagraphStyle(
            "H1",
            fontName="Arial-Bold",
            fontSize=19,
            leading=23,
            textColor=NAVY,
            spaceBefore=2,
            spaceAfter=10,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName="Arial-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=11,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            fontName="Arial-Bold",
            fontSize=11.2,
            leading=14,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "code": ParagraphStyle(
            "Code",
            fontName="Courier",
            fontSize=6.8,
            leading=9.2,
            textColor=colors.HexColor("#243B53"),
            leftIndent=7,
            rightIndent=7,
            borderColor=RULE,
            borderWidth=0.5,
            borderPadding=7,
            backColor=colors.HexColor("#F5F7FA"),
            spaceBefore=4,
            spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "Small",
            fontName="Arial",
            fontSize=8.1,
            leading=11,
            textColor=MUTED,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            fontName="Arial-Bold",
            fontSize=25,
            leading=31,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            fontName="Arial",
            fontSize=12,
            leading=17,
            textColor=MUTED,
            alignment=TA_LEFT,
        ),
        "status": ParagraphStyle(
            "Status",
            fontName="Arial-Bold",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#7A4E00"),
            backColor=WARN,
            borderColor=colors.HexColor("#E0B14B"),
            borderWidth=0.6,
            borderPadding=8,
            spaceBefore=12,
            spaceAfter=12,
        ),
    }


TOKEN = re.compile(r"(\*\*.+?\*\*|`.+?`|\[[^\]]+\]\([^)]+\)|\*[^*]+?\*)")


def inline_markup(text: str) -> str:
    output: list[str] = []
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            output.append(f"<b>{html.escape(part[2:-2])}</b>")
        elif part.startswith("`") and part.endswith("`"):
            output.append(f'<font name="Courier" size="8">{html.escape(part[1:-1])}</font>')
        elif part.startswith("[") and "](" in part and part.endswith(")"):
            label, url = part[1:-1].split("](", 1)
            output.append(f'<link href="{html.escape(url, quote=True)}" color="#2368A2">{html.escape(label)}</link>')
        elif part.startswith("*") and part.endswith("*"):
            output.append(f"<i>{html.escape(part[1:-1])}</i>")
        else:
            output.append(html.escape(part))
    return "".join(output)


def special(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or stripped.startswith("#")
        or stripped.startswith("```")
        or stripped.startswith("|")
        or stripped.startswith("- ")
        or bool(re.match(r"^\d+\.\s", stripped))
    )


def markdown_story(path: Path, st: dict[str, ParagraphStyle], *, skip_title: bool = True):
    lines = path.read_text(encoding="utf-8").splitlines()
    story = []
    index = 0
    title_skipped = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("```"):
            language = stripped[3:].strip()
            index += 1
            code: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            prefix = f"[{language}]\n" if language else ""
            story.append(Preformatted(prefix + "\n".join(code), st["code"], maxLineLength=105))
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if level == 1 and skip_title and not title_skipped:
                title_skipped = True
                index += 1
                continue
            style = st["h1"] if level == 1 else st["h2"] if level == 2 else st["h3"]
            story.append(Paragraph(inline_markup(text), style))
            index += 1
            continue
        if stripped.startswith("|"):
            raw_rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    raw_rows.append(cells)
                index += 1
            width = 174 * mm
            columns = max(len(row) for row in raw_rows)
            if columns >= 8:
                col_widths = [43 * mm] + [(width - 43 * mm) / (columns - 1)] * (columns - 1)
            elif columns == 5:
                col_widths = [46 * mm] + [(width - 46 * mm) / 4] * 4
            else:
                col_widths = [width / columns] * columns
            cell_style = ParagraphStyle(
                "TableCell", fontName="Arial", fontSize=7.4, leading=9.2, textColor=INK
            )
            header_style = ParagraphStyle(
                "TableHead", parent=cell_style, fontName="Arial-Bold", textColor=colors.white,
                alignment=TA_CENTER,
            )
            data = []
            for row_index, row in enumerate(raw_rows):
                padded = row + [""] * (columns - len(row))
                data.append([
                    Paragraph(inline_markup(cell), header_style if row_index == 0 else cell_style)
                    for cell in padded
                ])
            table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.35, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([Spacer(1, 2), table, Spacer(1, 8)])
            continue
        bullet = stripped.startswith("- ")
        numbered = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if bullet or numbered:
            marker = "•" if bullet else f"{numbered.group(1)}."
            text = stripped[2:] if bullet else numbered.group(2)
            index += 1
            continuation = []
            while index < len(lines) and lines[index].strip() and not special(lines[index]):
                continuation.append(lines[index].strip())
                index += 1
            if continuation:
                text += " " + " ".join(continuation)
            story.append(Paragraph(inline_markup(text), st["bullet"], bulletText=marker))
            continue
        paragraph = [stripped]
        index += 1
        while index < len(lines) and lines[index].strip() and not special(lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        story.append(Paragraph(inline_markup(" ".join(paragraph)), st["body"]))
    return story


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, height - 13 * mm, width - 18 * mm, height - 13 * mm)
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, height - 10 * mm, "BÁO CÁO ĐÁNH GIÁ PILOT S2G-RAG")
        canvas.drawRightString(width - 18 * mm, 10 * mm, f"Trang {doc.page}")
    canvas.restoreState()


def build() -> None:
    register_fonts()
    st = styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=17 * mm,
        title="Báo cáo kết quả và quy trình đánh giá pilot S2G-RAG",
        author="TDTU Regulatory S2G-RAG Project",
        subject="Đánh giá retrieval, độ bao phủ evidence, chất lượng câu trả lời, độ trễ và chi phí",
    )
    story = [
        Spacer(1, 27 * mm),
        HRFlowable(width="28%", thickness=4, color=BLUE, hAlign="LEFT"),
        Spacer(1, 10 * mm),
        Paragraph("BÁO CÁO ĐÁNH GIÁ<br/>PILOT S2G-RAG", st["cover_title"]),
        Paragraph(
            "Hiệu quả retrieval, độ bao phủ evidence, chất lượng câu trả lời, độ trễ, chi phí và quy trình đánh giá có thể tái lập",
            st["cover_subtitle"],
        ),
        Spacer(1, 16 * mm),
        Paragraph(
            "KẾT QUẢ TẠM THỜI - PILOT CÓ HỖ TRỢ CỦA AGENT<br/>Chưa phải benchmark khóa luận đã đóng băng cho đến khi hoàn tất kiểm chứng độc lập bởi con người.",
            st["status"],
        ),
        Spacer(1, 17 * mm),
        Table(
            [
                [Paragraph("Corpus", st["small"]), Paragraph("Canonical Corpus V1.1 - 27 PDF - 3.482 chunk", st["body"])],
                [Paragraph("Pilot", st["small"]), Paragraph("20 câu hỏi - 27 nhóm evidence - 23 GT chunk duy nhất", st["body"])],
                [Paragraph("Thực thi", st["small"]), Paragraph("Hoàn thành 20/20 - 89 phản hồi GPT-5 nano - 0 lỗi hạ tầng", st["body"])],
                [Paragraph("Ngày báo cáo", st["small"]), Paragraph(date.today().isoformat(), st["body"])],
            ],
            colWidths=[35 * mm, 125 * mm],
            style=TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.35, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        Spacer(1, 22 * mm),
        Paragraph("Chatbot S2G-RAG tra cứu quy định TDTU", st["small"]),
        PageBreak(),
        Paragraph("Phần I - Kết quả đánh giá pilot", st["h1"]),
        Paragraph(
            "Kết quả quan sát từ pilot kỹ thuật gồm 20 câu hỏi. Toàn bộ cảnh báo và giới hạn về tính hợp lệ được giữ nguyên.",
            st["small"],
        ),
        Spacer(1, 5 * mm),
    ]
    story.extend(markdown_story(RESULTS, st))
    story.extend([
        PageBreak(),
        Paragraph("Phụ lục A - Quy trình đánh giá", st["h1"]),
        Paragraph(
            "Định nghĩa, ground-truth firewall, lệnh tái lập, yêu cầu kiểm tra của con người và giới hạn diễn giải.",
            st["small"],
        ),
        Spacer(1, 5 * mm),
    ])
    story.extend(markdown_story(PROTOCOL, st))
    story.extend([
        Spacer(1, 8 * mm),
        HRFlowable(width="100%", thickness=0.5, color=RULE),
        Spacer(1, 3 * mm),
        Paragraph(
            "Nguồn nội dung: docs/s2g-pilot-evaluation-results.md và docs/evaluation-protocol.md. Bản tiếng Việt giữ nguyên số liệu và thuật ngữ kỹ thuật của hai tài liệu nguồn.",
            st["small"],
        ),
    ])
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


if __name__ == "__main__":
    build()
