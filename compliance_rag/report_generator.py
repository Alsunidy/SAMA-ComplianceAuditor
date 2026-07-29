"""
report_generator.py

Turns the gap-analysis results produced by compliance_engine.run_gap_analysis()
(or the equivalent JSON) into a polished PDF report: an executive summary
(counts + percentages per status) followed by a detailed, color-coded,
per-control findings table.

Run directly on a saved JSON result:
    python report_generator.py gap_analysis_result.json --out report.pdf --company "Al-Rawabi Finance Company"

Or import `generate_pdf_report()` and call it with the list of dicts returned
by compliance_engine.verdicts_to_dicts() directly (no need to round-trip
through a JSON file).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak,
)

# The base-14 PDF fonts (Helvetica, etc.) only cover the WinAnsi character
# set, which is missing glyphs LLM-generated text commonly includes (en/em
# dashes, curly quotes, etc.) - those render as a blank box in some viewers.
# DejaVu Sans has broad Unicode coverage (and Arabic script support, useful
# if this report is ever generated from the Arabic control set), so it is
# bundled here and registered as the report's font instead of Helvetica.
# DejaVu Sans is distributed under a free/permissive license (a Bitstream
# Vera derivative) that allows redistribution.
_FONTS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
pdfmetrics.registerFont(TTFont("DejaVuSans", str(_FONTS_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(_FONTS_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily(
    "DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
    italic="DejaVuSans", boldItalic="DejaVuSans-Bold",
)
FONT_NORMAL = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"

STATUS_COLORS = {
    "PASS": colors.HexColor("#d4edda"),      # light green
    "PARTIAL": colors.HexColor("#fff3cd"),   # light yellow
    "FAIL": colors.HexColor("#f8d7da"),      # light red
}
STATUS_TEXT_COLORS = {
    "PASS": colors.HexColor("#155724"),
    "PARTIAL": colors.HexColor("#856404"),
    "FAIL": colors.HexColor("#721c24"),
}

styles = getSampleStyleSheet()
cell_style = ParagraphStyle("cell", parent=styles["Normal"], fontSize=8, leading=10, fontName=FONT_NORMAL)
header_cell_style = ParagraphStyle("header_cell", parent=styles["Normal"], fontSize=9,
                                    leading=11, textColor=colors.white, fontName=FONT_BOLD)


def _summary_counts(results: List[dict]) -> dict:
    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    for r in results:
        counts[r["status_code"]] = counts.get(r["status_code"], 0) + 1
    return counts


def _build_summary_table(results: List[dict]) -> Table:
    counts = _summary_counts(results)
    total = len(results) or 1
    rows = [
        [Paragraph("Status", header_cell_style),
         Paragraph("Count", header_cell_style),
         Paragraph("Percentage", header_cell_style)],
    ]
    labels = {"PASS": "Compliant", "PARTIAL": "Partially Compliant", "FAIL": "Non-Compliant"}
    for code in ["PASS", "PARTIAL", "FAIL"]:
        pct = f"{counts[code] / total * 100:.1f}%"
        rows.append([labels[code], str(counts[code]), pct])

    table = Table(rows, colWidths=[7 * cm, 3 * cm, 3 * cm])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343a40")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 1), (-1, -1), FONT_NORMAL),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for i, code in enumerate(["PASS", "PARTIAL", "FAIL"], start=1):
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), STATUS_COLORS[code]))
        style_cmds.append(("TEXTCOLOR", (0, i), (-1, i), STATUS_TEXT_COLORS[code]))
    table.setStyle(TableStyle(style_cmds))
    return table


def _build_findings_table(results: List[dict]) -> Table:
    header = [Paragraph(h, header_cell_style) for h in
              ["Control", "Domain", "Status", "Justification", "Recommendation"]]
    rows = [header]
    for r in results:
        rows.append([
            Paragraph(f"{r['control_id']}<br/>{r['control_text'].splitlines()[0].split(' - ', 1)[-1]}", cell_style),
            Paragraph(r["control_domain"], cell_style),
            Paragraph(r["status_label"], cell_style),
            Paragraph(r["justification"], cell_style),
            Paragraph(r["recommendation"] or "-", cell_style),
        ])

    col_widths = [3.0 * cm, 2.4 * cm, 2.8 * cm, 5.1 * cm, 4.9 * cm]
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343a40")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i, r in enumerate(results, start=1):
        style_cmds.append(("BACKGROUND", (2, i), (2, i), STATUS_COLORS[r["status_code"]]))
    table.setStyle(TableStyle(style_cmds))
    return table


def generate_pdf_report(results: List[dict], output_path: str,
                         company_name: Optional[str] = None,
                         framework_name: str = "SAMA Cyber Security Framework") -> None:
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.2 * cm, rightMargin=1.2 * cm,
    )

    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=18, spaceAfter=4, fontName=FONT_BOLD)
    subtitle_style = ParagraphStyle("subtitle", parent=styles["Normal"], fontSize=11,
                                     textColor=colors.HexColor("#555555"), spaceAfter=2, fontName=FONT_NORMAL)
    section_style = ParagraphStyle("section", parent=styles["Heading2"], spaceBefore=14, spaceAfter=8, fontName=FONT_BOLD)

    story = []
    story.append(Paragraph("Compliance Gap Analysis Report", title_style))
    story.append(Paragraph(framework_name, subtitle_style))
    if company_name:
        story.append(Paragraph(f"Prepared for: {company_name}", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("Executive Summary", section_style))
    story.append(_build_summary_table(results))
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph("Detailed Findings", section_style))
    story.append(_build_findings_table(results))

    doc.build(story)


def main():
    parser = argparse.ArgumentParser(description="Generate a PDF Gap Analysis Report from compliance_engine JSON output.")
    parser.add_argument("input_json", help="Path to gap_analysis_result.json (from compliance_engine.py)")
    parser.add_argument("--out", default="gap_analysis_report.pdf")
    parser.add_argument("--company", default=None)
    args = parser.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        results = json.load(f)

    generate_pdf_report(results, args.out, company_name=args.company)
    print(f"Report saved to {args.out}")


if __name__ == "__main__":
    main()
