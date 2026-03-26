from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    HRFlowable,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]

VARIANTS_PATH = ROOT / "data" / "variants.jsonl"
RESULTS_PATH = ROOT / "outputs" / "results.jsonl"
OUT_DIR = ROOT / "outputs" / "session_previews"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_result_by_session_id(results: list[dict], session_id: str) -> Optional[dict]:
    for row in results:
        if row.get("session_id") == session_id:
            return row
    return None


def get_cv_by_id(cvs: list[dict], cv_id: str) -> Optional[dict]:
    for cv in cvs:
        if cv.get("cv_id") == cv_id:
            return cv
    return None


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE)
    return name.strip().replace(" ", "_")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value)).replace("\n", "<br/>")


def _normalize_bullets(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    txt = str(value).strip()
    if not txt:
        return []
    return [part.strip() for part in re.split(r"[;\n]+", txt) if part.strip()]


def _hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.black, spaceAfter=4, spaceBefore=4)


def _styles():
    return {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.black,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.grey,
        ),
        "section_title": ParagraphStyle(
            "section_title",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=10,
            textColor=colors.black,
        ),
        "candidate_title": ParagraphStyle(
            "candidate_title",
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.black,
            alignment=TA_CENTER,
        ),
        "name": ParagraphStyle(
            "name",
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.black,
            alignment=TA_CENTER,
        ),
        "subtitle_center": ParagraphStyle(
            "subtitle_center",
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.black,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.grey,
        ),
        "job_title": ParagraphStyle(
            "job_title",
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=colors.black,
        ),
        "job_meta": ParagraphStyle(
            "job_meta",
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=12,
            textColor=colors.grey,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            leftIndent=12,
            textColor=colors.black,
        ),
        "analysis": ParagraphStyle(
            "analysis",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            leftIndent=6,
            rightIndent=6,
            textColor=colors.black,
        ),
    }


def build_metadata_table(result_row: dict, cv1: dict, cv2: dict) -> Table:
    s = _styles()
    parsed = result_row.get("parsed_response") or {}
    scores = parsed.get("scores") or {}

    rows = [
        ["Session ID", result_row.get("session_id", "")],
        ["Matched Group", result_row.get("matched_group_id", "")],
        ["Provider / Model", f"{result_row.get('provider', '')} / {result_row.get('model', '')}"],
        ["Prompt", result_row.get("prompt_id", "")],
        ["Scenario", result_row.get("scenario", "")],
        ["Bias Variable", result_row.get("bias_variable", "")],
        ["Variant Values", " vs ".join(map(str, result_row.get("variant_values", [])))],
        ["Candidate 1", f"{cv1.get('full_name', '')} ({cv1.get('cv_id', '')})"],
        ["Candidate 2", f"{cv2.get('full_name', '')} ({cv2.get('cv_id', '')})"],
        ["Top Pick", str(parsed.get("top_pick", ""))],
        ["Scores", f"Candidate 1: {scores.get('1', '')}, Candidate 2: {scores.get('2', '')}"],
        ["Personal Signals Shown", str(result_row.get("include_personal_signals", False))],
    ]

    wrapped = [[Paragraph(f"<b>{_safe_text(k)}</b>", s["body"]), Paragraph(_safe_text(v), s["body"])] for k, v in rows]
    table = Table(wrapped, colWidths=[4.7 * cm, 11.3 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [ 
                ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_comparison_table(cv1: dict, cv2: dict) -> Table:
    s = _styles()

    rows = [
        ["Field", "Candidate 1", "Candidate 2"],
        ["Full Name", cv1.get("full_name", ""), cv2.get("full_name", "")],
        ["Gender", cv1.get("gender_signal", ""), cv2.get("gender_signal", "")],
        ["Name Origin", cv1.get("name_origin", ""), cv2.get("name_origin", "")],
        ["Age", str(cv1.get("age", "")), str(cv2.get("age", ""))],
        ["Variant Value", str(cv1.get("variant_value", "")), str(cv2.get("variant_value", ""))],
        ["Career Gap Months", str(cv1.get("career_gap_months", "")), str(cv2.get("career_gap_months", ""))],
        ["Gap Reason", str(cv1.get("gap_reason", "")), str(cv2.get("gap_reason", ""))],
        ["Hobbies", ", ".join(cv1.get("hobbies", [])), ", ".join(cv2.get("hobbies", []))],
    ]

    wrapped = []
    for i, row in enumerate(rows):
        wrapped.append([Paragraph(_safe_text(cell), s["body"]) for cell in row])

    table = Table(wrapped, colWidths=[3.8 * cm, 6.1 * cm, 6.1 * cm], repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_candidate_page(story: list, cv: dict, candidate_number: int) -> None:
    s = _styles()

    story.append(Paragraph(f"Candidate {candidate_number}", s["candidate_title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(_safe_text(cv.get("full_name", "Candidate")), s["name"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(_safe_text(cv.get("job_title_target", "Target Role")), s["subtitle_center"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            _safe_text(
                f"{cv.get('name_origin', '').replace('_', ' ').title()} | "
                f"{cv.get('gender_signal', '')} | Age {cv.get('age', '')}"
            ),
            s["subtitle_center"],
        )
    )
    story.append(Spacer(1, 8))
    story.append(_hr())

    if cv.get("professional_summary"):
        story.append(Paragraph("PROFESSIONAL SUMMARY", s["section_title"]))
        story.append(_hr())
        story.append(Spacer(1, 4))
        story.append(Paragraph(_safe_text(cv["professional_summary"]), s["body"]))
        story.append(Spacer(1, 6))

    experiences = cv.get("experience", [])
    if experiences:
        story.append(Paragraph("PROFESSIONAL EXPERIENCE", s["section_title"]))
        story.append(_hr())
        for exp in experiences:
            story.append(Spacer(1, 4))
            title = exp.get("title", "Role")
            company = exp.get("company", "Company")
            duration = exp.get("duration_months", "N/A")
            story.append(Paragraph(_safe_text(f"{title} — {company}"), s["job_title"]))
            story.append(Paragraph(_safe_text(f"{duration} months"), s["job_meta"]))

            bullets = _normalize_bullets(
                exp.get("missions") or exp.get("achievements") or exp.get("responsibilities")
            )
            if not bullets:
                bullets = [f"Worked as {title} at {company}."]

            for bullet in bullets:
                story.append(Paragraph(_safe_text(f"• {bullet}"), s["bullet"]))

        gap_months = int(cv.get("career_gap_months", 0) or 0)
        gap_reason = cv.get("gap_reason")
        if gap_months > 0 and gap_reason:
            story.append(Spacer(1, 4))
            story.append(
                Paragraph(
                    _safe_text(f"Career break ({gap_months} months) — {gap_reason}"),
                    s["small"],
                )
            )
        story.append(Spacer(1, 6))

    education = cv.get("education", [])
    if education:
        story.append(Paragraph("EDUCATION", s["section_title"]))
        story.append(_hr())
        for edu in education:
            story.append(
                Paragraph(
                    _safe_text(f"{edu.get('degree', 'Degree')} — {edu.get('school', 'Institution')}"),
                    s["job_title"],
                )
            )
            story.append(Paragraph(_safe_text(f"Graduated in {edu.get('year', 'N/A')}"), s["job_meta"]))
            story.append(Spacer(1, 4))

    for section_title, values in [
        ("TECHNICAL SKILLS", cv.get("skills", [])),
        ("SOFT SKILLS", cv.get("soft_skills", [])),
        ("LANGUAGES", cv.get("languages", [])),
        ("CERTIFICATIONS", cv.get("certifications", [])),
        ("INTERESTS", cv.get("hobbies", [])),
    ]:
        if values:
            story.append(Paragraph(section_title, s["section_title"]))
            story.append(_hr())
            story.append(Paragraph(_safe_text("  •  ".join(values)), s["body"]))
            story.append(Spacer(1, 5))


def build_report_pdf(result_row: dict, cv1: dict, cv2: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )

    s = _styles()
    story = []

    parsed = result_row.get("parsed_response") or {}
    reasoning = parsed.get("reasoning") or {}

    story.append(Paragraph("Session Preview", s["title"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(_safe_text(result_row.get("session_id", "")), s["subtitle"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Metadata", s["section_title"]))
    story.append(_hr())
    story.append(build_metadata_table(result_row, cv1, cv2))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Comparison", s["section_title"]))
    story.append(_hr())
    story.append(build_comparison_table(cv1, cv2))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Analysis", s["section_title"]))
    story.append(_hr())

    top_pick = parsed.get("top_pick", "")
    ranking = parsed.get("ranking", "")
    reason1 = reasoning.get("1", "")
    reason2 = reasoning.get("2", "")

    analysis_text = (
        f"<b>Top pick:</b> Candidate { _safe_text(top_pick) }<br/>"
        f"<b>Ranking:</b> { _safe_text(ranking) }<br/>"
        f"<b>Reasoning for Candidate 1:</b> { _safe_text(reason1) }<br/><br/>"
        f"<b>Reasoning for Candidate 2:</b> { _safe_text(reason2) }"
    )
    story.append(Paragraph(analysis_text, s["analysis"]))

    story.append(PageBreak())
    add_candidate_page(story, cv1, 1)

    story.append(PageBreak())
    add_candidate_page(story, cv2, 2)

    doc.build(story)


def main() -> None:
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Create a compact 3-page PDF preview for one session.")
    parser.add_argument("--session-id", help="Session ID from outputs/results.jsonl")
    parser.add_argument("--random", action="store_true", help="Pick a random session from outputs/results.jsonl")
    args = parser.parse_args()

    results = read_jsonl(RESULTS_PATH)
    cvs = read_jsonl(VARIANTS_PATH)

    if not results:
        raise ValueError("No results found in outputs/results.jsonl")

    if args.random:
        result_row = random.choice(results)
        print(f"[INFO] Random session selected: {result_row['session_id']}")
    elif args.session_id:
        result_row = get_result_by_session_id(results, args.session_id)
        if result_row is None:
            raise ValueError(f"No result found with session_id={args.session_id}")
    else:
        raise ValueError("Provide either --session-id or --random")

    cv_ids = result_row.get("cv_ids", [])
    if len(cv_ids) != 2:
        raise ValueError(f"Expected exactly 2 cv_ids in session {result_row.get('session_id')}, got {cv_ids}")

    cv1 = get_cv_by_id(cvs, cv_ids[0])
    cv2 = get_cv_by_id(cvs, cv_ids[1])

    if cv1 is None:
        raise ValueError(f"Could not find CV {cv_ids[0]} in variants.jsonl")
    if cv2 is None:
        raise ValueError(f"Could not find CV {cv_ids[1]} in variants.jsonl")

    filename = sanitize_filename(f"{result_row['session_id']}_{result_row.get('bias_variable', 'session')}.pdf")
    output_path = OUT_DIR / filename
    build_report_pdf(result_row, cv1, cv2, output_path)

    print(f"Session preview PDF created: {output_path}")


if __name__ == "__main__":
    main()