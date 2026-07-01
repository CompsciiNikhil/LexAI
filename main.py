"""
LexAI — FastAPI Entry Point

Serves:
  GET  /               → frontend/index.html
  POST /analyze        → upload PDF, run Explainer + Risk agents, return {summary, risks, document_text}
  POST /ask            → {question, document_text}, run Q&A agent, return {answer}
  POST /download-report → generate and download a PDF report from analysis data

Security:
  - 10 MB file size cap
  - PDF-only validation
  - Temp files deleted immediately after processing (no persistent storage)
  - CORS enabled for local dev
"""

import asyncio
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Must load before any google-adk import so GOOGLE_API_KEY is set

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ReportLab — server-side PDF generation
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import html as html_stdlib
import re as _re

from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.explainer_agent import explainer_agent
from agents.risk_agent import risk_agent
from agents.qa_agent import qa_agent
from agents.negotiation_agent import negotiation_agent
from mcp_server.tools.parse_document import parse_document
from mcp_server.tools.extract_clauses import extract_clauses

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LexAI] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validate critical env vars on startup
# ---------------------------------------------------------------------------

using_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"
if not os.environ.get("GOOGLE_API_KEY") and not using_vertex:
    logger.warning(
        "Neither GOOGLE_API_KEY nor GOOGLE_GENAI_USE_VERTEXAI are set. "
        "Gemini calls will fail. Copy .env.example to .env and add your credentials."
    )
else:
    auth_mode = "Vertex AI (ADC)" if using_vertex else "AI Studio API key"
    logger.info("Auth mode: %s", auth_mode)

# ---------------------------------------------------------------------------
# ADK session service and runners
# ---------------------------------------------------------------------------

APP_NAME = "lexai"
USER_ID = "api_user"

session_service = InMemorySessionService()


def _make_runner(agent) -> Runner:
    return Runner(agent=agent, app_name=APP_NAME, session_service=session_service)


explainer_runner = _make_runner(explainer_agent)
risk_runner = _make_runner(risk_agent)
qa_runner = _make_runner(qa_agent)
negotiation_runner = _make_runner(negotiation_agent)


# ---------------------------------------------------------------------------
# Helper: run an ADK agent and return its final text response
# ---------------------------------------------------------------------------

async def _run_agent(runner: Runner, prompt: str) -> str:
    """
    Create a fresh session, send prompt to the agent, collect the final
    response text and return it.
    """
    session_id = str(uuid.uuid4())
    agent_name = runner.agent.name
    logger.info("[_run_agent] START agent=%s session=%s", agent_name, session_id)

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=prompt)],
    )

    response_parts: list[str] = []
    event_count = 0

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=message,
    ):
        event_count += 1
        is_final = event.is_final_response()
        has_content = bool(event.content and event.content.parts)
        logger.info(
            "[_run_agent] agent=%s event#%d is_final=%s has_content=%s",
            agent_name, event_count, is_final, has_content,
        )

        if has_content:
            for i, part in enumerate(event.content.parts):
                has_text = hasattr(part, "text") and bool(part.text)
                logger.info(
                    "[_run_agent] agent=%s event#%d part[%d] has_text=%s text_preview=%r",
                    agent_name, event_count, i, has_text,
                    (part.text[:120] if has_text else None),
                )

        if is_final:
            if has_content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response_parts.append(part.text)
            break

    result = "".join(response_parts).strip()
    logger.info(
        "[_run_agent] DONE agent=%s events=%d result_len=%d result_preview=%r",
        agent_name, event_count, len(result), result[:200],
    )
    return result


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LexAI API",
    description="Multi-agent legal document intelligence system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent / "frontend"
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the single-page frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(500, "Frontend not found.")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    """Liveness check for Cloud Run."""
    return {"status": "ok", "service": "lexai"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accept a PDF upload, extract text + clauses, then run the Explainer and
    Risk agents concurrently. Returns { summary, risks, document_text,
    filename, page_count }.
    """
    # --- Validation ---
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds the 10 MB limit.")

    logger.info("Analyzing: %s (%d bytes)", file.filename, len(raw))

    # --- Parse document (direct Python — no MCP roundtrip needed server-side) ---
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        parsed = parse_document(tmp_path)
        clauses = extract_clauses(parsed["text"])

    finally:
        # Delete temp file immediately — no persistent user data on disk
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    doc_text: str = parsed["text"]
    clauses_formatted = "\n\n".join(
        f"[{c['heading']}]\n{c['content']}" for c in clauses
    )

    base_context = (
        f"Document: {parsed['filename']} ({parsed['page_count']} pages)\n\n"
        f"Full Document Text:\n{doc_text}\n\n"
        f"Extracted Clauses:\n{clauses_formatted}"
    )

    summary_prompt = (
        base_context
        + "\n\nPlease provide a plain-English, section-by-section summary of this legal document."
    )
    risk_prompt = (
        base_context
        + "\n\nPlease identify and flag all risky clauses in this legal document with severity ratings."
    )

    # --- Run both agents concurrently ---
    logger.info("[/analyze] Running Explainer and Risk agents concurrently…")
    summary, risks = await asyncio.gather(
        _run_agent(explainer_runner, summary_prompt),
        _run_agent(risk_runner, risk_prompt),
    )
    logger.info("[/analyze] summary_len=%d risks_len=%d", len(summary), len(risks))
    logger.info("[/analyze] summary_preview=%r", summary[:300])
    logger.info("[/analyze] risks_500=%r", risks[:500])

    # Calculate risk score metrics
    high_count = 0
    medium_count = 0
    low_count = 0
    score = 0.0
    label = "Low Risk"

    if risks.strip():
        # Match the first list item index
        first_match = re.search(r'(?:^|\n|<li[^>]*>)\s*(\d+)[\.\)]\s+', risks, re.IGNORECASE)
        if first_match:
            list_content = risks[first_match.start():]
            parts = re.split(r'(?:\n|<li[^>]*>)(?=\d+[\.\)]\s+)', list_content, flags=re.IGNORECASE)
            for part in parts:
                part_clean = re.sub(r'^\s*\d+[\.\)]\s+', '', part).strip()
                if len(part_clean) < 10:
                    continue
                upper_part = part_clean.upper()
                if "HIGH" in upper_part:
                    high_count += 1
                elif "LOW" in upper_part:
                    low_count += 1
                else:
                    medium_count += 1
        else:
            # Fallback if no numbered items but some text is present
            no_risk_phrases = ['no risks', 'no significant risk', 'no risky', 'could not find any risk', 'no risk was found']
            if not any(phrase in risks.lower() for phrase in no_risk_phrases):
                # Count as a single medium risk if it doesn't look like "no risks"
                medium_count = 1

        # ── New percentage-based scoring ───────────────────────────────────
        total_clauses = high_count + medium_count + low_count
        max_possible  = total_clauses * 15          # worst case: every clause HIGH
        raw           = (high_count * 15) + (medium_count * 8) + (low_count * 3)

        percentage = (raw / max_possible * 100) if max_possible > 0 else 0.0

        # Severity override — catastrophic contracts never buried by many LOW clauses
        if high_count >= 5:
            percentage = max(percentage, 70.0)
        elif high_count >= 3:
            percentage = max(percentage, 55.0)
        elif high_count >= 1:
            percentage = max(percentage, 35.0)

        score = round(percentage, 1)

        if score <= 30:
            label = "Low Risk"
        elif score <= 55:
            label = "Moderate Risk"
        elif score <= 75:
            label = "High Risk"
        else:
            label = "Critical Risk"
    else:
        total_clauses = 0

    logger.info(
        "[/analyze] calculated risk score: %.1f%% (%s), H:%d M:%d L:%d total:%d",
        score, label, high_count, medium_count, low_count,
        high_count + medium_count + low_count,
    )

    return JSONResponse({
        "summary": summary,
        "risks": risks,
        "document_text": doc_text,
        "filename": parsed["filename"],
        "page_count": parsed["page_count"],
        "score": score,
        "label": label,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "total_clauses": high_count + medium_count + low_count,
    })


@app.post("/analyze-demo")
async def analyze_demo():
    """
    Analyze the local fake_contract.pdf file directly to demonstrate the tool's capability.
    """
    demo_path = Path(__file__).parent / "fake_contract.pdf"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="Demo file not found.")
        
    parsed = parse_document(str(demo_path))
    clauses = extract_clauses(parsed["text"])
    
    doc_text = parsed["text"]
    clauses_formatted = "\n\n".join(
        f"[{c['heading']}]\n{c['content']}" for c in clauses
    )

    base_context = (
        f"Document: {parsed['filename']} ({parsed['page_count']} pages)\n\n"
        f"Full Document Text:\n{doc_text}\n\n"
        f"Extracted Clauses:\n{clauses_formatted}"
    )

    summary_prompt = (
        base_context
        + "\n\nPlease provide a plain-English, section-by-section summary of this legal document."
    )
    risk_prompt = (
        base_context
        + "\n\nPlease identify and flag all risky clauses in this legal document with severity ratings."
    )

    # Run both agents concurrently
    summary, risks = await asyncio.gather(
        _run_agent(explainer_runner, summary_prompt),
        _run_agent(risk_runner, risk_prompt),
    )

    # Calculate risk score metrics
    high_count = 0
    medium_count = 0
    low_count = 0
    score = 0
    label = "Low Risk"

    if risks.strip():
        first_match = re.search(r'(?:^|\n|<li[^>]*>)\s*(\d+)[\.\)]\s+', risks, re.IGNORECASE)
        if first_match:
            list_content = risks[first_match.start():]
            parts = re.split(r'(?:\n|<li[^>]*>)(?=\d+[\.\)]\s+)', list_content, flags=re.IGNORECASE)
            for part in parts:
                part_clean = re.sub(r'^\s*\d+[\.\)]\s+', '', part).strip()
                if len(part_clean) < 10:
                    continue
                upper_part = part_clean.upper()
                if "HIGH" in upper_part:
                    high_count += 1
                elif "LOW" in upper_part:
                    low_count += 1
                else:
                    medium_count += 1
        else:
            no_risk_phrases = ['no risks', 'no significant risk', 'no risky', 'could not find any risk', 'no risk was found']
            if not any(phrase in risks.lower() for phrase in no_risk_phrases):
                medium_count = 1

        # ── New percentage-based scoring ───────────────────────────────────
        total_clauses = high_count + medium_count + low_count
        max_possible  = total_clauses * 15
        raw           = (high_count * 15) + (medium_count * 8) + (low_count * 3)

        percentage = (raw / max_possible * 100) if max_possible > 0 else 0.0

        if high_count >= 5:
            percentage = max(percentage, 70.0)
        elif high_count >= 3:
            percentage = max(percentage, 55.0)
        elif high_count >= 1:
            percentage = max(percentage, 35.0)

        score = round(percentage, 1)

        if score <= 30:
            label = "Low Risk"
        elif score <= 55:
            label = "Moderate Risk"
        elif score <= 75:
            label = "High Risk"
        else:
            label = "Critical Risk"
    else:
        total_clauses = 0

    logger.info(
        "[/analyze-demo] calculated risk score: %.1f%% (%s), H:%d M:%d L:%d total:%d",
        score, label, high_count, medium_count, low_count,
        high_count + medium_count + low_count,
    )

    return JSONResponse({
        "summary": summary,
        "risks": risks,
        "document_text": doc_text,
        "filename": parsed["filename"],
        "page_count": parsed["page_count"],
        "score": score,
        "label": label,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "total_clauses": high_count + medium_count + low_count,
    })


class NegotiateRequest(BaseModel):
    risks_text: str


@app.post("/negotiate")
async def negotiate(body: NegotiateRequest):
    """
    Accept the raw risks text from the risk agent, pass it to the Negotiation
    Suggester Agent, and return { suggestions: string }.
    Run separately from /analyze to avoid free-tier rate limit conflicts.
    """
    if not body.risks_text.strip():
        raise HTTPException(status_code=400, detail="risks_text cannot be empty.")

    logger.info("[/negotiate] risks_text_len=%d", len(body.risks_text))

    prompt = (
        "Here is the risk analysis output from a legal document. "
        "Please provide negotiation suggestions for each HIGH and MEDIUM risk clause:\n\n"
        + body.risks_text
    )

    suggestions = await _run_agent(negotiation_runner, prompt)
    logger.info("[/negotiate] suggestions_len=%d suggestions_preview=%r", len(suggestions), suggestions[:300])

    return JSONResponse({"suggestions": suggestions})


class AskRequest(BaseModel):
    question: str
    document_text: str


@app.post("/ask")
async def ask(body: AskRequest):
    """
    Accept a natural language question and the document text, run the Q&A
    agent (which uses search_clause internally), and return { answer }.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not body.document_text.strip():
        raise HTTPException(status_code=400, detail="Document text is required.")

    logger.info("Q&A question: %s", body.question[:120])

    prompt = (
        f"Document text:\n{body.document_text}\n\n"
        f"User question: {body.question}"
    )

    answer = await _run_agent(qa_runner, prompt)
    logger.info("[/ask] answer_len=%d answer_preview=%r", len(answer), answer[:300])

    return JSONResponse({"answer": answer})


# ---------------------------------------------------------------------------
# Download PDF Report
# ---------------------------------------------------------------------------

class DownloadReportRequest(BaseModel):
    filename: str
    summary: str
    risks: str
    score: float
    label: str
    high_count: int
    medium_count: int
    low_count: int
    total_clauses: int = 0


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities for plain-text PDF rendering."""
    text = _re.sub(r"<[^>]+>", " ", text)
    text = html_stdlib.unescape(text)
    return _re.sub(r" {2,}", " ", text).strip()


def _build_report_pdf(req: DownloadReportRequest) -> BytesIO:
    """Build a polished A4 PDF and return it as an in-memory BytesIO."""
    buf = BytesIO()
    PAGE_W, PAGE_H = A4
    MARGIN = 20 * mm

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"LexAI Report — {req.filename}",
        author="LexAI",
    )

    # ── Palette ────────────────────────────────────────────────────────────
    DARK       = colors.HexColor("#ffffff")
    SURFACE    = colors.HexColor("#ffffff")
    ACCENT     = colors.HexColor("#6366f1")
    ACCENT_SOFT = colors.HexColor("#0f172a")
    TEXT       = colors.HexColor("#000000")
    TEXT_DIM   = colors.HexColor("#3c3c3c")
    WHITE      = colors.HexColor("#ffffff")
    C_HIGH     = colors.HexColor("#ef4444")
    C_MEDIUM   = colors.HexColor("#f97316")
    C_LOW      = colors.HexColor("#10b981")
    C_SUCCESS  = colors.HexColor("#10b981")
    DIVIDER    = colors.HexColor("#e2e8f0")

    label_lower = req.label.lower()
    if "critical" in label_lower:
        SCORE_COLOR = C_HIGH
    elif "high" in label_lower:
        SCORE_COLOR = C_HIGH
    elif "moderate" in label_lower:
        SCORE_COLOR = C_MEDIUM
    else:
        SCORE_COLOR = C_SUCCESS

    # ── Styles ─────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    def P(name, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, **kw)

    sty_header_title = P("HeaderTitle",
        fontSize=26, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#080b14"), leading=30, spaceAfter=2)
    sty_header_sub = P("HeaderSub",
        fontSize=11, fontName="Helvetica",
        textColor=colors.HexColor("#475569"), leading=14)
    sty_meta = P("Meta",
        fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#475569"), leading=13)
    sty_section_title = P("SectionTitle",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f172a"), leading=15,
        spaceBefore=14, spaceAfter=6,
        textTransform="uppercase", letterSpacing=0.8)
    sty_body = P("Body",
        fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#000000"), leading=15, spaceAfter=4)
    sty_risk_title = P("RiskTitle",
        fontSize=10, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#000000"), leading=14)
    sty_risk_body = P("RiskBody",
        fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#000000"), leading=13, spaceAfter=2)
    sty_footer = P("Footer",
        fontSize=8, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#3c3c3c"), leading=11, alignment=TA_CENTER)
    sty_score_num = P("ScoreNum",
        fontSize=40, fontName="Helvetica-Bold",
        textColor=SCORE_COLOR, leading=44, alignment=TA_CENTER)
    sty_score_label = P("ScoreLabel",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=SCORE_COLOR, leading=16, alignment=TA_CENTER)

    story = []

    # ── Header banner ──────────────────────────────────────────────────────
    header_data = [[
        Paragraph("⚖ LexAI", sty_header_title),
        Paragraph("Multi-agent legal document intelligence", sty_header_sub),
    ]]
    header_tbl = Table(header_data,
        colWidths=[90 * mm, None],
        rowHeights=[22 * mm])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), DARK),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [DARK]),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), [6, 6, 6, 6]),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 8 * mm))

    # ── Document meta ──────────────────────────────────────────────────────
    analysis_date = datetime.utcnow().strftime("%B %d, %Y at %H:%M UTC")
    story.append(Paragraph(f"Document: <b>{_strip_html(req.filename)}</b>", sty_meta))
    story.append(Paragraph(f"Analysis Date: {analysis_date}", sty_meta))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=DIVIDER, spaceAfter=6 * mm))

    # ── Risk Score card ────────────────────────────────────────────────────
    score_cell = [
        [Paragraph(f"{req.score}%", sty_score_num)],
        [Paragraph(f"Based on {req.total_clauses} clauses reviewed", sty_meta)],
        [Spacer(1, 2)],
        [Paragraph(req.label, sty_score_label)],
    ]

    def _badge(count: int, label: str, color: colors.Color):
        """Tiny coloured badge paragraph."""
        return Paragraph(
            f'<font color="{color.hexval()}" size="9"><b>{count} {label}</b></font>',
            P("Badge", alignment=TA_CENTER, leading=12))

    badge_data = [[
        _badge(req.high_count,   "HIGH",   C_HIGH),
        _badge(req.medium_count, "MEDIUM", C_MEDIUM),
        _badge(req.low_count,    "LOW",    C_LOW),
    ]]
    badge_tbl = Table(badge_data, colWidths=[50 * mm, 50 * mm, 50 * mm],
                      rowHeights=[12 * mm])
    badge_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#fee2e2")),
        ("BACKGROUND",    (1, 0), (1, -1), colors.HexColor("#ffedd5")),
        ("BACKGROUND",    (2, 0), (2, -1), colors.HexColor("#dcfce7")),
        ("BOX",           (0, 0), (0, -1), 0.5, C_HIGH),
        ("BOX",           (1, 0), (1, -1), 0.5, C_MEDIUM),
        ("BOX",           (2, 0), (2, -1), 0.5, C_LOW),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))

    score_section = Table(
        [[Table(score_cell, colWidths=[55 * mm]), badge_tbl]],
        colWidths=[60 * mm, None],
    )
    score_section.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), SURFACE),
        ("BOX",           (0, 0), (-1, -1), 1,
         SCORE_COLOR),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(score_section)
    story.append(Spacer(1, 7 * mm))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=DIVIDER, spaceAfter=4 * mm))

    # ── Summary section ────────────────────────────────────────────────────
    story.append(Paragraph("Plain English Summary", sty_section_title))
    clean_summary = _strip_html(req.summary)
    for para in clean_summary.split("\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, sty_body))
    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=DIVIDER, spaceAfter=4 * mm))

    # ── Risk Flags section ─────────────────────────────────────────────────
    story.append(Paragraph("Risk Flags", sty_section_title))
    clean_risks = _strip_html(req.risks)

    # Split numbered list items
    risk_parts = _re.split(r"(?:^|\n)(?=\d+[.)]\ )", clean_risks, flags=_re.M)
    risk_parts = [p.strip() for p in risk_parts if p.strip()]

    if not risk_parts:
        story.append(Paragraph("No significant risks were identified.", sty_body))
    else:
        for i, part in enumerate(risk_parts):
            upper = part.upper()
            if "HIGH" in upper:
                sev_color = C_HIGH
                sev_label = "HIGH"
            elif "MEDIUM" in upper:
                sev_color = C_MEDIUM
                sev_label = "MEDIUM"
            elif "LOW" in upper:
                sev_color = C_LOW
                sev_label = "LOW"
            else:
                sev_color = C_MEDIUM
                sev_label = "MEDIUM"

            # Strip leading number
            body_text = _re.sub(r"^\d+[.)]\s*", "", part).strip()

            badge_para = Paragraph(
                f'<font color="{sev_color.hexval()}"><b> {sev_label} </b></font>',
                P(f"RiskBadge{i}", fontSize=8, leading=10, alignment=TA_CENTER)
            )
            num_para   = Paragraph(f"<b>{i + 1}</b>",
                P(f"RiskNum{i}", fontSize=9, leading=10,
                  textColor=TEXT_DIM, alignment=TA_CENTER))

            lines = [l.strip() for l in body_text.split("\n") if l.strip()]
            title_text = lines[0] if lines else body_text
            rest_text  = " ".join(lines[1:]) if len(lines) > 1 else ""

            inner = [
                [Paragraph(title_text, sty_risk_title)],
            ]
            if rest_text:
                inner.append([Paragraph(rest_text, sty_risk_body)])

            content_tbl = Table(inner, colWidths=[None])
            content_tbl.setStyle(TableStyle([
                ("LEFTPADDING",  (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING",   (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 1),
            ]))

            row_data   = [[num_para, badge_para, content_tbl]]
            risk_row   = Table(row_data,
                colWidths=[10 * mm, 22 * mm, None],
                rowHeights=None)
            risk_row.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), SURFACE),
                ("LINEAFTER",     (0, 0), (0, -1), 3, sev_color),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING",    (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("BOX",           (0, 0), (-1, -1), 0.5, DIVIDER),
            ]))

            story.append(KeepTogether(risk_row))
            story.append(Spacer(1, 3 * mm))

    story.append(Spacer(1, 6 * mm))

    # ── Footer ─────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1,
                            color=DIVIDER, spaceBefore=4 * mm, spaceAfter=4 * mm))
    story.append(Paragraph(
        "Generated by LexAI — For legal clarity only, not legal advice",
        sty_footer))

    doc.build(story)
    buf.seek(0)
    return buf


@app.post("/download-report")
async def download_report(body: DownloadReportRequest):
    """
    Accept analysis data and return a styled PDF report as a file download.
    """
    logger.info("[/download-report] Generating PDF for %s", body.filename)
    try:
        pdf_buf = _build_report_pdf(body)
    except Exception as exc:
        logger.exception("[/download-report] PDF generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate PDF report.")

    safe_name = _re.sub(r"[^\w\-\.]", "_", body.filename.rsplit(".", 1)[0])
    download_name = f"lexai_report_{safe_name}.pdf"

    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"',
        },
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        reload=True,
    )
