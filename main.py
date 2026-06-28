"""
LexAI — FastAPI Entry Point

Serves:
  GET  /          → frontend/index.html
  POST /analyze   → upload PDF, run Explainer + Risk agents, return {summary, risks, document_text}
  POST /ask       → {question, document_text}, run Q&A agent, return {answer}

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
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.explainer_agent import explainer_agent
from agents.risk_agent import risk_agent
from agents.qa_agent import qa_agent
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
    score = 0
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

        score = (high_count * 15) + (medium_count * 8) + (low_count * 3)
        score = min(score, 100)

        if score <= 30:
            label = "Low Risk"
        elif score <= 60:
            label = "Moderate Risk"
        elif score <= 80:
            label = "High Risk"
        else:
            label = "Critical Risk"

    logger.info(
        "[/analyze] calculated risk score: %d (%s), H:%d M:%d L:%d",
        score, label, high_count, medium_count, low_count
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
        "low_count": low_count
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

        score = (high_count * 15) + (medium_count * 8) + (low_count * 3)
        score = min(score, 100)

        if score <= 30:
            label = "Low Risk"
        elif score <= 60:
            label = "Moderate Risk"
        elif score <= 80:
            label = "High Risk"
        else:
            label = "Critical Risk"

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
        "low_count": low_count
    })


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
