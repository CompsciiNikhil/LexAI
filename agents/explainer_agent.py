"""
LexAI — Document Explainer Agent (ADK Sub-agent / Skill)

Role: Takes raw extracted text and clause list and rewrites every section
      in plain, simple English using the exact system prompt from Section 7.

ADK concept: LlmAgent used as a sub-agent (AgentTool) inside the Orchestrator.
"""

import os
from google.adk.agents import LlmAgent

# ---------------------------------------------------------------------------
# System prompt — verbatim from Section 7
# ---------------------------------------------------------------------------

EXPLAINER_SYSTEM_PROMPT = """You are a legal document explainer. Your job is to take raw legal text and rewrite it in plain, simple English that anyone can understand.

Rules:
- Go section by section
- Use simple words. Avoid legal jargon.
- Keep each explanation to 2-4 sentences
- If a section is standard/harmless, say so
- Format output as: [Section Name] → [Plain English explanation]
- Do not add information that isn't in the document"""

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

explainer_agent = LlmAgent(
    name="document_explainer",
    description=(
        "Rewrites every section of a legal document in plain, simple English. "
        "Call this agent with the full document text and the list of extracted "
        "clauses to receive a section-by-section plain-English summary."
    ),
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    instruction=EXPLAINER_SYSTEM_PROMPT,
    tools=[],  # purely LLM-driven — no tool calls needed
)
