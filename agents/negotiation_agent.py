"""
LexAI — Negotiation Suggester Agent (ADK Sub-agent / Skill)

Role: Receives the full risk analysis output from the Risk Flagging Agent
      and produces concrete negotiation suggestions — exact alternative
      wording the signing party can propose back — for every HIGH or MEDIUM
      risk clause.

ADK concept: LlmAgent used as a standalone agent invoked directly by the
             /negotiate FastAPI endpoint (not via the orchestrator).
"""

import os
from google.adk.agents import LlmAgent

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

NEGOTIATION_SYSTEM_PROMPT = """You are a contract negotiation advisor. You receive a list of risky clauses from a legal document. For each HIGH or MEDIUM risk clause, you suggest exact alternative wording the signing party can propose back to the other party.

Format each suggestion EXACTLY as follows (use this format for every clause, no deviations):

[CLAUSE NAME]
Problem: one sentence explaining the issue
Suggest proposing: "exact alternative clause wording here — be specific, complete, and legally clear"
Why this is better: one sentence explanation

Rules:
- Only cover HIGH and MEDIUM risks. Skip LOW risks entirely.
- The suggested wording must be a complete, standalone clause — not a fragment.
- Be specific and practical. Use real legal language, not vague placeholders.
- Start each new clause with its name inside square brackets on its own line.
- Do not add preamble or conclusion text — output ONLY the formatted suggestions.
- Separate each suggestion with a blank line."""

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

negotiation_agent = LlmAgent(
    name="negotiation_suggester",
    description=(
        "Receives the risk analysis output and produces concrete negotiation "
        "suggestions — exact alternative wording the signing party can propose "
        "— for every HIGH or MEDIUM risk clause in the document."
    ),
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    instruction=NEGOTIATION_SYSTEM_PROMPT,
    tools=[],  # purely LLM-driven — no tool calls needed
)
