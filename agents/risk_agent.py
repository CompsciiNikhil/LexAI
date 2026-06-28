"""
LexAI — Risk Flagging Agent (ADK Sub-agent / Skill)

Role: Scans extracted clauses for risky, unfair, or unusual patterns and
      produces a numbered list of risks with severity (HIGH / MEDIUM / LOW)
      using the exact system prompt from Section 7.

ADK concept: LlmAgent used as a sub-agent (AgentTool) inside the Orchestrator.
"""

import os
from google.adk.agents import LlmAgent

# ---------------------------------------------------------------------------
# System prompt — verbatim from Section 7
# ---------------------------------------------------------------------------

RISK_SYSTEM_PROMPT = """You are a legal risk analyst. You read legal documents and flag clauses that may be risky, unfair, or unusual for the signing party.

For each risk found:
- Quote the relevant clause (shortened)
- Explain in plain English why it is risky
- Rate severity: HIGH / MEDIUM / LOW
- Suggest what the user should ask a lawyer about

Known risk categories to always check:
1. Auto-renewal clauses
2. IP/work product ownership assigned to company
3. Non-compete and non-solicitation clauses
4. Unlimited liability or indemnification
5. Termination without cause / at-will clauses
6. Mandatory arbitration (waives right to sue)
7. Penalty/liquidated damages clauses
8. Unilateral amendment rights (company can change terms anytime)
9. Broad confidentiality that restricts future employment
10. Jurisdiction clauses (laws of another state/country apply)

If no risks are found, say so clearly."""

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

risk_agent = LlmAgent(
    name="risk_flagging",
    description=(
        "Scans legal document clauses for risky, unfair, or unusual patterns. "
        "Call this agent with the full document text and extracted clauses to "
        "receive a numbered list of flagged risks with HIGH/MEDIUM/LOW severity "
        "ratings and quoted clause text."
    ),
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    instruction=RISK_SYSTEM_PROMPT,
    tools=[],  # purely LLM-driven — no tool calls needed
)
