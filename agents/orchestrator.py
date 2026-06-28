"""
LexAI — Orchestrator Agent (Root Agent)

Role: Entry point for all user interactions. Receives an uploaded document
      path and/or a user question, calls MCP tools to parse and structure
      the document, then delegates to the appropriate sub-agent(s).

Delegation flow (new document):
  1. parse_document  (MCP)  → extract raw text
  2. extract_clauses (MCP)  → structure into clauses
  3. document_explainer     → plain-English summary
  4. risk_flagging          → flagged risk list
  5. Assembles and returns both outputs to the user

Delegation flow (follow-up Q&A):
  1. qa_agent               → answers the question using search_clause (MCP)

ADK concepts used:
  - LlmAgent as root orchestrator with sub_agents
  - AgentTool wrapping each sub-agent for tool-style delegation
  - McpToolset for parse_document + extract_clauses tools
"""

import os
import sys

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, McpToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters, StdioConnectionParams

from .explainer_agent import explainer_agent
from .risk_agent import risk_agent
from .qa_agent import qa_agent

# ---------------------------------------------------------------------------
# System prompt — verbatim from Section 7
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """You are LexAI's orchestrator. You help users understand legal documents.

When a user uploads a document:
1. First call the Document Explainer Agent to generate a plain-English summary
2. Then call the Risk Flagging Agent to identify risky clauses
3. Present both results clearly to the user
4. Tell the user they can ask follow-up questions

When a user asks a question about the document:
- Call the Q&A Agent with the question and the document text
- Return its answer clearly

Always be empathetic. Remind users that LexAI provides clarity, not legal advice, and they should consult a lawyer for important decisions.

Never fabricate clauses. Only work with what is in the document.

---

TOOLS AVAILABLE TO YOU:

parse_document(file_path, api_key):
  Call this FIRST when a new document is uploaded.
  Returns: {text, page_count, filename}

extract_clauses(text, api_key):
  Call this SECOND after parse_document.
  Returns: list of {heading, content, position}

document_explainer(request):
  Call this THIRD. Pass the full document text and clause list.
  Returns: plain-English section-by-section summary.

risk_flagging(request):
  Call this FOURTH. Pass the full document text and clause list.
  Returns: numbered risk list with HIGH/MEDIUM/LOW severity ratings.

qa_agent(request):
  Call this when the user asks a follow-up question about the document.
  Pass the user's question and the full document text.
  Returns: plain-English answer referencing the relevant clause.

---

RESPONSE FORMAT for a new document upload:

## 📄 Document Overview
[filename, page count, brief description]

## 📝 Plain-English Summary
[Output from document_explainer]

## ⚠️ Risk Flags
[Output from risk_flagging]

---

*LexAI provides legal clarity, not legal advice. Please consult a qualified lawyer for important decisions.*"""

# ---------------------------------------------------------------------------
# MCP toolset — parse_document + extract_clauses tools for the orchestrator
# Filtered to only these two; search_clause belongs to the Q&A agent
# ---------------------------------------------------------------------------

_MCP_API_KEY = os.environ.get("MCP_API_KEY", "")

_orchestrator_mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env={
                **os.environ,
                "MCP_API_KEY": _MCP_API_KEY,
            },
        ),
        timeout=30.0,
    ),
    tool_filter=["parse_document", "extract_clauses"],
)

# ---------------------------------------------------------------------------
# Orchestrator agent
# ---------------------------------------------------------------------------

orchestrator = LlmAgent(
    name="lexai_orchestrator",
    description="LexAI root orchestrator — routes legal document analysis requests.",
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    instruction=ORCHESTRATOR_SYSTEM_PROMPT,
    tools=[
        # MCP tools (parse + extract)
        _orchestrator_mcp_toolset,
        # Sub-agents wrapped as AgentTools for tool-style delegation
        AgentTool(agent=explainer_agent),
        AgentTool(agent=risk_agent),
        AgentTool(agent=qa_agent),
    ],
)
