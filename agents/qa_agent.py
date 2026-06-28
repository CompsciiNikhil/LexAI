"""
LexAI — Q&A Agent (ADK Sub-agent / Skill)

Role: Answers natural language questions about an uploaded document.
      Uses the search_clause MCP tool to find the most relevant clause
      before forming a plain-English answer.

ADK concepts used:
  - LlmAgent as a sub-agent (AgentTool) inside the Orchestrator
  - McpToolset to connect to the LexAI MCP server (search_clause tool)
"""

import os
import sys
from google.adk.agents import LlmAgent
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters, StdioConnectionParams

# ---------------------------------------------------------------------------
# System prompt — verbatim from Section 7
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """You are a legal document Q&A assistant. A user has uploaded a legal document and wants to ask questions about it.

You will be given:
- The document text (or relevant clauses)
- The user's question

Your job:
- Find the most relevant part of the document that answers the question
- Answer in plain, simple English
- Reference which section or clause your answer comes from
- If the document doesn't address the question, say so clearly — do not guess

Always end with: "For decisions based on this, please consult a qualified lawyer.\""""

# ---------------------------------------------------------------------------
# MCP connection parameters — connects to LexAI MCP server via stdio
# Filtered to only expose the search_clause tool to this agent
# ---------------------------------------------------------------------------

_MCP_API_KEY = os.environ.get("MCP_API_KEY", "")

_mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,          # same Python interpreter
            args=["-m", "mcp_server.server"],  # run MCP server as a module
            env={
                **os.environ,
                "MCP_API_KEY": _MCP_API_KEY,
            },
        ),
        timeout=30.0,
    ),
    tool_filter=["search_clause"],       # only expose search_clause to this agent
)

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

qa_agent = LlmAgent(
    name="qa_agent",
    description=(
        "Answers natural language questions about an uploaded legal document. "
        "Call this agent with the user's question and the full document text. "
        "The agent will use search_clause to find the relevant clause and "
        "explain the answer in plain English."
    ),
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    instruction=QA_SYSTEM_PROMPT,
    tools=[_mcp_toolset],
)
