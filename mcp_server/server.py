"""
LexAI — MCP Server
Exposes three document-intelligence tools to ADK agents via the
Model Context Protocol (stdio transport).

Tools:
  • parse_document(file_path)      → {text, page_count, filename}
  • extract_clauses(text)          → [{heading, content, position}]
  • search_clause(text, query)     → {matched_clause, heading, relevance_score}

Security:
  • API-key gating: every tool call validates the MCP_API_KEY env var.
  • No document text is written to disk — all processing is in-memory.
  • Input is validated via Pydantic models before reaching tool logic.

Transport: stdio (subprocess) — compatible with Google ADK's MCPToolset.
"""

import os
import logging

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator

from .tools.parse_document import parse_document as _parse_document
from .tools.extract_clauses import extract_clauses as _extract_clauses
from .tools.search_clause import search_clause as _search_clause

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LexAI-MCP] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API-key security helper
# ---------------------------------------------------------------------------

_EXPECTED_KEY: str | None = os.environ.get("MCP_API_KEY")


def _require_api_key(provided_key: str | None) -> None:
    """
    Raise PermissionError if the provided key does not match MCP_API_KEY.
    Skips the check when MCP_API_KEY is not set (dev mode).
    """
    if _EXPECTED_KEY and provided_key != _EXPECTED_KEY:
        logger.warning("Rejected tool call: invalid or missing API key.")
        raise PermissionError(
            "Invalid or missing api_key. "
            "Set MCP_API_KEY in your environment and pass it as api_key."
        )


# ---------------------------------------------------------------------------
# Pydantic input models — validation & sanitization
# ---------------------------------------------------------------------------

class ParseDocumentInput(BaseModel):
    file_path: str = Field(..., min_length=1, max_length=512)
    api_key: str | None = Field(default=None)

    @field_validator("file_path")
    @classmethod
    def no_path_traversal(cls, v: str) -> str:
        """Block path-traversal attempts."""
        if ".." in v:
            raise ValueError("Path traversal ('..') is not allowed in file_path.")
        return v


class ExtractClausesInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2_000_000)  # 2 MB cap
    api_key: str | None = Field(default=None)


class SearchClauseInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2_000_000)
    query: str = Field(..., min_length=1, max_length=1_000)
    api_key: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="lexai-document-tools",
    description=(
        "LexAI document intelligence tools: parse PDFs, extract clauses, "
        "and search for relevant clauses by natural language query."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: parse_document
# ---------------------------------------------------------------------------

@mcp.tool()
def parse_document(file_path: str, api_key: str | None = None) -> dict:
    """
    Extract the full text content from a PDF file.

    Args:
        file_path: Absolute path to the uploaded PDF on the server's filesystem.
        api_key:   Optional MCP API key for authentication.

    Returns:
        {
          "text":       (str) full extracted text,
          "page_count": (int) number of pages in the PDF,
          "filename":   (str) base name of the file
        }
    """
    _require_api_key(api_key)

    # Validate input
    validated = ParseDocumentInput(file_path=file_path, api_key=api_key)
    logger.info("parse_document called: %s", validated.file_path)

    result = _parse_document(validated.file_path)

    logger.info(
        "parse_document OK — %d pages, %d chars",
        result["page_count"],
        len(result["text"]),
    )
    return result


# ---------------------------------------------------------------------------
# Tool 2: extract_clauses
# ---------------------------------------------------------------------------

@mcp.tool()
def extract_clauses(text: str, api_key: str | None = None) -> list[dict]:
    """
    Parse full document text into a structured list of clauses/sections.

    Args:
        text:    Full document text (output of parse_document).
        api_key: Optional MCP API key for authentication.

    Returns:
        List of dicts, each with:
          "heading"  (str): section/clause title,
          "content"  (str): clause body text,
          "position" (int): order in document (0-indexed)
    """
    _require_api_key(api_key)

    validated = ExtractClausesInput(text=text, api_key=api_key)
    logger.info("extract_clauses called — input length: %d chars", len(validated.text))

    clauses = _extract_clauses(validated.text)

    logger.info("extract_clauses OK — %d clauses found", len(clauses))
    return clauses


# ---------------------------------------------------------------------------
# Tool 3: search_clause
# ---------------------------------------------------------------------------

@mcp.tool()
def search_clause(text: str, query: str, api_key: str | None = None) -> dict:
    """
    Find the most relevant clause in the document for a given user query.

    Args:
        text:    Full document text (output of parse_document).
        query:   Natural language question or search term from the user.
        api_key: Optional MCP API key for authentication.

    Returns:
        {
          "matched_clause":  (str)   most relevant clause text,
          "heading":         (str)   section it came from,
          "relevance_score": (float) 0.0–1.0 confidence score
        }
    """
    _require_api_key(api_key)

    validated = SearchClauseInput(text=text, query=query, api_key=api_key)
    logger.info(
        "search_clause called — query: '%s' (doc: %d chars)",
        validated.query,
        len(validated.text),
    )

    result = _search_clause(validated.text, validated.query)

    logger.info(
        "search_clause OK — matched: '%s' (score: %.4f)",
        result["heading"],
        result["relevance_score"],
    )
    return result


# ---------------------------------------------------------------------------
# Entry point — run via stdio for ADK MCPToolset compatibility
# ---------------------------------------------------------------------------

def run():
    """Start the MCP server (stdio transport)."""
    logger.info(
        "Starting LexAI MCP server (API key gating: %s)",
        "ENABLED" if _EXPECTED_KEY else "DISABLED — dev mode",
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
