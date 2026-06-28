"""
LexAI — MCP Tool: extract_clauses
Parses full document text into a structured list of clauses/sections.

Input:  text (str) — full document text
Output: list of dicts, each with {
    heading  (str): clause/section title
    content  (str): clause text body
    position (int): order in document (0-indexed)
}
"""

import re


# ---------------------------------------------------------------------------
# Heading detection patterns (ordered by specificity — most specific first)
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    # "1.", "1.1", "1.1.1" — numbered sections
    re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[\.\)]\s+([A-Z][^\n]{0,80})", re.MULTILINE),
    # Roman numerals: "I.", "II.", "III."
    re.compile(
        r"^\s*((?:X{0,3})(?:IX|IV|V?I{0,3}))\.\s+([A-Z][^\n]{0,80})",
        re.MULTILINE | re.IGNORECASE,
    ),
    # ALL CAPS heading lines (at least 3 words or ≥10 chars)
    re.compile(r"^\s*([A-Z][A-Z\s\-\/]{9,79})\s*$", re.MULTILINE),
    # Title-case heading followed by a colon
    re.compile(r"^\s*([A-Z][A-Za-z\s]{4,60})\s*:\s*$", re.MULTILINE),
]


def _find_headings(text: str) -> list[tuple[int, str]]:
    """
    Return a sorted list of (char_offset, heading_text) for all detected
    section headings in *text*.
    """
    hits: dict[int, str] = {}  # offset → heading text (dedup by offset)

    for pattern in _HEADING_PATTERNS:
        for match in pattern.finditer(text):
            offset = match.start()
            # Use the full match as the heading label (strip whitespace)
            heading = match.group(0).strip()
            # Keep only the first hit at each offset
            if offset not in hits:
                hits[offset] = heading

    return sorted(hits.items())


def extract_clauses(text: str) -> list[dict]:
    """
    Split *text* into a structured list of clauses by detecting headings.

    Falls back to fixed-size paragraph chunking when no headings are found.

    Args:
        text: Full document text returned by parse_document.

    Returns:
        List of dicts with keys: heading, content, position.
    """
    if not text or not text.strip():
        return []

    headings = _find_headings(text)

    # -----------------------------------------------------------------
    # Fallback: no headings detected → split into 500-word paragraphs
    # -----------------------------------------------------------------
    if not headings:
        return _chunk_by_paragraphs(text)

    clauses: list[dict] = []

    for idx, (start_offset, heading) in enumerate(headings):
        # Content spans from end-of-current-heading to start-of-next-heading
        content_start = start_offset + len(heading)
        content_end = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        content = text[content_start:content_end].strip()

        if content or heading:  # skip fully empty slices
            clauses.append(
                {
                    "heading": heading,
                    "content": content,
                    "position": idx,
                }
            )

    return clauses


# ---------------------------------------------------------------------------
# Fallback chunker
# ---------------------------------------------------------------------------

def _chunk_by_paragraphs(text: str, words_per_chunk: int = 300) -> list[dict]:
    """
    Split text into chunks of roughly *words_per_chunk* words when no
    structural headings can be detected.
    """
    # Split on blank lines first for natural paragraph breaks
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    clauses: list[dict] = []
    buffer: list[str] = []
    word_count = 0
    chunk_index = 0

    for para in paragraphs:
        buffer.append(para)
        word_count += len(para.split())

        if word_count >= words_per_chunk:
            clauses.append(
                {
                    "heading": f"Section {chunk_index + 1}",
                    "content": "\n\n".join(buffer),
                    "position": chunk_index,
                }
            )
            buffer = []
            word_count = 0
            chunk_index += 1

    # Flush any remaining text
    if buffer:
        clauses.append(
            {
                "heading": f"Section {chunk_index + 1}",
                "content": "\n\n".join(buffer),
                "position": chunk_index,
            }
        )

    return clauses
