"""
LexAI — MCP Tool: parse_document
Extracts full text from an uploaded PDF using PyMuPDF (fitz).

Input:  file_path (str) — path to the uploaded PDF
Output: dict {
    text       (str): full extracted text
    page_count (int): number of pages
    filename   (str): original filename
}
"""

import os
import fitz  # PyMuPDF


def parse_document(file_path: str) -> dict:
    """
    Extract full text from a PDF file.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        Dict with keys: text, page_count, filename.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Expected a PDF file, got: {os.path.basename(file_path)}")

    doc = fitz.open(file_path)
    pages_text: list[str] = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pages_text.append(page.get_text("text"))

    doc.close()

    full_text = "\n".join(pages_text).strip()

    return {
        "text": full_text,
        "page_count": len(pages_text),
        "filename": os.path.basename(file_path),
    }
