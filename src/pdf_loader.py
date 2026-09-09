"""Utilities for extracting text and page metadata from PDF files."""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class ExtractedPage:
    """Text extracted from one PDF page."""

    page_number: int
    text: str


def extract_pages(pdf_path: str | Path) -> list[ExtractedPage]:
    """Extract non-empty page text from a PDF.

    Page numbers are one-based so they match the numbers shown to users in
    document citations.
    """

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.read_bytes()[:5] != b"%PDF-":
        raise ValueError(f"Could not open file as a PDF: {path.name}")
    pages: list[ExtractedPage] = []
    try:
        document = fitz.open(path)
    except Exception as error:
        raise ValueError(f"Could not open file as a PDF: {path.name}") from error

    with document:
        for index, page in enumerate(document):
            text = page.get_text("text").strip()
            if text:
                pages.append(ExtractedPage(page_number=index + 1, text=text))

    return pages
