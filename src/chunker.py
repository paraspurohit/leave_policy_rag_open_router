"""Split extracted PDF text into searchable chunks."""

from dataclasses import dataclass

from src.pdf_loader import ExtractedPage


@dataclass(frozen=True)
class TextChunk:
    """A searchable piece of a document with citation metadata."""

    chunk_id: int
    text: str
    page_start: int
    page_end: int


def _paragraphs(text: str) -> list[str]:
    """Return cleaned, non-empty paragraphs from page text."""

    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def chunk_pages(
    pages: list[ExtractedPage],
    max_characters: int = 1_200,
    overlap_characters: int = 200,
) -> list[TextChunk]:
    """Create chunks from extracted pages while keeping page references.

    Paragraphs are kept together where possible. If one paragraph is larger
    than ``max_characters``, it is split into smaller character-based pieces.
    """

    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero")
    if overlap_characters < 0 or overlap_characters >= max_characters:
        raise ValueError("overlap_characters must be between zero and max_characters")

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_pages: list[int] = []

    def add_chunk() -> None:
        if not current_parts:
            return
        chunks.append(
            TextChunk(
                chunk_id=len(chunks),
                text="\n\n".join(current_parts).strip(),
                page_start=min(current_pages),
                page_end=max(current_pages),
            )
        )

    for page in pages:
        for paragraph in _paragraphs(page.text):
            pieces = [
                paragraph[start : start + max_characters]
                for start in range(0, len(paragraph), max_characters - overlap_characters)
            ]
            for piece in pieces:
                proposed = "\n\n".join(current_parts + [piece])
                if current_parts and len(proposed) > max_characters:
                    add_chunk()
                    current_parts.clear()
                    current_pages.clear()
                current_parts.append(piece)
                current_pages.append(page.page_number)

    add_chunk()
    return chunks
