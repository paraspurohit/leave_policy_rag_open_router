from src.chunker import chunk_pages
from src.pdf_loader import ExtractedPage


def test_chunk_pages_preserves_page_metadata() -> None:
    pages = [ExtractedPage(page_number=3, text="First paragraph.\n\nSecond paragraph.")]
    chunks = chunk_pages(pages, max_characters=100, overlap_characters=20)

    assert len(chunks) == 1
    assert chunks[0].page_start == 3
    assert chunks[0].page_end == 3
    assert "First paragraph" in chunks[0].text


def test_chunk_pages_rejects_invalid_overlap() -> None:
    pages = [ExtractedPage(page_number=1, text="Some text")]

    try:
        chunk_pages(pages, max_characters=10, overlap_characters=10)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")
