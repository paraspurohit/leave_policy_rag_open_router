from pathlib import Path

import pytest

from src.pdf_loader import extract_pages


def test_extract_pages_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_pages(tmp_path / "missing.pdf")


def test_extract_pages_rejects_non_pdf(tmp_path: Path) -> None:
    text_file = tmp_path / "document.txt"
    text_file.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_pages(text_file)
