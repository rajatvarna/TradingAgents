"""User-uploaded research notes: extract → summarize → persist → list/delete.

Per-user storage lives under <user_home>/research/<TICKER>/ for the long-lived
library, and <user_home>/research/_shared_for_run/<run_id>/ for one-shot uploads.

This module starts with the extraction half (PDF / .md / .txt). Storage and
the LLM-summary ingest pipeline land in subsequent tasks.
"""

from __future__ import annotations

import io
import os


SUPPORTED_TEXT_EXT = {".txt", ".md", ".markdown"}
SUPPORTED_PDF_EXT = {".pdf"}
MAX_SUMMARY_INPUT_CHARS = 100_000


class ResearchExtractionError(Exception):
    """Raised when a file cannot be parsed."""


def _sniff_ext(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return ext


def _extract_text(file_bytes: bytes, filename: str) -> str:
    ext = _sniff_ext(filename)
    if ext in SUPPORTED_TEXT_EXT:
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            raise ResearchExtractionError(f"text decode failed: {e}") from e
    if ext in SUPPORTED_PDF_EXT:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            chunks = []
            for page in reader.pages:
                chunks.append(page.extract_text() or "")
            text = "\n".join(chunks).strip()
            if not text:
                raise ResearchExtractionError("PDF contained no extractable text")
            return text
        except ResearchExtractionError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ResearchExtractionError(f"PDF parse failed: {e}") from e
    raise ResearchExtractionError(f"Unsupported file type: {ext}")
