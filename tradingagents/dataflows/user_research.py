"""User-uploaded research notes: extract → summarize → persist → list/delete.

Per-user storage lives under <user_home>/research/<TICKER>/ for the long-lived
library, and <user_home>/research/_shared_for_run/<run_id>/ for one-shot uploads.

This module starts with the extraction half (PDF / .md / .txt). Storage and
the LLM-summary ingest pipeline land in subsequent tasks.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SUPPORTED_TEXT_EXT = {".txt", ".md", ".markdown"}
SUPPORTED_PDF_EXT = {".pdf"}
MAX_SUMMARY_INPUT_CHARS = 100_000  # consumed by ingest_research in a later task


class ResearchExtractionError(Exception):
    """Raised when a file cannot be parsed."""


def _sniff_ext(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return ext


def _extract_text(file_bytes: bytes, filename: str) -> str:
    ext = _sniff_ext(filename)
    if ext in SUPPORTED_TEXT_EXT:
        # errors="replace" guarantees decode can't raise, so no try/except needed.
        return file_bytes.decode("utf-8", errors="replace")
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


def _safe_filename(name: str) -> str:
    """Strip path separators and control chars."""
    name = os.path.basename(name)
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:200] or "file"


def _research_root(user_root: Path) -> Path:
    p = user_root / "research"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(
    file_bytes: bytes,
    summary_md: str,
    ticker: Optional[str],
    user_root: Path,
    original_filename: str,
    run_id: Optional[str],
) -> dict:
    """Persist original + summary + meta. Returns metadata dict."""
    safe_orig = _safe_filename(original_filename)
    ext = _sniff_ext(safe_orig) or ".bin"
    digest = hashlib.sha256(file_bytes).hexdigest()[:12]

    if ticker:
        target_dir = _research_root(user_root) / ticker.upper()
    else:
        if not run_id:
            raise ValueError("run_id is required when ticker is None")
        target_dir = _research_root(user_root) / "_shared_for_run" / run_id
    target_dir.mkdir(parents=True, exist_ok=True)

    original_path = target_dir / f"{digest}{ext}"
    summary_path = target_dir / f"{digest}.summary.md"
    meta_path = target_dir / f"{digest}.meta.json"

    deduped = original_path.exists()
    if not deduped:
        original_path.write_bytes(file_bytes)
        summary_path.write_text(summary_md, encoding="utf-8")
        meta_path.write_text(
            json.dumps({
                "filename": original_filename,
                "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "size": len(file_bytes),
                "hash": digest,
                "ticker": ticker.upper() if ticker else None,
                "run_id": run_id,
            }, indent=2),
            encoding="utf-8",
        )
        persisted_filename = original_filename
        persisted_summary = summary_md
    else:
        # Same bytes uploaded again — keep the original filename/summary on disk
        # and reflect that in the return dict so the caller doesn't believe a
        # new summary was saved.
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            persisted_filename = existing_meta.get("filename", original_filename)
        except Exception:
            persisted_filename = original_filename
        try:
            persisted_summary = summary_path.read_text(encoding="utf-8")
        except Exception:
            persisted_summary = summary_md

    return {
        "path": str(original_path),
        "summary_path": str(summary_path),
        "meta_path": str(meta_path),
        "hash": digest,
        "filename": persisted_filename,
        "summary": persisted_summary,
        "deduped": deduped,
    }


def list_research(user_root: Path, ticker: str) -> list[dict]:
    """List per-ticker library entries. Returns [] if none."""
    d = _research_root(user_root) / ticker.upper()
    if not d.exists():
        return []
    out: list[dict] = []
    for meta_path in sorted(d.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        digest = meta["hash"]
        summary_path = d / f"{digest}.summary.md"
        if not summary_path.exists():
            continue
        meta["summary"] = summary_path.read_text(encoding="utf-8")
        out.append(meta)
    return out


_DIGEST_RE = re.compile(r"^[a-f0-9]{12}$")


def delete_research(user_root: Path, ticker: str, digest: str) -> None:
    """Delete the original + summary + meta for a single library entry.

    The digest must be the 12-hex-char SHA prefix produced by `_save`.
    Anything else is rejected (defense against caller-supplied paths).
    """
    if not _DIGEST_RE.match(digest):
        return
    d = _research_root(user_root) / ticker.upper()
    if not d.exists():
        return
    for f in d.glob(f"{digest}*"):
        try:
            f.unlink()
        except OSError:
            pass


def clear_run_dir(user_root: Path, run_id: str) -> None:
    """Remove a per-run directory entirely."""
    import shutil
    d = _research_root(user_root) / "_shared_for_run" / run_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
