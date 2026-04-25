"""
PDF loading and text chunking utilities.
"""

import logging
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# PDF magic bytes: %PDF
PDF_MAGIC_BYTES: bytes = b"%PDF"


def validate_pdf_magic(file_path: str) -> bool:
    """Check that the file starts with the PDF magic bytes (%PDF)."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
        return header == PDF_MAGIC_BYTES
    except OSError:
        return False


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file, page by page."""
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text: str = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(f"[page {i + 1}]\n{text}")
    return "\n\n".join(pages).strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping chunks."""
    chunks: List[str] = []
    start: int = 0
    while start < len(text):
        end: int = start + chunk_size
        chunk: str = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)
        if start >= len(text):
            break
    return chunks


def load_pdfs(data_dir: str = DATA_DIR) -> List[Dict[str, object]]:
    """
    Load all PDFs from data_dir, extract text, and chunk them.
    Returns list of dicts: {'text': str, 'source': str, 'chunk_id': int}
    """
    docs: List[Dict[str, object]] = []
    pdf_files = sorted(Path(data_dir).glob("*.pdf"))

    if not pdf_files:
        logger.info("No PDFs found in %s", data_dir)
        return docs

    for pdf_path in pdf_files:
        # Validate magic bytes
        if not validate_pdf_magic(str(pdf_path)):
            logger.warning(
                "Skipping %s (not a valid PDF — missing %%PDF header)",
                pdf_path.name,
            )
            continue

        full_text: str = extract_text_from_pdf(str(pdf_path))
        if not full_text:
            logger.warning("Skipping %s (no extractable text)", pdf_path.name)
            continue

        chunks: List[str] = chunk_text(full_text)
        for i, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "source": pdf_path.name,
                "chunk_id": i,
            })
        logger.info("  %s: %d chunks", pdf_path.name, len(chunks))

    logger.info("Total: %d chunks from %d PDF(s)", len(docs), len(pdf_files))
    return docs
