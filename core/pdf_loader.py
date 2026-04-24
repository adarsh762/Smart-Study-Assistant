"""
PDF loading and text chunking utilities.
"""

import os
from pathlib import Path
from typing import List

from pypdf import PdfReader

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file, page by page."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(f"[page {i + 1}]\n{text}")
    return "\n\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)
        if start >= len(text):
            break
    return chunks


def load_pdfs(data_dir: str = DATA_DIR) -> List[dict]:
    """
    Load all PDFs from data_dir, extract text, and chunk them.
    Returns list of dicts: {'text': str, 'source': str, 'chunk_id': int}
    """
    docs = []
    pdf_files = sorted(Path(data_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {data_dir}")
        return docs

    for pdf_path in pdf_files:
        full_text = extract_text_from_pdf(str(pdf_path))
        if not full_text:
            print(f"  Skipping {pdf_path.name} (no extractable text)")
            continue

        chunks = chunk_text(full_text)
        for i, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "source": pdf_path.name,
                "chunk_id": i,
            })
        print(f"  {pdf_path.name}: {len(chunks)} chunks")

    print(f"Total: {len(docs)} chunks from {len(pdf_files)} PDF(s)")
    return docs
