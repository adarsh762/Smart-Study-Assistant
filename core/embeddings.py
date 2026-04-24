"""
Embedding generation and index management using Sentence Transformers + scikit-learn.
"""

import os
import json
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from config import EMBED_MODEL_NAME, INDEX_DIR, N_NEIGHBORS

# ── Singleton model instance ──
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def build_index(docs: List[dict], persist_dir: str = INDEX_DIR) -> Tuple[np.ndarray, List[dict], NearestNeighbors]:
    """
    Build embeddings and NearestNeighbors index from document chunks.
    Persists embeddings and docs to disk.
    """
    model = get_model()
    texts = [d["text"] for d in docs]

    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    nn = NearestNeighbors(n_neighbors=min(N_NEIGHBORS, len(docs)), metric="cosine", algorithm="auto")
    nn.fit(embeddings)

    # Persist
    np.save(os.path.join(persist_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(persist_dir, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Index saved to {persist_dir} ({len(docs)} chunks)")
    return embeddings, docs, nn


def load_index(persist_dir: str = INDEX_DIR) -> Optional[Tuple[np.ndarray, List[dict], NearestNeighbors]]:
    """
    Load a previously persisted index from disk.
    Returns None if no index exists.
    """
    emb_path = os.path.join(persist_dir, "embeddings.npy")
    docs_path = os.path.join(persist_dir, "docs.json")

    if not (os.path.exists(emb_path) and os.path.exists(docs_path)):
        return None

    embeddings = np.load(emb_path)
    with open(docs_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    nn = NearestNeighbors(n_neighbors=min(N_NEIGHBORS, len(docs)), metric="cosine", algorithm="auto")
    nn.fit(embeddings)

    print(f"Loaded index from {persist_dir} ({len(docs)} chunks)")
    return embeddings, docs, nn
