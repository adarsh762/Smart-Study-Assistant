"""
Embedding generation and index management using Sentence Transformers + scikit-learn.
"""

import os
import json
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from config import EMBED_MODEL_NAME, INDEX_DIR, N_NEIGHBORS

logger = logging.getLogger(__name__)

# ── Singleton model instance ──
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def build_index(
    docs: List[Dict[str, object]],
    persist_dir: str = INDEX_DIR,
) -> Tuple[np.ndarray, List[Dict[str, object]], NearestNeighbors]:
    """
    Build embeddings and NearestNeighbors index from document chunks.
    Persists embeddings and docs to disk.
    """
    model = get_model()
    texts: List[str] = [d["text"] for d in docs]

    logger.info("Encoding %d chunks...", len(texts))
    embeddings: np.ndarray = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True
    )

    nn = NearestNeighbors(
        n_neighbors=min(N_NEIGHBORS, len(docs)),
        metric="cosine",
        algorithm="auto",
    )
    nn.fit(embeddings)

    # Persist
    np.save(os.path.join(persist_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(persist_dir, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    logger.info("Index saved to %s (%d chunks)", persist_dir, len(docs))
    return embeddings, docs, nn


def load_index(
    persist_dir: str = INDEX_DIR,
) -> Optional[Tuple[np.ndarray, List[Dict[str, object]], NearestNeighbors]]:
    """
    Load a previously persisted index from disk.
    Returns None if no index exists or if the index files are corrupted.
    """
    emb_path = os.path.join(persist_dir, "embeddings.npy")
    docs_path = os.path.join(persist_dir, "docs.json")

    if not (os.path.exists(emb_path) and os.path.exists(docs_path)):
        return None

    try:
        embeddings: np.ndarray = np.load(emb_path)
        with open(docs_path, "r", encoding="utf-8") as f:
            docs: List[Dict[str, object]] = json.load(f)

        if len(embeddings) != len(docs):
            logger.warning(
                "Index mismatch: %d embeddings vs %d docs — rebuilding required",
                len(embeddings), len(docs),
            )
            return None

        nn = NearestNeighbors(
            n_neighbors=min(N_NEIGHBORS, len(docs)),
            metric="cosine",
            algorithm="auto",
        )
        nn.fit(embeddings)

        logger.info("Loaded index from %s (%d chunks)", persist_dir, len(docs))
        return embeddings, docs, nn

    except (ValueError, json.JSONDecodeError, OSError) as e:
        logger.error(
            "Corrupt index files in %s: %s — will need to rebuild", persist_dir, e
        )
        return None
