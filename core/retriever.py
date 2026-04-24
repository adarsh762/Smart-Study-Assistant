"""
Retrieval module: query embedding + nearest neighbor search.
"""

from typing import List

import numpy as np
from sklearn.neighbors import NearestNeighbors

from config import TOP_K
from core.embeddings import get_model


def retrieve(
    query: str,
    embeddings: np.ndarray,
    docs: List[dict],
    nn: NearestNeighbors,
    top_k: int = TOP_K,
) -> List[dict]:
    """
    Retrieve the top-k most relevant document chunks for a query.

    Returns list of dicts with keys: text, source, chunk_id, score
    """
    model = get_model()
    q_vec = model.encode([query], convert_to_numpy=True)

    k = min(top_k, len(docs))
    distances, indices = nn.kneighbors(q_vec, n_neighbors=k, return_distance=True)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc = docs[int(idx)]
        results.append({
            "text": doc["text"],
            "source": doc.get("source", "unknown"),
            "chunk_id": doc.get("chunk_id", 0),
            "score": round(float(1 - dist), 4),
        })

    return results
