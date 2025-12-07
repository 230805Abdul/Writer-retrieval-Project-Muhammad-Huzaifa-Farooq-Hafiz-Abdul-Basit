# src/reranking/qe.py
import numpy as np


def apply_qe(descs: np.ndarray, top_k: int = 5) -> np.ndarray:
    """
    Simple query expansion in descriptor space.

    For each descriptor i:
      - find its top_k nearest neighbours (cosine similarity)
      - replace desc[i] with the mean of {i} ∪ neighbours, L2 normalized
    """
    descs = np.asarray(descs, dtype=np.float32)
    sim = descs @ descs.T  # [N,N], assumes L2-normalized descs
    N, D = descs.shape

    new_descs = np.empty_like(descs)
    for i in range(N):
        idx = np.argsort(-sim[i])[: top_k + 1]  # include self
        vec = descs[idx].mean(axis=0)
        vec /= (np.linalg.norm(vec) + 1e-12)
        new_descs[i] = vec
    return new_descs
