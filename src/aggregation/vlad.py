# src/aggregation/vlad.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


@dataclass
class VladParams:
    centers: np.ndarray  # [K, D] cluster centers


def fit_kmeans_for_vlad(
    embeddings: np.ndarray,
    num_clusters: int = 32,
    max_iter: int = 100,
    random_state: int = 42,
) -> VladParams:
    """
    Fit KMeans to patch-level embeddings to obtain VLAD centers.

    embeddings: [N, D] array (ideally from many random patches).
    """
    kmeans = KMeans(
        n_clusters=num_clusters,
        max_iter=max_iter,
        random_state=random_state,
        n_init='auto',
    )
    kmeans.fit(embeddings)
    centers = kmeans.cluster_centers_.astype(np.float32)
    # L2-normalize centers (common practice)
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    return VladParams(centers=centers)


def vlad_aggregate_torch(
    emb: torch.Tensor,
    vlad_params: VladParams,
) -> torch.Tensor:
    """
    VLAD aggregation in PyTorch.

    emb: [N, D] patch embeddings (L2-normalized).
    vlad_params.centers: [K, D] numpy centers (L2-normalized).

    Returns:
        desc: [K*D] VLAD descriptor (L2-normalized).
    """
    if emb.ndim != 2:
        raise ValueError(f'emb must be [N, D], got {emb.shape}')

    device = emb.device
    centers = torch.from_numpy(vlad_params.centers).to(device)  # [K, D]

    N, D = emb.shape
    K = centers.shape[0]

    # Compute distances to centers (squared Euclidean)
    # dist[i, k] = || emb[i] - centers[k] ||^2
    # => use (x^2 + c^2 - 2x.c)
    emb_sq = (emb ** 2).sum(dim=1, keepdim=True)        # [N, 1]
    ctr_sq = (centers ** 2).sum(dim=1, keepdim=True).t()  # [1, K]
    cross = emb @ centers.t()                           # [N, K]
    dist = emb_sq + ctr_sq - 2 * cross                  # [N, K]

    # Hard assignment: nearest center
    assign = torch.argmin(dist, dim=1)  # [N]

    # Compute residuals per center
    # v_k = sum_{i: a_i=k} (emb_i - c_k)
    vlad = torch.zeros((K, D), device=device)
    for k in range(K):
        mask = (assign == k)
        if mask.any():
            diff = emb[mask] - centers[k]    # [Nk, D]
            vlad[k] = diff.sum(dim=0)

    # Intra-normalization (normalize each cluster vector)
    vlad = F.normalize(vlad, p=2, dim=1)  # [K, D]
    # Flatten and L2-normalize
    vlad = vlad.view(-1)                  # [K*D]
    vlad = F.normalize(vlad, p=2, dim=0)  # [K*D]
    return vlad
