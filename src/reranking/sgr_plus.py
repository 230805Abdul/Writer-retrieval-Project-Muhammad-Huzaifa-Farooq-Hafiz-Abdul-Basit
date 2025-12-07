# src/reranking/sgr_plus.py
"""
SGR++ (SGR Plus Plus): Enhanced reranking combining QE and SGR.

Reference: "Evaluating Feature Aggregation Methods for Deep Learning based 
           Writer Retrieval" (IJDAR 2024)

SGR++ combines:
1. Query Expansion (QE): Enrich each descriptor with top-k nearest neighbors
2. Similarity Graph Reranking (SGR): Propagate through weighted k-NN graph
3. Interpolation: Blend with original descriptors for stability

This typically provides 1-3% mAP improvement over SGR alone.
"""
import numpy as np
from .qe import apply_qe
from .sgr import apply_sgr


def apply_sgr_plus(descs: np.ndarray,
                   qe_top_k: int = 5,
                   sgr_k: int = 2,
                   sgr_gamma: float = 0.4,
                   alpha: float = 0.3,
                   use_qe: bool = True) -> np.ndarray:
    """
    SGR++ style reranking: QE followed by SGR and interpolation with original.

    Pipeline:
    1. Optional QE: average each descriptor with its top-k neighbors
    2. SGR: propagate through weighted k-NN similarity graph
    3. Interpolation: blend with original for stability
    
    Default parameters tuned to match icdar23 reference:
    - sgr_k=2: neighbors for SGR (reference uses 2)
    - sgr_gamma=0.4: SGR kernel bandwidth (reference uses 0.4)
    - alpha=0.3: blend 30% original, 70% reranked
    
    Args:
        descs: [N, D] L2-normalized descriptors
        qe_top_k: number of neighbors for query expansion (0 to disable)
        sgr_k: number of neighbors for SGR graph construction
        sgr_gamma: kernel bandwidth for SGR (smaller = sharper)
        alpha: interpolation weight (0 = pure reranked, 1 = pure original)
               Recommended: 0.3-0.5
        use_qe: whether to apply QE before SGR
        
    Returns:
        [N, D] reranked descriptors (L2-normalized)
    """
    descs = np.asarray(descs, dtype=np.float32)
    original = descs.copy()
    
    # Step 1: Query Expansion (optional)
    if use_qe and qe_top_k > 0:
        descs = apply_qe(descs, top_k=qe_top_k)
    
    # Step 2: SGR with proper weighted adjacency
    descs_sgr = apply_sgr(descs, k=sgr_k, gamma=sgr_gamma)
    
    # Step 3: Interpolation with original
    # alpha=0: pure SGR, alpha=1: pure original
    if alpha > 0:
        mixed = alpha * original + (1.0 - alpha) * descs_sgr
    else:
        mixed = descs_sgr
    
    # L2 normalize
    norms = np.linalg.norm(mixed, axis=1, keepdims=True) + 1e-12
    return mixed / norms


def apply_sgr_plus_iterative(descs: np.ndarray,
                             qe_top_k: int = 5,
                             sgr_k: int = 20,
                             sgr_gamma: float = 0.1,
                             num_iterations: int = 2,
                             alpha: float = 0.5) -> np.ndarray:
    """
    Iterative SGR++ for stronger smoothing.
    
    Applies SGR++ multiple times, which can help with noisy descriptors
    but may over-smooth if iterations are too high.
    
    Args:
        descs: [N, D] L2-normalized descriptors
        qe_top_k: neighbors for QE (only applied in first iteration)
        sgr_k: neighbors for SGR graph
        sgr_gamma: SGR kernel bandwidth
        num_iterations: number of SGR++ iterations (typically 1-3)
        alpha: interpolation weight per iteration
        
    Returns:
        [N, D] reranked descriptors (L2-normalized)
    """
    descs = np.asarray(descs, dtype=np.float32)
    
    for i in range(num_iterations):
        # Only apply QE in first iteration
        use_qe = (i == 0 and qe_top_k > 0)
        descs = apply_sgr_plus(
            descs,
            qe_top_k=qe_top_k if use_qe else 0,
            sgr_k=sgr_k,
            sgr_gamma=sgr_gamma,
            alpha=alpha,
            use_qe=use_qe
        )
    
    return descs
