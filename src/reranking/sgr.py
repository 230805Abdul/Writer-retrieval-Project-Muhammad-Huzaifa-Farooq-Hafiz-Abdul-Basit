# src/reranking/sgr.py
"""
Similarity Graph Reranking (SGR) for writer retrieval.

Reference: "Evaluating Feature Aggregation Methods for Deep Learning based 
           Writer Retrieval" (IJDAR 2024)

The key insight is that similar documents should have similar neighborhoods.
SGR propagates similarities through a weighted k-NN graph to produce
smoothed descriptors.

Algorithm:
1. Build k-NN graph from cosine similarities
2. Construct weighted adjacency matrix: A[i,j] = exp(-(1-sim)^2 / gamma)
3. Propagate: new_desc = A @ desc (row-normalized)
4. L2 normalize result
"""
import numpy as np
from typing import Optional
import logging
import time

logger = logging.getLogger(__name__)


def build_knn_graph(sim: np.ndarray, k: int) -> np.ndarray:
    """
    Build a k-NN adjacency matrix from similarity matrix.
    
    Args:
        sim: [N, N] cosine similarity matrix
        k: number of nearest neighbors
        
    Returns:
        adj: [N, N] binary adjacency matrix (1 if j in kNN of i)
    """
    N = sim.shape[0]
    # Get indices of k largest similarities per row (excluding self)
    # Set diagonal to -inf so self isn't selected
    sim_no_self = sim.copy()
    np.fill_diagonal(sim_no_self, -np.inf)
    
    topk_indices = np.argpartition(-sim_no_self, k, axis=1)[:, :k]
    
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        adj[i, topk_indices[i]] = 1.0
    
    return adj


def build_weighted_adjacency(sim: np.ndarray, adj: np.ndarray, 
                              gamma: float = 0.1) -> np.ndarray:
    """
    Build weighted adjacency matrix using exponential kernel.
    
    A[i,j] = exp(-(1 - sim[i,j])^2 / gamma) if j is neighbor of i, else 0
    
    Args:
        sim: [N, N] cosine similarity matrix
        adj: [N, N] binary k-NN adjacency matrix
        gamma: kernel bandwidth (smaller = sharper falloff)
        
    Returns:
        W: [N, N] weighted adjacency matrix
    """
    # Convert similarity to distance: d = 1 - sim (range [0, 2] for cosine)
    dist_sq = (1.0 - sim) ** 2
    
    # Apply exponential kernel
    W = np.exp(-dist_sq / gamma) * adj
    
    return W


def row_normalize(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize a matrix so each row sums to 1."""
    row_sums = W.sum(axis=1, keepdims=True)
    return W / (row_sums + eps)


def apply_sgr(descs: np.ndarray, 
              k: int = 2, 
              gamma: float = 0.4,
              num_iterations: int = 2,
              alpha: float = 0.0,
              verbose: bool = True,
              debug: bool = False) -> np.ndarray:
    """
    Apply Similarity Graph Reranking to GLOBAL DESCRIPTORS.
    
    SGR propagates descriptors through a weighted k-NN similarity graph,
    effectively smoothing each descriptor with its neighbors weighted by similarity.
    
    IMPORTANT: This operates on GLOBAL descriptors (one per page), not patches!
    
    Default parameters (FIXED from original k=20 which caused over-smoothing):
    - k=2-5: number of nearest neighbors (k=20 was too high!)
    - gamma=0.1-0.4: kernel bandwidth (smaller = sharper kernel)
    - num_iterations=2: propagation iterations
    
    Args:
        descs: [N, D] L2-normalized GLOBAL descriptors (one per page)
        k: number of nearest neighbors for graph construction (2-5 recommended)
        gamma: kernel bandwidth for exponential weighting (0.1-0.4)
        num_iterations: number of propagation iterations (usually 1-2)
        alpha: interpolation with original (0 = pure SGR, 1 = original only)
        verbose: whether to log progress
        debug: enable detailed debug logging
        
    Returns:
        new_descs: [N, D] reranked descriptors (L2-normalized)
    """
    descs = np.asarray(descs, dtype=np.float32)
    N, D = descs.shape
    
    if verbose:
        logger.info('🔀 Applying Similarity Graph Reranking (SGR)...')
        logger.info(f'   Descriptors: {N} × {D}')
        logger.info(f'   k-neighbors: {k}')
        logger.info(f'   γ (gamma): {gamma}')
        logger.info(f'   Iterations: {num_iterations}')
    
    if debug:
        logger.debug(f'[DEBUG] SGR input: shape={descs.shape}, '
                    f'norm_mean={np.linalg.norm(descs, axis=1).mean():.4f}')
        logger.debug(f'[DEBUG] SGR params: k={k}, gamma={gamma}, iters={num_iterations}, alpha={alpha}')
    
    if N <= k:
        # Not enough samples for k-NN
        if verbose:
            logger.warning(f'   ⚠️  Not enough samples ({N}) for k={k}, returning original')
        return descs.copy()
    
    start_time = time.time()
    
    # Step 1: Compute pairwise cosine similarities
    if verbose:
        logger.info('   Step 1/4: Computing similarity matrix...')
    sim = descs @ descs.T  # [N, N]
    
    if debug:
        logger.debug(f'[DEBUG] Similarity matrix: shape={sim.shape}, '
                    f'mean={sim.mean():.4f}, max={sim.max():.4f}')
    
    # Step 2: Build k-NN graph
    if verbose:
        logger.info('   Step 2/4: Building k-NN graph...')
    adj = build_knn_graph(sim, k=k)
    
    if debug:
        edges = adj.sum()
        logger.debug(f'[DEBUG] k-NN graph: {int(edges)} edges (avg {edges/N:.1f} per node)')
    
    # Step 3: Build weighted adjacency with exponential kernel
    if verbose:
        logger.info('   Step 3/4: Computing weighted adjacency...')
    W = build_weighted_adjacency(sim, adj, gamma=gamma)
    
    if debug:
        logger.debug(f'[DEBUG] Weighted adjacency: non-zero={np.count_nonzero(W)}, '
                    f'mean_weight={W[W > 0].mean():.4f}')
    
    # Add self-connections (important for stability)
    np.fill_diagonal(W, 1.0)
    
    # Step 4: Row-normalize to get transition matrix
    P = row_normalize(W)
    
    # Step 5: Propagate descriptors
    if verbose:
        logger.info(f'   Step 4/4: Propagating ({num_iterations} iterations)...')
    new_descs = descs.copy()
    for i in range(num_iterations):
        new_descs = P @ new_descs
        if debug:
            change = np.linalg.norm(new_descs - descs) / np.linalg.norm(descs)
            logger.debug(f'[DEBUG] Iteration {i+1}: relative_change={change:.4f}')
    
    # Step 6: Optional interpolation with original
    if alpha > 0:
        new_descs = alpha * descs + (1.0 - alpha) * new_descs
    
    # Step 7: L2 normalize
    norms = np.linalg.norm(new_descs, axis=1, keepdims=True) + 1e-12
    new_descs = new_descs / norms
    
    elapsed = time.time() - start_time
    if verbose:
        logger.info(f'   ✓ SGR complete in {elapsed:.2f}s')
    
    if debug:
        final_change = np.linalg.norm(new_descs - descs / np.linalg.norm(descs, axis=1, keepdims=True))
        logger.debug(f'[DEBUG] SGR final: total_change={final_change:.4f}')
    
    return new_descs


def apply_sgr_symmetric(descs: np.ndarray,
                        k: int = 20,
                        gamma: float = 0.1) -> np.ndarray:
    """
    Symmetric variant of SGR using k-reciprocal neighbors.
    
    Only keeps edges where both i->j and j->i are in each other's k-NN.
    This is more conservative but can be more robust.
    """
    descs = np.asarray(descs, dtype=np.float32)
    N, D = descs.shape
    
    if N <= k:
        return descs.copy()
    
    sim = descs @ descs.T
    
    # Build k-NN adjacency
    adj = build_knn_graph(sim, k=k)
    
    # Make symmetric (k-reciprocal): keep edge only if mutual
    adj_sym = adj * adj.T
    
    # Build weighted adjacency
    W = build_weighted_adjacency(sim, adj_sym, gamma=gamma)
    np.fill_diagonal(W, 1.0)
    
    # Row-normalize and propagate
    P = row_normalize(W)
    new_descs = P @ descs
    
    # L2 normalize
    norms = np.linalg.norm(new_descs, axis=1, keepdims=True) + 1e-12
    new_descs = new_descs / norms
    
    return new_descs
