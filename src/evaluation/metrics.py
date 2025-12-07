# src/evaluation/metrics.py
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict
import logging
import time

logger = logging.getLogger(__name__)


def average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Average Precision for a single query.

    y_true: [N] binary labels (1 = relevant, 0 = not)
    y_scores: [N] similarity scores (higher = more similar)
    """
    # sort by descending score
    order = np.argsort(-y_scores)
    y_true_sorted = y_true[order]

    cumsum = np.cumsum(y_true_sorted)
    rel_indices = np.nonzero(y_true_sorted)[0]
    if len(rel_indices) == 0:
        return 0.0

    precisions = cumsum[rel_indices] / (rel_indices + 1)
    ap = precisions.mean()
    return float(ap)


def mean_average_precision(
    all_labels: List[int],
    all_descs: np.ndarray,
    verbose: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Compute mAP and Top-1/Top-5/Top-10 for writer retrieval.

    all_labels: list of writer_id for each descriptor
    all_descs: [M, D] descriptors (L2-normalized).
               Each descriptor is used both as query and in the gallery.

    For each query i:
        - remove i from gallery
        - positives: same writer_id
    """
    labels = np.asarray(all_labels)
    descs = np.asarray(all_descs)
    M, D = descs.shape
    
    if verbose:
        logger.info('📊 Computing retrieval metrics...')
        logger.info(f'   Queries: {M}')
        logger.info(f'   Descriptor dim: {D}')
        unique_writers = len(set(all_labels))
        logger.info(f'   Unique writers: {unique_writers}')
    
    start_time = time.time()

    # cosine similarity
    if verbose:
        logger.info('   Computing similarity matrix...')
    sim = descs @ descs.T  # [M, M], since L2-normalized

    aps = []
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0

    for i in range(M):
        # gallery indices excluding self
        mask = np.ones(M, dtype=bool)
        mask[i] = False

        gallery_sim = sim[i, mask]       # [M-1]
        gallery_labels = labels[mask]    # [M-1]

        y_true = (gallery_labels == labels[i]).astype(np.int32)

        if y_true.sum() == 0:
            # no positives for this writer in gallery (should not happen in typical WR)
            continue

        ap = average_precision(y_true, gallery_sim)
        aps.append(ap)

        # sort for top-k
        order = np.argsort(-gallery_sim)
        y_sorted = y_true[order]

        # Top-1
        if y_sorted[:1].sum() > 0:
            top1_hits += 1
        # Top-5
        if y_sorted[:5].sum() > 0:
            top5_hits += 1
        # Top-10
        if y_sorted[:10].sum() > 0:
            top10_hits += 1

    mAP = float(np.mean(aps)) if aps else 0.0
    n_queries = len(aps)
    metrics = {
        'mAP': mAP,
        'Top1': top1_hits / n_queries if n_queries > 0 else 0.0,
        'Top5': top5_hits / n_queries if n_queries > 0 else 0.0,
        'Top10': top10_hits / n_queries if n_queries > 0 else 0.0,
        'n_queries': n_queries,
    }
    
    elapsed = time.time() - start_time
    if verbose:
        logger.info(f'   ✓ Metrics computed in {elapsed:.1f}s')
        logger.info('')
        logger.info('   ╔════════════════════════════════╗')
        logger.info(f'   ║  mAP:    {mAP * 100:6.2f}%              ║')
        logger.info(f'   ║  Top-1:  {metrics["Top1"] * 100:6.2f}%              ║')
        logger.info(f'   ║  Top-5:  {metrics["Top5"] * 100:6.2f}%              ║')
        logger.info(f'   ║  Top-10: {metrics["Top10"] * 100:6.2f}%              ║')
        logger.info('   ╚════════════════════════════════╝')
    
    return mAP, metrics
