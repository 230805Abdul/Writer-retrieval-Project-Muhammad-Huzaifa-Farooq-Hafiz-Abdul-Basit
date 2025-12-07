# src/aggregation/pooling.py
"""
Pooling and normalization layers for descriptor aggregation.

Reference: "Evaluating Feature Aggregation Methods for Deep Learning based 
           Writer Retrieval" (IJDAR 2024)

Key insight: Power normalization (signed square root) before L2 normalization
significantly improves retrieval performance by suppressing bursty features.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def power_normalize(x: torch.Tensor, alpha: float = 0.4, eps: float = 1e-12) -> torch.Tensor:
    """
    Apply signed power normalization: sign(x) * |x|^alpha
    
    This suppresses bursty features (large values) while preserving sign.
    Common values: alpha=0.5 (signed sqrt), alpha=0.4 (slightly stronger suppression)
    
    Args:
        x: input tensor
        alpha: power exponent (0 < alpha < 1 for suppression)
        eps: small constant for numerical stability
        
    Returns:
        Power-normalized tensor with same shape as input
    """
    return x.sign() * (x.abs() + eps).pow(alpha)


def mean_pool_aggregate(emb: torch.Tensor, 
                        use_power_norm: bool = True,
                        power_alpha: float = 0.4) -> torch.Tensor:
    """
    Mean pooling over patch embeddings with optional power normalization.

    Args:
        emb: [N, D] patch embeddings
        use_power_norm: whether to apply power normalization before L2 norm
        power_alpha: exponent for power normalization
        
    Returns:
        desc: [D] L2-normalized descriptor
    """
    if emb.ndim != 2:
        raise ValueError(f'emb must be [N, D], got {emb.shape}')

    if emb.shape[0] == 0:
        raise ValueError('No embeddings provided to mean_pool_aggregate')

    # Mean pooling
    desc = emb.mean(dim=0)
    
    # Power normalization (suppresses bursty features)
    if use_power_norm:
        desc = power_normalize(desc, alpha=power_alpha)
    
    # L2 normalization
    desc = F.normalize(desc, p=2, dim=0)
    return desc


def sum_pool_aggregate(emb: torch.Tensor,
                       use_power_norm: bool = True,
                       power_alpha: float = 0.4) -> torch.Tensor:
    """
    Sum pooling over patch embeddings with power normalization.
    
    This is the aggregation used in the reference implementation.
    Sum pooling followed by power-norm and L2-norm.

    Args:
        emb: [N, D] patch embeddings
        use_power_norm: whether to apply power normalization
        power_alpha: exponent for power normalization
        
    Returns:
        desc: [D] L2-normalized descriptor
    """
    if emb.ndim != 2:
        raise ValueError(f'emb must be [N, D], got {emb.shape}')

    if emb.shape[0] == 0:
        raise ValueError('No embeddings provided to sum_pool_aggregate')

    # Sum pooling
    desc = emb.sum(dim=0)
    
    # Power normalization
    if use_power_norm:
        desc = power_normalize(desc, alpha=power_alpha)
    
    # L2 normalization
    desc = F.normalize(desc, p=2, dim=0)
    return desc


def gem_pool_aggregate(emb: torch.Tensor, 
                       p: float = 3.0, 
                       eps: float = 1e-6,
                       use_power_norm: bool = True,
                       power_alpha: float = 0.4,
                       debug: bool = False) -> torch.Tensor:
    """
    GeM (Generalized Mean) pooling with sign-awareness and power normalization.

    Classic GeM assumes non-negative activations; our embeddings may be signed,
    so we use a signed variant: sign(x) * (mean(|x|^p))^(1/p).

    Args:
        emb: [N, D] patch embeddings
        p: GeM power parameter (higher = closer to max pooling)
        eps: small constant for numerical stability
        use_power_norm: whether to apply power normalization after GeM
        power_alpha: exponent for power normalization
        debug: enable debug logging
        
    Returns:
        desc: [D] L2-normalized descriptor
    """
    if emb.ndim != 2:
        raise ValueError(f'emb must be [N, D], got {emb.shape}')

    if emb.shape[0] == 0:
        raise ValueError('No embeddings provided to gem_pool_aggregate')
    
    if debug:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f'[DEBUG] gem_pool_aggregate: input={emb.shape}, p={p}')

    # GeM pooling: (mean(|x|^p))^(1/p)
    abs_p = torch.clamp(emb.abs(), min=eps) ** p
    mean_abs_p = abs_p.mean(dim=0)
    gem = mean_abs_p ** (1.0 / p)
    
    # Restore sign (use dominant sign per dimension)
    gem = gem * emb.sign().mean(dim=0).sign()
    
    # Power normalization
    if use_power_norm:
        gem = power_normalize(gem, alpha=power_alpha)
    
    # L2 normalization
    desc = F.normalize(gem, p=2, dim=0)
    
    if debug:
        logger = logging.getLogger(__name__)
        logger.debug(f'[DEBUG] gem_pool_aggregate: output norm={desc.norm():.4f}')
    
    return desc


def vlad_power_normalize(vlad: torch.Tensor, 
                         alpha: float = 0.4,
                         intra_norm: bool = True) -> torch.Tensor:
    """
    Apply power normalization to VLAD/NetVLAD descriptors.
    
    For VLAD with K clusters and D dimensions:
    1. Optionally apply intra-normalization (L2 per cluster)
    2. Apply power normalization
    3. Apply final L2 normalization
    
    Args:
        vlad: [K*D] or [K, D] VLAD descriptor
        alpha: power exponent
        intra_norm: whether to apply intra-normalization first
        
    Returns:
        Normalized VLAD descriptor with same shape
    """
    original_shape = vlad.shape
    
    # Power normalization (element-wise)
    vlad = power_normalize(vlad.view(-1), alpha=alpha)
    
    # Final L2 normalization
    vlad = F.normalize(vlad, p=2, dim=0)
    
    return vlad.view(original_shape)
