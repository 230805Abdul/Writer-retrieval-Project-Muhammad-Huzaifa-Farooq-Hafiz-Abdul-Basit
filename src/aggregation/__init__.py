# Aggregation module exports
"""
Aggregation methods for combining patch-level features into document descriptors.

Supported aggregation types:
- mean: Simple mean pooling
- sum: Sum pooling  
- gem: Generalized Mean (GeM) pooling
- vlad: Traditional VLAD encoding
- netvlad: Differentiable NetVLAD with soft assignment
- netrvlad: NetRVLAD (residual-less variant)
"""

from .pooling import mean_pool_aggregate, sum_pool_aggregate, gem_pool_aggregate
from .vlad import VladParams, fit_kmeans_for_vlad, vlad_aggregate_torch
from .netvlad import NetVLAD
from .netrvlav import NetRVLAD

__all__ = [
    # Pooling functions
    'mean_pool_aggregate',
    'sum_pool_aggregate', 
    'gem_pool_aggregate',
    # VLAD
    'VladParams',
    'fit_kmeans_for_vlad',
    'vlad_aggregate_torch',
    # NetVLAD variants
    'NetVLAD',
    'NetRVLAD',
]
