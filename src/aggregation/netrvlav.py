# src/aggregation/netrvlav.py
"""
NetRVLAD: Residual-less VLAD aggregation layer.

Reference: marco-peer/icdar23 implementation
- num_clusters = 100 (not 32)
- dim = 64 (ResNet20 feature dimension)
- No gating mechanism (simple soft-assignment only)

This layer produces GLOBAL DESCRIPTORS by aggregating local patch features.
Output dimension: num_clusters * dim (e.g., 100 * 64 = 6400)
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NetRVLAD(nn.Module):
    """
    NetRVLAD: learnable soft-assign weighted sum without residuals.
    
    Unlike NetVLAD, we don't compute residuals (x - centroid).
    Instead, we directly weight and sum the input features.
    
    Output dimension: num_clusters * dim
    
    This produces a GLOBAL DESCRIPTOR for the entire image/page.
    """

    def __init__(self, num_clusters: int = 100, dim: int = 64, 
                 normalize_input: bool = True, debug: bool = False):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input
        self.debug = debug

        # Soft assignment layer (no bias for stability)
        self.assignment = nn.Linear(dim, num_clusters, bias=False)
        # Initialize with small random values
        nn.init.normal_(self.assignment.weight, std=1.0 / num_clusters)
        
        if self.debug:
            logger.debug(f'[DEBUG] NetRVLAD initialized: clusters={num_clusters}, '
                        f'dim={dim}, output_dim={num_clusters * dim}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate local patch features into a global descriptor.
        
        Args:
            x: [N, D] input features (N patches, D dimensions)
        Returns:
            v: [K*D] aggregated GLOBAL descriptor (L2-normalized)
        """
        assert x.dim() == 2, f"Expected 2D input [N, D], got {x.shape}"
        N, D = x.shape
        K = self.num_clusters
        
        if self.debug:
            logger.debug(f'[DEBUG] NetRVLAD forward: input [N={N}, D={D}]')

        # Optional input normalization
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # Soft assignment: which cluster does each feature belong to?
        soft_assign = F.softmax(self.assignment(x), dim=1)  # [N, K]
        
        if self.debug:
            logger.debug(f'[DEBUG] Soft assignment: shape={soft_assign.shape}, '
                        f'max={soft_assign.max():.4f}, min={soft_assign.min():.6f}')

        # Weighted sum of features per cluster (no residuals)
        # v[k] = sum_n soft_assign[n,k] * x[n]
        v = soft_assign.t() @ x  # [K, D]

        # Intra-normalization (L2 normalize each cluster)
        v = F.normalize(v, p=2, dim=1)
        
        # Flatten and L2 normalize the full descriptor
        v = v.view(K * D)
        v = F.normalize(v, p=2, dim=0)
        
        if self.debug:
            logger.debug(f'[DEBUG] NetRVLAD output: shape={v.shape}, norm={v.norm():.4f}')
        
        return v
