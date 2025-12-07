# src/aggregation/netvlad.py
"""
NetVLAD: Vector of Locally Aggregated Descriptors with learned soft-assignment.

Reference: marco-peer/icdar23 implementation
- num_clusters = 100 (not 32)
- dim = 64 (ResNet20 feature dimension)

This layer produces GLOBAL DESCRIPTORS by aggregating local patch features.
Output dimension: num_clusters * dim (e.g., 100 * 64 = 6400)
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NetVLAD(nn.Module):
    """
    NetVLAD layer for aggregating local descriptors into a global descriptor.

    Computes residuals (x - centroid) weighted by soft assignment,
    then aggregates into a single GLOBAL descriptor.

    Input:  x [N, D]
    Output: vlad [K*D] (L2-normalized global descriptor)
    """

    def __init__(self, num_clusters: int = 100, dim: int = 64, 
                 normalize_input: bool = True, debug: bool = False):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input
        self.debug = debug

        # Learnable cluster centroids
        self.centroids = nn.Parameter(torch.randn(num_clusters, dim) * 0.1)
        # Soft assignment layer
        self.assignment = nn.Linear(dim, num_clusters, bias=True)
        
        # Initialize assignment to approximate hard assignment to nearest centroid
        self._init_assignment()
        
        if self.debug:
            logger.debug(f'[DEBUG] NetVLAD initialized: clusters={num_clusters}, '
                        f'dim={dim}, output_dim={num_clusters * dim}')

    def _init_assignment(self):
        """Initialize assignment layer to mimic nearest centroid assignment."""
        # W = 2 * centroids, b = -||centroids||^2
        # This makes assignment(x) ≈ -||x - centroid||^2 (up to constant)
        self.assignment.weight.data = 2.0 * self.centroids.data.clone()
        self.assignment.bias.data = -(self.centroids.data ** 2).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate local patch features into a global descriptor.
        
        Args:
            x: [N, D] input features (N patches, D dimensions)
        Returns:
            vlad: [K*D] aggregated GLOBAL descriptor (L2-normalized)
        """
        assert x.dim() == 2, f"Expected 2D input [N, D], got {x.shape}"
        N, D = x.shape
        K = self.num_clusters
        
        if self.debug:
            logger.debug(f'[DEBUG] NetVLAD forward: input [N={N}, D={D}]')

        # Optional input normalization
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # Soft assignment probabilities
        soft_assign = F.softmax(self.assignment(x), dim=1)  # [N, K]
        
        if self.debug:
            logger.debug(f'[DEBUG] Soft assignment: shape={soft_assign.shape}, '
                        f'max={soft_assign.max():.4f}, min={soft_assign.min():.6f}')

        # Compute residuals: (x - centroid) for each (feature, cluster) pair
        x_exp = x.unsqueeze(1)                 # [N, 1, D]
        c_exp = self.centroids.unsqueeze(0)    # [1, K, D]
        residual = x_exp - c_exp               # [N, K, D]

        # Weight residuals by soft assignment and sum over features
        # vlad[k] = sum_n soft_assign[n,k] * (x[n] - centroid[k])
        vlad = (soft_assign.unsqueeze(2) * residual).sum(dim=0)  # [K, D]

        # Intra-normalization (L2 normalize each cluster's residual sum)
        vlad = F.normalize(vlad, p=2, dim=1)
        
        # Flatten and L2 normalize the full descriptor
        vlad = vlad.view(K * D)
        vlad = F.normalize(vlad, p=2, dim=0)
        
        return vlad
