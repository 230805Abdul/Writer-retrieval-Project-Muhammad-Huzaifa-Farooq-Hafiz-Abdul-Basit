# src/models/e2e_writer_net.py
"""
End-to-End Writer Retrieval Network.

This module implements the key insight from SOTA writer retrieval:
- Train with PAGE-LEVEL loss, not PATCH-LEVEL loss
- Use differentiable aggregation (GeM, NetVLAD, Attention) inside the model
- Compute triplet loss on aggregated page descriptors

Architecture:
    Input: Bag of patches from a page [P, 1, H, W]
    → Patch Encoder (ResNet): [P, D] patch embeddings
    → Aggregator (GeM/NetVLAD/Attention): [D'] page descriptor
    → Triplet Loss on page descriptors

Why this matters:
- Patch-level training: Forces model to match individual letters (wrong objective)
- Page-level training: Forces model to match overall writing style (correct objective)
"""
import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features.resnet_patch_extractor import create_resnet_patch_encoder
from ..aggregation.netvlad import NetVLAD
from ..aggregation.netrvlav import NetRVLAD


logger = logging.getLogger(__name__)


class GeMAggregator(nn.Module):
    """
    Generalized Mean (GeM) pooling aggregator.
    
    GeM with learnable p parameter:
    - p=1: Mean pooling
    - p→∞: Max pooling
    - p=3 (typical): Balances mean and max
    
    Benefits:
    - Robust to varying patch counts
    - Suppresses noisy/uninformative patches
    - Differentiable (trainable p)
    """
    
    def __init__(self, dim: int, p: float = 3.0, eps: float = 1e-6, 
                 learnable_p: bool = True, power_norm: bool = True,
                 power_alpha: float = 0.4):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.power_norm = power_norm
        self.power_alpha = power_alpha
        
        # Learnable p parameter
        if learnable_p:
            self.p = nn.Parameter(torch.tensor(p))
        else:
            self.register_buffer('p', torch.tensor(p))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate patch embeddings into page descriptor.
        
        Args:
            x: [N, D] patch embeddings from one page
            
        Returns:
            desc: [D] aggregated page descriptor (L2-normalized)
        """
        assert x.dim() == 2, f"Expected [N, D], got {x.shape}"
        
        # GeM: (mean(|x|^p))^(1/p) with sign restoration
        p = self.p.clamp(min=1.0)  # p must be >= 1
        
        # Compute GeM
        x_abs = x.abs().clamp(min=self.eps)
        gem = x_abs.pow(p).mean(dim=0).pow(1.0 / p)
        
        # Restore dominant sign
        signs = x.sign().mean(dim=0).sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        gem = gem * signs
        
        # Power normalization (suppress bursty features)
        if self.power_norm:
            gem = gem.sign() * (gem.abs() + self.eps).pow(self.power_alpha)
        
        # L2 normalize
        desc = F.normalize(gem, p=2, dim=0)
        
        return desc
    
    def forward_batch(self, x: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Aggregate batch of pages with varying patch counts.
        
        Args:
            x: [total_patches, D] all patch embeddings
            batch_indices: [total_patches] indicating which page each patch belongs to
            
        Returns:
            descs: [B, D] aggregated descriptors for each page
        """
        unique_indices = batch_indices.unique()
        descs = []
        
        for idx in unique_indices:
            mask = batch_indices == idx
            patches = x[mask]
            desc = self.forward(patches)
            descs.append(desc)
        
        return torch.stack(descs, dim=0)


class AttentionAggregator(nn.Module):
    """
    Attention-based aggregator for patch embeddings.
    
    Learns to weight patches by their importance:
    - Important patches (clear writing) get high weight
    - Noisy patches (stains, artifacts) get low weight
    
    Architecture:
        Patches [N, D] → Attention scores [N, 1] → Weighted sum [D]
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 power_norm: bool = True, power_alpha: float = 0.4):
        super().__init__()
        self.dim = dim
        self.power_norm = power_norm
        self.power_alpha = power_alpha
        
        hidden = hidden_dim or dim // 2
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate with learned attention weights.
        
        Args:
            x: [N, D] patch embeddings
            
        Returns:
            desc: [D] attention-weighted descriptor
        """
        assert x.dim() == 2, f"Expected [N, D], got {x.shape}"
        
        # Compute attention weights
        attn_logits = self.attention(x)  # [N, 1]
        attn_weights = F.softmax(attn_logits, dim=0)  # [N, 1]
        
        # Weighted sum
        desc = (attn_weights * x).sum(dim=0)  # [D]
        
        # Power normalization
        if self.power_norm:
            desc = desc.sign() * (desc.abs() + 1e-6).pow(self.power_alpha)
        
        # L2 normalize
        desc = F.normalize(desc, p=2, dim=0)
        
        return desc
    
    def forward_batch(self, x: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """Aggregate batch of pages."""
        unique_indices = batch_indices.unique()
        descs = []
        
        for idx in unique_indices:
            mask = batch_indices == idx
            patches = x[mask]
            desc = self.forward(patches)
            descs.append(desc)
        
        return torch.stack(descs, dim=0)


class EndToEndWriterNet(nn.Module):
    """
    End-to-End Writer Retrieval Network.
    
    This is the CORRECT architecture for writer retrieval:
    - Takes a BAG of patches (representing a page/document)
    - Encodes each patch independently
    - Aggregates patch embeddings into a single page descriptor
    - Training loss is computed on PAGE descriptors (not patches!)
    
    Why this works:
    - Page A and Page B from the same writer may have different letters
    - Patch-level matching would fail (different letters look different)
    - Page-level matching succeeds (overall style is similar)
    
    Args:
        emb_dim: Patch embedding dimension (typically 64-128)
        backbone: Encoder architecture ('resnet18', 'resnet34', 'small')
        pretrained: Use ImageNet pretrained weights
        aggregator: Aggregation method ('gem', 'netvlad', 'netrvlad', 'attention')
        num_clusters: Number of clusters for VLAD-based aggregators
        gem_p: Initial p parameter for GeM
        freeze_encoder: Freeze encoder weights (for fine-tuning aggregator only)
    """
    
    def __init__(
        self,
        emb_dim: int = 64,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        aggregator: Literal['gem', 'netvlad', 'netrvlad', 'attention'] = 'gem',
        num_clusters: int = 32,
        gem_p: float = 3.0,
        power_norm: bool = True,
        power_alpha: float = 0.4,
        freeze_encoder: bool = False,
        freeze_bn: bool = False,
    ):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.aggregator_type = aggregator
        
        # Patch encoder
        self.encoder = create_resnet_patch_encoder(
            emb_dim=emb_dim,
            backbone=backbone,
            pretrained=pretrained,
            freeze_bn=freeze_bn,
        )
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen (aggregator-only training)")
        
        # Aggregator
        if aggregator == 'gem':
            self.aggregator = GeMAggregator(
                dim=emb_dim, p=gem_p, learnable_p=True,
                power_norm=power_norm, power_alpha=power_alpha
            )
            self.output_dim = emb_dim
        elif aggregator == 'netvlad':
            self.aggregator = NetVLAD(
                num_clusters=num_clusters, dim=emb_dim, normalize_input=True
            )
            self.output_dim = num_clusters * emb_dim
        elif aggregator == 'netrvlad':
            self.aggregator = NetRVLAD(
                num_clusters=num_clusters, dim=emb_dim, normalize_input=True
            )
            self.output_dim = num_clusters * emb_dim
        elif aggregator == 'attention':
            self.aggregator = AttentionAggregator(
                dim=emb_dim, power_norm=power_norm, power_alpha=power_alpha
            )
            self.output_dim = emb_dim
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
        
        logger.info(f"EndToEndWriterNet: encoder={backbone}, aggregator={aggregator}, "
                   f"emb_dim={emb_dim}, output_dim={self.output_dim}")
    
    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Encode patches without aggregation.
        
        Args:
            patches: [N, 1, H, W] grayscale patches
            
        Returns:
            embeddings: [N, D] patch embeddings
        """
        return self.encoder(patches)
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - handles both single page and batched input.
        
        Args:
            patches: Either:
                - [N, 1, H, W] patches from one page
                - [B, P, 1, H, W] batch of pages, each with P patches
            
        Returns:
            descriptor: [output_dim] or [B, output_dim] page descriptor(s)
        """
        if patches.dim() == 4:
            # Single page: [N, 1, H, W]
            embeddings = self.encoder(patches)  # [N, D]
            descriptor = self.aggregator(embeddings)  # [output_dim]
            return descriptor
        elif patches.dim() == 5:
            # Batched pages: [B, P, 1, H, W]
            B, P, C, H, W = patches.shape
            
            # Reshape to encode all patches at once: [B*P, 1, H, W]
            patches_flat = patches.view(B * P, C, H, W)
            embeddings = self.encoder(patches_flat)  # [B*P, D]
            
            # Reshape back and aggregate per page
            embeddings = embeddings.view(B, P, -1)  # [B, P, D]
            
            # Aggregate each page
            descriptors = []
            for i in range(B):
                desc = self.aggregator(embeddings[i])  # [D]
                descriptors.append(desc)
            
            return torch.stack(descriptors, dim=0)  # [B, output_dim]
        else:
            raise ValueError(f"Expected 4D or 5D input, got {patches.dim()}D")
    
    def forward_batch(self, patches: torch.Tensor, 
                      patch_counts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of pages with varying patch counts.
        
        This is the main training interface:
        - Input: All patches from B pages, concatenated
        - Output: B page descriptors
        
        Args:
            patches: [total_patches, 1, H, W] all patches concatenated
            patch_counts: [B] number of patches per page
            
        Returns:
            descriptors: [B, output_dim] page descriptors
        """
        # Encode all patches at once (efficient!)
        all_embeddings = self.encoder(patches)  # [total_patches, D]
        
        # Split by page and aggregate
        descriptors = []
        start_idx = 0
        
        for count in patch_counts:
            count = int(count.item())
            page_emb = all_embeddings[start_idx:start_idx + count]
            desc = self.aggregator(page_emb)
            descriptors.append(desc)
            start_idx += count
        
        return torch.stack(descriptors, dim=0)  # [B, output_dim]
    
    def get_encoder(self) -> nn.Module:
        """Return the patch encoder (for compatibility with existing eval code)."""
        return self.encoder


def create_e2e_model(
    emb_dim: int = 64,
    backbone: str = 'resnet18',
    pretrained: bool = True,
    aggregator: str = 'gem',
    num_clusters: int = 32,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
) -> EndToEndWriterNet:
    """
    Factory function to create End-to-End model.
    
    Args:
        emb_dim: Embedding dimension
        backbone: Encoder backbone
        pretrained: Use pretrained weights
        aggregator: Aggregation type
        num_clusters: VLAD clusters (if using VLAD)
        checkpoint_path: Optional checkpoint to load encoder weights from
        device: Target device
        
    Returns:
        Configured EndToEndWriterNet
    """
    model = EndToEndWriterNet(
        emb_dim=emb_dim,
        backbone=backbone,
        pretrained=pretrained,
        aggregator=aggregator,
        num_clusters=num_clusters,
    )
    
    # Load encoder weights from existing checkpoint if provided
    if checkpoint_path:
        logger.info(f"Loading encoder weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        else:
            state_dict = ckpt
        
        # Load into encoder
        model.encoder.load_state_dict(state_dict, strict=True)
        logger.info("Encoder weights loaded successfully")
    
    return model.to(device)
