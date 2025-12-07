# src/train/triplet_loss.py
"""
Improved triplet loss implementations for metric learning.

Includes:
- BatchHardTripletLoss: Original batch-hard mining
- BatchHardSoftMarginTripletLoss: Soft-margin variant (no margin hyperparameter)
- BatchSemiHardTripletLoss: Semi-hard mining (better convergence)
- MultiSimilarityLoss: State-of-the-art metric learning loss
- CircleLoss: Another SOTA option
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss with improved margin default.
    
    For each anchor:
    - Hardest positive = max distance to same-class sample
    - Hardest negative = min distance to different-class sample
    
    emb: [B, D], labels: [B]
    """

    def __init__(self, margin=0.3):  # INCREASED from 0.1 to 0.3
        super().__init__()
        self.margin = margin

    def forward(self, emb, labels):
        # pairwise distance matrix
        # dist[i, j] = ||emb_i - emb_j||_2
        dist = torch.cdist(emb, emb, p=2)  # [B, B]

        labels = labels.view(-1, 1)  # [B, 1]
        mask_pos = labels.eq(labels.t())  # [B, B]
        mask_neg = ~mask_pos

        # For each anchor, hardest positive = max dist over positives (excluding itself)
        dist_pos = dist.clone()
        dist_pos[~mask_pos] = -1.0  # ignore non-positives
        # set diagonal to -1 so it isn't selected
        diag = torch.arange(dist_pos.size(0), device=dist_pos.device)
        dist_pos[diag, diag] = -1.0
        hardest_pos = dist_pos.max(dim=1)[0]  # [B]

        # For each anchor, hardest negative = min dist over negatives
        dist_neg = dist.clone()
        dist_neg[~mask_neg] = 1e6  # ignore non-negatives
        hardest_neg = dist_neg.min(dim=1)[0]  # [B]

        # Triplet loss
        losses = F.relu(hardest_pos - hardest_neg + self.margin)
        # Ignore anchors that have no positive pairs
        valid_mask = hardest_pos > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=emb.device, requires_grad=True)
        loss = losses[valid_mask].mean()
        return loss


class BatchSemiHardTripletLoss(nn.Module):
    """
    Semi-hard triplet mining: select negatives that are harder than the positive
    but still within the margin.
    
    This provides a more stable training signal than pure hard mining.
    
    emb: [B, D], labels: [B]
    """
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, emb, labels):
        dist = torch.cdist(emb, emb, p=2)  # [B, B]
        
        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t())
        mask_neg = ~mask_pos
        
        B = emb.size(0)
        diag = torch.arange(B, device=emb.device)
        
        # For each anchor, find hardest positive
        dist_pos = dist.clone()
        dist_pos[~mask_pos] = -1.0
        dist_pos[diag, diag] = -1.0
        hardest_pos, _ = dist_pos.max(dim=1)  # [B]
        
        # For semi-hard: negative should satisfy d_pos < d_neg < d_pos + margin
        losses = []
        for i in range(B):
            if hardest_pos[i] <= 0:
                continue
            
            neg_mask = mask_neg[i]
            neg_dists = dist[i][neg_mask]
            
            if neg_dists.numel() == 0:
                continue
            
            # Semi-hard: d_pos < d_neg < d_pos + margin
            semi_hard_mask = (neg_dists > hardest_pos[i]) & (neg_dists < hardest_pos[i] + self.margin)
            
            if semi_hard_mask.any():
                # Use hardest semi-hard negative
                semi_hard_neg = neg_dists[semi_hard_mask].min()
            else:
                # Fallback to hardest negative
                semi_hard_neg = neg_dists.min()
            
            loss = F.relu(hardest_pos[i] - semi_hard_neg + self.margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=emb.device, requires_grad=True)
        
        return torch.stack(losses).mean()


class BatchHardSoftMarginTripletLoss(nn.Module):
    """
    Batch-hard triplet loss with soft margin (no margin hyperparameter).
    Uses softplus instead of hinge loss: log(1 + exp(d_pos - d_neg))
    
    This avoids the need to tune the margin hyperparameter.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, emb, labels):
        dist = torch.cdist(emb, emb, p=2)
        
        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t())
        mask_neg = ~mask_pos
        
        B = emb.size(0)
        diag = torch.arange(B, device=emb.device)
        
        # Hardest positive
        dist_pos = dist.clone()
        dist_pos[~mask_pos] = -1.0
        dist_pos[diag, diag] = -1.0
        hardest_pos = dist_pos.max(dim=1)[0]
        
        # Hardest negative
        dist_neg = dist.clone()
        dist_neg[~mask_neg] = 1e6
        hardest_neg = dist_neg.min(dim=1)[0]
        
        # Soft margin loss
        losses = F.softplus(hardest_pos - hardest_neg)
        valid_mask = hardest_pos > 0
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=emb.device, requires_grad=True)
        
        return losses[valid_mask].mean()


class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss for metric learning.
    
    Reference: Wang et al., "Multi-Similarity Loss with General Pair Weighting 
               for Deep Metric Learning" (CVPR 2019)
    
    This is a state-of-the-art loss that considers multiple aspects of similarity.
    """
    
    def __init__(self, alpha=2.0, beta=50.0, base=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
    
    def forward(self, emb, labels):
        # Cosine similarity (embeddings should be L2-normalized)
        sim = emb @ emb.t()  # [B, B]
        
        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t()).float()
        mask_neg = 1.0 - mask_pos
        
        B = emb.size(0)
        diag = torch.arange(B, device=emb.device)
        mask_pos[diag, diag] = 0  # Exclude self
        
        # For each anchor, find positive and negative pairs
        losses = []
        for i in range(B):
            pos_mask = mask_pos[i] > 0
            neg_mask = mask_neg[i] > 0
            
            if not pos_mask.any() or not neg_mask.any():
                continue
            
            pos_sim = sim[i][pos_mask]
            neg_sim = sim[i][neg_mask]
            
            # Hard mining with margin
            neg_hard = neg_sim[neg_sim + 0.1 > pos_sim.min()]
            pos_hard = pos_sim[pos_sim - 0.1 < neg_sim.max()]
            
            if neg_hard.numel() == 0:
                neg_hard = neg_sim
            if pos_hard.numel() == 0:
                pos_hard = pos_sim
            
            # MS loss
            pos_loss = torch.log(1 + torch.sum(torch.exp(-self.alpha * (pos_hard - self.base)))) / self.alpha
            neg_loss = torch.log(1 + torch.sum(torch.exp(self.beta * (neg_hard - self.base)))) / self.beta
            
            losses.append(pos_loss + neg_loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=emb.device, requires_grad=True)
        
        return torch.stack(losses).mean()


class CircleLoss(nn.Module):
    """
    Circle Loss for metric learning.
    
    Reference: Sun et al., "Circle Loss: A Unified Perspective of Pair Similarity 
               Optimization" (CVPR 2020)
    """
    
    def __init__(self, m=0.25, gamma=256):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
    
    def forward(self, emb, labels):
        # Cosine similarity
        sim = emb @ emb.t()
        
        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t()).float()
        mask_neg = 1.0 - mask_pos
        
        B = emb.size(0)
        diag = torch.arange(B, device=emb.device)
        mask_pos[diag, diag] = 0
        
        # Optimal points
        O_p = 1 + self.m
        O_n = -self.m
        Delta_p = 1 - self.m
        Delta_n = self.m
        
        # Compute weights
        alpha_p = F.relu(O_p - sim.detach())
        alpha_n = F.relu(sim.detach() - O_n)
        
        # Logits
        logit_p = -alpha_p * (sim - Delta_p) * self.gamma
        logit_n = alpha_n * (sim - Delta_n) * self.gamma
        
        # Apply masks
        logit_p = logit_p * mask_pos
        logit_n = logit_n * mask_neg
        
        # Loss
        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)
        ).mean()
        
        return loss


def get_loss_function(loss_type='triplet', margin=0.3, **kwargs):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: One of 'triplet', 'semi_hard', 'soft_margin', 'ms', 'circle'
        margin: Margin for triplet-based losses
        **kwargs: Additional arguments for specific losses
    
    Returns:
        Loss function module
    """
    loss_map = {
        'triplet': lambda: BatchHardTripletLoss(margin=margin),
        'batch_hard': lambda: BatchHardTripletLoss(margin=margin),
        'semi_hard': lambda: BatchSemiHardTripletLoss(margin=margin),
        'soft_margin': lambda: BatchHardSoftMarginTripletLoss(),
        'ms': lambda: MultiSimilarityLoss(**kwargs),
        'multi_similarity': lambda: MultiSimilarityLoss(**kwargs),
        'circle': lambda: CircleLoss(**kwargs),
    }
    
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_map.keys())}")
    
    return loss_map[loss_type]()
