# src/train/unsupervised_clustering.py
"""
Unsupervised clustering for writer pseudo-labels.

Used for HISIR19 dataset where writer labels are not available.
Following the icdar23 reference approach:
1. Extract patch embeddings from all images
2. Aggregate embeddings per image
3. Cluster image embeddings to get pseudo-labels
4. Train with pseudo-labels
5. Repeat clustering with improved embeddings
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for unsupervised clustering."""
    num_clusters: int = 500
    min_cluster_size: int = 10
    max_iter: int = 300
    batch_size: int = 10000  # For MiniBatchKMeans
    n_init: int = 10
    random_state: int = 42
    algorithm: str = "minibatch"  # "kmeans" or "minibatch"


def extract_embeddings(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract embeddings for all images in dataset.
    
    Returns:
        embeddings: [N, D] array of image embeddings
        indices: List of dataset indices corresponding to embeddings
    """
    model.eval()
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    all_embeddings = []
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, (patches, _) in enumerate(loader):
            # Handle multi-patch batches
            if patches.dim() == 5:  # [B, P, C, H, W]
                B, P = patches.shape[:2]
                patches = patches.view(B * P, *patches.shape[2:])
                
                patches = patches.to(device)
                emb = model(patches)  # [B*P, D]
                
                # Aggregate patches per image (mean pooling)
                emb = emb.view(B, P, -1).mean(dim=1)  # [B, D]
            else:
                patches = patches.to(device)
                emb = model(patches)
            
            all_embeddings.append(emb.cpu().numpy())
            
            # Track indices
            start_idx = batch_idx * batch_size
            end_idx = start_idx + emb.shape[0]
            all_indices.extend(range(start_idx, end_idx))
    
    embeddings = np.vstack(all_embeddings)
    logger.info(f"Extracted {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    return embeddings, all_indices


def cluster_embeddings(
    embeddings: np.ndarray,
    config: ClusteringConfig,
) -> Tuple[np.ndarray, Any]:
    """
    Cluster embeddings using K-Means.
    
    Returns:
        labels: Cluster assignments [N]
        kmeans: Fitted KMeans model
    """
    logger.info(f"Clustering {len(embeddings)} embeddings into {config.num_clusters} clusters...")
    
    if config.algorithm == "minibatch":
        kmeans = MiniBatchKMeans(
            n_clusters=config.num_clusters,
            max_iter=config.max_iter,
            batch_size=config.batch_size,
            n_init=config.n_init,
            random_state=config.random_state,
        )
    else:
        kmeans = KMeans(
            n_clusters=config.num_clusters,
            max_iter=config.max_iter,
            n_init=config.n_init,
            random_state=config.random_state,
        )
    
    labels = kmeans.fit_predict(embeddings)
    
    # Compute silhouette score on subset (for large datasets)
    if len(embeddings) > 10000:
        sample_idx = np.random.choice(len(embeddings), 10000, replace=False)
        sil_score = silhouette_score(embeddings[sample_idx], labels[sample_idx])
    else:
        sil_score = silhouette_score(embeddings, labels)
    
    # Count cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    logger.info(f"Clustering complete:")
    logger.info(f"  Silhouette score: {sil_score:.4f}")
    logger.info(f"  Cluster sizes: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}")
    
    return labels, kmeans


def filter_small_clusters(
    labels: np.ndarray,
    min_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out samples from clusters smaller than min_size.
    
    Returns:
        valid_mask: Boolean mask for valid samples
        new_labels: Remapped labels (with invalid samples marked as -1)
    """
    unique, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique[counts >= min_size]
    
    valid_mask = np.isin(labels, valid_clusters)
    
    # Remap labels
    new_labels = np.full_like(labels, -1)
    label_map = {old: new for new, old in enumerate(valid_clusters)}
    
    for old_label, new_label in label_map.items():
        new_labels[labels == old_label] = new_label
    
    n_removed = np.sum(~valid_mask)
    n_clusters_removed = len(unique) - len(valid_clusters)
    
    logger.info(f"Filtered clusters: {n_clusters_removed} clusters < {min_size} samples")
    logger.info(f"  Removed {n_removed} samples ({100*n_removed/len(labels):.1f}%)")
    logger.info(f"  Remaining: {len(valid_clusters)} clusters, {np.sum(valid_mask)} samples")
    
    return valid_mask, new_labels


class PseudoLabelDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that uses cluster pseudo-labels instead of ground truth.
    """
    
    def __init__(
        self,
        base_dataset,
        pseudo_labels: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ):
        """
        Args:
            base_dataset: Original dataset
            pseudo_labels: Cluster assignments
            valid_mask: Boolean mask for valid samples
        """
        self.base_dataset = base_dataset
        
        if valid_mask is not None:
            self.valid_indices = np.where(valid_mask)[0]
            self.pseudo_labels = pseudo_labels[valid_mask]
        else:
            self.valid_indices = np.arange(len(pseudo_labels))
            self.pseudo_labels = pseudo_labels
        
        self.samples = [
            (self.base_dataset.samples[i][0], int(self.pseudo_labels[j]))
            for j, i in enumerate(self.valid_indices)
        ]
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        orig_idx = self.valid_indices[idx]
        patches, _ = self.base_dataset[orig_idx]
        label = self.pseudo_labels[idx]
        return patches, label


class UnsupervisedTrainer:
    """
    Trainer for unsupervised writer identification using clustering.
    
    Training loop:
    1. Initialize with pretrained model
    2. Extract embeddings for all images
    3. Cluster embeddings -> pseudo-labels
    4. Train model with pseudo-labels
    5. Repeat from step 2 with updated model
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        dataset,
        device: torch.device,
        config: ClusteringConfig,
        train_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.config = config
        self.train_config = train_config or {}
        
        self.current_iteration = 0
        self.current_labels = None
        self.training_history = []
    
    def cluster_and_label(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings and cluster to get pseudo-labels.
        
        Returns:
            valid_mask: Boolean mask for valid samples
            pseudo_labels: Cluster assignments
        """
        # Extract embeddings
        embeddings, _ = extract_embeddings(
            self.model,
            self.dataset,
            self.device,
            batch_size=self.train_config.get('batch_size', 64),
            num_workers=self.train_config.get('num_workers', 4),
        )
        
        # L2 normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Cluster
        labels, kmeans = cluster_embeddings(embeddings, self.config)
        
        # Filter small clusters
        valid_mask, pseudo_labels = filter_small_clusters(
            labels, min_size=self.config.min_cluster_size
        )
        
        self.current_labels = pseudo_labels
        
        return valid_mask, pseudo_labels
    
    def get_pseudo_label_dataset(self) -> PseudoLabelDataset:
        """
        Get dataset with current pseudo-labels.
        """
        valid_mask, pseudo_labels = self.cluster_and_label()
        
        return PseudoLabelDataset(
            self.dataset,
            pseudo_labels,
            valid_mask,
        )
    
    def train_iteration(
        self,
        epochs: int = 10,
        save_dir: Optional[Path] = None,
    ):
        """
        Run one iteration of clustering + training.
        """
        self.current_iteration += 1
        logger.info(f"\n{'='*70}")
        logger.info(f"Clustering Iteration {self.current_iteration}")
        logger.info(f"{'='*70}")
        
        # Get pseudo-labeled dataset
        pseudo_dataset = self.get_pseudo_label_dataset()
        
        logger.info(f"Training on {len(pseudo_dataset)} samples "
                   f"with {len(set(pseudo_dataset.pseudo_labels))} pseudo-classes")
        
        # TODO: Integrate with train_v2.py training loop
        # This would require passing the pseudo_dataset to the training function
        
        return pseudo_dataset


def compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Args:
        embeddings: [N, D] embeddings
        labels: Predicted cluster labels
        true_labels: Optional ground truth labels for NMI/ARI
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    
    metrics = {}
    
    # Silhouette score (subset for speed)
    if len(embeddings) > 10000:
        sample_idx = np.random.choice(len(embeddings), 10000, replace=False)
        metrics['silhouette'] = silhouette_score(embeddings[sample_idx], labels[sample_idx])
    else:
        metrics['silhouette'] = silhouette_score(embeddings, labels)
    
    # Cluster statistics
    unique, counts = np.unique(labels, return_counts=True)
    metrics['n_clusters'] = len(unique)
    metrics['cluster_size_mean'] = np.mean(counts)
    metrics['cluster_size_std'] = np.std(counts)
    metrics['cluster_size_min'] = np.min(counts)
    metrics['cluster_size_max'] = np.max(counts)
    
    # If true labels available
    if true_labels is not None:
        metrics['nmi'] = normalized_mutual_info_score(true_labels, labels)
        metrics['ari'] = adjusted_rand_score(true_labels, labels)
    
    return metrics
