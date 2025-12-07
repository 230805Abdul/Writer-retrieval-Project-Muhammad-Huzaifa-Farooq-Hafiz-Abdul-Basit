# src/train/page_bag_dataset.py
"""
Dataset for End-to-End writer retrieval training.

CRITICAL DIFFERENCE FROM patch_dataset.py:
- PatchDataset: Returns individual patches with writer labels
  → Loss is computed on PATCHES → Model learns patch similarity
  
- PageBagDataset: Returns BAGS of patches per PAGE
  → Loss is computed on AGGREGATED PAGE descriptors  
  → Model learns WRITER similarity (page-level features)

This is the key insight: we want the model to learn that pages from 
the same writer are similar, NOT that individual patches are similar.

Usage:
    dataset = PageBagDataset(csv_path, patches_per_page=32)
    # Returns: (page_patches, writer_id)
    # page_patches: [patches_per_page, 1, H, W]
    # writer_id: scalar int64

In training loop:
    # Batch of pages from multiple writers
    pages, labels = batch  # [B, P, 1, H, W], [B]
    
    # Pass through E2E model (patch encoder + aggregation)
    page_descriptors = model(pages)  # [B, D]
    
    # Triplet loss on PAGE descriptors, not patches!
    loss = triplet_loss(page_descriptors, labels)
"""
import csv
from pathlib import Path
import random
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.preprocessing import load_image, binarize_otsu, resize_max_side
from ..sampling.adaptive_sampler import adaptive_sample, AdaptiveSamplingConfig

logger = logging.getLogger(__name__)


class PageBagDataset(Dataset):
    """
    Dataset that returns bags of patches for each page.
    
    Each __getitem__ returns:
        patches: [patches_per_page, 1, H, W] - bag of patches from one page
        label: scalar int64 - writer ID
    
    This is designed for End-to-End training where:
    1. Patches go through the encoder → [P, D] embeddings
    2. Embeddings are aggregated (GeM, NetVLAD) → [1, D] page descriptor
    3. Loss is computed on page descriptors → learns writer features
    
    Args:
        csv_path: Path to CSV file with image_path,writer_id
        root_dir: Optional root directory for image paths
        max_side: Maximum side length for resizing
        sampler_cfg: Adaptive sampling configuration
        transform: Optional transform to apply to patches
        min_patches: Minimum patches to ensure per page
        patches_per_page: Number of patches per page (CRITICAL for E2E training)
        cache_patches: If True, cache extracted patches in memory
        deterministic: If True, always return same patches for same idx
                       (useful for validation)
    """

    def __init__(
        self,
        csv_path,
        root_dir=None,
        max_side=1600,
        sampler_cfg=None,
        transform=None,
        min_patches=10,
        patches_per_page=32,  # Number of patches to sample from each page
        cache_patches=True,   # Cache for E2E training is beneficial
        deterministic=False,  # Random by default for training variety
        debug=False,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.max_side = max_side
        self.sampler_cfg = sampler_cfg or AdaptiveSamplingConfig()
        self.transform = transform
        self.min_patches = min_patches
        self.patches_per_page = patches_per_page
        self.cache_patches = cache_patches
        self.deterministic = deterministic
        self.debug = debug
        
        # Cache for extracted patches (image_idx -> patches array)
        self._patch_cache = {} if cache_patches else None

        self.samples = []
        with self.csv_path.open('r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row['image_path']
                writer_id = int(row['writer_id'])
                self.samples.append((img_path, writer_id))

        if not self.samples:
            raise ValueError(f'No samples loaded from {self.csv_path}')
        
        # Build writer -> indices mapping for potential future use
        self.writer_to_indices = {}
        for idx, (_, writer_id) in enumerate(self.samples):
            if writer_id not in self.writer_to_indices:
                self.writer_to_indices[writer_id] = []
            self.writer_to_indices[writer_id].append(idx)
        
        self.writers = list(self.writer_to_indices.keys())
        
        if self.debug:
            logger.debug(f'[DEBUG] PageBagDataset: loaded {len(self.samples)} pages')
            logger.debug(f'[DEBUG] patches_per_page={patches_per_page}, cache={cache_patches}')
            logger.debug(f'[DEBUG] Unique writers: {len(self.writers)}')
            pages_per_writer = [len(v) for v in self.writer_to_indices.values()]
            logger.debug(f'[DEBUG] Pages per writer: min={min(pages_per_writer)}, '
                        f'max={max(pages_per_writer)}, avg={np.mean(pages_per_writer):.1f}')

    def __len__(self):
        return len(self.samples)

    def _resolve_path(self, rel_or_abs):
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        if self.root_dir is not None:
            return self.root_dir / p
        return p

    def _extract_patches(self, idx):
        """Extract all patches from a page image, with optional caching."""
        # Check cache first
        if self._patch_cache is not None and idx in self._patch_cache:
            return self._patch_cache[idx]
        
        img_rel_path, writer_id = self.samples[idx]
        img_path = self._resolve_path(img_rel_path)

        img = load_image(img_path)
        if self.max_side is not None:
            img = resize_max_side(img, max_side=self.max_side)
        bw = binarize_otsu(img)

        centers, patches, info = adaptive_sample(bw, self.sampler_cfg)

        # Ensure we have enough patches
        if patches.shape[0] < self.min_patches:
            if patches.shape[0] == 0:
                patch = cv2.resize(bw, (self.sampler_cfg.patch_size, self.sampler_cfg.patch_size))
                patches = np.expand_dims(patch, axis=0)
            else:
                reps = int(np.ceil(self.min_patches / patches.shape[0]))
                patches = np.tile(patches, (reps, 1, 1))
        
        # Cache if enabled
        if self._patch_cache is not None:
            self._patch_cache[idx] = patches
        
        return patches

    def __getitem__(self, idx):
        """
        Get a bag of patches from one page.
        
        Returns:
            patches: [patches_per_page, 1, H, W] tensor
            label: writer_id (scalar int64)
        """
        img_rel_path, writer_id = self.samples[idx]
        patches = self._extract_patches(idx)
        
        num_available = patches.shape[0]
        
        if self.deterministic:
            # Use fixed seed based on idx for reproducibility
            rng = np.random.RandomState(idx)
            if num_available >= self.patches_per_page:
                selected_indices = rng.choice(num_available, self.patches_per_page, replace=False)
            else:
                selected_indices = rng.choice(num_available, self.patches_per_page, replace=True)
        else:
            # Random sampling for training
            if num_available >= self.patches_per_page:
                selected_indices = random.sample(range(num_available), self.patches_per_page)
            else:
                selected_indices = random.choices(range(num_available), k=self.patches_per_page)
        
        selected_patches = patches[selected_indices]  # [patches_per_page, H, W]

        # Convert to tensor
        selected_patches = selected_patches.astype(np.float32) / 255.0  # [0,1]
        selected_patches = 1.0 - selected_patches  # Make ink=1, background=0
        selected_patches = np.expand_dims(selected_patches, axis=1)  # [P, 1, H, W]

        patch_tensor = torch.from_numpy(selected_patches)

        if self.transform is not None:
            # Apply transform to each patch independently
            transformed = []
            for i in range(patch_tensor.shape[0]):
                transformed.append(self.transform(patch_tensor[i]))
            patch_tensor = torch.stack(transformed, dim=0)

        # Single label for the entire page
        label = torch.tensor(writer_id, dtype=torch.long)
        
        return patch_tensor, label
    
    def clear_cache(self):
        """Clear the patch cache to free memory."""
        if self._patch_cache is not None:
            self._patch_cache.clear()
            logger.info('PageBagDataset: cache cleared')

    def get_writer_indices(self, writer_id):
        """Get all page indices for a specific writer."""
        return self.writer_to_indices.get(writer_id, [])
    
    def get_all_labels(self):
        """Get list of all writer labels (one per page)."""
        return [wid for _, wid in self.samples]


class PageBagCollateFn:
    """
    Custom collate function for PageBagDataset.
    
    Handles batching of variable-size bags if needed.
    For fixed patches_per_page, this is just default stacking.
    """
    
    def __call__(self, batch):
        """
        Collate a batch of (patches, label) pairs.
        
        Args:
            batch: List of (patches, label) tuples
                   patches: [P, 1, H, W]
                   label: scalar
        
        Returns:
            patches: [B, P, 1, H, W]
            labels: [B]
        """
        patches_list = [item[0] for item in batch]
        labels_list = [item[1] for item in batch]
        
        # Stack into batch tensors
        patches = torch.stack(patches_list, dim=0)  # [B, P, 1, H, W]
        labels = torch.stack(labels_list, dim=0)    # [B]
        
        return patches, labels


def create_page_dataloader(
    csv_path,
    root_dir=None,
    patches_per_page=32,
    batch_size=16,
    m_per_class=2,
    sampler_cfg=None,
    transform=None,
    num_workers=4,
    cache_patches=True,
    deterministic=False,
    debug=False,
):
    """
    Create a DataLoader for End-to-End training.
    
    Uses MPerClassSampler to ensure each batch has m pages from each writer.
    This is crucial for triplet loss training.
    
    Args:
        csv_path: Path to CSV with image_path,writer_id
        root_dir: Root directory for images
        patches_per_page: Number of patches per page
        batch_size: Total batch size
        m_per_class: Pages per writer per batch (typically 2)
        sampler_cfg: Adaptive sampling configuration
        transform: Data augmentation transform
        num_workers: DataLoader workers
        cache_patches: Cache extracted patches
        deterministic: Use deterministic patch selection
        debug: Enable debug logging
    
    Returns:
        DataLoader, PageBagDataset
    """
    from .samplers import MPerClassSampler
    
    dataset = PageBagDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        sampler_cfg=sampler_cfg,
        patches_per_page=patches_per_page,
        cache_patches=cache_patches,
        deterministic=deterministic,
        transform=transform,
        debug=debug,
    )
    
    # Get all page labels for sampler
    all_labels = dataset.get_all_labels()
    
    # MPerClassSampler ensures each batch has m pages from each writer
    sampler = MPerClassSampler(
        labels=all_labels,
        m=m_per_class,
        batch_size=batch_size,
        length_before_new_iter=len(dataset),
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=PageBagCollateFn(),
    )
    
    logger.info(f'Created E2E DataLoader:')
    logger.info(f'   Pages: {len(dataset)}, Writers: {len(dataset.writers)}')
    logger.info(f'   Patches per page: {patches_per_page}')
    logger.info(f'   Batch size: {batch_size}, m_per_class: {m_per_class}')
    logger.info(f'   Batches per epoch: {len(loader)}')
    
    return loader, dataset
