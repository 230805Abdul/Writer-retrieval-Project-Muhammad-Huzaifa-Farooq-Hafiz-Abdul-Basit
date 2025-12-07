# src/train/patch_dataset.py
"""
Dataset for patch-based writer retrieval training.

IMPROVED VERSION:
- Returns MULTIPLE patches per image per __getitem__ call
- Configurable patches_per_call parameter (default: 8)
- This multiplies effective training data by patches_per_call
- Patches within a batch from the same writer come from DIFFERENT pages
- The model learns page-invariant style features, not page-specific patterns
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


class PatchDataset(Dataset):
    """
    Dataset that yields (patch_tensor, writer_id) pairs.
    
    IMPORTANT: When patches_per_call > 1, returns stacked tensors:
        patch_tensor: [patches_per_call, 1, H, W]
        label: [patches_per_call] (same writer_id repeated)

    Assumes a CSV with columns:
        image_path,writer_id

    For each __getitem__, we:
        - load image
        - resize (optional)
        - binarize
        - run adaptive_sample
        - pick patches_per_call random patches for this writer
        
    This ensures diversity: each call returns different random patches.
    
    Args:
        csv_path: Path to CSV file
        root_dir: Optional root directory for image paths
        max_side: Maximum side length for resizing
        sampler_cfg: Adaptive sampling configuration
        transform: Optional transform to apply to patches
        min_patches: Minimum patches to extract per image
        patches_per_call: Number of patches to return per __getitem__ (CRITICAL!)
        cache_patches: If True, cache extracted patches in memory
    """

    def __init__(
        self,
        csv_path,
        root_dir=None,
        max_side=1600,
        sampler_cfg=None,
        transform=None,
        min_patches=10,
        patches_per_call=8,  # CRITICAL: Return multiple patches per call!
        cache_patches=False,  # Cache patches in memory for faster training
        debug=False,  # Enable debug logging
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.max_side = max_side
        self.sampler_cfg = sampler_cfg or AdaptiveSamplingConfig()
        self.transform = transform
        self.min_patches = min_patches
        self.patches_per_call = patches_per_call
        self.cache_patches = cache_patches
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
        
        if self.debug:
            logger.debug(f'[DEBUG] PatchDataset: loaded {len(self.samples)} samples')
            logger.debug(f'[DEBUG] patches_per_call={patches_per_call}, cache={cache_patches}')
            writers = set(wid for _, wid in self.samples)
            logger.debug(f'[DEBUG] Unique writers: {len(writers)}')

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
        """Extract all patches from an image, with optional caching."""
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
        img_rel_path, writer_id = self.samples[idx]
        patches = self._extract_patches(idx)
        
        # Select multiple random patches from this image
        num_available = patches.shape[0]
        
        if num_available >= self.patches_per_call:
            # Sample without replacement
            selected_indices = random.sample(range(num_available), self.patches_per_call)
        else:
            # Sample with replacement if not enough patches
            selected_indices = random.choices(range(num_available), k=self.patches_per_call)
        
        selected_patches = patches[selected_indices]  # [patches_per_call, H, W]

        # Convert to tensor
        selected_patches = selected_patches.astype(np.float32) / 255.0  # [0,1]
        selected_patches = 1.0 - selected_patches  # Make ink=1, background=0
        selected_patches = np.expand_dims(selected_patches, axis=1)  # [patches_per_call, 1, H, W]

        patch_tensor = torch.from_numpy(selected_patches)

        if self.transform is not None:
            # Apply transform to each patch
            transformed = []
            for i in range(patch_tensor.shape[0]):
                transformed.append(self.transform(patch_tensor[i]))
            patch_tensor = torch.stack(transformed, dim=0)

        # Repeat label for each patch
        labels = torch.full((self.patches_per_call,), writer_id, dtype=torch.long)
        
        return patch_tensor, labels
    
    def clear_cache(self):
        """Clear the patch cache to free memory."""
        if self._patch_cache is not None:
            self._patch_cache.clear()


class SinglePatchDataset(Dataset):
    """
    Wrapper that returns single patches from a multi-patch dataset.
    Useful for validation where we don't need multi-patch batching.
    """
    
    def __init__(self, base_dataset: PatchDataset):
        self.base = base_dataset
        # Force single patch mode
        self._original_patches_per_call = base_dataset.patches_per_call
        
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        # Temporarily set to single patch
        self.base.patches_per_call = 1
        patch_tensor, labels = self.base[idx]
        self.base.patches_per_call = self._original_patches_per_call
        
        # Return single patch
        return patch_tensor.squeeze(0), labels.squeeze(0)

