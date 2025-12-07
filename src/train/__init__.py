# src/train/__init__.py
"""
Training module for CARA-WR writer retrieval.

This module provides two training paradigms:

1. PATCH-LEVEL Training (train_resnet_triplet.py):
   - Loss on individual patches
   - Faster to train, less memory
   - Suitable for pretraining
   
2. END-TO-END Training (train_e2e.py) [RECOMMENDED]:
   - Loss on aggregated PAGE descriptors
   - Learns writer-level features, not patch features
   - Significantly better retrieval performance

Usage:
    # End-to-End training (recommended)
    from src.train.page_bag_dataset import PageBagDataset, create_page_dataloader
    from src.models.e2e_writer_net import EndToEndWriterNet
    
    # Patch-level training (for pretraining or comparison)
    from src.train.patch_dataset import PatchDataset
"""

from .patch_dataset import PatchDataset, SinglePatchDataset
from .page_bag_dataset import PageBagDataset, PageBagCollateFn, create_page_dataloader
from .triplet_loss import BatchHardTripletLoss
from .samplers import MPerClassSampler
from .augmentation import get_train_augmentation

__all__ = [
    # Datasets
    'PatchDataset',
    'SinglePatchDataset',
    'PageBagDataset',
    'PageBagCollateFn',
    'create_page_dataloader',
    # Loss
    'BatchHardTripletLoss',
    # Sampling
    'MPerClassSampler',
    # Augmentation
    'get_train_augmentation',
]
