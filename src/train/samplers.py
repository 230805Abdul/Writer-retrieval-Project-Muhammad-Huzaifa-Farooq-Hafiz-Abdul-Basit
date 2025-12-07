# src/train/samplers.py
"""
Batch samplers for metric learning.

MPerClassSampler ensures each batch contains m samples from each of 
several classes, which is critical for effective triplet mining.
"""
import random
from collections import defaultdict
from typing import List, Iterator
import logging

import numpy as np
from torch.utils.data import Sampler


logger = logging.getLogger(__name__)


class MPerClassSampler(Sampler):
    """
    Samples m instances per class in each batch.
    
    This is crucial for triplet loss training - ensures each batch
    has multiple samples from the same writer for positive pairs.
    
    Reference: pytorch_metric_learning.samplers.MPerClassSampler
    """
    
    def __init__(self, labels: List[int], m: int = 4, 
                 batch_size: int = 128, 
                 length_before_new_iter: int = None):
        """
        Args:
            labels: List of class labels for each sample
            m: Number of samples per class in each batch
            batch_size: Total batch size (should be divisible by m)
            length_before_new_iter: How many samples before reshuffling
                                    (default: len(labels))
        """
        self.labels = np.array(labels)
        self.m = m
        self.batch_size = batch_size
        self.length_before_new_iter = length_before_new_iter or len(labels)
        
        # Build index mapping: class -> list of indices
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
        
        # Filter classes that have at least m samples
        self.valid_classes = [c for c in self.classes 
                              if len(self.class_to_indices[c]) >= m]
        
        if len(self.valid_classes) == 0:
            # Fallback: use all classes
            self.valid_classes = self.classes
            logger.warning(f'No classes have >= {m} samples, using all classes')
        
        # Number of classes per batch
        self.classes_per_batch = batch_size // m
        
        # Log sampler configuration
        logger.debug(f'MPerClassSampler initialized:')
        logger.debug(f'   Total samples: {len(labels)}')
        logger.debug(f'   Total classes: {len(self.classes)}')
        logger.debug(f'   Valid classes (>= {m} samples): {len(self.valid_classes)}')
        logger.debug(f'   Samples per class per batch: {m}')
        logger.debug(f'   Classes per batch: {self.classes_per_batch}')
        
    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch."""
        indices = []
        
        # Shuffle class order
        class_order = self.valid_classes.copy()
        random.shuffle(class_order)
        
        # For each class, shuffle its indices
        class_indices = {}
        for c in class_order:
            idx = self.class_to_indices[c].copy()
            random.shuffle(idx)
            class_indices[c] = idx
        
        # Pointer for each class
        class_ptr = {c: 0 for c in class_order}
        
        num_samples = 0
        while num_samples < self.length_before_new_iter:
            # Select random classes for this batch
            if len(class_order) >= self.classes_per_batch:
                batch_classes = random.sample(class_order, self.classes_per_batch)
            else:
                # Not enough classes, sample with replacement
                batch_classes = random.choices(class_order, k=self.classes_per_batch)
            
            for c in batch_classes:
                # Get m samples from this class
                for _ in range(self.m):
                    ptr = class_ptr[c]
                    if ptr >= len(class_indices[c]):
                        # Reshuffle this class
                        random.shuffle(class_indices[c])
                        ptr = 0
                    
                    indices.append(class_indices[c][ptr])
                    class_ptr[c] = ptr + 1
                    num_samples += 1
                    
                    if num_samples >= self.length_before_new_iter:
                        break
                
                if num_samples >= self.length_before_new_iter:
                    break
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.length_before_new_iter


class BalancedBatchSampler(Sampler):
    """
    Alternative: Sample equal number of instances from each class.
    Simpler version of MPerClassSampler.
    """
    
    def __init__(self, labels: List[int], n_classes: int = 16, 
                 n_samples: int = 4):
        """
        Args:
            labels: List of class labels
            n_classes: Number of classes per batch
            n_samples: Number of samples per class per batch
        """
        self.labels = np.array(labels)
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
        
        # How many batches can we make?
        self.n_batches = len(labels) // (n_classes * n_samples)
    
    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_batches):
            batch_classes = random.sample(self.classes, 
                                          min(self.n_classes, len(self.classes)))
            batch = []
            for c in batch_classes:
                indices = self.class_to_indices[c]
                if len(indices) >= self.n_samples:
                    batch.extend(random.sample(indices, self.n_samples))
                else:
                    batch.extend(random.choices(indices, k=self.n_samples))
            
            random.shuffle(batch)
            yield batch
    
    def __len__(self) -> int:
        return self.n_batches
