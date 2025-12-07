# src/sampling/__init__.py
"""
Sampling module for patch extraction from handwriting images.

This module provides various sampling strategies:
- DenseSampler: Grid-based sampling with stride
- ContourSampler: Sample along text contours  
- ComponentSampler: Connected component-based sampling (formerly CharacterSampler)
- AdaptiveSampler: Automatic mode selection based on text quantity

Usage:
    from src.sampling import adaptive_sample, AdaptiveSamplingConfig
    from src.sampling import ComponentSampler, ComponentSamplerConfig
    from src.sampling import dense_sample_patches, contour_sample_patches
"""

from .adaptive_sampler import adaptive_sample, AdaptiveSamplingConfig
from .component_sampler import ComponentSampler, ComponentSamplerConfig
from .dense_sampler import dense_sample_patches
from .contour_sampler import contour_sample_patches
from .text_quantity import estimate_text_quantity

# Backward compatibility alias
from .character_sampler import CharacterSampler, CharacterSamplerConfig

__all__ = [
    'adaptive_sample',
    'AdaptiveSamplingConfig', 
    'ComponentSampler',
    'ComponentSamplerConfig',
    'dense_sample_patches',
    'contour_sample_patches',
    'estimate_text_quantity',
    # Deprecated - use ComponentSampler instead
    'CharacterSampler',
    'CharacterSamplerConfig',
]
