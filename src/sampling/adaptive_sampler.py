# src/sampling/adaptive_sampler.py
"""
Adaptive sampling module that selects sampling strategy based on text quantity.

This module provides intelligent sampling mode selection:
- Dense: Grid-based sampling for text-heavy pages
- Contour: Sampling along text contours (best for writer retrieval)
- Component: Connected component-based sampling
- Auto: Automatic selection based on line count

Enhanced with comprehensive debug logging for analysis and debugging.
"""
from dataclasses import dataclass
import logging

from .text_quantity import estimate_text_quantity
from .dense_sampler import dense_sample_patches
from .contour_sampler import contour_sample_patches
from .component_sampler import ComponentSampler, ComponentSamplerConfig

# Setup module logger
logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSamplingConfig:
    """Configuration for adaptive sampling.
    
    Attributes:
        patch_size: Size of extracted patches (default: 32)
        dense_stride: Stride for dense sampling (default: 24)
        contour_step: Step size along contours (default: 12)
        max_patches: Maximum patches to extract (default: 1500)
        line_threshold: Lines threshold for auto mode switching (default: 4)
        mode: Sampling mode ('auto', 'dense', 'contour', 'component')
        debug: Enable detailed debug logging (default: False)
    """
    patch_size: int = 32
    dense_stride: int = 24
    contour_step: int = 12
    max_patches: int = 1500
    line_threshold: int = 4
    mode: str = 'auto'
    debug: bool = False


def _component_sample(bw_img, cfg: AdaptiveSamplingConfig):
    """Component-aware sampling using connected components.
    
    Args:
        bw_img: Binary grayscale image
        cfg: Adaptive sampling configuration
        
    Returns:
        centers: List of (x, y) center coordinates
        patches: Numpy array of patches [N, patch_size, patch_size]
    """
    comp_cfg = ComponentSamplerConfig(
        patch_size=cfg.patch_size,
        max_components=cfg.max_patches,
        debug=cfg.debug,
    )
    sampler = ComponentSampler(comp_cfg)
    centers_yx = sampler.sample_centers(bw_img)
    patches = sampler.extract_patches(bw_img)

    # Convert (y, x) → (x, y) for consistency
    centers = [(cx, cy) for (cy, cx) in centers_yx]
    
    if cfg.debug:
        logger.debug(f"[ComponentSampling] Extracted {len(centers)} patches")
    
    return centers, patches


def adaptive_sample(bw_img, cfg: AdaptiveSamplingConfig | None = None):
    """
    Quantity-adaptive sampling with comprehensive logging.

    Selects sampling strategy based on text quantity and configuration.
    
    Args:
        bw_img: Binary grayscale image (0/255), text = black
        cfg: AdaptiveSamplingConfig instance
        
    Returns:
        centers: List of (x, y) center coordinates
        patches: np.ndarray [N, patch_size, patch_size]
        info: Dict with metadata (mode used, line_count, etc.)
    """
    if cfg is None:
        cfg = AdaptiveSamplingConfig()

    if cfg.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"[AdaptiveSampling] Starting with mode='{cfg.mode}'")
        logger.debug(f"[AdaptiveSampling] Image shape: {bw_img.shape}")
        logger.debug(f"[AdaptiveSampling] Config: patch_size={cfg.patch_size}, "
                    f"max_patches={cfg.max_patches}")

    # Always compute text quantity for info dict
    q = estimate_text_quantity(bw_img)
    line_count = q['line_count']
    
    if cfg.debug:
        logger.debug(f"[AdaptiveSampling] Text quantity: lines={line_count}, "
                    f"area_ratio={q['text_area_ratio']:.4f}")

    # ---- Explicit modes ----
    if cfg.mode == 'dense':
        if cfg.debug:
            logger.debug("[AdaptiveSampling] Using DENSE sampling")
        centers, patches = dense_sample_patches(
            bw_img,
            patch_size=cfg.patch_size,
            stride=cfg.dense_stride,
            max_patches=cfg.max_patches,
        )
        mode_used = 'dense'

    elif cfg.mode == 'contour':
        if cfg.debug:
            logger.debug("[AdaptiveSampling] Using CONTOUR sampling")
        centers, patches = contour_sample_patches(
            bw_img,
            patch_size=cfg.patch_size,
            step=cfg.contour_step,
            max_patches=cfg.max_patches,
        )
        mode_used = 'contour'

    elif cfg.mode in ('char', 'component'):
        if cfg.debug:
            logger.debug("[AdaptiveSampling] Using COMPONENT sampling")
        centers, patches = _component_sample(bw_img, cfg)
        mode_used = 'component'

    else:
        # ---- AUTO: quantity-adaptive ----
        if cfg.debug:
            logger.debug(f"[AdaptiveSampling] AUTO mode: line_count={line_count}, "
                        f"threshold={cfg.line_threshold}")
        
        if line_count >= cfg.line_threshold:
            # Many lines → dense patches over page
            if cfg.debug:
                logger.debug("[AdaptiveSampling] AUTO → DENSE (many lines)")
            centers, patches = dense_sample_patches(
                bw_img,
                patch_size=cfg.patch_size,
                stride=cfg.dense_stride,
                max_patches=cfg.max_patches,
            )
            mode_used = 'dense-auto'
        else:
            # Few lines → component-aware focused sampling
            if cfg.debug:
                logger.debug("[AdaptiveSampling] AUTO → COMPONENT (few lines)")
            centers, patches = _component_sample(bw_img, cfg)
            mode_used = 'component-auto'

    # Build info dictionary
    info = {
        'mode': mode_used,
        'line_count': line_count,
        'text_area_ratio': q['text_area_ratio'],
        'num_patches': int(patches.shape[0]),
        'patch_size': cfg.patch_size,
    }
    
    if cfg.debug:
        logger.debug(f"[AdaptiveSampling] Result: mode={mode_used}, "
                    f"patches={patches.shape[0]}")
        logger.info(f"📦 Sampling complete: {mode_used} → {patches.shape[0]} patches")
    
    return centers, patches, info


# Backward compatibility - keep char_sample accessible
_char_sample = _component_sample
