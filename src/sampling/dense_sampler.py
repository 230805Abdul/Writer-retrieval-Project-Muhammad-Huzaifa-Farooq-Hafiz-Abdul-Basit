# src/sampling/dense_sampler.py
"""
Dense grid-based patch sampling.

Samples patches on a regular grid with configurable stride.
Skips mostly blank patches to save computational budget.
"""
import logging
import numpy as np

from ..utils.patch_ops import extract_patch_centered

# Setup module logger
logger = logging.getLogger(__name__)


def dense_sample_patches(bw_img, patch_size=32, stride=16, max_patches=2000,
                         min_ink_ratio=0.02, debug=False):
    """
    Dense sliding-window sampling over the whole page.
    
    Args:
        bw_img: Binary grayscale image (0/255), text = black
        patch_size: Size of patches to extract
        stride: Step size between patch centers
        max_patches: Maximum number of patches to return
        min_ink_ratio: Minimum ink ratio to keep a patch
        debug: Enable detailed debug logging
        
    Returns:
        centers: List of (x, y) centers
        patches: Numpy array [N, patch_size, patch_size]
    """
    h, w = bw_img.shape
    half = patch_size // 2
    
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"[DenseSampling] Image: {h}x{w}, patch_size={patch_size}, stride={stride}")
        logger.debug(f"[DenseSampling] Sampling region: y=[{half}, {h-half}], x=[{half}, {w-half}]")

    centers = []
    patches = []
    total_candidates = 0
    skipped_blank = 0

    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            total_candidates += 1
            
            # Skip mostly blank patches to save budget
            patch = extract_patch_centered(bw_img, x, y, patch_size=patch_size)
            # Fraction of ink pixels (dark = ink)
            ink_ratio = (patch < 128).mean()
            
            if ink_ratio < min_ink_ratio:
                skipped_blank += 1
                continue

            centers.append((x, y))
            patches.append(patch)

            if len(patches) >= max_patches:
                if debug:
                    logger.debug(f"[DenseSampling] Hit max_patches={max_patches}")
                    logger.debug(f"[DenseSampling] Total candidates: {total_candidates}, "
                                f"skipped blank: {skipped_blank}")
                return centers, np.stack(patches, axis=0)

    if debug:
        logger.debug(f"[DenseSampling] Grid complete: candidates={total_candidates}, "
                    f"skipped_blank={skipped_blank}, kept={len(patches)}")
        logger.info(f"📊 Dense sampling: {len(patches)}/{total_candidates} patches "
                   f"({100*len(patches)/max(1,total_candidates):.1f}%)")

    if patches:
        patches = np.stack(patches, axis=0)
    else:
        patches = np.zeros((0, patch_size, patch_size), dtype=bw_img.dtype)
        if debug:
            logger.warning("[DenseSampling] No patches extracted!")

    return centers, patches
