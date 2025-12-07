# src/sampling/contour_sampler.py
"""
Contour-based patch sampling.

Samples patches along the contours of handwriting strokes.
This approach focuses on the actual ink regions, providing 
better coverage of writing style characteristics.


What is contour?
    A contour is a curve joining all the continuous points (along the boundary),
    having same color or intensity. The contours are a useful tool for shape analysis
    and object detection and recognition.
    In simple layman terms, contours can be explained as the curve that joins all the dots
    having same intensity. For example, if we have a binary image consisting of white
    text on a black background, the contour of the text would be the curve that outlines
    the shape of the text.

    1. Find contours using cv2.findContours.
    2. Sample points along the contours.
    3. Extract patches centered at those points.
    4. Filter patches based on ink ratio.
    5. Return centers and patches.
"""
import logging
import cv2
import numpy as np

from ..utils.patch_ops import extract_patch_centered

# Setup module logger
logger = logging.getLogger(__name__)


def contour_sample_patches(bw_img, patch_size=32, step=10, max_patches=2000,
                           min_ink_ratio=0.01, min_contour_area=50, debug=False):
    """
    Sample patches along handwriting contours.
    
    Contour sampling extracts patches at points along the boundary of 
    connected ink regions. This provides good coverage of writing style
    characteristics like stroke beginnings, curves, and endings.
    
    Args:
        bw_img: Binary image with text as black (0) or white text on dark bg
        patch_size: Size of patches to extract
        step: Sample every 'step' points along contour
        max_patches: Maximum patches to extract
        min_ink_ratio: Minimum ink in patch (lowered from 0.05 to 0.01)
        min_contour_area: Skip tiny contours (noise)
        debug: Enable detailed debug logging
        
    Returns:
        centers: List of (x, y) center coordinates
        patches: np.ndarray [N, patch_size, patch_size]

    """
    h, w = bw_img.shape
    
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"[ContourSampling] Image: {h}x{w}, patch_size={patch_size}, step={step}")
        logger.debug(f"[ContourSampling] min_ink_ratio={min_ink_ratio}, min_area={min_contour_area}")
    
    # Normalize: ensure we work with a consistent format
    # If image is mostly dark (text is white), invert it
    if bw_img.mean() > 127:
        # Background is light, text is dark - standard case
        work_img = bw_img.copy()
        if debug:
            logger.debug("[ContourSampling] Light background detected (standard)")
    else:
        # Background is dark, text is light - invert for contour detection
        work_img = 255 - bw_img
        if debug:
            logger.debug("[ContourSampling] Dark background detected (inverted)")

    # Apply Otsu thresholding to get clean binary image
    _, thr = cv2.threshold(work_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # thr now has text as WHITE (255) for contour detection
    contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    if debug:
        logger.debug(f"[ContourSampling] Found {len(contours)} contours total")

    centers = []
    patches = []
    
    # Track statistics
    skipped_small_area = 0
    skipped_short = 0
    skipped_low_ink = 0

    # Sort contours by area (largest first) for better coverage
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt_idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Skip very small contours (noise)
        if area < min_contour_area:
            skipped_small_area += 1
            continue

        if len(cnt) < step:
            skipped_short += 1
            continue

        for i in range(0, len(cnt), step):
            x, y = cnt[i][0][0], cnt[i][0][1]  # OpenCV contours are (x, y)
            patch = extract_patch_centered(bw_img, x, y, patch_size=patch_size)

            # Check ink ratio - text pixels are dark (< 128)
            ink_ratio = (patch < 128).mean()
            if ink_ratio < min_ink_ratio:
                skipped_low_ink += 1
                continue

            centers.append((x, y))
            patches.append(patch)

            if len(patches) >= max_patches:
                if debug:
                    logger.debug(f"[ContourSampling] Hit max_patches={max_patches}")
                    logger.debug(f"[ContourSampling] Stats: small_area={skipped_small_area}, "
                                f"short={skipped_short}, low_ink={skipped_low_ink}")
                return centers, np.stack(patches, axis=0)

    if debug:
        logger.debug(f"[ContourSampling] Complete: {len(patches)} patches extracted")
        logger.debug(f"[ContourSampling] Skipped: small_area={skipped_small_area}, "
                    f"short={skipped_short}, low_ink={skipped_low_ink}")
        logger.info(f"🔲 Contour sampling: {len(patches)} patches from {len(contours)} contours")

    if patches:
        patches = np.stack(patches, axis=0)
    else:
        patches = np.zeros((0, patch_size, patch_size), dtype=bw_img.dtype)
        if debug:
            logger.warning("[ContourSampling] No patches extracted!")

    return centers, patches
