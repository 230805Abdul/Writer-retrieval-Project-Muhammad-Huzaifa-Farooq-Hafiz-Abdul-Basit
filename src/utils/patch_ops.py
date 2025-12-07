# src/utils/patch_ops.py
import cv2
import numpy as np


def extract_patch_centered(img, x, y, patch_size=32):
    """
    Extract a square patch of size patch_size x patch_size centered at (x, y).
    If the patch goes out of bounds, we pad with background (255).
    Returns uint8 patch.
    """
    h, w = img.shape[:2]
    half = patch_size // 2

    x1 = x - half
    y1 = y - half
    x2 = x + half
    y2 = y + half

    # define patch in output coordinates
    patch = np.full((patch_size, patch_size), 255, dtype=img.dtype)

    # intersection with image
    x1_img = max(x1, 0)
    y1_img = max(y1, 0)
    x2_img = min(x2, w)
    y2_img = min(y2, h)

    # corresponding coords in patch
    x1_p = x1_img - x1
    y1_p = y1_img - y1
    x2_p = x1_p + (x2_img - x1_img)
    y2_p = y1_p + (y2_img - y1_img)

    patch[y1_p:y2_p, x1_p:x2_p] = img[y1_img:y2_img, x1_img:x2_img]
    return patch
