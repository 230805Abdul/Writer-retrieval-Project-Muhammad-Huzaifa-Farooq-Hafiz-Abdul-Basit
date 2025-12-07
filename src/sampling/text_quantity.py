# src/sampling/text_quantity.py
import cv2
import numpy as np


def estimate_line_count(bw_img):
    """
    Roughly estimate line count using horizontal morphology and projection.
    bw_img: binary image with text as black (0), background as white (255).
    Returns an integer estimate of number of text lines.
    """
    # Ensure text is black
    if bw_img.mean() < 127:
        bw = 255 - bw_img
    else:
        bw = bw_img.copy()

    # Invert so text is white on black for morphology
    inv = 255 - bw

    # Horizontal dilation to connect characters into lines
    # kernel width should be a fraction of page width
    h, w = bw.shape
    k_w = max(15, w // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 3))
    merged = cv2.dilate(inv, kernel, iterations=1)

    # Horizontal projection
    proj = merged.sum(axis=1)

    # Threshold to detect lines
    # Normalize projection to [0,1], then threshold
    proj_norm = (proj - proj.min()) / (np.ptp(proj) + 1e-6)
    mask = proj_norm > 0.1  # row considered part of text line

    # Count connected components in 1D mask -> line segments
    line_count = 0
    in_segment = False
    for val in mask:
        if val and not in_segment:
            line_count += 1
            in_segment = True
        elif not val and in_segment:
            in_segment = False

    return line_count


def estimate_text_quantity(bw_img):
    """
    Return a small dict with basic quantity info.
    For now we mainly use 'line_count' but can extend (text area ratio, etc.).
    """
    h, w = bw_img.shape
    # text area ratio: proportion of dark pixels
    dark = (bw_img < 128).sum()
    ratio = dark / float(h * w)

    line_count = estimate_line_count(bw_img)

    return {
        'line_count': line_count,
        'text_area_ratio': ratio,
    }
