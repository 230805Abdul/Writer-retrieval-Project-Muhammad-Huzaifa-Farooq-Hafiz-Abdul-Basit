# src/utils/preprocessing.py
import cv2
import numpy as np
from pathlib import Path
import shutil


def clear_dir(path: Path):
    """Delete all contents of a directory (keeps the directory itself)."""
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)


def load_image(path):
    """Load an image from disk as BGR uint8 (OpenCV default)."""
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Could not read image at: {path}')
    return img


def to_gray(img_bgr):
    """Convert BGR image to grayscale uint8."""
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def binarize_otsu(img_bgr_or_gray, invert=True):
    """
    Binarize using Otsu's threshold (good for IAM/CVL style documents).
    Returns a uint8 image with values {0, 255} where 0 = background, 255 = ink (by default).
    """
    gray = to_gray(img_bgr_or_gray)
    # Otsu thresholding
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # In many WR pipelines, ink is black (0) and background white (255).
    # If 'invert' is True and the background seems dark, invert.
    if invert:
        # heuristic: if mean is low, invert
        if bw.mean() < 127:
            bw = 255 - bw
    return bw


def resize_max_side(img, max_side=1600):
    """Resize image so that max(height, width) <= max_side, keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = max(h, w) / max_side
    if scale <= 1.0:
        return img
    new_w = int(w / scale)
    new_h = int(h / scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
