# src/sampling/character_sampler.py

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CharacterSamplerConfig:
    patch_size: int = 64
    min_area: int = 10          # tune
    max_area: int = 2000        # tune
    min_aspect: float = 0.2     # w/h
    max_aspect: float = 5.0
    max_chars: int = 512        # max patches per page
    use_height_filter: bool = True
    height_tol: float = 0.5     # +- 50% around median height


class CharacterSampler:
    """
    Character-aware sampler: finds connected components in a binarized page
    and returns patch centers around character-like blobs.
    """

    def __init__(self, cfg: CharacterSamplerConfig | None = None):
        self.cfg = cfg or CharacterSamplerConfig()

    # ---------- public API ----------

    def sample_centers(self, bin_img: np.ndarray) -> List[Tuple[int, int]]:
        """
        bin_img: HxW uint8, 0=background, 255=text (or 0/1)
        returns: list of (y, x) centers for patches
        """
        bin_img = self._prepare_binary(bin_img)
        comps = self._connected_components(bin_img)
        comps = self._filter_components(comps)
        if self.cfg.use_height_filter:
            comps = self._filter_by_height(comps)

        centers = [(c["cy"], c["cx"]) for c in comps]
        # limit to max_chars, but try to distribute across page
        if len(centers) > self.cfg.max_chars:
            idx = np.linspace(0, len(centers) - 1,
                              self.cfg.max_chars).astype(int)
            centers = [centers[i] for i in idx]
        return centers

    def extract_patches(self, gray_img: np.ndarray) -> np.ndarray:
        """
        gray_img: HxW uint8 (original or binarized)
        returns: (N, patch_size, patch_size) array of uint8
        """
        h, w = gray_img.shape[:2]
        centers = self.sample_centers(gray_img)
        ps = self.cfg.patch_size
        r = ps // 2

        patches = []
        for cy, cx in centers:
            y1 = max(cy - r, 0)
            y2 = min(cy + r, h)
            x1 = max(cx - r, 0)
            x2 = min(cx + r, w)

            patch = np.zeros((ps, ps), dtype=gray_img.dtype)
            crop = gray_img[y1:y2, x1:x2]
            patch[0:crop.shape[0], 0:crop.shape[1]] = crop
            patches.append(patch)

        if not patches:
            return np.zeros((0, self.cfg.patch_size, self.cfg.patch_size),
                            dtype=gray_img.dtype)
        return np.stack(patches, axis=0)

    # ---------- internal helpers ----------

    def _prepare_binary(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # assume text is dark on light background
        if img.max() > 1:
            _, bin_img = cv2.threshold(img, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            bin_img = (img * 255).astype("uint8")

        # make text=1, background=0
        bin_inv = (bin_img == 0).astype("uint8")
        return bin_inv

    def _connected_components(self, bin_img: np.ndarray):
        # OpenCV returns labels + stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_img, connectivity=8
        )
        comps = []
        for label in range(1, num_labels):  # skip background
            x, y, w, h, area = stats[label]
            cx, cy = centroids[label]
            comps.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "cx": int(round(cx)),
                "cy": int(round(cy)),
            })
        return comps

    def _filter_components(self, comps):
        cfg = self.cfg
        filtered = []
        for c in comps:
            area = c["area"]
            if area < cfg.min_area or area > cfg.max_area:
                continue
            w, h = c["w"], c["h"]
            if h == 0:
                continue
            aspect = w / h
            if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
                continue
            filtered.append(c)
        return filtered

    def _filter_by_height(self, comps):
        if not comps:
            return comps
        heights = np.array([c["h"] for c in comps], dtype=float)
        median_h = np.median(heights)
        lo = median_h * (1.0 - self.cfg.height_tol)
        hi = median_h * (1.0 + self.cfg.height_tol)
        return [c for c in comps if lo <= c["h"] <= hi]
