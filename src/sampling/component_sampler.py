# src/sampling/component_sampler.py
"""
Component-based sampler for handwriting patches.

This sampler uses connected component analysis to find text regions
and extracts patches centered on these components.

NOTE: Previously named "CharacterSampler" but renamed to "ComponentSampler"
because connected components in cursive handwriting often represent
whole words or word fragments, not individual characters.

For true character-level sampling, you would need:
- MSER (Maximally Stable Extremal Regions)
- A dedicated character detector
- Stroke segmentation algorithms
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import logging

# Import our enhanced logging
try:
    from ..utils.logging_config import get_logger, Colors, timed, debug_call
except ImportError:
    # Fallback if logging module not available
    def get_logger():
        return logging.getLogger(__name__)
    class Colors:
        RESET = ''
        DEBUG = ''
        WARNING = ''
    def timed(func):
        return func
    def debug_call(func):
        return func

logger = logging.getLogger(__name__)


@dataclass
class ComponentSamplerConfig:
    """Configuration for component-based patch sampling."""
    patch_size: int = 64
    min_area: int = 10          # Minimum component area in pixels
    max_area: int = 2000        # Maximum component area in pixels
    min_aspect: float = 0.2     # Minimum width/height ratio
    max_aspect: float = 5.0     # Maximum width/height ratio
    max_components: int = 512   # Maximum patches per page
    use_height_filter: bool = True
    height_tol: float = 0.5     # Height tolerance: ±50% around median
    
    # Debug options
    debug: bool = False
    save_debug_images: bool = False
    debug_output_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.patch_size > 0, f"patch_size must be positive, got {self.patch_size}"
        assert self.min_area > 0, f"min_area must be positive, got {self.min_area}"
        assert self.max_area > self.min_area, f"max_area ({self.max_area}) must be > min_area ({self.min_area})"
        assert 0 < self.min_aspect < self.max_aspect, f"Invalid aspect ratio range: [{self.min_aspect}, {self.max_aspect}]"


# Alias for backwards compatibility
CharacterSamplerConfig = ComponentSamplerConfig


class ComponentSampler:
    """
    Component-aware sampler: finds connected components in a binarized page
    and returns patch centers around text-like blobs.
    
    This is effective for handwriting because it naturally focuses on
    ink regions rather than empty space.
    
    Debug Features:
    - Logs component statistics at each filtering stage
    - Can save visualization images showing detected components
    - Tracks timing for performance analysis
    """

    def __init__(self, cfg: Optional[ComponentSamplerConfig] = None):
        self.cfg = cfg or ComponentSamplerConfig()
        self._debug_stats: Dict[str, Any] = {}
        
        if self.cfg.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug(f"🔍 ComponentSampler initialized with config: {self.cfg}")

    # ---------- Public API ----------

    @timed
    def sample_centers(self, bin_img: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract patch centers from a binary image.
        
        Args:
            bin_img: HxW uint8 image, 0=background, 255=text (or 0/1)
            
        Returns:
            List of (y, x) centers for patches
            
        Debug Info:
            - Logs image dimensions and content statistics
            - Tracks component counts at each filtering stage
            - Warns if no components found
        """
        # Reset debug stats
        self._debug_stats = {
            'input_shape': bin_img.shape,
            'input_dtype': str(bin_img.dtype),
        }
        
        # Prepare binary image
        bin_img = self._prepare_binary(bin_img)
        self._debug_stats['ink_ratio'] = np.mean(bin_img > 0)
        
        if self.cfg.debug:
            logger.debug(f"📊 Input image: shape={bin_img.shape}, ink_ratio={self._debug_stats['ink_ratio']:.4f}")
        
        # Find connected components
        comps = self._connected_components(bin_img)
        self._debug_stats['total_components'] = len(comps)
        
        if self.cfg.debug:
            logger.debug(f"🔍 Found {len(comps)} raw connected components")
        
        if not comps:
            logger.warning(f"⚠️  No connected components found in image of shape {bin_img.shape}")
            return []
        
        # Filter by area and aspect ratio
        comps = self._filter_components(comps)
        self._debug_stats['after_area_filter'] = len(comps)
        
        if self.cfg.debug:
            logger.debug(f"🔍 After area/aspect filter: {len(comps)} components")
        
        # Filter by height consistency
        if self.cfg.use_height_filter:
            comps = self._filter_by_height(comps)
            self._debug_stats['after_height_filter'] = len(comps)
            
            if self.cfg.debug:
                logger.debug(f"🔍 After height filter: {len(comps)} components")
        
        # Extract centers
        centers = [(c["cy"], c["cx"]) for c in comps]
        
        # Subsample if too many
        if len(centers) > self.cfg.max_components:
            original_count = len(centers)
            idx = np.linspace(0, len(centers) - 1, self.cfg.max_components).astype(int)
            centers = [centers[i] for i in idx]
            
            if self.cfg.debug:
                logger.debug(f"🔍 Subsampled from {original_count} to {len(centers)} components")
        
        self._debug_stats['final_centers'] = len(centers)
        
        if self.cfg.debug:
            logger.debug(f"✅ ComponentSampler: {len(centers)} patch centers extracted")
            self._log_component_stats(comps)
        
        return centers

    @timed
    def extract_patches(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Extract patches centered on detected components.
        
        Args:
            gray_img: HxW uint8 grayscale image
            
        Returns:
            (N, patch_size, patch_size) array of uint8 patches
        """
        h, w = gray_img.shape[:2]
        centers = self.sample_centers(gray_img)
        ps = self.cfg.patch_size
        r = ps // 2

        patches = []
        boundary_patches = 0
        
        for cy, cx in centers:
            y1 = max(cy - r, 0)
            y2 = min(cy + r, h)
            x1 = max(cx - r, 0)
            x2 = min(cx + r, w)

            # Check if patch is at boundary
            if y1 == 0 or y2 == h or x1 == 0 or x2 == w:
                boundary_patches += 1

            patch = np.zeros((ps, ps), dtype=gray_img.dtype)
            crop = gray_img[y1:y2, x1:x2]
            patch[0:crop.shape[0], 0:crop.shape[1]] = crop
            patches.append(patch)

        if self.cfg.debug and boundary_patches > 0:
            logger.debug(f"⚠️  {boundary_patches}/{len(patches)} patches touch image boundary")

        if not patches:
            logger.warning(f"⚠️  No patches extracted from image of shape {gray_img.shape}")
            return np.zeros((0, self.cfg.patch_size, self.cfg.patch_size), dtype=gray_img.dtype)
        
        result = np.stack(patches, axis=0)
        
        if self.cfg.debug:
            logger.debug(f"✅ Extracted {result.shape[0]} patches of shape {result.shape[1:]}")
        
        return result

    def get_debug_stats(self) -> Dict[str, Any]:
        """Get statistics from the last sample_centers call."""
        return self._debug_stats.copy()

    # ---------- Internal Helpers ----------

    def _prepare_binary(self, img: np.ndarray) -> np.ndarray:
        """Convert input to binary image with text=1, background=0."""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Handle different input formats
        if img.max() > 1:
            _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            bin_img = (img * 255).astype("uint8")

        # Make text=1, background=0 (assumes dark text on light background)
        bin_inv = (bin_img == 0).astype("uint8")
        
        return bin_inv

    def _connected_components(self, bin_img: np.ndarray) -> List[Dict]:
        """Find connected components in binary image."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_img, connectivity=8
        )
        
        comps = []
        for label in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[label]
            cx, cy = centroids[label]
            comps.append({
                "label": label,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "cx": int(round(cx)),
                "cy": int(round(cy)),
            })
        
        return comps

    def _filter_components(self, comps: List[Dict]) -> List[Dict]:
        """Filter components by area and aspect ratio."""
        cfg = self.cfg
        filtered = []
        
        filter_reasons = {'too_small': 0, 'too_large': 0, 'bad_aspect': 0}
        
        for c in comps:
            area = c["area"]
            w, h = c["w"], c["h"]
            
            if area < cfg.min_area:
                filter_reasons['too_small'] += 1
                continue
            if area > cfg.max_area:
                filter_reasons['too_large'] += 1
                continue
            if h == 0:
                continue
                
            aspect = w / h
            if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
                filter_reasons['bad_aspect'] += 1
                continue
            
            filtered.append(c)
        
        if self.cfg.debug:
            logger.debug(f"   Filter reasons: {filter_reasons}")
        
        return filtered

    def _filter_by_height(self, comps: List[Dict]) -> List[Dict]:
        """Filter components to keep only those with consistent height."""
        if not comps:
            return comps
        
        heights = np.array([c["h"] for c in comps], dtype=float)
        median_h = np.median(heights)
        lo = median_h * (1.0 - self.cfg.height_tol)
        hi = median_h * (1.0 + self.cfg.height_tol)
        
        filtered = [c for c in comps if lo <= c["h"] <= hi]
        
        if self.cfg.debug:
            logger.debug(f"   Height filter: median={median_h:.1f}, range=[{lo:.1f}, {hi:.1f}]")
            logger.debug(f"   Kept {len(filtered)}/{len(comps)} components")
        
        return filtered

    def _log_component_stats(self, comps: List[Dict]):
        """Log detailed component statistics for debugging."""
        if not comps:
            return
        
        areas = [c["area"] for c in comps]
        widths = [c["w"] for c in comps]
        heights = [c["h"] for c in comps]
        
        logger.debug(f"   📊 Component Statistics:")
        logger.debug(f"      Areas:   min={min(areas)}, max={max(areas)}, mean={np.mean(areas):.1f}")
        logger.debug(f"      Widths:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
        logger.debug(f"      Heights: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")


# Alias for backwards compatibility
CharacterSampler = ComponentSampler
