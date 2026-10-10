"""Image preprocessing pipeline built on OpenCV.

Each product image is prepared for the rule-based defect detectors:

1. optional downscaling to ``max_display_width`` (aspect preserved),
2. grayscale conversion,
3. noise reduction (median or Gaussian blur),
4. optional percentile normalisation / CLAHE (both are disabled by
   default because they also amplify sensor noise), and
5. Canny edge detection used by the scratch detector.

Every parameter is read from ``settings.yaml``; nothing is hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    """Output of :func:`preprocess` for one image."""

    original_bgr: np.ndarray
    resized_bgr: np.ndarray
    gray: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    normalized: np.ndarray
    edges: np.ndarray
    scale: float
    original_size: Tuple[int, int]   # (width, height)
    processed_size: Tuple[int, int]  # (width, height)


def resize_keep_aspect(image: np.ndarray, max_width: int) -> Tuple[np.ndarray, float]:
    """Scale ``image`` down so its width <= ``max_width``. Never upscales."""
    height, width = image.shape[:2]
    if max_width <= 0 or width <= max_width:
        return image.copy(), 1.0
    scale = max_width / float(width)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def denoise_gray(gray: np.ndarray, method: str, ksize: int) -> np.ndarray:
    """Median (default) or Gaussian blur; the kernel size is forced odd."""
    ksize = max(3, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    if str(method).lower() == "gaussian":
        return cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return cv2.medianBlur(gray, ksize)


def normalize_intensity(gray: np.ndarray, percentiles=(1, 99)) -> np.ndarray:
    """Percentile contrast stretch to the full 0-255 range."""
    lo, hi = np.percentile(gray, list(percentiles))
    if float(hi) - float(lo) < 1:
        return gray.copy()
    stretched = (gray.astype(np.float32) - float(lo)) * 255.0 / float(hi - lo)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit=2.0, tile_grid=8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation."""
    grid = max(2, int(tile_grid))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid, grid))
    return clahe.apply(gray)


def compute_edges(gray: np.ndarray, canny_low: int, canny_high: int) -> np.ndarray:
    """Canny edge map with sane threshold ordering."""
    low = max(1, int(canny_low))
    high = max(low + 1, int(canny_high))
    return cv2.Canny(gray, low, high)


def preprocess(image_bgr: np.ndarray, settings: Dict[str, Any]) -> PreprocessResult:
    """Run the full preprocessing chain for one image."""
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        raise ValueError("preprocess() received an empty image")
    cfg = settings.get("processing", {})
    resized, scale = resize_keep_aspect(image_bgr, int(cfg.get("max_display_width", 800)))
    gray = to_gray(resized)
    denoised = denoise_gray(
        gray, cfg.get("denoise_method", "median"), int(cfg.get("denoise_ksize", 5))
    )
    enhanced = denoised
    if cfg.get("use_clahe"):
        enhanced = apply_clahe(
            denoised, cfg.get("clahe_clip_limit", 2.0), cfg.get("clahe_tile_grid", 8)
        )
    normalized = denoised
    if cfg.get("normalize"):
        normalized = normalize_intensity(
            denoised, tuple(cfg.get("normalize_percentiles", (1, 99)))
        )
    scratch_cfg = settings.get("detection", {}).get("scratch", {})
    # Edges are computed from the raw grayscale with only a light Gaussian
    # smoothing: the stronger median denoise erases thin (1-3 px) lines.
    edge_source = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = compute_edges(
        edge_source, scratch_cfg.get("canny_low", 45), scratch_cfg.get("canny_high", 140)
    )
    return PreprocessResult(
        original_bgr=image_bgr,
        resized_bgr=resized,
        gray=gray,
        denoised=denoised,
        enhanced=enhanced,
        normalized=normalized,
        edges=edges,
        scale=scale,
        original_size=(image_bgr.shape[1], image_bgr.shape[0]),
        processed_size=(resized.shape[1], resized.shape[0]),
    )
