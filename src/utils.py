"""Shared utilities for the manufacturing defect detection project.

Contains configuration loading, logging helpers, filesystem helpers and
the image-annotation drawing routine used by the pipeline and the
Streamlit dashboard.
"""
from __future__ import annotations

import copy
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_SUBDIRS: Tuple[str, ...] = (
    "annotated_images",
    "originals",
    "csv",
    "excel",
    "reports",
    "logs",
)

# -------------------------------------------------------------------------- #
# Status and severity constants (single source of truth for string values)
# -------------------------------------------------------------------------- #
STATUS_PASS = "Pass"
STATUS_FAIL = "Fail"
STATUS_REVIEW = "Review Required"
STATUS_ERROR = "Error"
ALL_STATUSES = (STATUS_PASS, STATUS_FAIL, STATUS_REVIEW, STATUS_ERROR)

SEVERITY_NONE = "None"
SEVERITY_MINOR = "Minor"
SEVERITY_MAJOR = "Major"
SEVERITY_CRITICAL = "Critical"
SEVERITY_ORDER = {
    SEVERITY_NONE: -1,
    SEVERITY_MINOR: 0,
    SEVERITY_MAJOR: 1,
    SEVERITY_CRITICAL: 2,
}


def default_settings() -> Dict[str, Any]:
    """Built-in defaults, mirrored by ``config/settings.yaml``."""
    return {
        "processing": {
            "max_display_width": 800,
            "denoise_method": "median",
            "denoise_ksize": 5,
            "normalize": False,
            "normalize_percentiles": [1, 99],
            "use_clahe": False,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": 8,
        },
        "detection": {
            "max_regions_per_image": 200,
            "dedupe_iou": 0.6,
            "min_background_std": 2.0,
            "spot": {"enabled": True, "background_blur_ksize": 31, "min_contrast": 18,
                     "min_area": 20, "max_area": 3000, "min_circularity": 0.25,
                     "min_solidity": 0.5},
            "scratch": {"enabled": True, "canny_low": 45, "canny_high": 140,
                        "connect_kernel": 15, "min_aspect_ratio": 5.0,
                        "max_thickness": 9, "min_length": 40, "min_area": 40},
            "crack": {"enabled": True, "canny_low": 40, "canny_high": 130,
                      "dilate_ksize": 3, "min_aspect_ratio": 3.0,
                      "max_solidity": 0.65, "min_area": 60},
            "dent": {"enabled": True, "min_contrast": 12,
                     "min_area_ratio": 0.004, "max_circularity": 0.65,
                     "max_solidity": 0.92},
            "discoloration": {"enabled": True, "min_area": 150, "saturation_offset": 30,
                              "hue_offset": 18, "min_abs_saturation": 45},
            "reference": {"enabled": True, "min_area": 60, "blur_ksize": 5,
                          "min_contrast": 25},
        },
        "severity": {
            "major_area_ratio": 0.01,
            "critical_area_ratio": 0.05,
            "category_floor": {"Crack": "Major"},
            "fail_total_area_ratio": 0.08,
            "fail_defect_count": 8,
            "review_total_area_ratio": 0.015,
            "review_defect_count": 3,
        },
        "ml": {
            "enabled": False,
            "model_path": "",
            "input_size": [224, 224],
            "scale": 0.00392,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "swap_rb": True,
            "class_names": [],
        },
        "report": {"max_table_rows": 25, "max_image_examples": 4},
        "output": {"root": "outputs"},
    }


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` on top of ``base`` (new dict returned)."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load ``config/settings.yaml`` merged over the built-in defaults."""
    settings_path = Path(path) if path else PROJECT_ROOT / "config" / "settings.yaml"
    settings = default_settings()
    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as handle:
            user_settings = yaml.safe_load(handle) or {}
        settings = deep_merge(settings, user_settings)
    else:  # pragma: no cover - defensive fallback
        logging.getLogger(__name__).warning(
            "Settings file not found at %s - using built-in defaults", settings_path
        )
    return settings


# -------------------------------------------------------------------------- #
# Filesystem helpers
# -------------------------------------------------------------------------- #
def ensure_directories(
    base: Path, subdirs: Sequence[str] = OUTPUT_SUBDIRS
) -> Dict[str, Path]:
    """Create ``base`` and its output sub-directories; return their paths."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {"root": base}
    for name in subdirs:
        path = base / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths


def create_batch_id(prefix: str = "BATCH") -> str:
    """Timestamped batch identifier, e.g. ``BATCH_20251014_093015``."""
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"


def unique_batch_dir(output_root: Path, batch_id: str) -> Path:
    """Create an unused directory ``output_root/batch_id[_N]`` (no overwrite)."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate = output_root / batch_id
    counter = 1
    while candidate.exists():
        candidate = output_root / f"{batch_id}_{counter}"
        counter += 1
    candidate.mkdir()
    return candidate


def setup_logger(
    name: str, log_file: Optional[Path] = None, level: int = logging.INFO
) -> logging.Logger:
    """Logger with a console handler and (optionally) one UTF-8 file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(console)
    if log_file is not None:
        log_file = Path(log_file)
        already = any(
            isinstance(handler, logging.FileHandler)
            and str(getattr(handler, "baseFilename", "")) == str(log_file)
            for handler in logger.handlers
        )
        if not already:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
            )
            logger.addHandler(file_handler)
    return logger


# -------------------------------------------------------------------------- #
# Geometry helpers
# -------------------------------------------------------------------------- #
def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = float(aw) * float(ah) + float(bw) * float(bh) - inter
    return inter / union if union > 0 else 0.0


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    return max(low, min(high, value))


# -------------------------------------------------------------------------- #
# Annotation drawing
# -------------------------------------------------------------------------- #
SEVERITY_COLORS = {
    SEVERITY_MINOR: (0, 220, 220),    # yellow (BGR)
    SEVERITY_MAJOR: (0, 165, 255),    # orange
    SEVERITY_CRITICAL: (0, 0, 255),   # red
}

STATUS_COLORS = {
    STATUS_PASS: (60, 170, 60),       # green
    STATUS_FAIL: (40, 40, 220),       # red
    STATUS_REVIEW: (30, 190, 255),    # amber
    STATUS_ERROR: (140, 140, 140),    # grey
}


def annotate_image(
    image_bgr: np.ndarray,
    regions: Sequence[Any],
    status: str,
    extra_text: str = "",
) -> np.ndarray:
    """Draw the status banner, bounding boxes and severity labels.

    ``regions`` are :class:`src.defect_detector.DefectRegion` objects; the
    function only relies on their ``bbox``, ``category`` and ``severity``
    attributes so it stays reusable.
    """
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    scale = max(0.55, width / 1000.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48 * scale
    thickness = max(1, int(round(1.3 * scale)))

    banner_height = int(34 * scale)
    banner_color = STATUS_COLORS.get(status, (160, 160, 160))
    cv2.rectangle(canvas, (0, 0), (width, banner_height), banner_color, -1)
    label = f"RESULT: {status.upper()}  |  DEFECT REGIONS: {len(regions)}"
    if extra_text:
        label += f"  |  {extra_text}"
    origin = (8, banner_height - 10)
    cv2.putText(canvas, label, origin, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, label, origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    for region in regions:
        x, y, w, h = [int(v) for v in region.bbox]
        x, y = max(0, x), max(0, y)
        w, h = min(width - x, w), min(height - y, h)
        if w <= 0 or h <= 0:
            continue
        color = SEVERITY_COLORS.get(region.severity, (255, 255, 255))
        cv2.rectangle(
            canvas, (x, y), (x + w, y + h), color, max(1, int(round(2 * scale)))
        )
        tag = f"{region.category} ({region.severity})"
        (text_w, text_h), _ = cv2.getTextSize(tag, font, font_scale, thickness)
        text_y = y - 4
        if text_y - text_h < banner_height + 2:
            text_y = y + text_h + 6
        text_x = max(0, x)
        cv2.rectangle(
            canvas,
            (text_x, text_y - text_h - 3),
            (text_x + text_w + 4, text_y + 3),
            (30, 30, 30),
            -1,
        )
        cv2.putText(
            canvas, tag, (text_x + 2, text_y), font, font_scale, color, thickness, cv2.LINE_AA
        )
    return canvas


