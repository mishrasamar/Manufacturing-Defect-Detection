"""Severity and inspection-status classification (rule-based decision layer).

Two decisions are implemented here and documented in README section 7:

* per-region severity (Minor / Major / Critical) derived from the defect
  area ratio with an optional per-category minimum severity, and
* the per-image inspection status (Pass / Fail / Review Required)
  derived from the severity mix, the cumulative defect area and the
  defect region count.

The module also provides an *optional* hook for a trained ONNX model
(:class:`ExternalModelClassifier`). The hook is working code, but no
model file ships with this project, so it stays disabled unless a user
supplies their own trained model. It must not be confused with the
rule-based pipeline, which is heuristic computer vision, not ML.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from src.defect_detector import DefectRegion
from src.utils import (
    SEVERITY_CRITICAL,
    SEVERITY_MAJOR,
    SEVERITY_MINOR,
    SEVERITY_ORDER,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_REVIEW,
)


def classify_region_severity(
    area_ratio: float, category: str, severity_cfg: Dict[str, Any]
) -> str:
    """Classify one defect region by its area ratio.

    1. ``area_ratio >= critical_area_ratio`` -> Critical;
    2. ``area_ratio >= major_area_ratio``    -> Major;
    3. otherwise Minor;
    4. categories listed in ``category_floor`` can never be classified
       below their floor (e.g. cracks are at least Major).
    """
    if area_ratio >= float(severity_cfg.get("critical_area_ratio", 0.05)):
        severity = SEVERITY_CRITICAL
    elif area_ratio >= float(severity_cfg.get("major_area_ratio", 0.01)):
        severity = SEVERITY_MAJOR
    else:
        severity = SEVERITY_MINOR
    floor = str(severity_cfg.get("category_floor", {}).get(category, ""))
    if floor and SEVERITY_ORDER.get(floor, -1) > SEVERITY_ORDER.get(severity, -1):
        severity = floor
    return severity


def classify_image_status(
    regions: Sequence["DefectRegion"],
    total_area_ratio: float,
    severity_cfg: Dict[str, Any],
) -> Tuple[str, str]:
    """Return ``(status, reason)`` for one inspected image.

    Evaluation order (first match wins):

    1. any Critical region -> Fail;
    2. cumulative defect area ratio >= ``fail_total_area_ratio`` -> Fail;
    3. defect region count >= ``fail_defect_count`` -> Fail;
    4. any Major region -> Review Required;
    5. cumulative area ratio >= ``review_total_area_ratio`` -> Review;
    6. defect count >= ``review_defect_count`` -> Review Required;
    7. otherwise Pass (zero regions -> clean pass).
    """
    counts = Counter(region.severity for region in regions)
    count = len(regions)
    if counts[SEVERITY_CRITICAL] > 0:
        return STATUS_FAIL, f"{counts[SEVERITY_CRITICAL]} critical defect region(s) present"
    fail_area = float(severity_cfg.get("fail_total_area_ratio", 0.08))
    if total_area_ratio >= fail_area:
        return (
            STATUS_FAIL,
            f"cumulative defect area ratio {total_area_ratio:.4f} >= {fail_area}",
        )
    fail_count = int(severity_cfg.get("fail_defect_count", 8))
    if count >= fail_count:
        return STATUS_FAIL, f"{count} defect regions >= fail threshold of {fail_count}"
    if counts[SEVERITY_MAJOR] > 0:
        return (
            STATUS_REVIEW,
            f"{counts[SEVERITY_MAJOR]} major defect region(s) require manual review",
        )
    review_area = float(severity_cfg.get("review_total_area_ratio", 0.015))
    if total_area_ratio >= review_area:
        return (
            STATUS_REVIEW,
            f"cumulative defect area ratio {total_area_ratio:.4f} >= {review_area}",
        )
    review_count = int(severity_cfg.get("review_defect_count", 3))
    if count >= review_count:
        return (
            STATUS_REVIEW,
            f"{count} defect regions >= review threshold of {review_count}",
        )
    if count == 0:
        return STATUS_PASS, "no defect regions detected"
    return STATUS_PASS, f"{count} minor defect region(s) below all review thresholds"


class ExternalModelClassifier:
    """Optional wrapper for a user-supplied trained ONNX model.

    A padded region crop is pushed through ``cv2.dnn``; the predicted
    label is attached to each region. This is *off by default*
    (``ml.enabled: false``) and raises a clear error when enabled without
    a valid model path and class list.
    """

    def __init__(self, ml_cfg: Dict[str, Any]):
        self.cfg = dict(ml_cfg or {})
        self.net = None
        self.class_names: List[str] = [str(c) for c in self.cfg.get("class_names", [])]
        if bool(self.cfg.get("enabled", False)):
            model_path = Path(str(self.cfg.get("model_path", "") or ""))
            if not model_path.is_file():
                raise ValueError(f"ml.model_path does not exist: {model_path}")
            if not self.class_names:
                raise ValueError("ml.enabled is true but ml.class_names is empty")
            self.net = cv2.dnn.readNet(str(model_path))

    @property
    def is_enabled(self) -> bool:
        return self.net is not None

    def predict_region(self, image_bgr: np.ndarray, bbox) -> Tuple[str, float]:
        """Return ``(label, probability)`` for one region crop."""
        if self.net is None:
            raise RuntimeError("ExternalModelClassifier is not enabled")
        x, y, w, h = [int(v) for v in bbox]
        img_h, img_w = image_bgr.shape[:2]
        pad = int(0.15 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img_w, x + w + pad), min(img_h, y + h + pad)
        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            raise ValueError("empty crop for region prediction")
        size = tuple(int(v) for v in self.cfg.get("input_size", (224, 224)))
        blob = cv2.dnn.blobFromImage(
            crop,
            scalefactor=float(self.cfg.get("scale", 1.0)),
            size=size,
            mean=tuple(float(m) for m in self.cfg.get("mean", (0.0, 0.0, 0.0))),
            swapRB=bool(self.cfg.get("swap_rb", True)),
            crop=False,
        )
        self.net.setInput(blob)
        output = np.asarray(self.net.forward()).reshape(-1)
        probs = self._softmax(output)
        best = int(np.argmax(probs))
        label = self.class_names[best] if best < len(self.class_names) else f"class_{best}"
        return label, float(probs[best])

    def annotate_regions(self, image_bgr: np.ndarray, regions) -> List["DefectRegion"]:
        """Run the model over regions; attach predictions without inventing data."""
        from src.defect_detector import CATEGORY_PRIORITY

        for region in regions:
            try:
                label, prob = self.predict_region(image_bgr, region.bbox)
            except Exception:
                continue  # keep the heuristic result when the model call fails
            region.method = f"{region.method} + ML({label}:{prob:.2f})"
            if label in CATEGORY_PRIORITY:
                region.category = label
        return regions

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        exps = np.exp(scores - float(np.max(scores)))
        return exps / float(exps.sum())


def describe_detection_rules(settings: Dict[str, Any]) -> List[str]:
    """Human-readable summary of the active detection rules (for reports)."""
    detection = settings.get("detection", {})
    lines: List[str] = []
    spot = detection.get("spot", {})
    if spot.get("enabled", True):
        lines.append(
            "Spots: deviation from a median-filtered background "
            f"(background kernel {spot.get('background_blur_ksize', 31)} px, minimum contrast "
            f"{spot.get('min_contrast', 18)}, area {spot.get('min_area', 20)}-"
            f"{spot.get('max_area', 3000)} px, circularity >= {spot.get('min_circularity', 0.25)}, "
            f"solidity >= {spot.get('min_solidity', 0.5)})."
        )
    scratch = detection.get("scratch", {})
    if scratch.get("enabled", True):
        lines.append(
            "Scratches: Canny edges "
            f"({scratch.get('canny_low', 60)}/{scratch.get('canny_high', 170)}) joined by anisotropic "
            f"closing (kernel {scratch.get('connect_kernel', 15)} px); accepted when length >= "
            f"{scratch.get('min_length', 40)} px, thickness <= {scratch.get('max_thickness', 7)} px and "
            f"aspect ratio >= {scratch.get('min_aspect_ratio', 5)}."
        )
    crack = detection.get("crack", {})
    if crack.get("enabled", True):
        lines.append(
            "Cracks: Canny edges "
            f"({crack.get('canny_low', 50)}/{crack.get('canny_high', 150)}) dilated by "
            f"{crack.get('dilate_ksize', 3)} px; accepted when solidity <= "
            f"{crack.get('max_solidity', 0.65)}, aspect ratio >= {crack.get('min_aspect_ratio', 3)} "
            f"and area >= {crack.get('min_area', 60)} px."
        )
    dent = detection.get("dent", {})
    if dent.get("enabled", True):
        lines.append(
            "Dents: broad irregular deviations from the global surface level "
            f"(minimum contrast {dent.get('min_contrast', 12)}, area >= "
            f"{float(dent.get('min_area_ratio', 0.004)):.1%} of the image, circularity <= "
            f"{dent.get('max_circularity', 0.65)}, solidity <= {dent.get('max_solidity', 0.92)})."
        )
    disc = detection.get("discoloration", {})
    if disc.get("enabled", True):
        lines.append(
            "Discoloration: HSV deviation from the image median (saturation offset > "
            f"{disc.get('saturation_offset', 30)} or hue distance > {disc.get('hue_offset', 18)}, "
            f"with saturation > {disc.get('min_abs_saturation', 45)}; area >= "
            f"{disc.get('min_area', 150)} px)."
        )
    reference = detection.get("reference", {})
    if reference.get("enabled", True):
        lines.append(
            "Reference difference: absolute difference against the configured reference image "
            f"(minimum contrast {reference.get('min_contrast', 25)}, area >= "
            f"{reference.get('min_area', 60)} px); active only when a reference is supplied."
        )
    return lines


def describe_severity_rules(settings: Dict[str, Any]) -> List[str]:
    """Human-readable summary of the severity / status rules (for reports)."""
    severity = settings.get("severity", {})
    major = float(severity.get("major_area_ratio", 0.01))
    critical = float(severity.get("critical_area_ratio", 0.05))
    floor = severity.get("category_floor", {})
    floor_text = ", ".join(f"{key} >= {value}" for key, value in floor.items()) or "none"
    return [
        "Region severity: Minor below " f"{major:.1%} of the image area, Major at or above "
        f"{major:.1%}, Critical at or above {critical:.1%}; category floors: {floor_text}.",
        "Fail when: any Critical region, cumulative defect area >= "
        f"{float(severity.get('fail_total_area_ratio', 0.08)):.1%} of the image, or region count >= "
        f"{int(severity.get('fail_defect_count', 8))}.",
        "Review Required when: any Major region, cumulative defect area >= "
        f"{float(severity.get('review_total_area_ratio', 0.015)):.1%}, or region count >= "
        f"{int(severity.get('review_defect_count', 3))}; otherwise Pass.",
    ]

