"""Rule-based (heuristic) defect detection on preprocessed images.

The detector combines six independent, explainable OpenCV methods:

===============  ==========================================================
Category          Principle
===============  ==========================================================
Spot              deviation from a median-filtered background, Otsu-like
                  threshold with an absolute contrast floor, blob filters
Scratch           Canny edges, anisotropic closing, thin elongated shapes
Crack             Canny edges, dilated contours with low convex solidity
Dent              broad irregular regions deviating from the background
Discoloration     saturation / hue deviation from the image median (HSV)
Reference         absolute difference against an optional reference image
===============  ==========================================================

Every threshold lives in ``config/settings.yaml``. These methods are
heuristics, not a trained classifier - see README section 7 for the
accuracy limitations that come with that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.preprocessing import compute_edges
from src.utils import box_iou

CATEGORY_PRIORITY: Tuple[str, ...] = (
    "Spot",
    "Crack",
    "Scratch",
    "Dent",
    "Discoloration",
    "Reference Difference",
)


@dataclass
class DefectRegion:
    """One detected defect region (in processed-image coordinates)."""

    category: str
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    area: int
    area_ratio: float
    score: float
    method: str
    severity: str = "None"
    note: str = "heuristic score, not an ML probability"


def _odd(value: int, minimum: int = 3) -> int:
    """Return the next odd integer >= max(value, minimum)."""
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def circularity(contour: np.ndarray) -> float:
    """Isoperimetric quotient ``4*pi*Area / Perimeter^2`` clipped to [0, 1]."""
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter <= 0 or area <= 0:
        return 0.0
    return min(1.0, 4 * math.pi * area / (perimeter * perimeter))


def solidity(contour: np.ndarray) -> float:
    """Contour area divided by its convex-hull area, clipped to [0, 1]."""
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area <= 0:
        return 0.0
    return min(1.0, area / hull_area)


def _rect_shape(contour: np.ndarray) -> Tuple[float, float]:
    """Return ``(length, thickness)`` of the minimum-area rectangle."""
    (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(contour)
    return max(rw, rh), min(rw, rh)


class RuleBasedDefectDetector:
    """Applies the configured rule-based detectors to one preprocessed image."""

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.cfg = settings.get("detection", {})

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def detect(
        self, pre: Any, reference_bgr: Optional[np.ndarray] = None
    ) -> List[DefectRegion]:
        """Run every enabled detector and return the merged, deduped regions."""
        image_area = int(pre.processed_size[0]) * int(pre.processed_size[1])
        spot_cfg = self.cfg.get("spot", {})
        dent_cfg = self.cfg.get("dent", {})
        scratch_cfg = self.cfg.get("scratch", {})
        crack_cfg = self.cfg.get("crack", {})
        disc_cfg = self.cfg.get("discoloration", {})
        ref_cfg = self.cfg.get("reference", {})

        # Statistical flatness gate: skip blob detectors on uniform images so
        # pure sensor noise cannot produce phantom defects.
        flat = self._background_std(pre.denoised) < float(
            self.cfg.get("min_background_std", 2.0)
        )

        regions: List[DefectRegion] = []
        if spot_cfg.get("enabled", True) and not flat:
            regions += self._detect_spots(pre, spot_cfg, image_area)
        if crack_cfg.get("enabled", True):
            regions += self._detect_cracks(pre, crack_cfg, image_area)
        if scratch_cfg.get("enabled", True):
            regions += self._detect_scratches(pre, scratch_cfg, image_area)
        if dent_cfg.get("enabled", True) and not flat:
            regions += self._detect_dents(pre, dent_cfg, image_area)
        if disc_cfg.get("enabled", True) and getattr(pre.resized_bgr, "ndim", 2) == 3:
            regions += self._detect_discoloration(pre, disc_cfg, image_area)
        if reference_bgr is not None and ref_cfg.get("enabled", True):
            regions += self._detect_reference_differences(
                pre, reference_bgr, ref_cfg, image_area
            )

        regions = self._dedupe(regions)
        max_regions = int(self.cfg.get("max_regions_per_image", 200))
        if len(regions) > max_regions:
            regions = sorted(
                regions, key=lambda r: r.score * r.area, reverse=True
            )[:max_regions]
            regions = self._sort_regions(regions)
        return regions

    # ------------------------------------------------------------------ #
    # shared helpers
    # ------------------------------------------------------------------ #
    def _background_std(self, gray: np.ndarray) -> float:
        """Std-dev of |image - global surface level| (the flatness metric).

        Uses the global level (not a local median) so that broad, smooth
        deviations are still visible to the gate.
        """
        background_level = int(round(float(np.median(gray))))
        return float(cv2.absdiff(gray, np.full_like(gray, background_level)).std())

    def _bg_kernel(self, gray: np.ndarray, requested) -> int:
        """A median-blur kernel that always fits the image size."""
        return _odd(min(int(requested), max(3, min(gray.shape[:2]) // 2)))

    def _otsu_mask(self, diff: np.ndarray, contrast_floor: float) -> np.ndarray:
        """Binary mask of ``diff`` above max(Otsu threshold, contrast floor)."""
        otsu_t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(float(otsu_t), float(contrast_floor))
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return mask

    @staticmethod
    def _masked_mean(intensity: np.ndarray, contour: np.ndarray) -> float:
        """Mean intensity of ``intensity`` inside one contour."""
        mask = np.zeros(intensity.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean = cv2.mean(intensity, mask=mask)
        return float(mean[0]) if isinstance(mean, tuple) else float(mean)

    def _dedupe(self, regions: List[DefectRegion]) -> List[DefectRegion]:
        """Drop later regions that overlap an already-kept region (IoU)."""
        threshold = float(self.cfg.get("dedupe_iou", 0.6))
        kept: List[DefectRegion] = []
        for region in self._sort_regions(regions):
            if all(box_iou(region.bbox, other.bbox) < threshold for other in kept):
                kept.append(region)
        return kept

    @staticmethod
    def _sort_regions(regions: List[DefectRegion]) -> List[DefectRegion]:
        """Stable order: category priority first, then larger area first."""
        priority = {name: idx for idx, name in enumerate(CATEGORY_PRIORITY)}
        return sorted(regions, key=lambda r: (priority.get(r.category, 99), -r.area))

    # ------------------------------------------------------------------ #
    # spot detector (spots / stains: blobs deviating from the background)
    # ------------------------------------------------------------------ #
    def _detect_spots(
        self, pre: Any, cfg: Dict[str, Any], image_area: int
    ) -> List[DefectRegion]:
        gray = pre.denoised
        background = cv2.medianBlur(gray, self._bg_kernel(gray, cfg.get("background_blur_ksize", 31)))
        diff = cv2.absdiff(gray, background)
        mask = self._otsu_mask(diff, cfg.get("min_contrast", 18))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return self._mask_to_regions(
            mask,
            diff,
            "Spot",
            "background-difference + blob filters",
            image_area,
            min_area=int(cfg.get("min_area", 20)),
            max_area=int(cfg.get("max_area", 3000)),
            min_circularity=float(cfg.get("min_circularity", 0.25)),
            min_solidity=float(cfg.get("min_solidity", 0.5)),
            score_divisor=max(1.0, float(cfg.get("min_contrast", 18)) * 4.0),
        )

    # ------------------------------------------------------------------ #
    # shared mask -> region conversion with geometric filters
    # ------------------------------------------------------------------ #
    def _mask_to_regions(
        self,
        mask: np.ndarray,
        intensity: np.ndarray,
        category: str,
        method: str,
        image_area: int,
        *,
        min_area: int,
        max_area: Optional[int] = None,
        min_circularity: Optional[float] = None,
        max_circularity: Optional[float] = None,
        min_solidity: Optional[float] = None,
        max_solidity: Optional[float] = None,
        score_divisor: float = 100.0,
    ) -> List[DefectRegion]:
        """Convert contours of ``mask`` into filtered DefectRegion objects.

        A region is rejected when its area or any configured geometry
        bound (circularity / solidity) is violated. The severity score is
        the mean deviation intensity normalised by ``score_divisor``.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: List[DefectRegion] = []
        for contour in contours:
            area = int(round(cv2.contourArea(contour)))
            if area < min_area or (max_area is not None and area > max_area):
                continue
            circ = circularity(contour)
            solid = solidity(contour)
            if min_circularity is not None and circ < min_circularity:
                continue
            if max_circularity is not None and circ > max_circularity:
                continue
            if min_solidity is not None and solid < min_solidity:
                continue
            if max_solidity is not None and solid > max_solidity:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            mean_level = self._masked_mean(intensity, contour)
            score = float(np.clip(mean_level / max(score_divisor, 1e-6), 0.05, 1.0))
            regions.append(
                DefectRegion(
                    category=category,
                    bbox=(int(x), int(y), int(w), int(h)),
                    area=area,
                    area_ratio=area / float(max(image_area, 1)),
                    score=score,
                    method=method,
                )
            )
        return regions

    # ------------------------------------------------------------------ #
    # dent detector (broad irregular deviations from the background)
    # ------------------------------------------------------------------ #
    def _detect_dents(self, pre, cfg, image_area):
        # Dents are BROAD deviations: a local median background would
        # self-cancel them (defect size >= kernel size), so the comparison
        # is made against the robust global surface level instead.
        gray = pre.denoised
        background_level = int(round(float(np.median(gray))))
        diff = cv2.absdiff(gray, np.full_like(gray, background_level))
        mask = self._otsu_mask(diff, cfg.get("min_contrast", 12))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        min_area = max(int(round(float(cfg.get("min_area_ratio", 0.004)) * image_area)), 1)
        return self._mask_to_regions(
            mask,
            diff,
            "Dent",
            "broad irregular deviation from the global surface level",
            image_area,
            min_area=min_area,
            max_circularity=float(cfg.get("max_circularity", 0.65)),
            max_solidity=float(cfg.get("max_solidity", 0.92)),
            score_divisor=max(1.0, float(cfg.get("min_contrast", 12)) * 4.0),
        )

    # ------------------------------------------------------------------ #
    # crack detector (jagged edge structures with low convex solidity)
    # ------------------------------------------------------------------ #
    def _detect_cracks(self, pre, cfg, image_area):
        # Edges come from a lightly blurred grayscale image: the median
        # denoise erases thin (1-3 px) crack lines entirely.
        edge_source = cv2.GaussianBlur(pre.gray, (3, 3), 0)
        edges = compute_edges(edge_source, cfg.get("canny_low", 40), cfg.get("canny_high", 130))
        dilate_k = max(1, int(cfg.get("dilate_ksize", 3)))
        mask = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        min_area = float(cfg.get("min_area", 60))
        max_solidity = float(cfg.get("max_solidity", 0.65))
        min_aspect = float(cfg.get("min_aspect_ratio", 3.0))
        regions: List[DefectRegion] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = int(round(cv2.contourArea(contour)))
            if area < min_area:
                continue
            solid = solidity(contour)
            if solid > max_solidity:
                continue
            length, thickness = _rect_shape(contour)
            if thickness <= 0 or length / thickness < min_aspect:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            score = float(np.clip((1.0 - solid) * min(1.0, (length / thickness) / 10.0), 0.05, 1.0))
            regions.append(
                DefectRegion(
                    category="Crack",
                    bbox=(int(x), int(y), int(w), int(h)),
                    area=area,
                    area_ratio=area / float(max(image_area, 1)),
                    score=score,
                    method="Canny edges + low-solidity contour filter",
                )
            )
        return regions

    # ------------------------------------------------------------------ #
    # scratch detector (thin elongated marks from Canny edges)
    # ------------------------------------------------------------------ #
    def _detect_scratches(self, pre, cfg, image_area):
        edges = pre.edges  # Canny edges computed with the scratch thresholds
        connect_kernel = _odd(int(cfg.get("connect_kernel", 15)))
        horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (connect_kernel, 1))
        vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, connect_kernel))
        joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal)
        joined = cv2.bitwise_or(joined, cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical))
        mask = cv2.morphologyEx(joined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        min_length = float(cfg.get("min_length", 40))
        min_aspect = float(cfg.get("min_aspect_ratio", 5))
        max_thickness = float(cfg.get("max_thickness", 7))
        min_area = float(cfg.get("min_area", 40))
        regions: List[DefectRegion] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = int(round(cv2.contourArea(contour)))
            if area < min_area:
                continue
            length, thickness = _rect_shape(contour)
            if thickness <= 0 or length < min_length or thickness > max_thickness:
                continue
            if length / thickness < min_aspect:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            score = float(np.clip((length / thickness) / 25.0, 0.05, 1.0))
            regions.append(
                DefectRegion(
                    category="Scratch",
                    bbox=(int(x), int(y), int(w), int(h)),
                    area=area,
                    area_ratio=area / float(max(image_area, 1)),
                    score=score,
                    method="Canny edges + anisotropic closing",
                )
            )
        return regions

    # ------------------------------------------------------------------ #
    # discoloration detector (HSV deviation from the image median)
    # ------------------------------------------------------------------ #
    def _detect_discoloration(self, pre, cfg, image_area):
        hue, saturation, _ = cv2.split(cv2.cvtColor(pre.resized_bgr, cv2.COLOR_BGR2HSV))
        sat_f = saturation.astype(np.float32)
        median_sat = float(np.median(sat_f))
        sat_deviation = np.abs(sat_f - median_sat)

        sat_mask = (sat_f > float(cfg.get("min_abs_saturation", 45))) & (
            sat_deviation > float(cfg.get("saturation_offset", 30))
        )

        hue_f = hue.astype(np.float32)
        median_hue = float(np.median(hue_f))
        hue_distance = np.abs(hue_f - median_hue)
        hue_distance = np.minimum(hue_distance, 180.0 - hue_distance)  # circular
        hue_mask = (sat_f > float(cfg.get("min_abs_saturation", 45))) & (
            hue_distance > float(cfg.get("hue_offset", 18))
        )

        mask = cv2.bitwise_or(sat_mask, hue_mask).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        intensity = np.clip(sat_deviation, 0, 255).astype(np.uint8)
        return self._mask_to_regions(
            mask,
            intensity,
            "Discoloration",
            "HSV saturation/hue deviation",
            image_area,
            min_area=int(cfg.get("min_area", 150)),
            score_divisor=90.0,
        )

    # ------------------------------------------------------------------ #
    # reference detector (difference against a reference image)
    # ------------------------------------------------------------------ #
    def _detect_reference_differences(self, pre, reference_bgr, cfg, image_area):
        target_size = (int(pre.processed_size[0]), int(pre.processed_size[1]))
        reference = cv2.resize(reference_bgr, target_size, interpolation=cv2.INTER_AREA)
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        blur_k = _odd(int(cfg.get("blur_ksize", 5)))
        ref_blur = cv2.GaussianBlur(ref_gray, (blur_k, blur_k), 0)
        img_blur = cv2.GaussianBlur(pre.denoised, (blur_k, blur_k), 0)
        diff = cv2.absdiff(img_blur, ref_blur)
        mask = self._otsu_mask(diff, cfg.get("min_contrast", 25))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return self._mask_to_regions(
            mask,
            diff,
            "Reference Difference",
            "absolute difference vs reference image",
            image_area,
            min_area=int(cfg.get("min_area", 60)),
            score_divisor=max(1.0, float(cfg.get("min_contrast", 25)) * 3.0),
        )


def detect_defects(pre, settings, reference_bgr=None) -> List[DefectRegion]:
    """Convenience wrapper around :class:`RuleBasedDefectDetector`."""
    return RuleBasedDefectDetector(settings).detect(pre, reference_bgr)




