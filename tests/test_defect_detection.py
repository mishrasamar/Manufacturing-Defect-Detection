"""Unit tests for the rule-based defect detectors and status classification."""
import cv2
import numpy as np

from src.defect_classifier import classify_image_status, classify_region_severity
from src.defect_detector import DefectRegion, RuleBasedDefectDetector
from src.preprocessing import preprocess
from src.utils import default_settings


def _surface(w=480, h=360, seed=1, noise=6.0):
    rng = np.random.default_rng(seed)
    return np.clip(178 + rng.normal(0.0, noise, (h, w)), 0, 255).astype(np.uint8)


def _detect(gray, settings=None, reference_bgr=None):
    settings = settings or default_settings()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
    pre = preprocess(bgr, settings)
    return RuleBasedDefectDetector(settings).detect(pre, reference_bgr)


def test_clean_surface_has_no_regions():
    assert _detect(_surface(seed=11)) == []


def test_dark_spots_detected():
    img = _surface(seed=12)
    for cx, cy, radius in [(120, 100, 14), (300, 180, 12), (400, 90, 10)]:
        cv2.circle(img, (cx, cy), radius, 120, -1)
    regions = _detect(cv2.GaussianBlur(img, (3, 3), 0))
    spots = [r for r in regions if r.category == "Spot"]
    assert len(spots) >= 3


def test_straight_scratch_detected():
    img = _surface(seed=13)
    cv2.line(img, (80, 300), (420, 300), 125, 2)
    regions = _detect(cv2.GaussianBlur(img, (3, 3), 0))
    assert any(r.category == "Scratch" for r in regions)


def test_jagged_crack_detected():
    img = _surface(seed=14)
    points = [(60, 150), (110, 120), (160, 165), (210, 125), (260, 170), (310, 130), (360, 160), (410, 135)]
    for p1, p2 in zip(points[:-1], points[1:]):
        cv2.line(img, p1, p2, 115, 2)
    regions = _detect(cv2.GaussianBlur(img, (3, 3), 0))
    assert any(r.category == "Crack" for r in regions)


def test_irregular_dent_detected():
    # An L-shaped depression: irregular enough for the dent shape filters.
    img = _surface(seed=15)
    cv2.rectangle(img, (140, 120), (340, 180), 120, -1)
    cv2.rectangle(img, (140, 120), (200, 280), 120, -1)
    regions = _detect(cv2.GaussianBlur(img, (3, 3), 0))
    dents = [r for r in regions if r.category == "Dent"]
    assert dents and max(r.area for r in dents) > 4000


def test_reference_difference_detected():
    settings = default_settings()
    base = _surface(seed=16)
    modified = base.copy()
    cv2.circle(modified, (240, 180), 22, 110, -1)
    reference = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    regions = _detect(cv2.cvtColor(modified, cv2.COLOR_GRAY2BGR), settings, reference)
    assert any(r.category == "Reference Difference" for r in regions)


def test_disabled_detectors_produce_no_regions():
    settings = default_settings()
    for key in ("spot", "scratch", "crack", "dent", "discoloration", "reference"):
        settings["detection"][key]["enabled"] = False
    img = _surface(seed=17)
    cv2.circle(img, (240, 180), 25, 110, -1)
    cv2.line(img, (60, 300), (420, 300), 120, 2)
    assert _detect(cv2.GaussianBlur(img, (3, 3), 0), settings) == []


def test_region_severity_classification(settings):
    severity_cfg = settings["severity"]
    assert classify_region_severity(0.001, "Spot", severity_cfg) == "Minor"
    assert classify_region_severity(0.02, "Spot", severity_cfg) == "Major"
    assert classify_region_severity(0.06, "Spot", severity_cfg) == "Critical"
    assert classify_region_severity(0.001, "Crack", severity_cfg) == "Major"  # category floor


def test_image_status_rules(settings):
    severity_cfg = settings["severity"]

    def region(severity, ratio=0.001):
        return DefectRegion("Spot", (0, 0, 10, 10), 100, ratio, 0.5, "test", severity)

    assert classify_image_status([], 0.0, severity_cfg)[0] == "Pass"
    assert classify_image_status([region("Critical", 0.06)], 0.06, severity_cfg)[0] == "Fail"
    assert classify_image_status([region("Major")], 0.001, severity_cfg)[0] == "Review Required"
    minors = [region("Minor") for _ in range(8)]
    assert classify_image_status(minors, 0.008, severity_cfg)[0] == "Fail"  # count rule
