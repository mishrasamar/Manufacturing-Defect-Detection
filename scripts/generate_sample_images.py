"""Generate synthetic sample product images for testing the pipeline.

Creates one image per defect category plus a clean reference image and a
deliberately corrupted file, so the whole pipeline - including error
handling - can be demonstrated end to end:

    clean_01.png          no defects (must produce status Pass)
    spot_01.png           three dark round spots
    scratch_01.png        three straight scratches
    crack_01.png          one jagged crack
    dent_01.png           one broad irregular dent
    discoloration_01.png  one rust-like colour patch
    mixed_02.png          several defect types at once
    corrupted_sample.jpg  invalid bytes (must be reported as unreadable)

Usage:  .venv\\Scripts\\python.exe scripts\\generate_sample_images.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_images"
BASE_LEVEL = 178


def _base_surface(w: int = 800, h: int = 600, seed: int = 0) -> np.ndarray:
    """Brushed-metal-like surface: gradient + noise + faint horizontal lines."""
    rng = np.random.default_rng(seed)
    rows = np.arange(h).reshape(-1, 1)
    cols = np.arange(w).reshape(1, -1)
    surface = (
        BASE_LEVEL
        - 12.0 * (rows / float(h))
        + 2.5 * np.sin(cols / 9.0)
        + rng.normal(0.0, 6.0, (h, w))
    )
    return np.clip(surface, 0, 255).astype(np.uint8)


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _soften(img: np.ndarray, k: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), 0)


def draw_spots(gray: np.ndarray, rng, count: int = 3) -> np.ndarray:
    """Round dark spots (stains / pits)."""
    h, w = gray.shape
    img = gray.copy()
    for _ in range(count):
        cx = int(rng.integers(70, w - 70))
        cy = int(rng.integers(70, h - 70))
        radius = int(rng.integers(10, 17))
        depth = int(rng.integers(45, 70))
        cv2.circle(img, (cx, cy), radius, BASE_LEVEL - depth, -1)
    return _soften(img)


def draw_scratches(gray: np.ndarray, rng, count: int = 3) -> np.ndarray:
    """Thin elongated scratches."""
    h, w = gray.shape
    img = gray.copy()
    for _ in range(count):
        x1 = int(rng.integers(60, max(61, w - 320)))
        y1 = int(rng.integers(60, h - 60))
        length = int(rng.integers(150, 300))
        angle = rng.uniform(0.0, np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), 120, 3)
    return _soften(img)


def draw_crack(gray: np.ndarray, rng) -> np.ndarray:
    """One jagged crack path confined to a horizontal band."""
    h, w = gray.shape
    img = gray.copy()
    x = int(rng.integers(70, 160))
    y0 = int(rng.integers(120, h - 120))
    y = y0
    points = [(x, y)]
    for _ in range(12):
        x += int(rng.integers(22, 38))
        y = int(np.clip(y + int(rng.integers(-30, 31)), y0 - 35, y0 + 35))
        points.append((x, y))
    for p1, p2 in zip(points[:-1], points[1:]):
        cv2.line(img, p1, p2, 108, 2)
    return _soften(img)


def draw_dent(gray: np.ndarray, rng) -> np.ndarray:
    """Broad, irregular soft depression (two overlapping plateaus = L shape)."""
    img = gray.astype(np.float32)
    h, w = gray.shape
    cx = int(rng.integers(280, w - 280))
    cy = int(rng.integers(210, h - 280))
    cv2.rectangle(img, (cx - 180, cy - 40), (cx + 100, cy + 40), BASE_LEVEL - 58, -1)
    cv2.rectangle(img, (cx - 180, cy - 40), (cx - 100, cy + 160), BASE_LEVEL - 58, -1)
    img = cv2.GaussianBlur(img, (41, 41), 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_discoloration(gray: np.ndarray, rng) -> np.ndarray:
    """Rust-like colour patch blended over the grey surface."""
    img = _to_bgr(gray)
    h, w = gray.shape
    cx = int(rng.integers(240, w - 240))
    cy = int(rng.integers(180, h - 180))
    axes = (int(rng.integers(62, 80)), int(rng.integers(42, 55)))
    angle = float(rng.uniform(0.0, 180.0))
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
    rust = np.array([64.0, 88.0, 150.0])  # BGR brown-rust tint
    blended = img.astype(np.float32) * (1.0 - mask[..., None]) + rust * mask[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def main() -> int:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)

    images = {
        "clean_01.png": _to_bgr(_base_surface(seed=1)),
        "spot_01.png": _to_bgr(draw_spots(_base_surface(seed=2), rng, count=3)),
        "scratch_01.png": _to_bgr(draw_scratches(_base_surface(seed=3), rng, count=3)),
        "crack_01.png": _to_bgr(draw_crack(_base_surface(seed=4), rng)),
        "dent_01.png": _to_bgr(draw_dent(_base_surface(seed=5), rng)),
        "discoloration_01.png": draw_discoloration(_base_surface(seed=6), rng),
        "mixed_02.png": draw_discoloration(
            draw_spots(_base_surface(seed=8), rng, count=6), rng
        ),
    }
    # mixed_02.png additionally carries one straight scratch.
    mixed_gray = cv2.cvtColor(images["mixed_02.png"], cv2.COLOR_BGR2GRAY)
    mixed_gray[519:522, 100:420] = 120
    images["mixed_02.png"] = _to_bgr(_soften(mixed_gray))

    for name, image in images.items():
        path = SAMPLE_DIR / name
        if not cv2.imwrite(str(path), image):
            print(f"FAILED to write {path}")
            return 1
        print(f"wrote {path.name} ({image.shape[1]}x{image.shape[0]})")

    corrupted = SAMPLE_DIR / "corrupted_sample.jpg"
    corrupted.write_bytes(b"this is not a real JPEG file - it must fail to decode")
    print(f"wrote {corrupted.name} (deliberately corrupted, {corrupted.stat().st_size} bytes)")
    print(f"Done: {len(images) + 1} files in {SAMPLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

