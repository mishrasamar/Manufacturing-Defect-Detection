"""Unit tests for the preprocessing pipeline."""
import cv2
import numpy as np

from src.preprocessing import (
    compute_edges,
    denoise_gray,
    normalize_intensity,
    preprocess,
    resize_keep_aspect,
    to_gray,
)


def test_to_gray_single_channel():
    bgr = np.zeros((10, 12, 3), np.uint8)
    gray = to_gray(bgr)
    assert gray.ndim == 2
    assert gray.shape == (10, 12)


def test_resize_keep_aspect_and_scale():
    img = np.zeros((100, 400, 3), np.uint8)
    resized, scale = resize_keep_aspect(img, 200)
    assert resized.shape[1] == 200
    assert resized.shape[0] == 50
    assert abs(scale - 0.5) < 1e-6


def test_resize_never_upscales():
    img = np.zeros((50, 100, 3), np.uint8)
    resized, scale = resize_keep_aspect(img, 200)
    assert scale == 1.0
    assert resized.shape == img.shape


def test_denoise_reduces_noise():
    rng = np.random.default_rng(0)
    noisy = np.clip(np.full((200, 200), 128.0) + rng.normal(0, 25, (200, 200)), 0, 255)
    noisy = noisy.astype(np.uint8)
    denoised = denoise_gray(noisy, "median", 5)
    assert float(denoised.std()) < float(noisy.std())


def test_normalize_intensity_stretches_range():
    gray = np.full((50, 50), 100, np.uint8)
    gray[10:20, 10:20] = 150
    normalized = normalize_intensity(gray, (1, 99))
    assert int(normalized.max()) > int(gray.max())


def test_compute_edges_orders_thresholds():
    gray = np.zeros((60, 60), np.uint8)
    gray[20:40, 20:40] = 255
    edges = compute_edges(gray, 400, 50)  # swapped thresholds are ordered internally
    assert edges.shape == gray.shape
    assert edges.max() > 0  # the step edge is still detected, and no error is raised


def test_preprocess_shapes_and_edges(settings):
    img = np.zeros((120, 160, 3), np.uint8)
    cv2.rectangle(img, (40, 30), (120, 90), (255, 255, 255), -1)
    result = preprocess(img, settings)
    assert result.processed_size == (160, 120)
    assert result.original_size == (160, 120)
    assert result.denoised.shape == (120, 160)
    assert result.edges.max() > 0  # the rectangle border must produce edges


def test_preprocess_rejects_empty_image(settings):
    try:
        preprocess(None, settings)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty image")
