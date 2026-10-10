"""Shared pytest fixtures and image factories."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def settings():
    """Default settings dict (independent of the YAML file)."""
    from src.utils import default_settings

    return default_settings()


def make_surface(w: int = 480, h: int = 360, seed: int = 1, noise: float = 6.0) -> np.ndarray:
    """A synthetic brushed-metal surface as a grayscale image."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 178.0) + rng.normal(0.0, noise, (h, w))
    return np.clip(base, 0, 255).astype(np.uint8)
