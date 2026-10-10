"""Loading and validation of product images (folders and in-memory uploads).

The loader is deliberately defensive: corrupted, empty or unreadable files
never raise into the pipeline. They are returned as failed records so the
batch can continue, the failure counter stays accurate and the problem is
visible in the dashboard and the PDF report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class ImageRecord:
    """One candidate image in a batch, loaded or failed."""

    image_id: str
    filename: str
    path: str
    image: Optional[np.ndarray] = None
    error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.image is not None


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """Normalise any decoded image to 3-channel BGR."""
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    """Decode raw bytes into a BGR image, or ``None`` when unreadable."""
    if not data:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    try:
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception:  # pragma: no cover - defensive, cv2 rarely raises here
        return None
    if image is None or image.size == 0:
        return None
    return _to_bgr(image)


def load_image_file(path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Read one image from disk. Returns ``(image, None)`` or ``(None, error)``."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, f"could not read file: {exc}"
    if not data:
        return None, "file is empty"
    image = decode_image_bytes(data)
    if image is None:
        return None, "corrupted or unsupported image data"
    return image, None


def scan_folder(folder) -> Dict[str, object]:
    """List supported images in ``folder``.

    Raises ``FileNotFoundError`` / ``NotADirectoryError`` for invalid paths.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file())
    supported = [p for p in files if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    unsupported = [p.name for p in files if p.suffix.lower() not in SUPPORTED_EXTENSIONS]
    return {
        "supported": supported,
        "total_files": len(files),
        "unsupported": unsupported,
    }


def load_folder(
    folder, batch_id: str = ""
) -> Tuple[List[ImageRecord], Dict[str, object]]:
    """Load every supported image in ``folder`` with unique image IDs."""
    info = scan_folder(folder)
    records: List[ImageRecord] = []
    for index, path in enumerate(info["supported"], start=1):
        image, error = load_image_file(path)
        records.append(
            ImageRecord(
                image_id=f"IMG_{index:04d}",
                filename=path.name,
                path=str(path),
                image=image,
                error=error,
            )
        )
    stats = {
        "total_files": info["total_files"],
        "supported": len(info["supported"]),
        "unsupported": list(info["unsupported"]),
        "loaded": sum(1 for r in records if r.loaded),
        "failed": sum(1 for r in records if not r.loaded),
    }
    return records, stats


def load_uploaded_files(
    uploads: Sequence[Tuple[str, bytes]], batch_id: str = ""
) -> Tuple[List[ImageRecord], Dict[str, object]]:
    """Load in-memory uploads (``name, bytes`` pairs) with unique image IDs."""
    records: List[ImageRecord] = []
    unsupported: List[str] = []
    for index, (filename, data) in enumerate(uploads, start=1):
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            unsupported.append(filename)
            records.append(
                ImageRecord(
                    image_id=f"IMG_{index:04d}",
                    filename=filename,
                    path="(upload)",
                    error=f"unsupported file format '{suffix or '(none)'}'",
                )
            )
            continue
        image = decode_image_bytes(data)
        records.append(
            ImageRecord(
                image_id=f"IMG_{index:04d}",
                filename=filename,
                path="(upload)",
                image=image,
                error=None if image is not None else "corrupted or unsupported image data",
            )
        )
    stats = {
        "total_files": len(uploads),
        "supported": len(uploads) - len(unsupported),
        "unsupported": unsupported,
        "loaded": sum(1 for r in records if r.loaded),
        "failed": sum(1 for r in records if not r.loaded),
    }
    return records, stats
