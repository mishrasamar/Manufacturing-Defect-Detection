"""Unit tests for the defensive image loader."""
import cv2
import numpy as np
import pytest

from src.image_loader import SUPPORTED_EXTENSIONS, load_folder, load_uploaded_files


def _write_png(path, w=30, h=20):
    cv2.imwrite(str(path), np.zeros((h, w, 3), np.uint8))


def test_supported_extensions_constant():
    assert ".jpg" in SUPPORTED_EXTENSIONS
    assert ".jpeg" in SUPPORTED_EXTENSIONS
    assert ".png" in SUPPORTED_EXTENSIONS
    assert ".bmp" in SUPPORTED_EXTENSIONS


def test_load_folder_reports_supported_and_unsupported(tmp_path):
    _write_png(tmp_path / "ok.png")
    (tmp_path / "notes.txt").write_text("hello")
    records, stats = load_folder(tmp_path)
    assert stats["total_files"] == 2
    assert stats["supported"] == 1
    assert stats["loaded"] == 1
    assert stats["failed"] == 0
    assert records[0].image_id == "IMG_0001"


def test_corrupted_image_is_reported_not_raised(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"not an image at all")
    _write_png(tmp_path / "ok.png")
    records, stats = load_folder(tmp_path)
    assert stats["loaded"] == 1
    assert stats["failed"] == 1
    failed = [record for record in records if not record.loaded]
    assert failed and failed[0].error
    assert failed[0].image_id == "IMG_0001"  # ids stay unique and ordered


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_folder(tmp_path / "does_not_exist")


def test_load_uploaded_files_validates_content_and_extension():
    ok, encoded = cv2.imencode(".png", np.zeros((20, 30, 3), np.uint8))
    assert ok
    records, stats = load_uploaded_files(
        [
            ("good.png", encoded.tobytes()),
            ("bad.jpg", b"junk data"),
            ("skip.txt", b"text file"),
        ]
    )
    assert stats["loaded"] == 1
    assert stats["failed"] == 2
    assert stats["unsupported"] == ["skip.txt"]
    by_name = {r.filename: r for r in records}
    assert by_name["bad.jpg"].error
