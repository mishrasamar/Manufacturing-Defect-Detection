"""Unit tests for the CSV / Excel persistence layer."""
import pandas as pd

from src.data_logger import (
    DEFECT_COLUMNS,
    INSPECTION_COLUMNS,
    DataLogger,
)


def _frames():
    inspection = pd.DataFrame(
        [
            {
                "image_id": "IMG_0001", "filename": "a.png", "batch_id": "B1",
                "width": 100, "height": 80, "num_defects": 1,
                "total_defect_area_px": 50, "defect_area_ratio": 0.006,
                "categories_detected": "Spot", "max_severity": "Minor",
                "status": "Pass", "status_reason": "ok", "processing_time_ms": 12.0,
                "error_message": "",
            }
        ],
        columns=INSPECTION_COLUMNS,
    )
    defects = pd.DataFrame(
        [
            {
                "defect_id": "IMG_0001_D01", "image_id": "IMG_0001", "filename": "a.png",
                "batch_id": "B1", "category": "Spot", "method": "m",
                "bbox_x": 1, "bbox_y": 2, "bbox_w": 5, "bbox_h": 6,
                "area_px": 50, "area_ratio": 0.006, "severity": "Minor", "score": 0.4,
            }
        ],
        columns=DEFECT_COLUMNS,
    )
    summary = pd.DataFrame(
        [
            {
                "batch_id": "B1", "generated_at": "t", "images_found": 1,
                "images_processed": 1, "images_failed_load": 0, "images_inspected": 1,
                "defective_images": 1, "total_defect_regions": 1, "passed": 0,
                "failed": 0, "review_required": 1, "processing_errors": 0,
                "defect_rate_pct": 100.0, "pass_rate_pct": 0.0,
                "avg_defects_per_image": 1.0, "most_frequent_defect_type": "Spot",
                "most_frequent_defect_count": 1,
            }
        ]
    )
    category = pd.DataFrame([{"category": "Spot", "region_count": 1, "image_count": 1, "share_pct": 100.0}])
    severity = pd.DataFrame([{"severity": "Minor", "region_count": 1, "share_pct": 100.0}])
    batch_stats = pd.DataFrame(
        [
            {
                "batch_id": "B1", "images_inspected": 1, "defective_images": 1,
                "total_defect_regions": 1, "passed": 0, "failed": 0,
                "review_required": 1, "defect_rate_pct": 100.0,
                "pass_rate_pct": 0.0, "avg_defects_per_image": 1.0,
            }
        ]
    )
    return inspection, defects, summary, category, severity, batch_stats


def test_csv_export_roundtrip(tmp_path):
    frames = _frames()
    logger = DataLogger(tmp_path / "csv", tmp_path / "excel")
    paths = logger.export_csv(*frames)
    reloaded = pd.read_csv(paths["inspection"])
    assert list(reloaded.columns) == INSPECTION_COLUMNS
    assert reloaded.iloc[0]["image_id"] == "IMG_0001"
    assert (tmp_path / "csv" / "defect_details.csv").exists()
    assert (tmp_path / "csv" / "quality_summary.csv").exists()


def test_excel_export_sheet_names(tmp_path):
    frames = _frames()
    logger = DataLogger(tmp_path / "csv", tmp_path / "excel")
    path = logger.export_excel(*frames)
    workbook = pd.ExcelFile(path)
    assert set(workbook.sheet_names) == {
        "Inspection Log",
        "Defect Details",
        "Quality Summary",
        "Category Distribution",
        "Severity Distribution",
        "Batch Statistics",
    }
