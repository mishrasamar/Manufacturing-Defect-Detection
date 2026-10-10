"""CSV / Excel persistence for inspection records.

The logger writes the project datasets to the timestamped batch output
directories:

* ``inspection_log.csv``       - one row per processed image;
* ``defect_details.csv``       - one row per detected defect region;
* ``quality_summary.csv``      - one row of batch KPIs;
* ``category_distribution.csv``- defect categories with shares;
* ``severity_distribution.csv``- defect severities with shares;
* ``batch_statistics.csv``     - one summary row per batch id;
* ``inspection_data.xlsx``     - all of the above as Excel sheets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

INSPECTION_COLUMNS = [
    "image_id",
    "filename",
    "batch_id",
    "width",
    "height",
    "num_defects",
    "total_defect_area_px",
    "defect_area_ratio",
    "categories_detected",
    "max_severity",
    "status",
    "status_reason",
    "processing_time_ms",
    "error_message",
]

DEFECT_COLUMNS = [
    "defect_id",
    "image_id",
    "filename",
    "batch_id",
    "category",
    "method",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "area_px",
    "area_ratio",
    "severity",
    "score",
]

SUMMARY_COLUMNS = [
    "batch_id",
    "generated_at",
    "images_found",
    "images_processed",
    "images_failed_load",
    "images_inspected",
    "defective_images",
    "total_defect_regions",
    "passed",
    "failed",
    "review_required",
    "processing_errors",
    "defect_rate_pct",
    "pass_rate_pct",
    "avg_defects_per_image",
    "most_frequent_defect_type",
    "most_frequent_defect_count",
]

CATEGORY_COLUMNS = ["category", "region_count", "image_count", "share_pct"]
SEVERITY_COLUMNS = ["severity", "region_count", "share_pct"]
BATCH_STATS_COLUMNS = [
    "batch_id",
    "images_inspected",
    "defective_images",
    "total_defect_regions",
    "passed",
    "failed",
    "review_required",
    "defect_rate_pct",
    "pass_rate_pct",
    "avg_defects_per_image",
]

CSV_FILENAMES = {
    "inspection": "inspection_log.csv",
    "defects": "defect_details.csv",
    "summary": "quality_summary.csv",
    "category": "category_distribution.csv",
    "severity": "severity_distribution.csv",
    "batch_stats": "batch_statistics.csv",
}

EXCEL_FILENAME = "inspection_data.xlsx"


class DataLogger:
    """Writes the batch datasets to the CSV and Excel output directories."""

    def __init__(self, csv_dir: Path, excel_dir: Path):
        self.csv_dir = Path(csv_dir)
        self.excel_dir = Path(excel_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.excel_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(
        self,
        inspection_df: pd.DataFrame,
        defect_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        category_df: pd.DataFrame,
        severity_df: pd.DataFrame,
        batch_stats_df: pd.DataFrame,
    ) -> Dict[str, Path]:
        """Write all CSV files; returns a mapping of role -> file path."""
        frames = {
            "inspection": inspection_df,
            "defects": defect_df,
            "summary": summary_df,
            "category": category_df,
            "severity": severity_df,
            "batch_stats": batch_stats_df,
        }
        paths: Dict[str, Path] = {}
        for role, frame in frames.items():
            path = self.csv_dir / CSV_FILENAMES[role]
            frame.to_csv(path, index=False, encoding="utf-8")
            paths[role] = path
        return paths

    def export_excel(
        self,
        inspection_df: pd.DataFrame,
        defect_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        category_df: pd.DataFrame,
        severity_df: pd.DataFrame,
        batch_stats_df: pd.DataFrame,
    ) -> Path:
        """Write all datasets as sheets of one Excel workbook."""
        path = self.excel_dir / EXCEL_FILENAME
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            inspection_df.to_excel(writer, sheet_name="Inspection Log", index=False)
            defect_df.to_excel(writer, sheet_name="Defect Details", index=False)
            summary_df.to_excel(writer, sheet_name="Quality Summary", index=False)
            category_df.to_excel(writer, sheet_name="Category Distribution", index=False)
            severity_df.to_excel(writer, sheet_name="Severity Distribution", index=False)
            batch_stats_df.to_excel(writer, sheet_name="Batch Statistics", index=False)
        return path
