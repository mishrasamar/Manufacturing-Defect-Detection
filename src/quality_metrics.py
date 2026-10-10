"""Pandas-based quality metrics and aggregations.

Definitions used consistently across the whole project (README section 8):

* an *inspected image* is one successfully processed image;
* a *defect region* is one detected region inside one image;
* an image is *defective* when it contains at least one defect region;
* defect rate = defective images / inspected images * 100 (image level);
* pass rate   = images with status "Pass" / inspected images * 100.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.utils import (
    SEVERITY_CRITICAL,
    SEVERITY_MAJOR,
    SEVERITY_MINOR,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_REVIEW,
)

SEVERITY_LABELS = (SEVERITY_MINOR, SEVERITY_MAJOR, SEVERITY_CRITICAL)
STATUS_ORDER = (STATUS_FAIL, STATUS_REVIEW, STATUS_PASS)


def compute_summary(
    inspection_df: pd.DataFrame,
    defect_df: pd.DataFrame,
    batch_id: str,
    images_found: int,
    images_failed_load: int,
    generated_at: str,
) -> Dict[str, Any]:
    """Aggregate the headline KPIs for one batch.

    "Error" rows (images that could not be processed) are counted
    separately and excluded from inspection-rate denominators so the
    statistics only describe images that were actually inspected.
    """
    processed = int(len(inspection_df)) if inspection_df is not None else 0
    inspected_df = (
        inspection_df[inspection_df["status"].ne(STATUS_ERROR)]
        if processed
        else inspection_df
    )
    inspected = int(len(inspected_df))
    defective = int((inspected_df["num_defects"] > 0).sum()) if inspected else 0
    total_regions = int(len(defect_df)) if defect_df is not None else 0
    passed = int((inspected_df["status"] == STATUS_PASS).sum()) if inspected else 0
    failed = int((inspected_df["status"] == STATUS_FAIL).sum()) if inspected else 0
    review = int((inspected_df["status"] == STATUS_REVIEW).sum()) if inspected else 0
    errors = int((inspection_df["status"] == STATUS_ERROR).sum()) if processed else 0
    if inspected:
        defect_rate = defective / inspected * 100.0
        pass_rate = passed / inspected * 100.0
        avg_defects = total_regions / inspected
    else:
        defect_rate = pass_rate = avg_defects = 0.0
    if total_regions:
        top = defect_df["category"].value_counts()
        most_frequent, most_count = str(top.index[0]), int(top.iloc[0])
    else:
        most_frequent, most_count = "None", 0
    return {
        "batch_id": batch_id,
        "generated_at": generated_at,
        "images_found": int(images_found),
        "images_processed": processed,
        "images_failed_load": int(images_failed_load),
        "images_inspected": inspected,
        "defective_images": defective,
        "total_defect_regions": total_regions,
        "passed": passed,
        "failed": failed,
        "review_required": review,
        "processing_errors": errors,
        "defect_rate_pct": round(defect_rate, 2),
        "pass_rate_pct": round(pass_rate, 2),
        "avg_defects_per_image": round(avg_defects, 2),
        "most_frequent_defect_type": most_frequent,
        "most_frequent_defect_count": most_count,
    }


def compute_category_distribution(defect_df: pd.DataFrame) -> pd.DataFrame:
    """Defect region counts per category (regions and images, with share)."""
    columns = ["category", "region_count", "image_count", "share_pct"]
    if defect_df is None or defect_df.empty:
        return pd.DataFrame(columns=columns)
    counts = (
        defect_df.groupby("category")
        .agg(region_count=("defect_id", "size"), image_count=("image_id", "nunique"))
        .reset_index()
    )
    total = int(counts["region_count"].sum())
    counts["share_pct"] = (counts["region_count"] / total * 100.0).round(2) if total else 0.0
    return counts.sort_values("region_count", ascending=False).reset_index(drop=True)


def compute_severity_distribution(defect_df: pd.DataFrame) -> pd.DataFrame:
    """Defect region counts per severity level, ordered Minor -> Critical."""
    columns = ["severity", "region_count", "share_pct"]
    if defect_df is None or defect_df.empty:
        return pd.DataFrame(columns=columns)
    counts = defect_df.groupby("severity").agg(region_count=("defect_id", "size")).reset_index()
    total = int(counts["region_count"].sum())
    counts["share_pct"] = (counts["region_count"] / total * 100.0).round(2) if total else 0.0
    order = {label: idx for idx, label in enumerate(SEVERITY_LABELS)}
    counts = counts.sort_values("severity", key=lambda s: s.map(order).fillna(99))
    return counts.reset_index(drop=True)


def compute_batch_statistics(
    inspection_df: pd.DataFrame, defect_df: pd.DataFrame
) -> pd.DataFrame:
    """One summary row per batch id (supports multi-batch concatenation)."""
    if inspection_df is None or inspection_df.empty:
        return pd.DataFrame(
            columns=[
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
        )
    rows = []
    for batch_id, group in inspection_df.groupby("batch_id"):
        inspected_df = group[group["status"].ne(STATUS_ERROR)]
        inspected = int(len(inspected_df))
        defective = int((inspected_df["num_defects"] > 0).sum()) if inspected else 0
        if defect_df is not None and len(defect_df):
            if "batch_id" in defect_df.columns:
                regions = int((defect_df["batch_id"] == batch_id).sum())
            else:  # frames without a batch column: count by image membership
                image_ids = set(group["image_id"].astype(str))
                regions = int(defect_df["image_id"].astype(str).isin(image_ids).sum())
        else:
            regions = 0
        passed = int((inspected_df["status"] == STATUS_PASS).sum()) if inspected else 0
        failed = int((inspected_df["status"] == STATUS_FAIL).sum()) if inspected else 0
        review = int((inspected_df["status"] == STATUS_REVIEW).sum()) if inspected else 0
        rows.append(
            {
                "batch_id": batch_id,
                "images_inspected": inspected,
                "defective_images": defective,
                "total_defect_regions": regions,
                "passed": passed,
                "failed": failed,
                "review_required": review,
                "defect_rate_pct": round(defective / inspected * 100.0, 2)
                if inspected
                else 0.0,
                "pass_rate_pct": round(passed / inspected * 100.0, 2) if inspected else 0.0,
                "avg_defects_per_image": round(regions / inspected, 2) if inspected else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_defect_count_series(inspection_df: pd.DataFrame) -> pd.Series:
    """Defect-region counts per inspected image (dashboard histogram input)."""
    if inspection_df is None or inspection_df.empty:
        return pd.Series(dtype=int)
    inspected_df = inspection_df[inspection_df["status"].ne(STATUS_ERROR)]
    return inspected_df["num_defects"].astype(int)

