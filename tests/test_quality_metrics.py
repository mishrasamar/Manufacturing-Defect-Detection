"""Unit tests for the Pandas quality metrics."""
import pandas as pd
import pytest

from src.quality_metrics import (
    compute_batch_statistics,
    compute_category_distribution,
    compute_severity_distribution,
    compute_summary,
)


def _inspection_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"image_id": "IMG_0001", "filename": "a.png", "batch_id": "B1", "status": "Pass", "num_defects": 0},
            {"image_id": "IMG_0002", "filename": "b.png", "batch_id": "B1", "status": "Review Required", "num_defects": 2},
            {"image_id": "IMG_0003", "filename": "c.png", "batch_id": "B1", "status": "Fail", "num_defects": 1},
            {"image_id": "IMG_0004", "filename": "d.png", "batch_id": "B1", "status": "Pass", "num_defects": 0},
            {"image_id": "IMG_0005", "filename": "e.png", "batch_id": "B1", "status": "Error", "num_defects": 0},
        ]
    )


def _defect_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"defect_id": "D1", "image_id": "IMG_0002", "category": "Scratch", "severity": "Minor"},
            {"defect_id": "D2", "image_id": "IMG_0002", "category": "Scratch", "severity": "Minor"},
            {"defect_id": "D3", "image_id": "IMG_0003", "category": "Dent", "severity": "Critical"},
        ]
    )


def test_summary_counts():
    summary = compute_summary(
        _inspection_df(), _defect_df(), "B1", images_found=6, images_failed_load=1, generated_at="t"
    )
    assert summary["images_inspected"] == 4  # the Error row is excluded
    assert summary["defective_images"] == 2
    assert summary["total_defect_regions"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["review_required"] == 1
    assert summary["defect_rate_pct"] == 50.0
    assert summary["pass_rate_pct"] == 50.0
    assert summary["avg_defects_per_image"] == 0.75
    assert summary["most_frequent_defect_type"] == "Scratch"


def test_summary_handles_empty_frames():
    empty_inspection = pd.DataFrame(columns=["image_id", "filename", "batch_id", "status", "num_defects"])
    empty_defects = pd.DataFrame(columns=["defect_id", "image_id", "category", "severity"])
    summary = compute_summary(empty_inspection, empty_defects, "B", 0, 0, "t")
    assert summary["images_inspected"] == 0
    assert summary["defect_rate_pct"] == 0.0
    assert summary["most_frequent_defect_type"] == "None"


def test_category_distribution():
    category = compute_category_distribution(_defect_df())
    scratch = category.loc[category["category"] == "Scratch"].iloc[0]
    assert scratch["region_count"] == 2
    assert scratch["image_count"] == 1
    assert category["share_pct"].sum() == pytest.approx(100.0)


def test_category_distribution_empty():
    empty = pd.DataFrame(columns=["defect_id", "image_id", "category", "severity"])
    category = compute_category_distribution(empty)
    assert list(category.columns) == ["category", "region_count", "image_count", "share_pct"]


def test_severity_distribution_ordered():
    severity = compute_severity_distribution(_defect_df())
    labels = severity["severity"].tolist()
    assert labels.index("Minor") < labels.index("Critical")


def test_batch_statistics_groups_by_batch():
    stats = compute_batch_statistics(_inspection_df(), _defect_df())
    assert len(stats) == 1
    row = stats.iloc[0]
    assert row["batch_id"] == "B1"
    assert row["images_inspected"] == 4
    assert row["total_defect_regions"] == 3
    assert row["defect_rate_pct"] == 50.0
