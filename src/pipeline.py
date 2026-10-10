"""End-to-end batch pipeline.

Coordinates the full flow for one batch of product images:

1. load + validate images (folder or in-memory uploads),
2. preprocess each image,
3. detect defect regions with the rule-based detectors,
4. classify region severity and the per-image inspection status,
5. save the resized original and the annotated image,
6. build the three Pandas datasets plus distribution tables,
7. export CSV + Excel, and
8. generate the executive PDF report.

Every batch writes into its own unique ``outputs/<batch_id>/`` folder so
repeated runs never overwrite previous results. Images that cannot be
read are recorded as "Error" rows and reported - they never crash the
batch and they never silently disappear from the statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from src.data_logger import (
    DEFECT_COLUMNS,
    INSPECTION_COLUMNS,
    SUMMARY_COLUMNS,
    DataLogger,
)
from src.defect_classifier import classify_image_status, classify_region_severity
from src.defect_detector import DefectRegion, RuleBasedDefectDetector
from src.image_loader import ImageRecord, load_folder, load_image_file, load_uploaded_files
from src.preprocessing import preprocess
from src.quality_metrics import (
    compute_batch_statistics,
    compute_category_distribution,
    compute_severity_distribution,
    compute_summary,
)
from src.report_generator import ExecutiveReportGenerator
from src.utils import (
    PROJECT_ROOT,
    SEVERITY_NONE,
    SEVERITY_ORDER,
    STATUS_ERROR,
    annotate_image,
    create_batch_id,
    ensure_directories,
    load_settings,
    setup_logger,
    unique_batch_dir,
)


class PipelineError(RuntimeError):
    """Raised when the batch cannot be produced (no input, no readable images)."""


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class BatchResult:
    """Everything one batch produced (paths and data frames)."""

    batch_id: str
    batch_dir: Path
    dirs: Dict[str, Path]
    inspection_df: pd.DataFrame
    defect_df: pd.DataFrame
    summary_df: pd.DataFrame
    category_df: pd.DataFrame
    severity_df: pd.DataFrame
    batch_stats_df: pd.DataFrame
    summary: Dict[str, Any]
    load_stats: Dict[str, Any]
    failed_loads: List[Dict[str, str]]
    files: Dict[str, Path]
    originals: Dict[str, Path]
    annotated: Dict[str, Path]


def _max_severity(regions: Sequence[DefectRegion]) -> str:
    """Highest severity among regions ("None" when the list is empty)."""
    if not regions:
        return SEVERITY_NONE
    return max((r.severity for r in regions), key=lambda s: SEVERITY_ORDER.get(s, -1))


def _error_row(record: ImageRecord, batch_id: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Inspection-log row for an image that could not be processed."""
    return {
        "image_id": record.image_id,
        "filename": record.filename,
        "batch_id": batch_id,
        "width": 0,
        "height": 0,
        "num_defects": 0,
        "total_defect_area_px": 0,
        "defect_area_ratio": 0.0,
        "categories_detected": "",
        "max_severity": SEVERITY_NONE,
        "status": STATUS_ERROR,
        "status_reason": "",
        "processing_time_ms": 0.0,
        "error_message": error or record.error or "unknown error",
    }


def _load_inputs(
    input_folder: Optional[Path],
    uploaded_files: Optional[Sequence[Tuple[str, bytes]]],
    logger,
) -> Tuple[List[ImageRecord], Dict[str, Any], List[Dict[str, str]]]:
    """Load images from a folder or from in-memory uploads."""
    if uploaded_files:
        records, stats = load_uploaded_files(uploaded_files)
    elif input_folder:
        records, stats = load_folder(input_folder)
    else:
        raise PipelineError("No input provided: pass input_folder or uploaded_files.")
    failed_loads = [
        {"image_id": r.image_id, "filename": r.filename, "error": r.error or "unknown"}
        for r in records
        if not r.loaded
    ]
    logger.info(
        "Input: %d file(s) found, %d supported, %d loaded, %d failed",
        stats.get("total_files", 0),
        stats.get("supported", 0),
        stats.get("loaded", 0),
        stats.get("failed", 0),
    )
    return records, stats, failed_loads


def _load_reference(reference_path: Optional[Path], logger) -> Optional[np.ndarray]:
    """Load the optional reference image for the reference-based detector."""
    if not reference_path:
        return None
    path = Path(reference_path)
    if not path.is_file():
        raise PipelineError(f"Reference image not found: {path}")
    image, error = load_image_file(path)
    if image is None:
        raise PipelineError(f"Reference image could not be read: {error}")
    logger.info("Reference image loaded: %s", path.name)
    return image


def run_batch(
    input_folder: Optional[Path] = None,
    uploaded_files: Optional[Sequence[Tuple[str, bytes]]] = None,
    settings: Optional[Dict[str, Any]] = None,
    reference_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    generate_report: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> BatchResult:
    """Process one batch of images end to end and persist every output.

    Raises :class:`PipelineError` when no input exists or when no image
    could be processed at all (for example a folder of corrupt files).
    """
    settings = settings if settings is not None else load_settings()
    if output_root is None:
        configured = str(settings.get("output", {}).get("root", "outputs"))
        output_root = Path(configured)
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    batch_dir = unique_batch_dir(output_root, create_batch_id())
    batch_id = batch_dir.name  # may differ from the raw id if the folder existed
    dirs = ensure_directories(batch_dir)
    logger = setup_logger("pipeline", dirs["logs"] / f"{batch_id}.log")
    notify = progress_callback or (lambda done, total, message: None)
    logger.info("Batch %s started", batch_id)

    records, load_stats, failed_loads = _load_inputs(input_folder, uploaded_files, logger)
    if not records:
        raise PipelineError(
            "No supported image files (JPG, JPEG, PNG, BMP) were found in the input."
        )
    reference_bgr = _load_reference(reference_path, logger)

    detector = RuleBasedDefectDetector(settings)
    severity_cfg = settings.get("severity", {})
    inspection_rows: List[Dict[str, Any]] = []
    defect_rows: List[Dict[str, Any]] = []
    originals: Dict[str, Path] = {}
    annotated: Dict[str, Path] = {}
    total = len(records)

    for index, record in enumerate(records, start=1):
        notify(index - 1, total, f"Processing {record.filename} ({index}/{total})")
        started = perf_counter()
        if not record.loaded:
            inspection_rows.append(_error_row(record, batch_id))
            logger.warning("%s skipped: %s", record.image_id, record.error)
            continue
        try:
            pre = preprocess(record.image, settings)
            regions = detector.detect(pre, reference_bgr)
            for region in regions:
                region.severity = classify_region_severity(
                    region.area_ratio, region.category, severity_cfg
                )
            total_area = int(sum(r.area for r in regions))
            area_ratio = total_area / float(pre.processed_size[0] * pre.processed_size[1])
            status, reason = classify_image_status(regions, area_ratio, severity_cfg)

            inspection_rows.append(
                {
                    "image_id": record.image_id,
                    "filename": record.filename,
                    "batch_id": batch_id,
                    "width": int(pre.processed_size[0]),
                    "height": int(pre.processed_size[1]),
                    "num_defects": len(regions),
                    "total_defect_area_px": total_area,
                    "defect_area_ratio": round(area_ratio, 6),
                    "categories_detected": ", ".join(sorted({r.category for r in regions})),
                    "max_severity": _max_severity(regions),
                    "status": status,
                    "status_reason": reason,
                    "processing_time_ms": round((perf_counter() - started) * 1000.0, 1),
                    "error_message": "",
                }
            )
            for position, region in enumerate(regions, start=1):
                x, y, w, h = region.bbox
                defect_rows.append(
                    {
                        "defect_id": f"{record.image_id}_D{position:02d}",
                        "image_id": record.image_id,
                        "filename": record.filename,
                        "batch_id": batch_id,
                        "category": region.category,
                        "method": region.method,
                        "bbox_x": int(x),
                        "bbox_y": int(y),
                        "bbox_w": int(w),
                        "bbox_h": int(h),
                        "area_px": int(region.area),
                        "area_ratio": round(region.area_ratio, 6),
                        "severity": region.severity,
                        "score": round(region.score, 3),
                    }
                )
            original_path = dirs["originals"] / f"{record.image_id}_original.png"
            annotated_path = dirs["annotated_images"] / f"{record.image_id}_annotated.png"
            cv2.imwrite(str(original_path), pre.resized_bgr)
            cv2.imwrite(
                str(annotated_path),
                annotate_image(pre.resized_bgr, regions, status, extra_text=record.filename),
            )
            originals[record.image_id] = original_path
            annotated[record.image_id] = annotated_path
            logger.info(
                "%s (%s): %d region(s) -> %s",
                record.image_id, record.filename, len(regions), status,
            )
        except Exception as exc:  # keep the batch running, record the failure
            inspection_rows.append(_error_row(record, batch_id, error=str(exc)))
            logger.exception("%s failed during processing", record.image_id)
    notify(total, total, "Building datasets and outputs...")

    inspection_df = pd.DataFrame(inspection_rows, columns=INSPECTION_COLUMNS)
    defect_df = pd.DataFrame(defect_rows, columns=DEFECT_COLUMNS)
    inspected_any = bool(inspection_df["status"].ne(STATUS_ERROR).any()) if not inspection_df.empty else False
    if not inspected_any:
        raise PipelineError(
            f"None of the {total} image file(s) could be processed "
            "(corrupted or unreadable). No datasets or report were produced."
        )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = compute_summary(
        inspection_df,
        defect_df,
        batch_id,
        images_found=load_stats.get("total_files", 0),
        images_failed_load=load_stats.get("failed", 0),
        generated_at=generated_at,
    )
    summary_df = pd.DataFrame([summary], columns=SUMMARY_COLUMNS)
    category_df = compute_category_distribution(defect_df)
    severity_df = compute_severity_distribution(defect_df)
    batch_stats_df = compute_batch_statistics(inspection_df, defect_df)

    data_logger = DataLogger(dirs["csv"], dirs["excel"])
    csv_paths = data_logger.export_csv(
        inspection_df, defect_df, summary_df, category_df, severity_df, batch_stats_df
    )
    excel_path = data_logger.export_excel(
        inspection_df, defect_df, summary_df, category_df, severity_df, batch_stats_df
    )
    files: Dict[str, Path] = {f"csv_{role}": path for role, path in csv_paths.items()}
    files["excel"] = excel_path
    files["log"] = dirs["logs"] / f"{batch_id}.log"

    if generate_report:
        report_path = ExecutiveReportGenerator(settings).generate(
            batch_id=batch_id,
            inspection_df=inspection_df,
            defect_df=defect_df,
            summary=summary,
            category_df=category_df,
            severity_df=severity_df,
            originals=originals,
            annotated=annotated,
            out_dir=dirs["reports"],
            failed_loads=failed_loads,
        )
        files["report"] = report_path

    logger.info("Batch %s finished: %s", batch_id, summary)
    notify(total, total, "Done.")
    return BatchResult(
        batch_id=batch_id,
        batch_dir=batch_dir,
        dirs=dirs,
        inspection_df=inspection_df,
        defect_df=defect_df,
        summary_df=summary_df,
        category_df=category_df,
        severity_df=severity_df,
        batch_stats_df=batch_stats_df,
        summary=summary,
        load_stats=load_stats,
        failed_loads=failed_loads,
        files=files,
        originals=originals,
        annotated=annotated,
    )

