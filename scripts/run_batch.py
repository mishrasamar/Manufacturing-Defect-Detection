"""Command-line batch runner (no user interface).

Examples:
    .venv\\Scripts\\python.exe scripts\\run_batch.py --input data\\sample_images
    .venv\\Scripts\\python.exe scripts\\run_batch.py --input C:\\imgs --no-report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PipelineError, run_batch  # noqa: E402
from src.utils import PROJECT_ROOT, load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one inspection batch from the command line."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "sample_images"),
        help="Folder containing the input images.",
    )
    parser.add_argument("--reference", default="", help="Optional reference image path.")
    parser.add_argument("--settings", default="", help="Optional settings YAML path.")
    parser.add_argument("--output-root", default="", help="Optional output root folder.")
    parser.add_argument("--no-report", action="store_true", help="Skip the executive PDF report.")
    args = parser.parse_args()

    settings = load_settings(Path(args.settings)) if args.settings else load_settings()
    try:
        result = run_batch(
            input_folder=Path(args.input),
            settings=settings,
            reference_path=Path(args.reference) if args.reference else None,
            output_root=Path(args.output_root) if args.output_root else None,
            generate_report=not args.no_report,
        )
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 1

    summary = result.summary
    print("\n=== Batch summary ===")
    for key in (
        "batch_id",
        "images_found",
        "images_processed",
        "images_failed_load",
        "images_inspected",
        "defective_images",
        "total_defect_regions",
        "passed",
        "failed",
        "review_required",
        "defect_rate_pct",
        "pass_rate_pct",
        "avg_defects_per_image",
        "most_frequent_defect_type",
    ):
        print(f"{key:28s}: {summary[key]}")

    print("\n=== Output files ===")
    for role, path in sorted(result.files.items()):
        print(f"{role:12s}: {path}")
    print(f"\nBatch folder: {result.batch_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
