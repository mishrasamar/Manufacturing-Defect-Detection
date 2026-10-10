"""Streamlit dashboard for the Automated Manufacturing Defect Detection
and Quality Audit System.

Run with:
    .venv\\Scripts\\python.exe -m streamlit run app.py

The dashboard supports two input modes (folder path or file upload),
exposes the main detection / severity thresholds (defaults come from
config/settings.yaml), runs the batch pipeline with live progress and
presents every actual output: KPI metrics, charts, filterable logs,
original-vs-annotated previews and download buttons for the CSV, Excel,
PDF report and annotated-image ZIP files.
"""
from __future__ import annotations

import sys
import traceback
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import PipelineError, run_batch  # noqa: E402
from src.report_generator import (  # noqa: E402
    category_bar_chart,
    defects_per_image_chart,
    severity_bar_chart,
    status_pie_chart,
)
from src.utils import load_settings  # noqa: E402

st.set_page_config(
    page_title="Manufacturing Defect Detection",
    page_icon=":material/factory:",
    layout="wide",
)

SUPPORTED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp"]

DETECTOR_LABELS = [
    ("spot", "Spot detector"),
    ("scratch", "Scratch detector"),
    ("crack", "Crack detector"),
    ("dent", "Dent detector"),
    ("discoloration", "Discoloration detector"),
    ("reference", "Reference comparison"),
]


def _sidebar_controls() -> dict:
    """Collect the input mode, optional reference image and thresholds."""
    with st.sidebar:
        st.header("1 - Image source")
        mode = st.radio("Input mode", ["Folder on this computer", "Upload image files"])
        folder_path = ""
        uploads = []
        if mode == "Folder on this computer":
            folder_path = st.text_input(
                "Folder path", value=str(ROOT / "data" / "sample_images")
            )
            st.caption("Supported formats: JPG, JPEG, PNG, BMP")
        else:
            uploads = st.file_uploader(
                "Upload product images",
                type=SUPPORTED_UPLOAD_TYPES,
                accept_multiple_files=True,
            )
            if uploads:
                st.caption(f"{len(uploads)} file(s) selected")
        reference = st.text_input("Reference image path (optional)")

        st.header("2 - Detection settings")
        settings = load_settings()
        detection = settings["detection"]
        with st.expander("Detector toggles", expanded=False):
            for key, label in DETECTOR_LABELS:
                detection[key]["enabled"] = st.toggle(
                    label,
                    value=bool(detection[key].get("enabled", True)),
                    key=f"toggle_{key}",
                )
        with st.expander("Severity and status thresholds", expanded=False):
            severity = settings["severity"]
            severity["major_area_ratio"] = st.slider(
                "Major defect area ratio", 0.001, 0.10,
                float(severity["major_area_ratio"]), 0.001, format="%.3f",
            )
            severity["critical_area_ratio"] = st.slider(
                "Critical defect area ratio", 0.01, 0.30,
                float(severity["critical_area_ratio"]), 0.005, format="%.3f",
            )
            severity["fail_total_area_ratio"] = st.slider(
                "Fail: cumulative area ratio", 0.01, 0.50,
                float(severity["fail_total_area_ratio"]), 0.01, format="%.2f",
            )
            severity["fail_defect_count"] = st.slider(
                "Fail: defect region count", 2, 30, int(severity["fail_defect_count"])
            )
            severity["review_total_area_ratio"] = st.slider(
                "Review: cumulative area ratio", 0.001, 0.20,
                float(severity["review_total_area_ratio"]), 0.001, format="%.3f",
            )
            severity["review_defect_count"] = st.slider(
                "Review: defect region count", 1, 15, int(severity["review_defect_count"])
            )

        run_clicked = st.button("Run inspection", type="primary", use_container_width=True)
        st.caption("Defaults come from config/settings.yaml.")
    return {
        "mode": mode,
        "folder_path": folder_path,
        "uploads": uploads,
        "reference": reference,
        "settings": settings,
        "run_clicked": run_clicked,
    }


def _run_inspection(controls: dict) -> None:
    """Validate the sidebar inputs and execute one batch."""
    settings = controls["settings"]
    reference_path = str(controls["reference"] or "").strip()
    if reference_path and not Path(reference_path).is_file():
        st.error(f"Reference image not found: {reference_path}")
        return
    uploads = controls["uploads"]
    folder = str(controls["folder_path"] or "").strip()
    payload = None
    if controls["mode"] == "Folder on this computer":
        if not folder:
            st.error("Please enter a folder path.")
            return
        if not Path(folder).is_dir():
            st.error(f"Folder does not exist: {folder}")
            return
    else:
        if not uploads:
            st.error("Please upload at least one image file.")
            return
        payload = [(f.name, f.getvalue()) for f in uploads]

    progress = st.progress(0.0, text="Preparing batch...")

    def callback(done: int, total: int, message: str) -> None:
        progress.progress(min(1.0, done / max(total, 1)), text=message)

    try:
        result = run_batch(
            input_folder=folder if payload is None else None,
            uploaded_files=payload,
            settings=settings,
            reference_path=reference_path or None,
            progress_callback=callback,
        )
        st.session_state["batch_result"] = result
        progress.progress(1.0, text="Inspection complete.")
    except PipelineError as exc:
        st.error(f"Inspection stopped: {exc}")
    except Exception as exc:  # show unexpected errors without killing the app
        st.error(f"Unexpected error: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


def _annotated_zip(result) -> bytes:
    """Build an in-memory ZIP of all annotated images of the batch."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for image_id, path in sorted(result.annotated.items()):
            path = Path(path)
            if path.exists():
                archive.write(path, arcname=path.name)
    return buffer.getvalue()


def _file_bytes(path) -> bytes:
    path = Path(path)
    return path.read_bytes() if path.exists() else b""


def _render_results(result) -> None:
    """Render every actual output of the last batch."""
    st.success(f"Batch {result.batch_id} completed - outputs in {result.batch_dir}")
    summary = result.summary
    top = st.columns(4)
    top[0].metric("Images found", summary["images_found"])
    top[1].metric("Processed", summary["images_processed"])
    top[2].metric("Failed to load", summary["images_failed_load"])
    top[3].metric("Defect regions", summary["total_defect_regions"])
    mid = st.columns(4)
    mid[0].metric("Defect rate", f"{summary['defect_rate_pct']:.1f} %")
    mid[1].metric("Pass rate", f"{summary['pass_rate_pct']:.1f} %")
    mid[2].metric("Avg defects / image", f"{summary['avg_defects_per_image']:.2f}")
    mid[3].metric("Most frequent type", summary["most_frequent_defect_type"])
    bottom = st.columns(3)
    bottom[0].metric("Pass", summary["passed"])
    bottom[1].metric("Fail", summary["failed"])
    bottom[2].metric("Review Required", summary["review_required"])

    if result.failed_loads:
        with st.expander(
            f"Images that could not be read ({len(result.failed_loads)})", expanded=False
        ):
            st.dataframe(pd.DataFrame(result.failed_loads), hide_index=True, use_container_width=True)


    st.divider()
    st.subheader("Quality statistics")
    left, right = st.columns(2)
    with left:
        fig_category = category_bar_chart(result.category_df)
        if fig_category is not None:
            st.pyplot(fig_category)
            plt.close(fig_category)
        else:
            st.info("No defect regions were detected in this batch.")
        fig_severity = severity_bar_chart(result.severity_df)
        if fig_severity is not None:
            st.pyplot(fig_severity)
            plt.close(fig_severity)
    with right:
        fig_status = status_pie_chart(result.summary)
        if fig_status is not None:
            st.pyplot(fig_status)
            plt.close(fig_status)
        fig_histogram = defects_per_image_chart(result.inspection_df)
        if fig_histogram is not None:
            st.pyplot(fig_histogram)
            plt.close(fig_histogram)

    st.subheader("Batch statistics")
    st.dataframe(result.batch_stats_df, hide_index=True, use_container_width=True)
    st.divider()

    st.subheader("Inspection log")
    query = st.text_input("Filter rows (matches any column value)")
    status_filter = st.multiselect(
        "Status filter",
        sorted(result.inspection_df["status"].unique().tolist()),
        default=[],
    )
    view = result.inspection_df
    if status_filter:
        view = view[view["status"].isin(status_filter)]
    if query:
        haystack = view.astype(str).apply(
            lambda row: query.lower() in " ".join(row.values).lower(), axis=1
        )
        view = view[haystack]
    st.dataframe(view, hide_index=True, use_container_width=True)
    with st.expander("Defect details log", expanded=False):
        defect_view = result.defect_df
        if query:
            haystack = defect_view.astype(str).apply(
                lambda row: query.lower() in " ".join(row.values).lower(), axis=1
            )
            defect_view = defect_view[haystack]
        st.dataframe(defect_view, hide_index=True, use_container_width=True)
    st.divider()

    st.subheader("Image review (original vs annotated)")
    inspected = result.inspection_df[result.inspection_df["status"].ne("Error")]
    image_ids = inspected["image_id"].tolist()
    if image_ids:
        selected = st.selectbox("Select image", image_ids)
        row = inspected[inspected["image_id"] == selected].iloc[0]
        col_original, col_annotated = st.columns(2)
        for column, path, caption in (
            (col_original, result.originals.get(selected), "Original (processed size)"),
            (col_annotated, result.annotated.get(selected), "Annotated (result)"),
        ):
            with column:
                if path and Path(path).exists():
                    image = cv2.imread(str(path))
                    if image is not None:
                        st.image(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                            caption=caption,
                            use_container_width=True,
                        )
        st.caption(
            f"{row['filename']} - status {row['status']} - "
            f"{int(row['num_defects'])} defect region(s) - {row['status_reason']}"
        )
    else:
        st.info("No inspectable images in this batch.")
    st.divider()

    st.subheader("Downloads")
    files = result.files
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button(
            "Inspection log (CSV)", _file_bytes(files["csv_inspection"]),
            file_name=Path(files["csv_inspection"]).name, key="dl_inspection",
        )
        st.download_button(
            "Defect details (CSV)", _file_bytes(files["csv_defects"]),
            file_name=Path(files["csv_defects"]).name, key="dl_defects",
        )
        st.download_button(
            "Quality summary (CSV)", _file_bytes(files["csv_summary"]),
            file_name=Path(files["csv_summary"]).name, key="dl_summary",
        )
    with col_b:
        st.download_button(
            "All datasets (Excel)", _file_bytes(files["excel"]),
            file_name=Path(files["excel"]).name, key="dl_excel",
        )
        if "report" in files:
            st.download_button(
                "Executive report (PDF)", _file_bytes(files["report"]),
                file_name=Path(files["report"]).name, key="dl_report",
            )
    with col_c:
        st.download_button(
            "Annotated images (ZIP)", _annotated_zip(result),
            file_name=f"{result.batch_id}_annotated_images.zip", key="dl_zip",
        )
        st.caption(f"All files are also stored in:\n{result.batch_dir}")


def main() -> None:
    st.title("Automated Manufacturing Defect Detection & Quality Audit")
    st.caption(
        "Rule-based OpenCV inspection pipeline with Pandas analytics, "
        "CSV/Excel export and an automated executive PDF report."
    )
    controls = _sidebar_controls()
    if controls["run_clicked"]:
        _run_inspection(controls)
    if "batch_result" in st.session_state:
        st.divider()
        _render_results(st.session_state["batch_result"])
    elif not controls["run_clicked"]:
        st.info(
            "Select an image source in the sidebar and press **Run inspection**. "
            "Sample images are available in `data/sample_images`."
        )


if __name__ == "__main__":
    main()


