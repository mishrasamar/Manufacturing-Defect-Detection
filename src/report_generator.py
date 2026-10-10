"""Automated executive PDF report generation (ReportLab).

The report is a management-oriented summary of one inspection batch:
title block, KPI summary, defect statistics with charts, the inspection
table, failed / review cases, original-vs-annotated image examples,
automatically derived observations, suggested follow-up actions and a
method-and-limitations section.

Report generation is *refused* when there are no valid inspection
records (:class:`ReportGenerationError`), so an empty batch can never
produce a fabricated-looking report. Only values that are actually
present in the provided data frames are written into the report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from PIL import Image as PILImage  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.defect_classifier import describe_detection_rules, describe_severity_rules
from src.utils import (
    SEVERITY_CRITICAL,
    SEVERITY_MAJOR,
    SEVERITY_MINOR,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_REVIEW,
)

REPORT_TITLE = "Automated Manufacturing Defect Detection - Quality Audit Report"

BRAND = colors.HexColor("#1F3864")
ACCENT = colors.HexColor("#2C6FBB")
LIGHT = colors.HexColor("#EEF3F9")
MUTED = colors.HexColor("#5A6472")


class ReportGenerationError(RuntimeError):
    """Raised when a report cannot be built from the available records."""


# -------------------------------------------------------------------------- #
# Matplotlib charts (saved as PNG files next to the PDF report)
# -------------------------------------------------------------------------- #
def _save_chart(fig, out_dir: Path, name: str) -> Optional[Path]:
    path = out_dir / f"{name}.png"
    try:
        fig.savefig(path, bbox_inches="tight")
        return path
    finally:
        plt.close(fig)


def category_bar_chart(category_df: pd.DataFrame, out_dir: Optional[Path] = None):
    """Horizontal bar chart of defect regions per category.

    With ``out_dir`` the chart is saved and its path returned; without it
    the live Matplotlib figure is returned (caller closes it).
    """
    if category_df is None or category_df.empty:
        return None
    data = category_df.sort_values("region_count", ascending=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=150)
    bars = ax.barh(
        data["category"].astype(str), data["region_count"].astype(int), color="#2C6FBB"
    )
    ax.set_xlabel("Detected regions")
    ax.set_title("Defect regions by category")
    xmax = max(int(data["region_count"].max()), 1)
    ax.set_xlim(0, xmax * 1.18)
    for bar, count in zip(bars, data["region_count"].astype(int)):
        ax.text(
            bar.get_width() + xmax * 0.02,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if out_dir is None:
        return fig
    return _save_chart(fig, out_dir, "category_distribution")


def severity_bar_chart(severity_df: pd.DataFrame, out_dir: Optional[Path] = None):
    """Vertical bar chart of defect regions per severity level."""
    if severity_df is None or severity_df.empty:
        return None
    color_map = {
        SEVERITY_MINOR: "#F9C74F",
        SEVERITY_MAJOR: "#F8961E",
        SEVERITY_CRITICAL: "#E63946",
    }
    labels = severity_df["severity"].astype(str).tolist()
    values = severity_df["region_count"].astype(int).tolist()
    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=150)
    ax.bar(labels, values, color=[color_map.get(lbl, "#8D99AE") for lbl in labels])
    for x, value in enumerate(values):
        ax.text(x, value, str(value), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Detected regions")
    ax.set_title("Defect regions by severity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if out_dir is None:
        return fig
    return _save_chart(fig, out_dir, "severity_distribution")


def status_pie_chart(summary: Dict[str, Any], out_dir: Optional[Path] = None):
    """Pie chart of the inspection status distribution."""
    counts = {
        STATUS_PASS: int(summary.get("passed", 0)),
        STATUS_FAIL: int(summary.get("failed", 0)),
        STATUS_REVIEW: int(summary.get("review_required", 0)),
    }
    pairs = [(label, value) for label, value in counts.items() if value > 0]
    if not pairs:
        return None
    labels, sizes = zip(*pairs)
    palette = {STATUS_PASS: "#2E7D32", STATUS_FAIL: "#C62828", STATUS_REVIEW: "#F9A825"}
    fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=150)
    ax.pie(
        sizes,
        labels=[f"{label}\n({value})" for label, value in pairs],
        colors=[palette[label] for label in labels],
        autopct="%1.0f%%",
        startangle=90,
        counterclock=False,
    )
    ax.set_title("Inspection status distribution")
    fig.tight_layout()
    if out_dir is None:
        return fig
    return _save_chart(fig, out_dir, "status_distribution")


def defects_per_image_chart(inspection_df: pd.DataFrame, out_dir: Optional[Path] = None):
    """Histogram: how many inspected images contain N defect regions."""
    if inspection_df is None or inspection_df.empty:
        return None
    inspected = inspection_df[inspection_df["status"].ne(STATUS_ERROR)]
    counts = inspected["num_defects"].astype(int).value_counts().sort_index()
    if counts.empty or (counts.index == 0).all():
        return None
    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=150)
    ax.bar([str(idx) for idx in counts.index], counts.values, color="#457B9D")
    for x, value in enumerate(counts.values):
        ax.text(x, value, str(value), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Defect regions per image")
    ax.set_ylabel("Images")
    ax.set_title("Defect regions per image")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if out_dir is None:
        return fig
    return _save_chart(fig, out_dir, "defects_per_image")


# -------------------------------------------------------------------------- #
# Executive PDF report generator
# -------------------------------------------------------------------------- #
DOC_WIDTH = A4[0] - 92  # usable content width with 46pt side margins


class ExecutiveReportGenerator:
    """Builds the executive PDF report for one inspection batch."""

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        report_cfg = settings.get("report", {})
        self.max_table_rows = int(report_cfg.get("max_table_rows", 25))
        self.max_examples = int(report_cfg.get("max_image_examples", 4))

    # ------------------------------------------------------------------ #
    def generate(
        self,
        batch_id: str,
        inspection_df: pd.DataFrame,
        defect_df: pd.DataFrame,
        summary: Dict[str, Any],
        category_df: pd.DataFrame,
        severity_df: pd.DataFrame,
        originals: Dict[str, Path],
        annotated: Dict[str, Path],
        out_dir: Path,
        failed_loads: Optional[List[Dict[str, str]]] = None,
    ) -> Path:
        """Create ``quality_report_<batch_id>.pdf`` and return its path."""
        if inspection_df is None or inspection_df.empty:
            raise ReportGenerationError(
                "No valid inspection records - the report was not generated."
            )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        charts = self._build_charts(out_dir, summary, category_df, severity_df, inspection_df)
        report_path = out_dir / f"quality_report_{batch_id}.pdf"
        self._build_pdf(
            report_path,
            batch_id,
            inspection_df,
            defect_df,
            summary,
            category_df,
            severity_df,
            charts,
            originals,
            annotated,
            failed_loads or [],
        )
        return report_path

    def _build_charts(self, out_dir, summary, category_df, severity_df, inspection_df) -> Dict[str, Path]:
        """Render every applicable chart; a failing chart never blocks the report."""
        builders = {
            "category": lambda: category_bar_chart(category_df, out_dir),
            "severity": lambda: severity_bar_chart(severity_df, out_dir),
            "status": lambda: status_pie_chart(summary, out_dir),
            "histogram": lambda: defects_per_image_chart(inspection_df, out_dir),
        }
        charts: Dict[str, Path] = {}
        for name, build in builders.items():
            try:
                path = build()
            except Exception:
                path = None
            if path is not None:
                charts[name] = path
        return charts

    # ------------------------------------------------------------------ #
    @staticmethod
    def _styles() -> Dict[str, ParagraphStyle]:
        dark_text = colors.HexColor("#20242B")
        return {
            "title": ParagraphStyle("RptTitle", fontName="Helvetica-Bold", fontSize=19, leading=24, alignment=TA_CENTER, textColor=BRAND, spaceAfter=4),
            "subtitle": ParagraphStyle("RptSubtitle", fontName="Helvetica", fontSize=10.5, leading=14, alignment=TA_CENTER, textColor=MUTED, spaceAfter=14),
            "h1": ParagraphStyle("RptH1", fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=BRAND, spaceBefore=13, spaceAfter=6),
            "body": ParagraphStyle("RptBody", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=dark_text, spaceAfter=5),
            "small": ParagraphStyle("RptSmall", fontName="Helvetica", fontSize=8, leading=11, textColor=MUTED),
            "th": ParagraphStyle("RptTh", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.white),
            "td": ParagraphStyle("RptTd", fontName="Helvetica", fontSize=8.5, leading=11, textColor=dark_text),
            "bullet": ParagraphStyle("RptBullet", fontName="Helvetica", fontSize=9.5, leading=13.5, leftIndent=14, bulletIndent=4, spaceAfter=3, textColor=dark_text),
            "caption": ParagraphStyle("RptCaption", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=MUTED, spaceBefore=2, spaceAfter=3),
        }

    @staticmethod
    def _footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B7BFC9"))
        canvas.setLineWidth(0.6)
        canvas.line(doc.leftMargin, 38, A4[0] - doc.rightMargin, 38)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 27, "Automated Manufacturing Defect Detection - Quality Audit")
        canvas.drawRightString(A4[0] - doc.rightMargin, 27, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    @staticmethod
    def _data_table(header, rows, widths, styles) -> Table:
        data = [[Paragraph(f"<b>{h}</b>", styles["th"]) for h in header]]
        for row in rows:
            data.append([Paragraph(str(c), styles["td"]) for c in row])
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB4C0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return table

    @staticmethod
    def _kv_table(rows, styles) -> Table:
        data = [
            [Paragraph(f"<b>{label}</b>", styles["td"]), Paragraph(str(value), styles["td"])]
            for label, value in rows
        ]
        table = Table(
            data,
            colWidths=[int(DOC_WIDTH * 0.42), int(DOC_WIDTH * 0.58)],
            hAlign="LEFT",
        )
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB4C0")),
            ("BACKGROUND", (0, 0), (0, -1), LIGHT),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return table

    def _build_pdf(self, path, batch_id, inspection_df, defect_df, summary, category_df, severity_df, charts, originals, annotated, failed_loads):
        doc = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=46,
            rightMargin=46,
            topMargin=50,
            bottomMargin=52,
            title=REPORT_TITLE,
            author="Automated Defect Detection System",
            subject=f"Quality audit report for batch {batch_id}",
        )
        styles = self._styles()
        story: List[Any] = []
        story += self._cover(batch_id, summary, styles)
        story += self._summary_section(summary, styles)
        story += self._statistics_section(category_df, severity_df, charts, styles)
        story += self._inspection_table_section(inspection_df, styles)
        story += self._failed_review_section(inspection_df, defect_df, failed_loads, styles)
        story += self._examples_section(inspection_df, originals, annotated, styles)
        story += self._observations_section(inspection_df, defect_df, summary, category_df, severity_df, styles)
        story += self._actions_section(summary, styles)
        story += self._method_section(styles)
        doc.build(story, onFirstPage=self._footer, onLaterPages=self._footer)


    # ------------------------------------------------------------------ #
    # report sections
    # ------------------------------------------------------------------ #
    @staticmethod
    def _overall_status(summary: Dict[str, Any]) -> str:
        if int(summary.get("failed", 0)) > 0:
            return "FAIL"
        if int(summary.get("review_required", 0)) > 0:
            return "REVIEW REQUIRED"
        return "PASS"

    def _cover(self, batch_id, summary, styles) -> List[Any]:
        return [
            Spacer(1, 26),
            Paragraph(REPORT_TITLE, styles["title"]),
            Paragraph(
                "Rule-based OpenCV inspection batch - automated executive summary",
                styles["subtitle"],
            ),
            self._kv_table(
                [
                    ("Batch ID", batch_id),
                    ("Generated at", summary.get("generated_at", "")),
                    ("Overall batch status", self._overall_status(summary)),
                    ("Images processed", summary.get("images_processed", 0)),
                    ("Defect regions detected", summary.get("total_defect_regions", 0)),
                ],
                styles,
            ),
            Spacer(1, 10),
            Paragraph(
                "This document was generated automatically from the inspection "
                "records of the batch above. Every figure reflects the actual "
                "pipeline outputs for this batch; no values are estimated or "
                "simulated.",
                styles["small"],
            ),
            PageBreak(),
        ]

    def _summary_section(self, summary, styles) -> List[Any]:
        story = [Paragraph("1. Inspection Summary", styles["h1"])]
        rows = [
            ("Images found in input", summary.get("images_found", 0)),
            ("Images processed", summary.get("images_processed", 0)),
            ("Images that failed to load", summary.get("images_failed_load", 0)),
            ("Images inspected", summary.get("images_inspected", 0)),
            ("Defective images (>= 1 defect region)", summary.get("defective_images", 0)),
            ("Total defect regions", summary.get("total_defect_regions", 0)),
            ("Passed", summary.get("passed", 0)),
            ("Failed", summary.get("failed", 0)),
            ("Review required", summary.get("review_required", 0)),
            ("Processing errors", summary.get("processing_errors", 0)),
            ("Defect rate (image level)", f"{float(summary.get('defect_rate_pct', 0.0)):.2f} %"),
            ("Pass rate", f"{float(summary.get('pass_rate_pct', 0.0)):.2f} %"),
            ("Avg defect regions per image", f"{float(summary.get('avg_defects_per_image', 0.0)):.2f}"),
            ("Most frequent defect type", summary.get("most_frequent_defect_type", "None")),
        ]
        story.append(self._kv_table(rows, styles))
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "Definitions: an image is <i>defective</i> when at least one defect "
                "region was detected in it. The defect rate is the share of inspected "
                "images that are defective; the pass rate is the share of inspected "
                'images with status "Pass". Images that could not be read are '
                "excluded from both rates.",
                styles["body"],
            )
        )
        return story

    @staticmethod
    def _chart_image(path: Path, max_width: float, max_height: float) -> RLImage:
        with PILImage.open(path) as img:
            width, height = img.size
        scale = min(max_width / width, max_height / height)
        return RLImage(str(path), width=width * scale, height=height * scale)

    def _statistics_section(self, category_df, severity_df, charts, styles) -> List[Any]:
        story = [Paragraph("2. Defect Statistics", styles["h1"])]
        if category_df is not None and not category_df.empty:
            rows = [
                [row["category"], int(row["region_count"]), int(row["image_count"]),
                 f"{float(row['share_pct']):.2f}"]
                for _, row in category_df.iterrows()
            ]
            widths = [DOC_WIDTH * 0.4, DOC_WIDTH * 0.2, DOC_WIDTH * 0.2, DOC_WIDTH * 0.2]
            story.append(self._data_table(["Category", "Regions", "Images", "Share %"], rows, widths, styles))
            story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("No defect regions were detected in this batch.", styles["body"]))
        if severity_df is not None and not severity_df.empty:
            rows = [
                [row["severity"], int(row["region_count"]), f"{float(row['share_pct']):.2f}"]
                for _, row in severity_df.iterrows()
            ]
            widths = [DOC_WIDTH * 0.4, DOC_WIDTH * 0.3, DOC_WIDTH * 0.3]
            story.append(Spacer(1, 4))
            story.append(self._data_table(["Severity", "Regions", "Share %"], rows, widths, styles))
        story.append(Spacer(1, 8))
        if "category" in charts:
            story.append(self._chart_image(charts["category"], DOC_WIDTH, 300))
            story.append(Spacer(1, 6))
        pair = []
        if "status" in charts:
            pair.append(self._chart_image(charts["status"], DOC_WIDTH * 0.48, 230))
        if "histogram" in charts:
            pair.append(self._chart_image(charts["histogram"], DOC_WIDTH * 0.48, 230))
        if pair:
            story.append(Table([pair], colWidths=[DOC_WIDTH * 0.5, DOC_WIDTH * 0.5], hAlign="CENTER"))
            story.append(Spacer(1, 6))
        if "severity" in charts:
            story.append(self._chart_image(charts["severity"], DOC_WIDTH * 0.55, 235))
        return story


    def _inspection_table_section(self, inspection_df, styles) -> List[Any]:
        story = [Paragraph("3. Inspection Results", styles["h1"])]
        inspected = inspection_df[inspection_df["status"].ne(STATUS_ERROR)]
        rows = [
            [
                row["image_id"],
                row["filename"],
                row["status"],
                int(row["num_defects"]),
                row["max_severity"],
                f"{float(row['defect_area_ratio']):.4f}",
                f"{float(row['processing_time_ms']):.1f}",
            ]
            for _, row in inspected.iterrows()
        ]
        if not rows:
            story.append(Paragraph("No image could be inspected in this batch.", styles["body"]))
            return story
        if len(rows) > self.max_table_rows:
            story.append(Paragraph(
                f"Showing the first {self.max_table_rows} of {len(inspected)} inspected "
                "images; the complete table is available in inspection_log.csv and "
                "inspection_data.xlsx.",
                styles["small"],
            ))
            story.append(Spacer(1, 4))
            rows = rows[: self.max_table_rows]
        fractions = [0.11, 0.24, 0.15, 0.10, 0.14, 0.13, 0.13]
        widths = [DOC_WIDTH * f for f in fractions]
        header = ["Image ID", "Filename", "Status", "Regions", "Max severity", "Area ratio", "Time (ms)"]
        story.append(self._data_table(header, rows, widths, styles))
        return story

    def _failed_review_section(self, inspection_df, defect_df, failed_loads, styles) -> List[Any]:
        story = [Paragraph("4. Failed and Review-Required Inspections", styles["h1"])]
        if failed_loads:
            names = ", ".join(item["filename"] for item in failed_loads)
            story.append(Paragraph(
                f"{len(failed_loads)} image(s) could not be read and were excluded from "
                f"the inspection: {names}.",
                styles["body"],
            ))
        defect_lookup: Dict[str, List[str]] = {}
        if defect_df is not None and not defect_df.empty:
            for image_id, group in defect_df.groupby("image_id"):
                defect_lookup[str(image_id)] = [
                    f"{row['category']} ({row['severity']})" for _, row in group.iterrows()
                ]
        interesting = inspection_df[inspection_df["status"].isin([STATUS_FAIL, STATUS_REVIEW])]
        if interesting.empty:
            story.append(Paragraph(
                "No image ended with status Fail or Review Required in this batch.",
                styles["body"],
            ))
            return story
        for _, row in interesting.iterrows():
            details = ", ".join(defect_lookup.get(str(row["image_id"]), [])) or "no region details recorded"
            story.append(Paragraph(
                f"- <b>{row['image_id']}</b> - {row['filename']} - <b>{row['status']}</b> - "
                f"{int(row['num_defects'])} region(s): {details}. Reason: {row['status_reason']}.",
                styles["bullet"],
            ))
        return story


    def _examples_section(self, inspection_df, originals, annotated, styles) -> List[Any]:
        story = [Paragraph("5. Original vs Annotated Examples", styles["h1"])]
        inspected = inspection_df[inspection_df["status"].ne(STATUS_ERROR)]
        if inspected.empty:
            story.append(Paragraph("No images were inspected, so no examples exist.", styles["body"]))
            return story
        status_rank = {STATUS_FAIL: 0, STATUS_REVIEW: 1, STATUS_PASS: 2}
        ordered = inspected.copy()
        ordered["_rank"] = ordered["status"].map(status_rank).fillna(3)
        ordered = ordered.sort_values(["_rank", "num_defects"], ascending=[True, False])
        half_width = DOC_WIDTH * 0.5 - 6
        examples_added = 0
        for _, row in ordered.iterrows():
            if examples_added >= self.max_examples:
                break
            annotated_path = annotated.get(str(row["image_id"]))
            original_path = originals.get(str(row["image_id"]))
            right = self._image_flowable(annotated_path, half_width, 200)
            if right is None:
                continue
            left = self._image_flowable(original_path, half_width, 200)
            story.append(Paragraph(
                f"{row['image_id']} - {row['filename']} - status {row['status']} "
                f"({int(row['num_defects'])} defect region(s))",
                styles["caption"],
            ))
            if left is not None:
                labels = Table(
                    [[Paragraph("Original", styles["small"]), Paragraph("Annotated", styles["small"])]],
                    colWidths=[DOC_WIDTH * 0.5, DOC_WIDTH * 0.5],
                )
                labels.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
                story.append(labels)
                story.append(Table(
                    [[left, right]],
                    colWidths=[DOC_WIDTH * 0.5, DOC_WIDTH * 0.5],
                    hAlign="CENTER",
                ))
            else:
                story.append(right)
            story.append(Spacer(1, 8))
            examples_added += 1
        if examples_added == 0:
            story.append(Paragraph("No annotated image previews were available on disk.", styles["body"]))
        return story

    @staticmethod
    def _image_flowable(path, max_width: float, max_height: float):
        if not path:
            return None
        path = Path(path)
        if not path.exists():
            return None
        try:
            with PILImage.open(path) as img:
                width, height = img.size
            scale = min(max_width / width, max_height / height)
            return RLImage(str(path), width=width * scale, height=height * scale)
        except Exception:
            return None


    def _observations_section(self, inspection_df, defect_df, summary, category_df, severity_df, styles) -> List[Any]:
        story = [Paragraph("6. Key Observations", styles["h1"])]
        inspected = int(summary.get("images_inspected", 0))
        observations: List[str] = []
        if inspected:
            observations.append(
                f"{int(summary.get('defective_images', 0))} of {inspected} inspected "
                f"images ({float(summary.get('defect_rate_pct', 0.0)):.2f} %) contain at "
                "least one detected defect region."
            )
        if int(summary.get("images_failed_load", 0)):
            observations.append(
                f"{int(summary['images_failed_load'])} image(s) could not be read "
                "(corrupted or unsupported); they are excluded from all statistics."
            )
        if category_df is not None and not category_df.empty:
            top = category_df.iloc[0]
            observations.append(
                f"The most frequent defect category is \"{top['category']}\" with "
                f"{int(top['region_count'])} region(s) across {int(top['image_count'])} image(s)."
            )
        if severity_df is not None and not severity_df.empty:
            sev = {str(r["severity"]): int(r["region_count"]) for _, r in severity_df.iterrows()}
            critical = sev.get(SEVERITY_CRITICAL, 0)
            major = sev.get(SEVERITY_MAJOR, 0)
            minor = sev.get(SEVERITY_MINOR, 0)
            observations.append(
                f"Severity mix: {critical} critical, {major} major and {minor} minor region(s)."
            )
            if critical:
                observations.append(
                    "Critical regions fail their images immediately and should be "
                    "verified by a human inspector before any disposition decision."
                )
        if defect_df is not None and not defect_df.empty and inspected:
            worst = defect_df.groupby("image_id")["area_px"].sum().sort_values(ascending=False)
            if not worst.empty:
                top_image = str(worst.index[0])
                filename = str(defect_df.loc[defect_df["image_id"] == top_image, "filename"].iloc[0])
                observations.append(
                    f"The largest cumulative defect area appears in {top_image} "
                    f"({filename}) with {int(worst.iloc[0])} affected pixels."
                )
        if not observations:
            observations.append("No defects were detected; all inspected images passed.")
        for text in observations:
            story.append(Paragraph(f"- {text}", styles["bullet"]))
        return story

    def _actions_section(self, summary, styles) -> List[Any]:
        story = [Paragraph("7. Suggested Follow-up Actions", styles["h1"])]
        actions: List[str] = []
        if int(summary.get("failed", 0)):
            actions.append(
                f"Manually inspect the {int(summary['failed'])} failed image(s); "
                "critical regions trigger automatic failure."
            )
        if int(summary.get("review_required", 0)):
            actions.append(
                f"Schedule a quality-engineer review for the "
                f"{int(summary['review_required'])} image(s) flagged \"Review Required\"."
            )
        if int(summary.get("images_failed_load", 0)):
            actions.append(
                "Re-export or re-scan the unreadable image file(s) and rerun the batch "
                "so they are included in the statistics."
            )
        if int(summary.get("total_defect_regions", 0)):
            actions.append(
                "Cross-check the top defect category against the process step most "
                "likely to cause it (e.g. handling scratches vs machining dents)."
            )
            actions.append(
                "If thresholds need tuning, adjust the values in config/settings.yaml "
                "and rerun the batch; no code change is required."
            )
        if not actions:
            actions.append(
                "No defects were detected. Continue routine sampling and keep the "
                "current thresholds under periodic review."
            )
        for text in actions:
            story.append(Paragraph(f"- {text}", styles["bullet"]))
        return story

    def _method_section(self, styles) -> List[Any]:
        story = [Paragraph("8. Detection Method and Limitations", styles["h1"])]
        story.append(Paragraph(
            "Detection is <b>rule-based heuristic computer vision</b>, not a trained "
            "machine-learning model. Every threshold used below is defined in "
            "config/settings.yaml.",
            styles["body"],
        ))
        story.append(Paragraph("Active detection rules for this batch:", styles["body"]))
        for line in describe_detection_rules(self.settings):
            story.append(Paragraph(f"- {line}", styles["bullet"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph("Severity and status rules:", styles["body"]))
        for line in describe_severity_rules(self.settings):
            story.append(Paragraph(f"- {line}", styles["bullet"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Limitations: heuristic thresholds are sensitive to lighting, surface "
            "texture and camera setup; categories are geometric interpretations and "
            "can be confused (e.g. a deep scratch may register as a crack); defects "
            "below the configured contrast or area floors are missed; and no "
            "ground-truth labels were used, so no accuracy metric is claimed. For "
            "statistically validated classification, connect the optional ONNX model "
            "hook described in README section 7.",
            styles["body"],
        ))
        return story







