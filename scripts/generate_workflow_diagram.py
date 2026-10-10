"""Render the system workflow diagram (docs/assets/workflow_diagram.png)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT_PATH = PROJECT_ROOT / "docs" / "assets" / "workflow_diagram.png"

STAGES = [
    ("Input", "Folder / uploaded files\n(JPG, JPEG, PNG, BMP)"),
    ("Image Loader", "Validate + decode,\nunique image IDs,\ncorruption handling"),
    ("Preprocessing", "Resize + grayscale, denoise,\nnormalize / CLAHE, Canny edges"),
    ("Defect Detection", "Spots, scratches, cracks, dents,\ndiscoloration, reference comparison"),
    ("Classification", "Region severity + image status\n(Pass / Fail / Review Required)"),
    ("Outputs", "Annotated images, resized\noriginals, batch log"),
    ("Analysis + Export", "Pandas datasets, quality metrics,\nCSV + Excel export"),
    ("Executive Report", "Charts, tables, examples\n(PDF via ReportLab)"),
]


def main() -> int:
    fig, ax = plt.subplots(figsize=(8.6, 11.0), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    count = len(STAGES)
    top, bottom = 10.0, 0.5
    slot = (top - bottom) / count
    box_h = slot * 0.74
    x0, box_w = 1.2, 4.9
    centers = []
    for index, (title, body) in enumerate(STAGES):
        cy = top - slot * (index + 0.5)
        centers.append(cy)
        box = FancyBboxPatch(
            (x0, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.3, edgecolor="#1F3864", facecolor="#EEF3F9",
        )
        ax.add_patch(box)
        ax.text(x0 + box_w / 2, cy + box_h * 0.24, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#1F3864")
        ax.text(x0 + box_w / 2, cy - box_h * 0.22, body, ha="center", va="center",
                fontsize=8.2, color="#333B45")
        if index > 0:
            ax.add_patch(FancyArrowPatch(
                (x0 + box_w / 2, centers[index - 1] - box_h / 2),
                (x0 + box_w / 2, cy + box_h / 2),
                arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#2C6FBB",
            ))

    dash_x, dash_w = 7.25, 2.5
    dash = FancyBboxPatch(
        (dash_x, 3.4), dash_w, 3.5,
        boxstyle="round,pad=0.08,rounding_size=0.14",
        linewidth=1.3, edgecolor="#2C6FBB", facecolor="#E8F1FB", linestyle="--",
    )
    ax.add_patch(dash)
    ax.text(dash_x + dash_w / 2, 6.55, "Streamlit Dashboard", ha="center",
            fontsize=10.5, fontweight="bold", color="#1F3864")
    labels = ["Upload / folder input", "Live progress", "KPI metrics", "Charts",
              "Filterable tables", "Image previews", "CSV/Excel/PDF downloads"]
    for offset, label in enumerate(labels):
        ax.text(dash_x + 0.14, 6.05 - offset * 0.42, "- " + label,
                fontsize=8.0, color="#333B45", ha="left")

    for cy in (centers[4], centers[6]):
        ax.add_patch(FancyArrowPatch(
            (x0 + box_w, cy), (dash_x, cy),
            arrowstyle="-|>", mutation_scale=13, linewidth=1.1,
            color="#7A93B5", linestyle="--",
        ))

    ax.text(5.0, 10.4, "Manufacturing Defect Detection - Workflow",
            ha="center", fontsize=13, fontweight="bold", color="#1F3864")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
