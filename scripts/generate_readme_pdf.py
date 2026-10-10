"""Generate README.pdf from README.md via src/readme_pdf.py.

Usage:
    .venv\\Scripts\\python.exe scripts\\generate_readme_pdf.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.readme_pdf import build_readme_pdf  # noqa: E402

PROJECT_NAME = "Automated Manufacturing Defect Detection and Quality Audit System"
DESCRIPTION = (
    "Batch image inspection with OpenCV, Pandas quality analytics, a Streamlit "
    "dashboard and an automated executive PDF report."
)
FOOTER_NAME = "Manufacturing Defect Detection - Quality Audit"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate README.pdf from README.md.")
    parser.add_argument("--readme", default=str(PROJECT_ROOT / "README.md"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "README.pdf"))
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()

    readme_path = Path(args.readme)
    if not readme_path.exists():
        print(f"ERROR: {readme_path} not found")
        return 1
    out_path = Path(args.out)
    generated = datetime.now().strftime("%d %B %Y")
    path = build_readme_pdf(
        readme_path,
        out_path,
        project_name=PROJECT_NAME,
        description=DESCRIPTION,
        version=args.version,
        generated=generated,
        footer_name=FOOTER_NAME,
    )
    size_kb = path.stat().st_size / 1024.0
    print(f"wrote {path} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
