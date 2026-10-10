"""Integration tests: full batch run, outputs and report generation guard."""
import cv2
import numpy as np
import pandas as pd
import pytest

from src.data_logger import DEFECT_COLUMNS, INSPECTION_COLUMNS
from src.pipeline import PipelineError, run_batch
from src.report_generator import ExecutiveReportGenerator, ReportGenerationError
from src.utils import default_settings


def _write_sample(folder, name, painter, seed=11):
    rng = np.random.default_rng(seed)
    img = np.clip(178 + rng.normal(0.0, 5.0, (360, 480)), 0, 255).astype(np.uint8)
    painter(img)
    cv2.imwrite(str(folder / name), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


def _paint_l_dent(img):
    cv2.rectangle(img, (120, 150), (330, 200), 120, -1)
    cv2.rectangle(img, (120, 150), (180, 290), 120, -1)


def test_full_batch_produces_all_outputs(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    _write_sample(folder, "clean.png", lambda img: None, seed=21)
    _write_sample(folder, "defect.png", _paint_l_dent, seed=22)

    result = run_batch(
        input_folder=folder,
        settings=default_settings(),
        output_root=tmp_path / "outputs",
    )

    assert (result.batch_dir / "csv" / "inspection_log.csv").exists()
    assert (result.batch_dir / "csv" / "defect_details.csv").exists()
    assert (result.batch_dir / "excel" / "inspection_data.xlsx").exists()
    assert result.annotated and result.originals
    for path in result.annotated.values():
        assert path.exists()
    assert "report" in result.files
    report_bytes = result.files["report"].read_bytes()
    assert report_bytes[:5] == b"%PDF-"
    assert len(report_bytes) > 20000

    defect_row = result.inspection_df[result.inspection_df["filename"] == "defect.png"].iloc[0]
    assert defect_row["status"] in ("Fail", "Review Required")
    clean_row = result.inspection_df[result.inspection_df["filename"] == "clean.png"].iloc[0]
    assert clean_row["status"] == "Pass"


def test_empty_folder_raises(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(PipelineError):
        run_batch(input_folder=folder, settings=default_settings(), output_root=tmp_path / "out")


def test_folder_with_only_corrupt_files_raises(tmp_path):
    folder = tmp_path / "corrupt"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"junk")
    (folder / "b.jpg").write_bytes(b"more junk")
    with pytest.raises(PipelineError):
        run_batch(input_folder=folder, settings=default_settings(), output_root=tmp_path / "out")


def test_report_blocked_without_records(tmp_path):
    generator = ExecutiveReportGenerator(default_settings())
    with pytest.raises(ReportGenerationError):
        generator.generate(
            batch_id="B",
            inspection_df=pd.DataFrame(columns=INSPECTION_COLUMNS),
            defect_df=pd.DataFrame(columns=DEFECT_COLUMNS),
            summary={},
            category_df=pd.DataFrame(),
            severity_df=pd.DataFrame(),
            originals={},
            annotated={},
            out_dir=tmp_path,
        )
