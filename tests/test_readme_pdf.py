"""Tests for the Markdown -> PDF README converter."""
from src.readme_pdf import build_readme_pdf

SAMPLE_MD = """# Title

Intro paragraph with **bold**, `inline code` and a [link](https://example.com).

## Section

- item one
- item two
  - nested item

1. first
2. second

| Col A | Col B |
| ----- | ----- |
| a     | b     |

```python
def hello():
    return "world"  # trailing comment
```

---

> a quoted remark
"""


def test_build_readme_pdf_produces_valid_pdf(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(SAMPLE_MD, encoding="utf-8")
    out = tmp_path / "README.pdf"
    build_readme_pdf(
        readme,
        out,
        project_name="Test Project",
        description="A short description.",
        version="1.0.0",
        generated="2026-01-01",
        footer_name="Test Project",
    )
    data = out.read_bytes()
    assert data[:5] == b"%PDF-"
    assert len(data) > 5000
