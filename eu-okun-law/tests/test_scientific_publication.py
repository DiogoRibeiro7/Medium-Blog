"""Publication regression tests that do not depend on private Medium prose."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from eu_okun.publication import (
    build_scientific,
    check_scientific,
    scientific_outputs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_scientific_outputs_never_read_medium_template() -> None:
    """Reproduce frozen numerical artifacts without reading editorial source."""
    original_read_text = Path.read_text

    def checked_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.name == "article.template.md":
            raise AssertionError("Scientific checks must not read Medium templates")
        return original_read_text(path, *args, **kwargs)

    with patch.object(Path, "read_text", checked_read_text):
        outputs = scientific_outputs(ROOT)

    assert "publication/medium/article.md" not in outputs
    assert "publication/manuscript/generated/values.tex" in outputs
    assert "publication/manifest/values.json" in outputs
    assert "publication/AUDIT.md" in outputs
    assert all((ROOT / name).read_text(encoding="utf-8") == content
               for name, content in outputs.items())


def test_scientific_build_preserves_existing_editorial_notice(tmp_path: Path) -> None:
    """Numerical-only generation cannot replace a public article-path notice."""
    publication = tmp_path / "publication"
    publication.mkdir()
    notice = publication / "medium" / "article.md"
    notice.parent.mkdir()
    notice.write_text("Private editorial manuscript.\n", encoding="utf-8")

    expected = {"publication/manifest/values.json": "{}\n"}
    with patch("eu_okun.publication.scientific_outputs", return_value=expected):
        result = build_scientific(tmp_path, figures_too=False)

    assert result == expected
    assert (publication / "manifest" / "values.json").read_text(encoding="utf-8") == "{}\n"
    assert notice.read_text(encoding="utf-8") == "Private editorial manuscript.\n"
    assert not (publication / "medium" / "article.template.md").exists()


def test_scientific_check_detects_missing_and_stale_research_files(tmp_path: Path) -> None:
    """Scientific checks still fail for missing or incorrect committed values."""
    rel = "publication/manifest/values.json"
    expected = {rel: '{"estimate": 1}\n'}
    path = tmp_path / rel
    with patch("eu_okun.publication.scientific_outputs", return_value=expected):
        assert check_scientific(tmp_path) == [rel]
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
        assert check_scientific(tmp_path) == [rel]
        path.write_text(expected[rel], encoding="utf-8")
        assert check_scientific(tmp_path) == []


def test_cli_scientific_modes_work_without_medium_source(tmp_path: Path) -> None:
    """The CLI can write and validate scientific artifacts in an isolated directory."""
    from eu_okun.cli import main

    rel = "publication/manifest/values.json"
    expected = {rel: '{"estimate": 1}\n'}
    with (
        patch("eu_okun.publication_values.audit", return_value=[]),
        patch("eu_okun.publication.scientific_outputs", return_value=expected),
    ):
        assert main(["publication", "--root", str(tmp_path),
                     "--scientific-only", "--no-figures"]) == 0
        assert (tmp_path / rel).read_text(encoding="utf-8") == expected[rel]
        assert main(["publication", "--root", str(tmp_path), "--check-scientific"]) == 0
        assert not (tmp_path / "publication" / "medium" / "article.md").exists()
