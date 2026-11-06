"""Regression tests for the public, manuscript-independent publication contract."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from iberian_lp.publication import check_scientific, scientific_outputs, write_publication


def test_scientific_values_do_not_require_a_medium_template(tmp_path: Path) -> None:
    """Scientific values can be produced with no editorial source or article."""
    with patch("iberian_lp.publication.values", return_value={"estimate": "−1.25"}):
        generated = scientific_outputs(tmp_path)

    assert list(generated) == ["publication/values.json"]
    assert json.loads(generated["publication/values.json"]) == {"estimate": "−1.25"}
    assert not (tmp_path / "publication" / "article.template.md").exists()


def test_default_publication_preserves_editorial_notice(tmp_path: Path) -> None:
    """The ordinary publication command cannot replace a historical-path notice."""
    publication = tmp_path / "publication"
    publication.mkdir()
    notice = publication / "article.md"
    notice.write_text("Private editorial archive notice.\n", encoding="utf-8")

    with patch("iberian_lp.publication.values", return_value={"estimate": "−1.25"}):
        written = write_publication(tmp_path, figures=False)

    assert written == ["publication/values.json"]
    assert notice.read_text(encoding="utf-8") == "Private editorial archive notice.\n"
    assert not (publication / "article.template.md").exists()


def test_scientific_check_detects_missing_and_stale_values(tmp_path: Path) -> None:
    """The research-values check remains strict after the manuscript moves."""
    destination = tmp_path / "publication" / "values.json"
    with patch("iberian_lp.publication.values", return_value={"estimate": "−1.25"}):
        assert check_scientific(tmp_path) == ["publication/values.json"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}\n", encoding="utf-8")
        assert check_scientific(tmp_path) == ["publication/values.json"]
        destination.write_text(scientific_outputs(tmp_path)["publication/values.json"],
                               encoding="utf-8")
        assert check_scientific(tmp_path) == []


def test_cli_values_only_and_check_values_without_manuscript(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI generation and verification do not need editorial Markdown."""
    from iberian_lp.cli import main

    root_flag = ["--root", str(tmp_path)]
    with patch("iberian_lp.publication.values", return_value={"estimate": "−1.25"}):
        main(["publication", *root_flag, "--values-only"])
        assert (tmp_path / "publication" / "values.json").is_file()
        assert not (tmp_path / "publication" / "article.md").exists()
        assert not (tmp_path / "publication" / "article.template.md").exists()
        main(["publication", *root_flag, "--check-values"])
        assert "current" in capsys.readouterr().out

        (tmp_path / "publication" / "values.json").write_text("{}\n", encoding="utf-8")
        with pytest.raises(SystemExit) as error:
            main(["publication", *root_flag, "--check-values"])
        assert error.value.code == 2
