"""Regression tests for the public, manuscript-independent publication contract."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from iberian_lp.publication import check_scientific, scientific_outputs


def test_scientific_values_do_not_require_a_medium_template(tmp_path: Path) -> None:
    """Scientific values can be produced with no editorial source or article."""
    with patch("iberian_lp.publication.values", return_value={"estimate": "−1.25"}):
        generated = scientific_outputs(tmp_path)

    assert list(generated) == ["publication/values.json"]
    assert json.loads(generated["publication/values.json"]) == {"estimate": "−1.25"}
    assert not (tmp_path / "publication" / "article.template.md").exists()


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
