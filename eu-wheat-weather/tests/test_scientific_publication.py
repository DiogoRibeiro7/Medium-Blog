"""Numerical publication contracts independent of wheat and maize manuscripts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from eu_wheat_weather.cli import build_parser
from eu_wheat_weather.publication import (
    check_scientific,
    scientific_outputs,
    write_scientific,
)


def test_scientific_outputs_do_not_need_article_templates(tmp_path: Path) -> None:
    """Wheat values are available even when neither editorial template exists."""
    with patch("eu_wheat_weather.publication.collect_values", return_value={"e1": "2.6%"}) as wheat:
        with patch("eu_wheat_weather.publication.maize_values") as maize:
            actual = scientific_outputs(tmp_path)

    assert actual == {"publication/values.json": '{\n  "e1": "2.6%"\n}\n'}
    wheat.assert_called_once_with(tmp_path)
    maize.assert_not_called()


def test_maize_manifest_is_controlled_by_results_not_template(tmp_path: Path) -> None:
    """Maize values must remain reproducible after its article has moved."""
    maize_results = tmp_path / "docs" / "maize" / "results" / "results.json"
    maize_results.parent.mkdir(parents=True)
    maize_results.write_text("{}\n", encoding="utf-8")

    with patch("eu_wheat_weather.publication.collect_values", return_value={"wheat": "1"}):
        with patch("eu_wheat_weather.publication.maize_values", return_value={"maize": "2"}):
            generated = scientific_outputs(tmp_path)
            assert set(generated) == {
                "publication/values.json",
                "publication/maize/values.json",
            }
            assert json.loads(generated["publication/maize/values.json"]) == {"maize": "2"}
            assert set(check_scientific(tmp_path)) == set(generated)
            assert set(write_scientific(tmp_path)) == set(generated)
            assert check_scientific(tmp_path) == []

            maize_manifest = tmp_path / "publication" / "maize" / "values.json"
            maize_manifest.write_text('{"maize": "stale"}\n', encoding="utf-8")
            assert check_scientific(tmp_path) == ["publication/maize/values.json"]


def test_cli_accepts_independent_numerical_modes() -> None:
    """The public scientific modes never imply an editorial publication check."""
    parser = build_parser()
    values = parser.parse_args(["publication", "--values-only", "--no-figures"])
    check = parser.parse_args(["publication", "--check-values"])

    assert values.values_only is True
    assert values.check_values is False
    assert check.check_values is True
    assert check.check is False
