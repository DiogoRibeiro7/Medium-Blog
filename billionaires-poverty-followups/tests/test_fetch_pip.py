"""Offline contracts for fetch_pip.py using a fake PIP server."""

from __future__ import annotations

import gzip
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_pip  # noqa: E402

SURVEYS = [
    ("AAA", 2022, "income", "national", 20.0, 15.0),
    ("BBB", 2021, "consumption", "national", 5.0, 3.5),
    ("CCC", 2023, "income", "urban", 40.0, 32.0),
]


def fake_csv(line: float, drop: str | None = None) -> bytes:
    rows = []
    for code, year, welfare, level, mean, median in SURVEYS:
        if code == drop:
            continue
        head = min(1.0, line / (2 * median))
        rows.append({"country_code": code, "reporting_year": year, "welfare_type": welfare,
                     "reporting_level": level, "country_name": code, "poverty_line": line,
                     "headcount": head, "poverty_gap": head / 3, "mean": mean, "median": median,
                     "gini": 0.4, "decile1": 0.03, "decile10": 0.3, "survey_year": year})
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode()


def fake_fetcher(url: str, *, name: str, **_: object) -> bytes:
    if "versions" in url or "pip-info" in url:
        return json.dumps({"release": "test"}).encode()
    line = float(url.split("povline=")[1].split("&")[0])
    return fake_csv(line)


class FetchPipTests(unittest.TestCase):
    def test_url_uses_2021_ppp_and_no_gap_filling(self) -> None:
        url = fetch_pip.pip_url(8.3)
        self.assertIn("povline=8.3", url)
        self.assertIn("ppp_version=2021", url)
        self.assertIn("fill_gaps=false", url)

    def test_grid_contains_official_and_first_article_lines(self) -> None:
        for line in (3.0, 4.2, 8.3, 30.0):
            self.assertIn(line, fetch_pip.POVERTY_LINES)
        self.assertEqual(list(fetch_pip.POVERTY_LINES), sorted(fetch_pip.POVERTY_LINES))

    def test_fetch_writes_tables_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = fetch_pip.fetch(out, fetcher=fake_fetcher, lines=(3.0, 30.0))
            surveys = pd.read_csv(out / "pip_surveys.csv.gz")
            heads = pd.read_csv(out / "pip_headcounts.csv.gz")
            self.assertEqual(len(surveys), 3)
            self.assertEqual(len(heads), 6)
            self.assertEqual(manifest["welfare_type_counts"], {"income": 2, "consumption": 1})
            self.assertEqual(len(manifest["responses"]), 2)
            raw = gzip.decompress((out / "raw_pip" / "pip_3.csv.gz").read_bytes())
            self.assertEqual(raw, fake_csv(3.0))

    def test_rejects_wrong_line_and_changing_survey_set(self) -> None:
        with self.assertRaises(ValueError):
            fetch_pip.parse_pip_csv(fake_csv(3.0), 30.0)
        responses = {3.0: fetch_pip.parse_pip_csv(fake_csv(3.0), 3.0),
                     30.0: fetch_pip.parse_pip_csv(fake_csv(30.0, drop="BBB"), 30.0)}
        with self.assertRaises(ValueError):
            fetch_pip.build_tables(responses)

    def test_rejects_decreasing_headcount(self) -> None:
        low = fetch_pip.parse_pip_csv(fake_csv(3.0), 3.0)
        high = fetch_pip.parse_pip_csv(fake_csv(30.0), 30.0)
        high["headcount"] = 0.0
        with self.assertRaises(ValueError):
            fetch_pip.build_tables({3.0: low, 30.0: high})


if __name__ == "__main__":
    unittest.main()
