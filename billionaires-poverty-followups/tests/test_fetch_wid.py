"""Offline contract tests for fetch_wid.py, using a synthetic WID archive."""

from __future__ import annotations

import gzip
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_wid  # noqa: E402

HEADER = "country;variable;percentile;year;value;age;pop\n"


def _data(country: str, rows: list[tuple[str, str, int, str]], split: bool = False) -> bytes:
    lines = [HEADER]
    for var, pct, year, value in rows:
        if split:
            lines.append(f"{country};{var[:6]};{pct};{year};{value};{var[6:9]};{var[9]}\n")
        else:
            lines.append(f"{country};{var};{pct};{year};{value};{var[6:9]};{var[9]}\n")
    return "".join(lines).encode()


def _archive(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)


US_ROWS = [
    ("shweal992j", "p99p100", 2022, "0.348"),
    ("shweal992j", "p90p100", 2022, "0.705"),
    ("shweal992j", "p99p100", 1970, "0.30"),   # before MIN_YEAR: dropped
    ("shweal992j", "p20p30", 2022, "0.01"),    # percentile not kept
    ("sfiinc992j", "p99p100", 2022, "0.2"),    # variable not kept
    ("anninc992i", "p0p100", 2022, "70000"),
    ("sptinc992j", "p99p100", 2023, ""),       # empty value: dropped
]


class ParseTests(unittest.TestCase):
    def test_keeps_only_selected_series(self) -> None:
        out = fetch_wid.parse_data_member(_data("US", US_ROWS), "US")
        self.assertEqual(len(out), 3)
        self.assertEqual(set(out.variable), {"shweal992j", "anninc992i"})
        self.assertTrue((out.year >= fetch_wid.MIN_YEAR).all())

    def test_split_codes_are_joined(self) -> None:
        out = fetch_wid.parse_data_member(_data("US", US_ROWS, split=True), "US")
        self.assertEqual(sorted(out.variable.unique()), ["anninc992i", "shweal992j"])

    def test_rejects_foreign_rows(self) -> None:
        raw = _data("US", US_ROWS[:1]) + b"FR;shweal992j;p99p100;2022;0.27;992;j\n"
        with self.assertRaises(ValueError):
            fetch_wid.parse_data_member(raw, "US")

    def test_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            fetch_wid.parse_data_member(_data("US", US_ROWS[:1] * 2), "US")

    def test_rejects_impossible_share(self) -> None:
        with self.assertRaises(ValueError):
            fetch_wid.parse_data_member(_data("US", [("shweal992j", "p99p100", 2022, "1.7")]), "US")


class ExtractTests(unittest.TestCase):
    def test_writes_tables_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            meta = ("country;variable;shortname;source;method\n"
                    "US;shweal992j;Net personal wealth;Saez and Zucman;tax data\n"
                    "US;sfiinc992j;Fiscal income;IRS;tax data\n").encode()
            _archive(tmp / "wid.zip", {
                "WID_data_US.csv": _data("US", US_ROWS),
                "WID_data_FR.csv": _data("FR", [("shweal992j", "p99p100", 2022, "0.27")]),
                "WID_data_US-CA.csv": _data("US-CA", [("shweal992j", "p99p100", 2022, "0.4")]),
                "WID_metadata_US.csv": meta,
                "WID_countries.csv": b"alpha2;titlename\nUS;USA\n",
            })
            manifest = fetch_wid.extract(tmp / "wid.zip", tmp / "out")
            series = pd.read_csv(io.BytesIO(gzip.decompress((tmp / "out" / "wid_series.csv.gz").read_bytes())))
            self.assertEqual(sorted(series.country.unique()), ["FR", "US"])  # no subnational
            self.assertEqual(manifest["n_rows"], 4)
            self.assertEqual(manifest["latest_year_by_variable"]["shweal992j"], 2022)
            meta_out = pd.read_csv(io.BytesIO(gzip.decompress((tmp / "out" / "wid_metadata.csv.gz").read_bytes())))
            self.assertEqual(list(meta_out.variable), ["shweal992j"])
            saved = json.loads((tmp / "out" / "wid_manifest.json").read_text())
            self.assertEqual(saved["archive_sha256"], manifest["archive_sha256"])
            self.assertIn("WID_countries.csv", saved["other_members"])

    def test_fails_without_us_wealth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _archive(tmp / "wid.zip", {"WID_data_FR.csv": _data("FR", [("shweal992j", "p99p100", 2022, "0.27")])})
            with self.assertRaises(ValueError):
                fetch_wid.extract(tmp / "wid.zip", tmp / "out")


if __name__ == "__main__":
    unittest.main()
