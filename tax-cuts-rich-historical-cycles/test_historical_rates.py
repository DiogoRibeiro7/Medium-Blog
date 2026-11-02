"""Independent structural and landmark tests for the IRS transcription."""

import csv
import tempfile
import unittest
from pathlib import Path

from historical_rates import DATA_FILE, Observation, load_rates, render_svg


class HistoricalRateTests(unittest.TestCase):
    """Check source-year coverage and known statutory turning points."""

    def test_complete_years_and_landmarks(self) -> None:
        observations = load_rates()
        self.assertEqual(len(observations), 55)
        rates = {row.year: row.top_rate_pct for row in observations}
        for year, rate in {
            1964: 77.0,
            1965: 70.0,
            1980: 70.0,
            1982: 50.0,
            1987: 38.5,
            1988: 28.0,
            1993: 39.6,
            2003: 35.0,
            2013: 39.6,
            2018: 37.0,
        }.items():
            with self.subTest(year=year):
                self.assertAlmostEqual(rates[year], rate)

    def test_schema_and_gaps_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "altered.csv"
            with DATA_FILE.open(encoding="utf-8", newline="") as handle:
                original = list(csv.reader(handle))
            altered = [row for row in original if row[0] != "1982"]
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(altered)
            with self.assertRaisesRegex(ValueError, "unique, sorted, and complete"):
                load_rates(path)

    def test_invalid_values_rejected(self) -> None:
        for values in ((1960, 70), (1980, -1), (1980, float("nan")), (1980.0, 70)):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                Observation(*values)

    def test_svg_deterministic_and_caveated(self) -> None:
        observations = load_rates()
        image = render_svg(observations)
        self.assertEqual(image, render_svg(observations))
        self.assertIn("Statutory rates differ from effective burdens", image)
        self.assertIn("1964–2018", image)


if __name__ == "__main__":
    unittest.main()
