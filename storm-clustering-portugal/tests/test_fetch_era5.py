"""Offline tests of the ERA5 download helpers (no network)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_era5  # noqa: E402


class Era5(unittest.TestCase):
    def test_parse_response(self):
        body = (b'{"latitude":38.5,"longitude":-9.0,"elevation":120.0,'
                b'"daily":{"time":["2026-01-27","2026-01-28"],"wind_speed_10m_max":[9.4,null]}}')
        meta, rows = fetch_era5.parse_response(body)
        self.assertEqual(meta, {"lat_out": 38.5, "lon_out": -9.0, "elevation_m": 120.0})
        self.assertEqual(rows, [("2026-01-27", 9.4), ("2026-01-28", None)])

    def test_points_unique_and_in_portugal(self):
        self.assertEqual(len(set(fetch_era5.POINTS)), len(fetch_era5.POINTS))
        for lat, lon in fetch_era5.POINTS:
            self.assertTrue(36.9 <= lat <= 42.2 and -9.6 <= lon <= -6.1)

    def test_point_file_name(self):
        p = fetch_era5.point_file(Path("data"), 37.0, -8.5)
        self.assertEqual(p.name, "era5_37.00_-8.50.csv.gz")


if __name__ == "__main__":
    unittest.main()
