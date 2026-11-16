"""Offline tests of the download helpers (no network)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch  # noqa: E402


class OpenMeteo(unittest.TestCase):
    def test_parse_response_two_variables(self):
        body = (b'{"latitude":39.75,"longitude":-8.75,"elevation":60.0,'
                b'"daily":{"time":["2026-01-27","2026-01-28"],'
                b'"wind_speed_10m_max":[9.4,21.0],"wind_gusts_10m_max":[18.0,null]}}')
        meta, rows = fetch.parse_response(body, ["wind_speed_10m_max", "wind_gusts_10m_max"])
        self.assertEqual(meta, {"lat_out": 39.75, "lon_out": -8.75, "elevation_m": 60.0})
        self.assertEqual(rows, [("2026-01-27", 9.4, 18.0), ("2026-01-28", 21.0, None)])

    def test_jobs_unique_paths_and_order(self):
        js = fetch.jobs(Path("data"))
        self.assertEqual(len({j["path"] for j in js}), len(js))
        self.assertEqual(len(js), len(fetch.KRISTIN_SITES) + len(fetch.SITES) + 2 * len(fetch.NODES))
        self.assertTrue(js[0]["key"].startswith("kristin_"))
        self.assertTrue(js[-1]["key"].startswith("early_"))

    def test_url(self):
        j = [j for j in fetch.jobs(Path("data")) if j["key"].startswith("gust_")][-1]
        u = fetch.url_for(j)
        self.assertIn("daily=wind_gusts_10m_max", u)
        self.assertIn("models=era5", u)
        self.assertIn("start_date=1950-10-01", u)

    def test_points_in_portugal(self):
        pts = list(fetch.NODES) + [v[1:] for v in fetch.SITES.values()] + [v[:2] for v in fetch.KRISTIN_SITES.values()]
        for lat, lon in pts:
            self.assertTrue(36.9 <= lat <= 42.2 and -9.6 <= lon <= -6.1, (lat, lon))


class Reports(unittest.TestCase):
    def test_parse_reports_window(self):
        psv = ("Station_ID|DATE|Report_Type|wind_direction|wind_speed|wind_speed_Quality_Code|"
               "wind_speed_Source_Code|wind_gust|wind_gust_Quality_Code|wind_gust_Source_Code|sea_level_pressure\n"
               "X|2026-01-25T23:00:00|FM-15|250|5.1||343|||343|1001.0\n"
               "X|2026-01-28T05:00:00|FM-15|260|27.0||343|38.0||343|975.2\n"
               "X|2026-01-31T00:00:00|FM-15|270|6.0||343|||343|1010.0\n").encode()
        rows = fetch.parse_reports(psv, "2026-01-26", "2026-01-30")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["wind_speed"], "27.0")
        self.assertEqual(rows[0]["wind_gust"], "38.0")


if __name__ == "__main__":
    unittest.main()
