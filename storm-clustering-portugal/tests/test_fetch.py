"""Offline tests of the parsers in fetch.py (no network)."""

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch  # noqa: E402


class Parsers(unittest.TestCase):
    def test_psv_daily_aggregates_and_quality_codes(self):
        header = ("Station_ID|Station_name|DATE|Latitude|Longitude|Elevation|"
                  "wind_speed|wind_speed_Measurement_Code|wind_speed_Quality_Code|"
                  "wind_gust|wind_gust_Measurement_Code|wind_gust_Quality_Code|"
                  "sea_level_pressure|sea_level_pressure_Measurement_Code|sea_level_pressure_Quality_Code")
        rows = [
            "POM00008535|LISBOA|2026-01-28T00:00:00|38.7|-9.1|95|8.5|||15.0|||1001.2||",
            "POM00008535|LISBOA|2026-01-28T03:00:00|38.7|-9.1|95|21.5||5|30.1|||986.1||1",
            "POM00008535|LISBOA|2026-01-28T04:00:00|38.7|-9.1|95|40.0||L|50.0||L|950.0||L",
            "POM00008535|LISBOA|2026-01-29T00:00:00|38.7|-9.1|95|4.0||||||1010.2||",
        ]
        raw = ("\n".join([header] + rows) + "\n").encode()
        qc = Counter()
        days = fetch.parse_psv(raw, qc)
        self.assertEqual(days["2026-01-28"], [2, 21.5, 30.1, 986.1])
        self.assertEqual(days["2026-01-29"], [1, 4.0, None, 1010.2])
        self.assertEqual(qc, Counter({"": 2, "5": 1, "L": 1}))

    def test_psv_with_separate_date_columns(self):
        header = "Station_ID|Year|Month|Day|Hour|wind_speed|wind_speed_Quality_Code"
        raw = (header + "\nPOX|2026|2|7|19|17.2|\n").encode()
        self.assertEqual(fetch.parse_psv(raw)["2026-02-07"], [1, 17.2, None, None])

    def test_station_list_csv(self):
        text = ("GHCN_ID,LATITUDE,LONGITUDE,ELEVATION,STATE,NAME,GSN,HCN_CRN,WMO_ID,ICAO\n"
                "POM00008535,38.717,-9.150,95.0,,LISBOA GEOFISICA,GSN,,08535,\n"
                "POM00008515,37.741,-25.698,79.0,,PONTA DELGADA,,,08512,LPPD\n"
                "SPE00120287,40.000,-7.000,500.0,,SPAIN STATION,,,,\n")
        rows = fetch.parse_station_list(text)
        self.assertEqual([r["id"] for r in rows], ["POM00008535"])
        self.assertEqual(rows[0]["wmo"], "08535")

    def test_nao(self):
        text = " 2026  1 -1.23\n 2026  2  0.45\nnot a row\n"
        self.assertEqual(fetch.parse_nao(text), [(2026, 1, -1.23), (2026, 2, 0.45)])


if __name__ == "__main__":
    unittest.main()
