"""Offline tests of the parsers in fetch.py (no network)."""

import gzip
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch  # noqa: E402


class Parsers(unittest.TestCase):
    def test_isdlite_daily_max_and_min(self):
        text = (
            "2026  1 28  0   120    90 10012   220    85     8 -9999 -9999\n"
            "2026  1 28  3   110    80  9861   230   215     8 -9999 -9999\n"
            "2026  1 28  6 -9999 -9999 -9999   240 -9999     8 -9999 -9999\n"
            "2026  1 29  0   100    70 10102   250    40     4 -9999 -9999\n"
        )
        days = fetch.parse_isdlite(gzip.compress(text.encode()))
        self.assertEqual(days["2026-01-28"], [2, 21.5, 986.1])
        self.assertEqual(days["2026-01-29"], [1, 4.0, 1010.2])

    def test_gsod_units_and_missing(self):
        text = ('"STATION","DATE","MXSPD","GUST","PRCP"\n'
                '"08535099999","2026-01-28","30.0","58.3","1.20"\n'
                '"08535099999","2026-01-29","999.9","999.9","99.99"\n')
        out = fetch.parse_gsod(text.encode())
        self.assertEqual(out["2026-01-28"], (15.43, 29.99, 30.5))
        self.assertEqual(out["2026-01-29"], (None, None, None))

    def test_nao(self):
        text = " 2026  1 -1.23\n 2026  2  0.45\nnot a row\n"
        self.assertEqual(fetch.parse_nao(text), [(2026, 1, -1.23), (2026, 2, 0.45)])

    def test_history_selection(self):
        text = ('"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"\n'
                '"085360","99999","LISBOA","PO","","LPPT","+38.767","-009.133","+0114.0","19730101","20261110"\n'
                '"085090","99999","FLORES","PO","","LPFL","+39.450","-031.133","+0028.0","19730101","20261110"\n'
                '"085450","99999","PORTO","PO","","LPPR","+41.233","-008.667","+0073.0","19900101","20261110"\n'
                '"085540","99999","FARO","PO","","LPFR","+37.017","-007.967","+0007.0","19730101","20200101"\n')
        rows = fetch.parse_history(text)
        self.assertEqual([r["usaf"] for r in rows], ["085360"])


if __name__ == "__main__":
    unittest.main()
