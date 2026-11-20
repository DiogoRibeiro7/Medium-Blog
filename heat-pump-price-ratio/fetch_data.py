"""Download the Eurostat series behind the article, and compare them with the committed snapshot.

python fetch_data.py           # download into data/raw/ and report whether the snapshot matches
python fetch_data.py --write   # same, then overwrite the committed snapshot in data/

- nrg_pc_204  household electricity prices, half-yearly, bands DC and DD, euros per kWh  -> data/electricity.csv
- nrg_pc_202  household gas prices, half-yearly, band D2, euros per kWh (gross calorific value) -> data/gas.csv
- nrg_chdd_a  heating degree days, annual                                               -> data/hdd.csv

Prices come with all taxes and levies (I_TAX), without VAT and other recoverable taxes (X_VAT), and without any
taxes or levies (X_TAX). The period is fixed (2007-S2 to 2026-S1; degree days 2015 to 2024) so that later
releases only show up as revisions.
"""

import csv
import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import heat_core as hc

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
AGENT = "heat-pump-price-ratio/0.1 (+https://github.com/DiogoRibeiro7/Medium-Blog)"
GEOS = "+".join((*hc.PANEL, "EU27_2020"))
TAXES = "I_TAX+X_VAT+X_TAX"
QUERIES = {
    "electricity": (
        f"nrg_pc_204/S.E7000.{hc.ELECTRICITY_SMALL}+{hc.ELECTRICITY_BAND}.KWH.{TAXES}.EUR.{GEOS}",
        f"startPeriod={hc.FIRST}&endPeriod={hc.LAST}",
    ),
    "gas": (
        f"nrg_pc_202/S.G3000.{hc.GAS_BAND}.KWH.{TAXES}.EUR.{GEOS}",
        f"startPeriod={hc.FIRST}&endPeriod={hc.LAST}",
    ),
    "hdd": (f"nrg_chdd_a/A.NR.HDD.{GEOS}", "startPeriod=2015&endPeriod=2024"),
}


def download(name, address):
    target = RAW / f"{name}.csv"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(address, headers={"User-Agent": AGENT})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=300) as r:
                    body = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise
                time.sleep(30 * (attempt + 1))
        target.write_bytes(body)
    return target.read_text(encoding="utf-8")


def tidy(name, text):
    """Keep the columns the analysis uses, sorted, so the snapshot only changes when the data do."""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r["OBS_VALUE"]]
    if name == "hdd":
        w.writerow(["country", "year", "value", "flag"])
        w.writerows(sorted((r["geo"], r["TIME_PERIOD"], r["OBS_VALUE"], r["OBS_FLAG"]) for r in rows))
    else:
        w.writerow(["country", "half", "band", "tax", "value", "flag"])
        w.writerows(
            sorted(
                (
                    r["geo"],
                    r["TIME_PERIOD"],
                    r["nrg_cons"],
                    r["tax"],
                    r["OBS_VALUE"],
                    r["OBS_FLAG"],
                )
                for r in rows
            )
        )
    return out.getvalue()


def main(write):
    for name, (path, period) in QUERIES.items():
        text = tidy(name, download(name, f"{EUROSTAT}{path}?format=SDMX-CSV&{period}"))
        target = DATA / f"{name}.csv"
        old = target.read_text() if target.exists() else ""
        if old == text:
            print(f"{name}.csv: identical to Eurostat")
            continue
        print(
            f"{name}.csv: differs from Eurostat ({len(old.splitlines())} lines committed, {len(text.splitlines())} now)"
        )
        if write:
            target.write_text(text)


if __name__ == "__main__":
    main("--write" in sys.argv)
