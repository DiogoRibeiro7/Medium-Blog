"""Download the Eurostat series behind the article and compare them with the snapshot in data/.

python fetch_data.py           # download into data/raw/, rebuild the snapshot, report any difference
python fetch_data.py --write   # same, then overwrite the committed snapshot

The analysis reads only the snapshot (data/*.csv), so it runs offline and gives the same numbers after
Eurostat revises a series. Seasonally adjusted survey balances are re-estimated every month, so a later
download will differ in the last decimals; this script says which files and how many values changed.

Sources, all from the Eurostat dissemination API (free reuse with acknowledgement):
- prc_hicp_midx   HICP monthly indices, ECOICOP ver.1, 2015 = 100, to December 2025
- prc_hicp_inw    HICP item weights, ECOICOP ver.1, to 2025
- prc_hicp_minr   HICP monthly indices, ECOICOP ver.2 (COICOP 2018), 2015 = 100, to October 2026
- prc_hicp_iw     HICP item weights, ECOICOP ver.2
- ei_bsco_m       European Commission consumer survey: price trends over the last 12 months (balance)
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import felt_core as fc

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
LAST_MONTH = "2026-10"


def url(dataset, **params):
    parts = []
    for key, value in params.items():
        values = value if isinstance(value, list | tuple) else [value]
        parts += [f"{key}={v}" for v in values]
    return API + dataset + "?" + "&".join(parts + ["lang=EN"])


def download(name, address):
    target = RAW / f"{name}.json"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            with urllib.request.urlopen(address, timeout=300) as r:
                body = r.read()
            if b'"error"' not in body[:200]:
                break
            time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError(f"{name}: Eurostat returned an error: {body[:200]!r}")
        target.write_bytes(body)
    return json.loads(target.read_text())


def requests():
    """Every request the snapshot needs: (snapshot file, raw name, URL)."""
    geos = fc.EU27 + ["EA20"]
    out = []
    for code in fc.PANEL_CODES:
        out.append(("hicp_index_v1", f"midx_{code}", url("prc_hicp_midx", unit="I15", coicop=code, geo=geos)))
    showcase = url("prc_hicp_midx", unit="I15", coicop=fc.SHOWCASE_EXTRA, geo="EA20")
    out.append(("hicp_index_v1", "midx_showcase", showcase))
    out.append(("hicp_weights_v1", "inw_panel", url("prc_hicp_inw", coicop=fc.PANEL_CODES, geo=geos)))
    out.append(
        (
            "hicp_index_v2",
            "minr",
            url(
                "prc_hicp_minr",
                unit="I15",
                coicop18=fc.V2_CODES,
                geo=["EA20", "EA21"],
                untilTimePeriod=LAST_MONTH,
            ),
        )
    )
    out.append(("hicp_weights_v2", "iw", url("prc_hicp_iw", coicop18=["TOTAL", "FROOPP"], geo=["EA20", "EA21"])))
    out.append(
        (
            "perceived_prices",
            "bsco",
            url(
                "ei_bsco_m",
                indic="BS-PT-LY",
                s_adj="SA",
                unit="BAL",
                geo=fc.EU27 + ["EA20", "EA21"],
                untilTimePeriod=LAST_MONTH,
            ),
        )
    )
    return out


def snapshot():
    tables = {}
    for file, name, address in requests():
        rows = fc.jsonstat_rows(download(name, address))
        tables.setdefault(file, []).extend(rows)
    return {file: fc.wide_csv(rows) for file, rows in tables.items()}


def main(write):
    new = snapshot()
    changed = 0
    for file, text in sorted(new.items()):
        path = DATA / f"{file}.csv"
        old = path.read_text() if path.exists() else ""
        if old == text:
            print(f"{file}.csv: identical")
            continue
        changed += 1
        old_cells = fc.cells(old)
        new_cells = fc.cells(text)
        diff = sum(1 for k in new_cells.keys() | old_cells.keys() if old_cells.get(k) != new_cells.get(k))
        print(f"{file}.csv: {diff} values differ from the snapshot")
        if write:
            path.write_text(text)
    if changed and not write:
        print("run with --write to replace the snapshot (the numbers in output.txt may then change)")


if __name__ == "__main__":
    main("--write" in sys.argv)
