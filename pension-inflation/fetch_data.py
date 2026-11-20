"""Download the series behind the article from the OECD and Eurostat, and compare them with the committed snapshot.

python fetch_data.py           # download into data/raw/ and report whether the snapshot still matches the sources
python fetch_data.py --write   # same, then overwrite the committed snapshot in data/

The analysis reads only the committed snapshot, so it gives the same numbers after the sources revise their data:
- OECD DF_PRICES_ALL      consumer price index, all items, annual average, 2015 = 100        -> data/cpi.csv
- OECD DF_FINMARK         long-term interest rates (10-year government bonds), annual, %     -> data/long_rates.csv
- OECD AV_AN_WAGE         average annual wages, constant prices and PPPs                      -> data/wages.csv
- Eurostat demo_mlifetable  probability of dying (qx) by age, and life expectancy at 65       -> data/life_tables.csv
- Eurostat prc_hicp_aind    HICP all items, annual average index, 2015 = 100                  -> data/hicp.csv

The Portuguese pension updates (data/portugal_updates.csv, data/portugal_first_bracket_coefficients.csv) were
transcribed by hand from the ordinances published in the Diario da Republica; each row names its source.
"""

import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pension_core as pc

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
OECD = "https://sdmx.oecd.org/public/rest/data/"
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
COUNTRIES = "+".join(pc.PANEL + pc.EXTRA)
EU = "BE BG CZ DK DE EE IE EL ES FR HR IT CY LV LT LU HU MT NL AT PL PT RO SI SK FI SE".split()
LIFE_YEARS = (2019, 2024)
LAST_YEAR = 2025
AGENT = "pension-inflation/0.1 (+https://github.com/DiogoRibeiro7/Medium-Blog)"


def download(name, address, accept=None):
    target = RAW / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": AGENT} | ({"Accept": accept} if accept else {})
        request = urllib.request.Request(address, headers=headers)
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


def oecd(name, flow, key):
    text = download(
        f"{name}.csv",
        f"{OECD}{flow}/{key}?dimensionAtObservation=AllDimensions&endPeriod={LAST_YEAR}",
        accept="application/vnd.sdmx.data+csv; charset=utf-8",
    )
    return list(csv.DictReader(io.StringIO(text)))


def long_csv(header, rows):
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(header)
    w.writerows(sorted(rows))
    return out.getvalue()


def oecd_series(name, flow, key, keep=lambda r: True):
    rows = [
        (r["REF_AREA"], int(r["TIME_PERIOD"]), r["OBS_VALUE"])
        for r in oecd(name, flow, key)
        if r["OBS_VALUE"] and keep(r)
    ]
    return long_csv(["country", "year", "value"], rows)


def eurostat(name, dataset, **params):
    parts = []
    for k, v in params.items():
        parts += [f"{k}={x}" for x in (v if isinstance(v, list | tuple) else [v])]
    d = json.loads(download(f"{name}.json", f"{EUROSTAT}{dataset}?format=JSON&lang=EN&" + "&".join(parts)))
    dims = d["id"]
    sizes = d["size"]
    labels = {k: {i: c for c, i in d["dimension"][k]["category"]["index"].items()} for k in dims}
    out = []
    for flat, value in d["value"].items():
        rest, coords = int(flat), {}
        for k, n in zip(reversed(dims), reversed(sizes), strict=True):
            coords[k] = labels[k][rest % n]
            rest //= n
        out.append((coords, value))
    return out


def life_tables():
    ages = [f"Y{a}" for a in range(60, 95)] + ["Y_GE95"]
    qx = eurostat(
        "qx",
        "demo_mlifetable",
        indic_de="PROBDEATH",
        geo=[pc.LIFE_GEO, "PT"],
        sex=["T", "F", "M"],
        time=list(LIFE_YEARS),
        age=ages,
    )
    e65 = eurostat(
        "e65",
        "demo_mlifetable",
        indic_de="LIFEXP",
        geo=[pc.LIFE_GEO, "PT"],
        sex=["T", "F", "M"],
        time=list(LIFE_YEARS),
        age="Y65",
    )
    rows = [(c["geo"], int(c["time"]), c["sex"], "qx", c["age"], repr(float(v))) for c, v in qx]
    rows += [(c["geo"], int(c["time"]), c["sex"], "e65", c["age"], repr(float(v))) for c, v in e65]
    return long_csv(["geo", "year", "sex", "indicator", "age", "value"], rows)


def hicp():
    d = eurostat(
        "hicp",
        "prc_hicp_aind",
        unit="INX_A_AVG",
        coicop="CP00",
        geo=["EA20", "EU27_2020"] + EU,
        sinceTimePeriod=2010,
        untilTimePeriod=LAST_YEAR,
    )
    rows = [(c["geo"], int(c["time"]), repr(float(v))) for c, v in d]
    return long_csv(["geo", "year", "value"], rows)


def snapshot():
    return {
        "cpi": oecd_series("cpi", "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0", f"{COUNTRIES}.A.N.CPI.IX._T.N._Z"),
        "long_rates": oecd_series("long_rates", "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0", f"{COUNTRIES}.A.IRLT......"),
        "wages": oecd_series(
            "wages",
            "OECD.ELS.SAE,DSD_EARNINGS@AV_AN_WAGE,1.0",
            f"{COUNTRIES}.WG.USD_PPP.A.Q.MEAN._Z",
        ),
        "life_tables": life_tables(),
        "hicp": hicp(),
    }


def main(write):
    for name, text in snapshot().items():
        path = DATA / f"{name}.csv"
        old = path.read_text() if path.exists() else ""
        if old == text:
            print(f"{name}.csv: identical to the sources")
            continue
        print(
            f"{name}.csv: differs from the sources ({len(old.splitlines())} lines committed,"
            f" {len(text.splitlines())} now)"
        )
        if write:
            path.write_text(text)


if __name__ == "__main__":
    main("--write" in sys.argv)
