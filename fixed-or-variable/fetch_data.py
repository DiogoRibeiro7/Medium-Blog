"""Download the series behind the article from Eurostat.

python fetch_data.py           # download Euribor into data/raw/ and check it; compare the snapshot with Eurostat
python fetch_data.py --write   # same, then overwrite the committed snapshot

Euribor is a benchmark of the European Money Markets Institute, so it is downloaded each time (Eurostat's
monthly averages, `irt_st_m`) and checked against a pinned digest instead of being committed. The other series
are committed in data/ so that the analysis gives the same numbers after Eurostat revises them:
- irt_h_euryld_m, irt_euryld_m  euro yield curves (par and spot rates), 1999-2005 and 2004-2026
- prc_hicp_minr                 HICP all items, euro area, 2015 = 100
- nama_10_lp_ulc                compensation per employee, euro area (20 countries), euros
"""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import mortgage_core as mc

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
EURIBOR_SHA256 = "ca30dd155bb7bf4e5323521d6e60add4cbb31ef7afeb8d51c17cbd1a556ad76a"


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


def euribor_text():
    d = download(
        "euribor",
        url("irt_st_m", geo="EA", int_rt=["IRT_M3", "IRT_M6", "IRT_M12"], untilTimePeriod=mc.LAST),
    )
    return mc.wide_csv(mc.jsonstat_rows(d, ["int_rt"]))


def euribor_csv():
    """Path of the checked Euribor file, downloading it if needed."""
    path = RAW / "euribor.csv"
    text = euribor_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != EURIBOR_SHA256:
        raise SystemExit(f"Euribor download differs from the pinned digest: {digest}")
    path.write_text(text)
    return path


def snapshot():
    curves = mc.jsonstat_rows(
        download("euryld_hist", url("irt_h_euryld_m", geo="EA")), ["yld_curv", "HIST", "maturity"]
    ) + mc.jsonstat_rows(
        download(
            "euryld",
            url("irt_euryld_m", geo="EA", yld_curv=["PAR", "SPOT_RT"], untilTimePeriod=mc.LAST),
        ),
        ["yld_curv", "bonds", "maturity"],
    )
    hicp = download("hicp", url("prc_hicp_minr", geo="EA", coicop18="TOTAL", unit="I15", untilTimePeriod=mc.LAST))
    wage = download("wage", url("nama_10_lp_ulc", geo="EA20", na_item="D1_SAL_PER", unit="EUR"))
    prices = mc.jsonstat_rows(hicp, ["HICP"]) + mc.jsonstat_rows(wage, ["COMP_EMP"])
    return {"yield_curves": mc.wide_csv(curves), "prices_income": mc.wide_csv(prices)}


def main(write):
    print(f"euribor: matches the pinned digest ({euribor_csv().name})")
    for name, text in snapshot().items():
        path = DATA / f"{name}.csv"
        old = path.read_text() if path.exists() else ""
        if old == text:
            print(f"{name}.csv: identical")
            continue
        print(f"{name}.csv: differs from the snapshot")
        if write:
            path.write_text(text)


if __name__ == "__main__":
    main("--write" in sys.argv)
