"""Download the Eurostat series behind the article, and compare them with the committed snapshot.

python fetch_data.py           # download into data/raw/ and report whether the snapshot matches
python fetch_data.py --write   # same, then overwrite the committed snapshot in data/

- namq_10_a10     gross value added, total economy, chain-linked volumes (2020), SCA  -> gva.csv
- namq_10_a10_e   hours worked, total employment, domestic concept, SCA               -> hours_na.csv
- namq_10_pe      employment in persons, domestic and national concept, unadjusted    -> employment_na.csv
- lfsi_ahw_q      LFS total actual hours worked in the main job, index 2021=100, SA   -> hours_lfs.csv
- lfsi_emp_q      LFS employment, persons aged 15-74, SA                              -> employment_lfs.csv
- lfsi_abt_q      LFS persons absent from work in the reference week, % employed, SA  -> absences_lfs.csv
- lfsq_pganws     LFS population and employment aged 15-74, by citizenship            -> population_lfs.csv
- demo_pjangroup  population on 1 January, five-year age groups 15-19 to 70-74        -> population_demo.csv
- namq_10_lp_ulc  Eurostat's real labour productivity per hour worked, index 2020=100 -> productivity_eurostat.csv

All files go to data/. The periods are fixed so that later releases only show up as revisions.
"""

import csv
import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import hours_core as hc

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
AGENT = "productivity-hours-denominator/0.1 (+https://github.com/DiogoRibeiro7/Medium-Blog)"
GEOS = "+".join((hc.EU, *hc.MEMBERS))
NA_PERIOD = f"startPeriod={hc.NA_FIRST}&endPeriod={hc.LAST}"
LFS_PERIOD = f"startPeriod={hc.LFS_FIRST}&endPeriod={hc.LAST}"
AGE_GROUPS = "+".join(hc.DEMO_GROUPS)

# name: (key, period, columns kept besides country, period, value and flag)
QUERIES = {
    "gva": (f"namq_10_a10/Q.CLV20_MNAC.SCA.TOTAL.B1G.{GEOS}", NA_PERIOD, ()),
    "hours_na": (f"namq_10_a10_e/Q.THS_HW.TOTAL.SCA.EMP_DC.{GEOS}", NA_PERIOD, ()),
    "employment_na": (f"namq_10_pe/Q.THS_PER.NSA.EMP_DC+EMP_NC.{GEOS}", NA_PERIOD, ("na_item",)),
    "hours_lfs": (f"lfsi_ahw_q/Q.SA.T.Y15-74+Y20-64.I21.{GEOS}", LFS_PERIOD, ("age",)),
    "employment_lfs": (f"lfsi_emp_q/Q.EMP_LFS.SA.T.Y15-74.THS_PER.{GEOS}", LFS_PERIOD, ()),
    "absences_lfs": (f"lfsi_abt_q/Q.SA.T.Y15-74.PC_EMP.{GEOS}", LFS_PERIOD, ()),
    "population_lfs": (
        f"lfsq_pganws/Q.THS_PER.T.TOTAL+FOR.Y15-74.POP+EMP.{GEOS}",
        LFS_PERIOD,
        ("citizen", "wstatus"),
    ),
    "population_demo": (
        f"demo_pjangroup/A.NR.T.{AGE_GROUPS}.{GEOS}",
        f"startPeriod={hc.LFS_FIRST[:4]}&endPeriod={hc.LAST[:4]}",
        ("age",),
    ),
    "productivity_eurostat": (f"namq_10_lp_ulc/Q.I20.SCA.RLPR_HW.{GEOS}", NA_PERIOD, ()),
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


def tidy(text, extra):
    """Keep the columns the analysis uses, sorted, so the snapshot only changes when the data do."""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r["OBS_VALUE"]]
    w.writerow(["country", "period", *extra, "value", "flag"])
    w.writerows(
        sorted((r["geo"], r["TIME_PERIOD"], *(r[c] for c in extra), r["OBS_VALUE"], r["OBS_FLAG"]) for r in rows)
    )
    return out.getvalue()


def main(write):
    for name, (key, period, extra) in QUERIES.items():
        text = tidy(download(name, f"{EUROSTAT}{key}?format=SDMX-CSV&{period}"), extra)
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
