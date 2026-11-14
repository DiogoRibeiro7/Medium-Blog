"""Download the inputs for the france-repeat-sales-floor analysis.

    python fetch.py --out data                 # French monthly CPI only (small, committed)
    python fetch.py --out data --dvf DIR       # also the geolocated DVF files into DIR

Sources
-------
- OECD Consumer price indices (OECD.SDD.TPS, DSD_PRICES@DF_PRICES_ALL): France, monthly CPI,
  2015 = 100, January 2018 onwards -> ``data/fr_cpi_monthly.csv``.
- Demandes de valeurs foncieres (DVF), geolocated version published by Etalab on data.gouv.fr
  from the DGFiP files: every sale of land and buildings registered by the land registry
  services in France except Alsace-Moselle and Mayotte. The DGFiP publishes a rolling window of
  about five years; this script downloads every yearly file that is currently online
  (``geo-dvf/latest/csv/<year>/full.csv.gz``). The files go to a scratch directory and are never
  committed: they carry addresses and cadastral parcel numbers, and the DGFiP terms of use forbid
  re-identifying the parties. ``build.py`` turns them into aggregate tables.

Licence: Licence Ouverte / Open Licence 2.0 (Etalab), source DGFiP.

``data/fetch_log.txt`` records each URL, HTTP status, size and SHA-256, and the dataset's
last-modified date on data.gouv.fr.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UA = {"User-Agent": "france-repeat-sales-floor/1.0 (research companion)"}

CPI_URL = (
    "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/"
    "FRA.M.N.CPI.IX._T.N._Z?startPeriod=2018-01&format=csvfilewithlabels"
)
DVF_BASE = "https://files.data.gouv.fr/geo-dvf/latest/csv/{year}/full.csv.gz"
DATASET_API = "https://www.data.gouv.fr/api/1/datasets/demandes-de-valeurs-foncieres-geolocalisees/"
YEARS = range(2014, datetime.now(timezone.utc).year + 1)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_cpi(out: Path, log: list[str]) -> bool:
    req = urllib.request.Request(CPI_URL, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            status, body = r.status, r.read()
    except Exception as e:
        log.append(f"{CPI_URL}\terror {type(e).__name__}\t0\t-")
        return False
    log.append(f"{CPI_URL}\t{status}\t{len(body)}\t{hashlib.sha256(body).hexdigest()}")
    raw = pd.read_csv(io.BytesIO(body))
    cpi = (raw[["TIME_PERIOD", "OBS_VALUE"]].rename(columns={"TIME_PERIOD": "month", "OBS_VALUE": "cpi"})
           .assign(month=lambda d: d.month + "-01").sort_values("month"))
    cpi.to_csv(out / "fr_cpi_monthly.csv", index=False, float_format="%.10g")
    return True


def dataset_metadata(log: list[str]) -> None:
    req = urllib.request.Request(DATASET_API, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            meta = json.load(r)
    except Exception as e:
        log.append(f"# data.gouv.fr metadata unavailable: {type(e).__name__}")
        return
    log.append(f"# data.gouv.fr dataset last_modified {meta.get('last_modified', '?')[:10]}, "
               f"temporal_coverage {json.dumps(meta.get('temporal_coverage'))}")
    for res in meta.get("resources", []):
        log.append(f"#   resource '{res.get('title')}' {res.get('filesize')} bytes, "
                   f"modified {str(res.get('last_modified', ''))[:10]}")


def fetch_dvf(dest: Path, log: list[str]) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    for year in YEARS:
        url = DVF_BASE.format(year=year)
        target = dest / f"dvf_{year}.csv.gz"
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=900) as r, target.open("wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
                status = r.status
        except urllib.error.HTTPError as e:
            log.append(f"{url}\t{e.code}\t0\t-")
            target.unlink(missing_ok=True)
            continue
        except Exception as e:
            log.append(f"{url}\terror {type(e).__name__}: {e}\t0\t-")
            target.unlink(missing_ok=True)
            continue
        size = target.stat().st_size
        log.append(f"{url}\t{status}\t{size}\t{_sha256_file(target)}")
        if status == 200 and size > 1_000_000:
            got.append(target)
        else:
            target.unlink(missing_ok=True)
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--dvf", default=None, help="scratch directory for the DVF files")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = [f"# fetched {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}",
           "source\tstatus\tbytes\tsha256"]
    ok = fetch_cpi(out, log)
    if args.dvf:
        dataset_metadata(log)
        files = fetch_dvf(Path(args.dvf), log)
        ok = ok and len(files) >= 3
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
