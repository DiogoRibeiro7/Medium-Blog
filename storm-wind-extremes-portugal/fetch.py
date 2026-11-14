"""Download the data that storm-wind-extremes-portugal adds to its sibling project.

Run by the GitHub Actions workflow storm-wind-extremes-portugal-data.yml, which commits
what it writes to data/. The analysis itself never touches the network.

The daily maximum 10 m wind from ERA5 at 38 nodes over mainland Portugal (1950-10-01 to
2026-03-31) and the station records from NOAA's GHCNh are already in the sibling folder
storm-clustering-portugal/data and are read from there. This script adds three things:

1. ERA5 daily maximum 10 m wind gust at the same 38 nodes, same period.
2. ERA5 daily maximum 10 m wind at the grid cell of nine long-record stations, so the
   reanalysis and the anemometer can be compared cell by station rather than node by
   station.
3. ERA5 daily maximum wind and gust on 26-30 January 2026 at the cells of the stations
   whose gusts during storm Kristin were reported by IPMA, and every GHCNh report from
   the five airports that kept reporting through that storm.
4. ERA5 daily maximum 10 m wind at the 38 nodes from 1940-01-01 to 1950-09-30, the early
   part of ERA5's back extension, so the record can be tested with the cyclone of
   15 February 1941 in it. ERA5 before 1950 assimilates far fewer observations.

ERA5 (Hersbach et al. 2020, Copernicus Climate Change Service) is served at 0.25 degrees
by the Open-Meteo historical weather API (archive-api.open-meteo.com, CC BY 4.0). Days are
in UTC and speeds in metres per second. Open-Meteo weighs a request by its days and
variables: one variable over 76 years costs about 200 of the free tier's 10,000 daily
calls. So the requests run one after another in a single job, the script waits out the
per-minute and per-hour limits, and on the daily limit it stops and keeps what it has.
Files already on disk are skipped, so running the workflow again resumes.

GHCNh: NOAA NCEI Global Historical Climatology Network hourly, doi:10.25921/jp3d-3v19.

Outputs in --out:
  era5_gust/gust_<lat>_<lon>.csv.gz      date, gust_max_ms
  era5_sites/site_<key>.csv.gz           date, wind_max_ms
  era5_kristin/kristin_<key>.csv.gz      date, wind_max_ms, gust_max_ms
  era5_early/early_<lat>_<lon>.csv.gz    date, wind_max_ms (1940-01-01 to 1950-09-30)
  era5_cells.csv                         every request: key, requested and returned
                                         coordinates, model elevation, days, missing
  kristin_reports.csv                    GHCNh reports from the airports, 26-30 Jan 2026
  fetch_log.txt                          time, HTTP status, bytes, SHA-256 and URL
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://archive-api.open-meteo.com/v1/archive"
START, END = "1950-10-01", "2026-03-31"
K_START, K_END = "2026-01-26", "2026-01-30"
E_START, E_END = "1940-01-01", "1950-09-30"
GHCNH = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-station/GHCNh_{sid}_por.psv"
AGENT = "storm-wind-extremes-portugal (research; github.com/DiogoRibeiro7)"

# The sibling project's 38 ERA5 nodes (0.5-degree lattice inside mainland Portugal plus
# two Algarve coastal nodes).
NODES = [
    (37.0, -8.5), (37.0, -8.0),
    (37.5, -8.5), (37.5, -8.0), (37.5, -7.5),
    (38.0, -8.5), (38.0, -8.0), (38.0, -7.5),
    (38.5, -9.0), (38.5, -8.5), (38.5, -8.0), (38.5, -7.5),
    (39.0, -9.0), (39.0, -8.5), (39.0, -8.0), (39.0, -7.5),
    (39.5, -9.0), (39.5, -8.5), (39.5, -8.0), (39.5, -7.5),
    (40.0, -8.5), (40.0, -8.0), (40.0, -7.5), (40.0, -7.0),
    (40.5, -8.5), (40.5, -8.0), (40.5, -7.5), (40.5, -7.0),
    (41.0, -8.5), (41.0, -8.0), (41.0, -7.5), (41.0, -7.0),
    (41.5, -8.5), (41.5, -8.0), (41.5, -7.5), (41.5, -7.0), (41.5, -6.5),
    (42.0, -8.5),
]

# Stations with at least 19 complete winters of synoptic wind in the sibling's GHCNh
# table (coordinates from that table's stations.csv).
SITES = {
    "porto": ("POMU0068545", 41.23, -8.68),
    "faro": ("POMU0068554", 37.02, -7.92),
    "beja": ("POI0000LPBJ", 38.0789, -7.9324),
    "monte_real": ("POI0000LPMR", 39.8312, -8.8873),
    "montijo": ("POI0000LPMT", 38.7039, -9.0359),
    "braganca": ("POMU0068575", 41.82, -6.67),
    "portalegre": ("POM00008571", 39.2833, -7.4167),
    "lisboa_gc": ("POM00008579", 38.7667, -9.1333),
    "vila_real": ("POMU0068566", 41.32, -7.73),
}
AIRPORTS = ["porto", "faro", "beja", "monte_real", "montijo"]

# Stations whose gusts on 28 January 2026 were reported by IPMA (via Lusa). Coordinates
# are approximate (to about 0.05 degree); at ERA5's 0.25-degree spacing they only pick
# the grid cell. Gusts in km/h as reported.
KRISTIN_SITES = {
    "monte_real": (39.83, -8.89, 178.0),       # Air Force station, Monte Real air base
    "degracias_soure": (39.98, -8.52, 208.8),  # CIM Coimbra station at 524 m, wind farm
    "cabo_carvoeiro": (39.36, -9.41, 149.0),
    "ansiao": (39.91, -8.43, 146.0),
    "leiria": (39.78, -8.82, 142.0),
    "castelo_branco": (39.84, -7.48, 137.0),
    "foia": (37.32, -8.59, 135.0),
    "tomar_vale_donas": (39.60, -8.41, 133.0),
    "cabo_da_roca": (38.78, -9.50, 131.0),
    "santa_cruz": (39.13, -9.38, 128.9),
}


def gust_file(out: Path, lat: float, lon: float) -> Path:
    return out / "era5_gust" / f"gust_{lat:.2f}_{lon:.2f}.csv.gz"


def site_file(out: Path, key: str) -> Path:
    return out / "era5_sites" / f"site_{key}.csv.gz"


def kristin_file(out: Path, key: str) -> Path:
    return out / "era5_kristin" / f"kristin_{key}.csv.gz"


def early_file(out: Path, lat: float, lon: float) -> Path:
    return out / "era5_early" / f"early_{lat:.2f}_{lon:.2f}.csv.gz"


def jobs(out: Path) -> list[dict]:
    """Every Open-Meteo request, cheapest and most needed first."""
    js = []
    for key, (lat, lon, _) in KRISTIN_SITES.items():
        js.append({"key": f"kristin_{key}", "lat": lat, "lon": lon, "start": K_START, "end": K_END,
                   "vars": ["wind_speed_10m_max", "wind_gusts_10m_max"],
                   "cols": ["wind_max_ms", "gust_max_ms"], "path": kristin_file(out, key)})
    for key, (_, lat, lon) in SITES.items():
        js.append({"key": f"site_{key}", "lat": lat, "lon": lon, "start": START, "end": END,
                   "vars": ["wind_speed_10m_max"], "cols": ["wind_max_ms"], "path": site_file(out, key)})
    for lat, lon in NODES:
        js.append({"key": f"gust_{lat:.2f}_{lon:.2f}", "lat": lat, "lon": lon, "start": START, "end": END,
                   "vars": ["wind_gusts_10m_max"], "cols": ["gust_max_ms"], "path": gust_file(out, lat, lon)})
    for lat, lon in NODES:
        js.append({"key": f"early_{lat:.2f}_{lon:.2f}", "lat": lat, "lon": lon, "start": E_START, "end": E_END,
                   "vars": ["wind_speed_10m_max"], "cols": ["wind_max_ms"], "path": early_file(out, lat, lon)})
    return js


def url_for(job: dict) -> str:
    return API + "?" + urlencode({
        "latitude": job["lat"], "longitude": job["lon"],
        "start_date": job["start"], "end_date": job["end"],
        "daily": ",".join(job["vars"]), "wind_speed_unit": "ms", "timezone": "GMT", "models": "era5",
    })


def parse_response(body: bytes, variables: list[str]) -> tuple[dict, list[tuple]]:
    """Returned coordinates and (date, value, ...) rows from an Open-Meteo archive reply."""
    js = json.loads(body)
    meta = {"lat_out": js.get("latitude"), "lon_out": js.get("longitude"), "elevation_m": js.get("elevation")}
    daily = js["daily"]
    rows = list(zip(daily["time"], *(daily[v] for v in variables)))
    return meta, rows


def parse_reports(raw: bytes, first: str, last: str) -> list[dict]:
    """GHCNh reports dated first..last (inclusive, YYYY-MM-DD) with the wind fields kept."""
    text = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="replace")
    reader = csv.reader(text, delimiter="|")
    header = [h.strip() for h in next(reader)]
    idx = {h: i for i, h in enumerate(header)}
    keep = ["DATE", "Report_Type", "wind_direction", "wind_speed", "wind_speed_Quality_Code",
            "wind_speed_Source_Code", "wind_gust", "wind_gust_Quality_Code", "wind_gust_Source_Code",
            "sea_level_pressure"]
    i_date = idx.get("DATE")
    out = []
    for r in reader:
        if not r or i_date is None or i_date >= len(r):
            continue
        day = r[i_date][:10]
        if first <= day <= last:
            out.append({k: (r[idx[k]].strip() if k in idx and idx[k] < len(r) else "") for k in keep})
    return out


class Limited(Exception):
    def __init__(self, scope: str):
        super().__init__(scope)
        self.scope = scope


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get(url: str, log: list[str], open_meteo: bool) -> bytes:
    req = Request(url, headers={"User-Agent": AGENT})
    try:
        with urlopen(req, timeout=600) as resp:
            body = resp.read()
        log.append(f"{stamp()}\t200\t{len(body)}\t{hashlib.sha256(body).hexdigest()}\t{url}")
        return body
    except HTTPError as err:
        reason = err.read().decode("utf-8", errors="replace")[:300]
        log.append(f"{stamp()}\t{err.code}\t0\t-\t{url}\t{reason}")
        if open_meteo and err.code == 429:
            low = reason.lower()
            raise Limited("daily" if "daily" in low else "hourly" if "hourly" in low else "minutely")
        raise


def write_gz(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="", compresslevel=9) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    tmp.replace(path)


def is_lfs_pointer(path: Path) -> bool:
    """True for a Git LFS pointer file (a CSV committed before the folder was exempted from LFS)."""
    try:
        with open(path, "rb") as fh:
            return fh.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def fetch_reports(out: Path, log: list[str]) -> None:
    path = out / "kristin_reports.csv"
    if path.exists() and not is_lfs_pointer(path):
        print("kristin_reports.csv already on disk")
        return
    rows = []
    for key in AIRPORTS:
        sid = SITES[key][0]
        body = None
        for attempt in range(4):
            try:
                body = get(GHCNH.format(sid=sid), log, open_meteo=False)
                break
            except (HTTPError, URLError, TimeoutError, ConnectionError) as err:
                print(f"  {sid} attempt {attempt + 1}: {err!r}")
                time.sleep(10 * (attempt + 1))
        if body is None:
            print(f"  {sid}: no file; kristin_reports.csv not written")
            return
        reps = parse_reports(body, K_START, K_END)
        rows += [{"site": key, "station": sid, **r} for r in reps]
        print(f"  {key}: {len(reps)} reports on {K_START} to {K_END}")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-minutes", type=float, default=320,
                    help="stop starting new requests after this long, so the workflow can commit")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path, cells_path = out / "fetch_log.txt", out / "era5_cells.csv"
    log = log_path.read_text().splitlines() if log_path.exists() else []
    cells = {}
    if cells_path.exists() and not is_lfs_pointer(cells_path):
        with open(cells_path) as fh:
            for r in csv.DictReader(fh):
                cells[r["key"]] = r

    fetch_reports(out, log)

    t0 = time.monotonic()
    todo = [j for j in jobs(out) if not j["path"].exists()]
    total = len(jobs(out))
    print(f"{total - len(todo)} of {total} ERA5 requests already on disk; {len(todo)} to fetch")
    stopped = None
    for job in todo:
        if (time.monotonic() - t0) / 60 > args.max_minutes:
            stopped = "time budget"
            break
        body = None
        for attempt in range(6):
            try:
                body = get(url_for(job), log, open_meteo=True)
                break
            except Limited as lim:
                if lim.scope == "daily":
                    stopped = "daily request limit"
                    break
                wait = 65 if lim.scope == "minutely" else 20 * 60
                print(f"  {lim.scope} limit reached; waiting {wait} s")
                time.sleep(wait)
            except (HTTPError, URLError, TimeoutError, ConnectionError) as err:
                print(f"  {job['key']} attempt {attempt + 1}: {err!r}")
                time.sleep(10 * (attempt + 1))
        if stopped:
            break
        if body is None:
            continue
        meta, rows = parse_response(body, job["vars"])
        write_gz(job["path"], ["date"] + job["cols"],
                 ((r[0], *("" if v is None else v for v in r[1:])) for r in rows))
        cells[job["key"]] = {"key": job["key"], "lat": job["lat"], "lon": job["lon"], **meta,
                             "days": len(rows), "missing": sum(any(v is None for v in r[1:]) for r in rows)}
        print(f"  {job['key']}: ({job['lat']}, {job['lon']}) -> ({meta['lat_out']}, {meta['lon_out']}), "
              f"{meta['elevation_m']} m, {len(rows)} days")
    with open(cells_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "lat", "lon", "lat_out", "lon_out", "elevation_m", "days", "missing"],
                           lineterminator="\n")
        w.writeheader()
        for key in sorted(cells):
            w.writerow(cells[key])
    log_path.write_text("\n".join(log) + "\n")
    left = [j for j in jobs(out) if not j["path"].exists()]
    print(f"{total - len(left)} of {total} ERA5 requests on disk"
          + (f"; stopped early ({stopped}); run the workflow again to resume" if stopped else ""))
    return 0 if not left else (0 if stopped else 1)


if __name__ == "__main__":
    sys.exit(main())
