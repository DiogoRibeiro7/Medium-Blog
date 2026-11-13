"""Download daily maximum 10 m wind from ERA5 for a grid of points over mainland Portugal.

Run by the GitHub Actions workflow storm-clustering-portugal-era5.yml, which commits
the files it writes to data/era5/. The analysis itself never touches the network.

Why a reanalysis next to the stations: most of IPMA's synoptic stations stop appearing
in NOAA's GHCNh archive after September 2022, so the station network that covers the
2025/26 winter is reduced to a handful of airports. ERA5 is spatially complete and
produced with one model throughout, which makes it the better record for counting
storms; the stations are used to check it.

Source: ERA5 (Hersbach et al. 2020, Copernicus Climate Change Service), served at
0.25 degrees by the Open-Meteo historical weather API (archive-api.open-meteo.com,
CC BY 4.0). Daily maximum of the hourly 10 m wind speed, days in UTC, metres per
second. One request per point, so the download can stop and resume: points already
on disk are skipped. Open-Meteo weighs a request by days and variables (one variable
over 76 years is about 200 "calls"), and the free tier allows 10,000 a day, so the
script pauses on the per-minute and per-hour limits and stops cleanly on the daily one.

Outputs in --out:
  era5/era5_<lat>_<lon>.csv.gz   date, wind_max_ms
  era5_points.csv                requested and returned coordinates, model elevation
  era5_fetch_log.txt             time, HTTP status, bytes, SHA-256 and URL of each request
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
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

# ERA5 nodes on a 0.5-degree lattice inside mainland Portugal, plus two Algarve
# coastal nodes (37.0 N, 8.5 W and 8.0 W) so the south coast is represented.
POINTS = [
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


def point_file(out: Path, lat: float, lon: float) -> Path:
    return out / "era5" / f"era5_{lat:.2f}_{lon:.2f}.csv.gz"


def parse_response(body: bytes) -> tuple[dict, list[tuple[str, float | None]]]:
    """Returned coordinates and the (date, value) series from an Open-Meteo archive reply."""
    js = json.loads(body)
    meta = {"lat_out": js.get("latitude"), "lon_out": js.get("longitude"),
            "elevation_m": js.get("elevation")}
    daily = js["daily"]
    rows = list(zip(daily["time"], daily["wind_speed_10m_max"]))
    return meta, rows


class Limited(Exception):
    def __init__(self, scope: str):
        super().__init__(scope)
        self.scope = scope


def request(url: str, log: list[str]) -> bytes:
    stamp = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    req = Request(url, headers={"User-Agent": "storm-clustering-portugal (research; github.com/DiogoRibeiro7)"})
    try:
        with urlopen(req, timeout=600) as resp:
            body = resp.read()
        log.append(f"{stamp()}\t200\t{len(body)}\t{hashlib.sha256(body).hexdigest()}\t{url}")
        return body
    except HTTPError as err:
        reason = err.read().decode("utf-8", errors="replace")[:300]
        log.append(f"{stamp()}\t{err.code}\t0\t-\t{url}\t{reason}")
        if err.code == 429:
            low = reason.lower()
            raise Limited("daily" if "daily" in low else "hourly" if "hourly" in low else "minutely")
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-minutes", type=float, default=320,
                    help="stop starting new requests after this long, so the workflow can commit")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "era5").mkdir(parents=True, exist_ok=True)
    log_path, points_path = out / "era5_fetch_log.txt", out / "era5_points.csv"
    log = log_path.read_text().splitlines() if log_path.exists() else []
    meta_rows = {}
    if points_path.exists():
        with open(points_path) as fh:
            for r in csv.DictReader(fh):
                meta_rows[(float(r["lat"]), float(r["lon"]))] = r

    t0 = time.monotonic()
    todo = [p for p in POINTS if not point_file(out, *p).exists()]
    print(f"{len(POINTS) - len(todo)} of {len(POINTS)} points already on disk; {len(todo)} to fetch")
    stopped = None
    for lat, lon in todo:
        if (time.monotonic() - t0) / 60 > args.max_minutes:
            stopped = "time budget"
            break
        url = API + "?" + urlencode({
            "latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
            "daily": "wind_speed_10m_max", "wind_speed_unit": "ms", "timezone": "GMT",
            "models": "era5",
        })
        body = None
        for attempt in range(6):
            try:
                body = request(url, log)
                break
            except Limited as lim:
                if lim.scope == "daily":
                    stopped = "daily request limit"
                    break
                wait = 65 if lim.scope == "minutely" else 20 * 60
                print(f"  {lim.scope} limit reached; waiting {wait} s")
                time.sleep(wait)
            except (HTTPError, URLError, TimeoutError, ConnectionError) as err:
                print(f"  ({lat}, {lon}) attempt {attempt + 1}: {err!r}")
                time.sleep(10 * (attempt + 1))
        if stopped:
            break
        if body is None:
            continue
        meta, rows = parse_response(body)
        with gzip.open(point_file(out, lat, lon), "wt", newline="", compresslevel=9) as fh:
            w = csv.writer(fh)
            w.writerow(["date", "wind_max_ms"])
            w.writerows((d, "" if v is None else v) for d, v in rows)
        meta_rows[(lat, lon)] = {"lat": lat, "lon": lon, **meta,
                                 "days": len(rows), "missing": sum(v is None for _, v in rows)}
        print(f"  ({lat:.1f}, {lon:.1f}) -> ({meta['lat_out']}, {meta['lon_out']}), "
              f"{meta['elevation_m']} m, {len(rows)} days")
    with open(points_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["lat", "lon", "lat_out", "lon_out", "elevation_m", "days", "missing"])
        w.writeheader()
        for key in sorted(meta_rows):
            w.writerow(meta_rows[key])
    log_path.write_text("\n".join(log) + "\n")
    left = [p for p in POINTS if not point_file(out, *p).exists()]
    print(f"{len(POINTS) - len(left)} of {len(POINTS)} points on disk"
          + (f"; stopped early ({stopped}); run the workflow again to resume" if stopped else ""))
    return 0 if not left else (0 if stopped else 1)


if __name__ == "__main__":
    sys.exit(main())
