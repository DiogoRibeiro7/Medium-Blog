"""Download the station records and the NAO index used by storm_clustering.py.

Run by the GitHub Actions workflow storm-clustering-portugal-data.yml, which
commits the compact tables it writes to data/. The analysis itself never
touches the network.

Sources (all public domain, U.S. National Oceanic and Atmospheric Administration):

* ISD station history: https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv
  used to select the mainland Portuguese stations.
* ISD-Lite hourly observations (wind speed, sea-level pressure):
  https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{usaf}-{wban}-{year}.gz
* Global Summary of the Day (daily maximum sustained wind, maximum gust, precipitation):
  https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{usaf}{wban}.csv
* Monthly North Atlantic Oscillation index, Climate Prediction Center:
  https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii

Outputs in --out:
  stations.csv          selected stations and their ISD coverage
  daily_isdlite.csv.gz  one row per station-day: hourly observations with a valid wind,
                        maximum hourly wind speed (m/s), minimum sea-level pressure (hPa)
  daily_gsod.csv.gz     one row per station-day: GSOD maximum sustained wind and maximum
                        gust (m/s, converted from knots), precipitation (mm)
  nao_monthly.csv       year, month, NAO index
  fetch_log.txt         time, HTTP status, bytes, SHA-256 and URL of every download
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ISD_HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ISD_LITE = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{usaf}-{wban}-{year}.gz"
GSOD = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{usaf}{wban}.csv"
NAO = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"

# Mainland Portugal (excludes the Azores and Madeira).
LAT_MIN, LAT_MAX = 36.8, 42.3
LON_MIN, LON_MAX = -9.7, -6.0
FIRST_YEAR = 1940
LAST_YEAR = 2026
MIN_END = 20260301     # station must still report in March 2026
MAX_BEGIN = 19850101   # and have started by 1985

KNOT = 0.514444


class Fetcher:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.lines: list[str] = []

    def get(self, url: str, tries: int = 4) -> bytes | None:
        for attempt in range(tries):
            try:
                req = Request(url, headers={"User-Agent": "storm-clustering-portugal (research; github.com/DiogoRibeiro7)"})
                with urlopen(req, timeout=120) as resp:
                    body = resp.read()
                self._log(200, body, url)
                return body
            except HTTPError as err:
                if err.code == 404:
                    self._log(404, b"", url)
                    return None
                last = err.code
            except (URLError, TimeoutError, ConnectionError) as err:  # noqa: PERF203
                last = type(err).__name__
            time.sleep(2 * (attempt + 1))
        self._log(f"failed:{last}", b"", url)
        return None

    def _log(self, status, body: bytes, url: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        digest = hashlib.sha256(body).hexdigest() if body else "-"
        self.lines.append(f"{stamp}\t{status}\t{len(body)}\t{digest}\t{url}")

    def write_log(self) -> None:
        self.log_path.write_text("\n".join(sorted(self.lines, key=lambda l: l.split("\t")[-1])) + "\n")


def parse_history(text: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            lat, lon = float(r["LAT"]), float(r["LON"])
            begin, end = int(r["BEGIN"]), int(r["END"])
        except (ValueError, KeyError):
            continue
        if r["CTRY"] != "PO":
            continue
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            continue
        if end < MIN_END or begin > MAX_BEGIN:
            continue
        if r["USAF"] == "999999":
            continue
        rows.append({
            "usaf": r["USAF"], "wban": r["WBAN"], "name": r["STATION NAME"].strip(),
            "icao": r["ICAO"].strip(), "lat": lat, "lon": lon,
            "elev_m": r["ELEV(M)"].strip(), "begin": begin, "end": end,
        })
    return rows


def parse_isdlite(raw: bytes) -> dict[str, list]:
    """Daily aggregates from one ISD-Lite station-year file.

    ISD-Lite columns: year month day hour air-temp dew-point sea-level-pressure
    wind-direction wind-speed sky-cover precip-1h precip-6h. Scaling factor 10
    for pressure (hPa) and wind speed (m/s); -9999 is missing.
    """
    days: dict[str, list] = {}
    for line in gzip.decompress(raw).decode("ascii", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        y, m, d = parts[0], parts[1].zfill(2), parts[2].zfill(2)
        key = f"{y}-{m}-{d}"
        slp, wspd = int(parts[6]), int(parts[8])
        rec = days.setdefault(key, [0, None, None])
        if wspd != -9999:
            rec[0] += 1
            v = wspd / 10.0
            rec[1] = v if rec[1] is None else max(rec[1], v)
        if slp != -9999:
            p = slp / 10.0
            rec[2] = p if rec[2] is None else min(rec[2], p)
    return days


def parse_gsod(raw: bytes) -> dict[str, tuple]:
    out = {}
    for r in csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace"))):
        def val(name, missing):
            try:
                x = float(r[name])
            except (ValueError, KeyError):
                return None
            return None if x >= missing else x
        mx = val("MXSPD", 999.0)
        gust = val("GUST", 999.0)
        prcp = val("PRCP", 99.0)
        out[r["DATE"]] = (
            None if mx is None else round(mx * KNOT, 2),
            None if gust is None else round(gust * KNOT, 2),
            None if prcp is None else round(prcp * 25.4, 1),
        )
    return out


def parse_nao(text: str) -> list[tuple[int, int, float]]:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3:
            try:
                rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(out / "fetch_log.txt")

    history = fetcher.get(ISD_HISTORY)
    if history is None:
        fetcher.write_log()
        print("could not download the ISD station history", file=sys.stderr)
        return 1
    stations = parse_history(history.decode("utf-8", errors="replace"))
    with open(out / "stations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stations[0]))
        w.writeheader()
        w.writerows(stations)
    print(f"{len(stations)} mainland stations reporting from {MAX_BEGIN // 10000} or earlier to 2026")

    jobs = []
    for s in stations:
        first = max(FIRST_YEAR, s["begin"] // 10000)
        for year in range(first, LAST_YEAR + 1):
            jobs.append((s, year))

    def work(job):
        s, year = job
        lite = fetcher.get(ISD_LITE.format(year=year, usaf=s["usaf"], wban=s["wban"]))
        gsod = fetcher.get(GSOD.format(year=year, usaf=s["usaf"], wban=s["wban"]))
        return s, (parse_isdlite(lite) if lite else {}), (parse_gsod(gsod) if gsod else {})

    lite_rows, gsod_rows = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for s, lite, gsod in pool.map(work, jobs):
            sid = f"{s['usaf']}-{s['wban']}"
            for day, (n, wmax, pmin) in lite.items():
                lite_rows.append((sid, day, n, "" if wmax is None else wmax, "" if pmin is None else pmin))
            for day, (mx, gust, prcp) in gsod.items():
                gsod_rows.append((sid, day, "" if mx is None else mx, "" if gust is None else gust,
                                  "" if prcp is None else prcp))
    lite_rows.sort()
    gsod_rows.sort()
    with gzip.open(out / "daily_isdlite.csv.gz", "wt", newline="", compresslevel=9) as fh:
        w = csv.writer(fh)
        w.writerow(["station", "date", "n_wind_obs", "wind_max_ms", "slp_min_hpa"])
        w.writerows(lite_rows)
    with gzip.open(out / "daily_gsod.csv.gz", "wt", newline="", compresslevel=9) as fh:
        w = csv.writer(fh)
        w.writerow(["station", "date", "mxspd_ms", "gust_ms", "prcp_mm"])
        w.writerows(gsod_rows)

    nao = fetcher.get(NAO)
    if nao is not None:
        with open(out / "nao_monthly.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["year", "month", "nao"])
            w.writerows(parse_nao(nao.decode("ascii", errors="replace")))
    fetcher.write_log()
    ok = sum(1 for l in fetcher.lines if l.split("\t")[1] == "200")
    failed = [l for l in fetcher.lines if l.split("\t")[1].startswith("failed")]
    print(f"{ok} files downloaded; {len(failed)} failed after retries; "
          f"{len(lite_rows)} station-days (ISD-Lite), {len(gsod_rows)} station-days (GSOD)")
    for l in failed[:20]:
        print("  failed:", l.split("\t")[-1])
    return 0 if nao is not None and lite_rows else 1


if __name__ == "__main__":
    sys.exit(main())
