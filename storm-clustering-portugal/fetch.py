"""Download the station records and the NAO index used by storm_clustering.py.

Run by the GitHub Actions workflow storm-clustering-portugal-data.yml, which
commits the compact tables it writes to data/. The analysis itself never
touches the network.

Sources (public domain, U.S. National Oceanic and Atmospheric Administration):

* Global Historical Climatology Network hourly (GHCNh), NOAA NCEI, the
  successor of the Integrated Surface Database, doi:10.25921/jp3d-3v19.
  Station list: .../hourly/doc/ghcnh-station-list.csv
  Station records: .../hourly/access/by-station/GHCNh_<id>_por.psv
* Monthly North Atlantic Oscillation index, NOAA Climate Prediction Center.

Outputs in --out:
  stations.csv        mainland Portuguese GHCNh stations with any hourly wind data
  daily_wind.csv.gz   one row per station-day (UTC): hourly reports with a valid wind
                      speed, maximum wind speed (m/s), maximum gust (m/s), minimum
                      sea-level pressure (hPa)
  qc_codes.csv        how often each quality code appeared for wind speed, and whether
                      it was accepted
  nao_monthly.csv     year, month, NAO index
  fetch_log.txt       time, HTTP status, bytes, SHA-256 and URL of every download
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly"
STATION_LISTS = [f"{BASE}/doc/ghcnh-station-list.csv", f"{BASE}/doc/ghcnh-station-list.txt"]
POR = BASE + "/access/by-station/GHCNh_{sid}_por.psv"
NAO = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"

# Mainland Portugal (excludes the Azores and Madeira).
LAT_MIN, LAT_MAX = 36.8, 42.3
LON_MIN, LON_MAX = -9.7, -6.0

# GHCNh quality codes: blank means no test failed; ISD-derived numeric flags 0, 1, 4, 5, 9
# mean passed or not checked. Anything else (a letter naming a failed test, or ISD's
# 2, 3, 6, 7 for suspect or erroneous) is rejected.
ACCEPT_QC = {"", "0", "1", "4", "5", "9"}


class Fetcher:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.lines: list[str] = []

    def get(self, url: str, tries: int = 4) -> bytes | None:
        last = None
        for attempt in range(tries):
            try:
                req = Request(url, headers={"User-Agent": "storm-clustering-portugal (research; github.com/DiogoRibeiro7)"})
                with urlopen(req, timeout=300) as resp:
                    body = resp.read()
                self._log(200, body, url)
                return body
            except HTTPError as err:
                if err.code == 404:
                    self._log(404, b"", url)
                    return None
                last = err.code
            except (URLError, TimeoutError, ConnectionError) as err:
                last = type(err).__name__
            time.sleep(3 * (attempt + 1))
        self._log(f"failed:{last}", b"", url)
        return None

    def _log(self, status, body: bytes, url: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        digest = hashlib.sha256(body).hexdigest() if body else "-"
        self.lines.append(f"{stamp}\t{status}\t{len(body)}\t{digest}\t{url}")

    def write_log(self) -> None:
        self.log_path.write_text("\n".join(sorted(self.lines, key=lambda l: l.split("\t")[-1])) + "\n")


def parse_station_list(text: str) -> list[dict]:
    """Mainland Portuguese stations from the GHCNh station list (CSV or fixed width)."""
    rows = []
    first_line = text.splitlines()[0] if text else ""
    if "," in first_line:
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
        upper = [h.strip().upper() for h in header]

        def col(*names):
            for n in names:
                if n in upper:
                    return upper.index(n)
            return None

        i_id, i_lat, i_lon = col("GHCN_ID", "ID", "STATION_ID"), col("LATITUDE", "LAT"), col("LONGITUDE", "LON")
        i_elev, i_name = col("ELEVATION", "ELEV"), col("NAME", "STATION_NAME")
        i_wmo, i_icao = col("WMO_ID", "WMO"), col("ICAO")
        for r in reader:
            if not r or i_id is None or max(i_id, i_lat, i_lon) >= len(r):
                continue
            rows.append({
                "id": r[i_id].strip(), "lat": r[i_lat], "lon": r[i_lon],
                "elev_m": r[i_elev].strip() if i_elev is not None and i_elev < len(r) else "",
                "name": r[i_name].strip() if i_name is not None and i_name < len(r) else "",
                "wmo": r[i_wmo].strip() if i_wmo is not None and i_wmo < len(r) else "",
                "icao": r[i_icao].strip() if i_icao is not None and i_icao < len(r) else "",
            })
    else:  # fixed width, GHCN style: ID(1-11) LAT(13-20) LON(22-30) ELEV(32-37) NAME(42-71)
        for line in text.splitlines():
            if len(line) < 38:
                continue
            rows.append({"id": line[0:11].strip(), "lat": line[12:20], "lon": line[21:30],
                         "elev_m": line[31:37].strip(), "name": line[41:71].strip(), "wmo": "", "icao": ""})
    out = []
    for r in rows:
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
        except ValueError:
            continue
        if not r["id"].startswith("PO"):
            continue
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            out.append({**r, "lat": lat, "lon": lon})
    return out


def _float(x: str):
    try:
        return float(x)
    except ValueError:
        return None


def parse_psv(raw: bytes, qc_counter: Counter | None = None) -> dict[str, list]:
    """Daily aggregates (UTC) from one GHCNh pipe-separated station file.

    Returns {date: [n_valid_wind, wind_max, gust_max, slp_min]}.
    """
    text = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="replace")
    reader = csv.reader(text, delimiter="|")
    header = [h.strip() for h in next(reader)]
    idx = {h: i for i, h in enumerate(header)}
    i_date = idx.get("DATE")
    i_y, i_m, i_d = idx.get("Year"), idx.get("Month"), idx.get("Day")
    cols = {var: (idx.get(var), idx.get(f"{var}_Quality_Code"))
            for var in ("wind_speed", "wind_gust", "sea_level_pressure")}
    days: dict[str, list] = {}
    for r in reader:
        if not r:
            continue
        if i_date is not None and i_date < len(r) and r[i_date]:
            key = r[i_date][:10]
        elif None not in (i_y, i_m, i_d):
            try:
                key = f"{int(r[i_y]):04d}-{int(r[i_m]):02d}-{int(r[i_d]):02d}"
            except (ValueError, IndexError):
                continue
        else:
            continue
        rec = days.setdefault(key, [0, None, None, None])
        for slot, var in ((1, "wind_speed"), (2, "wind_gust"), (3, "sea_level_pressure")):
            iv, iq = cols[var]
            if iv is None or iv >= len(r):
                continue
            v = _float(r[iv])
            if v is None:
                continue
            q = r[iq].strip() if iq is not None and iq < len(r) else ""
            if var == "wind_speed" and qc_counter is not None:
                qc_counter[q] += 1
            if q not in ACCEPT_QC:
                continue
            if var == "wind_speed":
                if 0 <= v <= 75:
                    rec[0] += 1
                    rec[1] = v if rec[1] is None else max(rec[1], v)
            elif var == "wind_gust":
                if 0 <= v <= 100:
                    rec[2] = v if rec[2] is None else max(rec[2], v)
            elif 900 <= v <= 1080:
                rec[3] = v if rec[3] is None else min(rec[3], v)
    return days


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
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(out / "fetch_log.txt")

    stations = []
    for url in STATION_LISTS:
        body = fetcher.get(url)
        if body is not None:
            text = body.decode("utf-8-sig", errors="replace")
            print(f"station list {url}: first line {text.splitlines()[0][:160]!r}")
            stations = parse_station_list(text)
            if stations:
                break
    if not stations:
        fetcher.write_log()
        print("no mainland Portuguese station found in the GHCNh station list", file=sys.stderr)
        return 1
    print(f"{len(stations)} GHCNh stations in mainland Portugal")

    qc = Counter()

    def work(s):
        raw = fetcher.get(POR.format(sid=s["id"]))
        if raw is None:
            return s, {}, Counter(), 0
        local = Counter()
        try:
            days = parse_psv(raw, local)
        except Exception as err:  # keep going; report the station
            print(f"  could not parse {s['id']}: {err!r}", file=sys.stderr)
            return s, {}, local, len(raw)
        return s, days, local, len(raw)

    rows, kept = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for s, days, local, size in pool.map(work, stations):
            qc.update(local)
            wind_days = [d for d, v in days.items() if v[0] > 0]
            if not wind_days:
                print(f"  {s['id']} {s['name'][:28]:<28} no wind data ({size} bytes)")
                continue
            first, last = min(wind_days), max(wind_days)
            kept.append({**s, "first_wind_day": first, "last_wind_day": last, "wind_days": len(wind_days)})
            for day, (n, w, g, p) in days.items():
                if n == 0 and g is None and p is None:
                    continue
                rows.append((s["id"], day, n, "" if w is None else w, "" if g is None else g,
                             "" if p is None else p))
            print(f"  {s['id']} {s['name'][:28]:<28} {first} to {last}, {len(wind_days)} days with wind")
    rows.sort()
    kept.sort(key=lambda s: s["id"])
    with open(out / "stations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0]) if kept else ["id"])
        w.writeheader()
        w.writerows(kept)
    with gzip.open(out / "daily_wind.csv.gz", "wt", newline="", compresslevel=9) as fh:
        w = csv.writer(fh)
        w.writerow(["station", "date", "n_wind_obs", "wind_max_ms", "gust_max_ms", "slp_min_hpa"])
        w.writerows(rows)
    with open(out / "qc_codes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wind_speed_quality_code", "count", "accepted"])
        for code, n in qc.most_common():
            w.writerow([code if code else "(blank)", n, int(code in ACCEPT_QC)])

    nao = fetcher.get(NAO)
    if nao is not None:
        with open(out / "nao_monthly.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["year", "month", "nao"])
            w.writerows(parse_nao(nao.decode("ascii", errors="replace")))
    fetcher.write_log()
    failed = [l for l in fetcher.lines if l.split("\t")[1].startswith("failed")]
    print(f"{len(kept)} stations with wind data; {len(rows)} station-days; {len(failed)} downloads failed")
    for l in failed[:20]:
        print("  failed:", l.split("\t")[-1])
    return 0 if nao is not None and rows else 1


if __name__ == "__main__":
    sys.exit(main())
