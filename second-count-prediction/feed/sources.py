"""Snapshot of the outside counts, and of what each source publishes.

    python -m feed.sources --out sources --metadata-only   # names, sizes and HTTP status only
    python -m feed.sources --out sources                   # the values (only once PLAN.md is committed)

Citi Bike (New York):
- the public bucket of trip files: keys and sizes (the files are counted by ``feed.citibike``);
- the operator's monthly operating reports to NYC DOT (PDF): the month's total trips, read from the text;
- NYC DOT's quarterly Bike Share Usage Data Report: the table "Citi Bike Customer Trips by Month".

Chicago, Transportation Network Providers (rideshare):
- the trips datasets, counted by month of trip start with the portal's own aggregation;
- the vehicles and drivers datasets: NUMBER_OF_TRIPS summed by MONTH_REPORTED.

Every document is recorded with its URL, retrieval date and SHA-256.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import hashlib
import io
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

from feed.http import get, status

BUCKET = "https://s3.amazonaws.com/tripdata"
REPORTS_PAGE = "https://citibikenyc.com/system-data/operating-reports"
DOT_CANDIDATES = [
    f"https://{host}/html/dot/downloads/excel/bike-share-usage-data-report-{q}.xlsx"
    for q in ("q3-2026", "q2-2026", "q1-2026", "q4-2025", "q3-2025")
    for host in ("home4.nyc.gov", "www.nyc.gov")
] + [
    "https://a860-gpp.nyc.gov/downloads/gx41mp52w?locale=en",  # Q2 2025, PDF
    "https://a860-gpp.nyc.gov/downloads/8k71nn50q?locale=en",  # Q1 2025, PDF
]
PORTAL = "https://data.cityofchicago.org"
TRIPS = {"m6dm-c72p": ("2018-11", "2022-12"), "n26f-ihde": ("2023-01", "2024-12"), "6dvr-xwnh": ("2025-01", None)}
PER_UNIT = {"vehicles": "bc6b-sq4u", "drivers": "j6wf-834c"}
MIN_MONTHLY = 10_000  # a parsed operator-report total below this is not a month's trips (PLAN.md)

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
REPORT_PATTERNS = (
    r"There\s+were\s+([\d,]{5,})\s+(?:total\s+)?trips\s+in\b",
    r"riders\s+took\s+([\d,]{5,})\s+trips",
    r"There\s+were\s+([\d,]{5,})\s+total\s+trips",
    r"([\d,]{5,})\s+total\s+trips",
)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --- Citi Bike -------------------------------------------------------------------------------------------


def bucket_keys() -> list[dict]:
    """Every key in the public trip-data bucket, with its size (S3 ListObjectsV2, paginated)."""
    out, token = [], None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        q = {"list-type": "2"} | ({"continuation-token": token} if token else {})
        root = ET.fromstring(get(f"{BUCKET}?{urllib.parse.urlencode(q)}"))
        for c in root.findall("s3:Contents", ns):
            out.append({"key": c.findtext("s3:Key", namespaces=ns), "size": int(c.findtext("s3:Size", namespaces=ns))})
        if root.findtext("s3:IsTruncated", namespaces=ns) != "true":
            return out
        token = root.findtext("s3:NextContinuationToken", namespaces=ns)


def report_month(url: str) -> str | None:
    """The month an operating-report URL is for, from its file name."""
    name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    if m := re.search(r"(\d{4})_(\d{2})_[A-Za-z]+_Citi", name):
        return f"{m[1]}-{m[2]}"
    if m := re.search(r"([A-Za-z]+)[-_ ](\d{4})", name):
        if m[1].lower() in MONTHS:
            return f"{m[2]}-{MONTHS[m[1].lower()]:02d}"
    if m := re.search(r"(\d{4})[-_ ]([A-Za-z]+)", name):
        if m[2].lower() in MONTHS:
            return f"{m[1]}-{MONTHS[m[2].lower()]:02d}"
    return None


def report_links(html: str) -> dict[str, str]:
    """Month -> PDF link, from the operating-reports page (first link for a month wins)."""
    out: dict[str, str] = {}
    for href in re.findall(r'href="([^"]+?\.pdf[^"]*)"', html, flags=re.I):
        href = href.replace("&amp;", "&")
        if (m := report_month(href)) and m not in out:
            out[m] = href
    return dict(sorted(out.items()))


def report_total(text: str) -> tuple[int | None, int | None]:
    """The month's total trips in an operating report, and which pattern found it (PLAN.md order)."""
    t = re.sub(r"\s+", " ", text)
    for i, p in enumerate(REPORT_PATTERNS):
        if m := re.search(p, t, flags=re.I):
            n = int(m[1].replace(",", ""))
            if n >= MIN_MONTHLY:
                return n, i
    return None, None


def pdf_text(data: bytes, pages: int | None = None) -> str:
    from pypdf import PdfReader

    r = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in r.pages[:pages])


def dot_months_from_text(text: str) -> dict[str, int]:
    """'Citi Bike Customer Trips by Month' from a PDF's text; months marked * (partial) are left out."""
    out: dict[str, int] = {}
    for m, star, n in re.findall(r"\b(\d{4}-\d{2})(\*?)\s+([\d,]{4,})\b", text):
        if not star and m not in out:
            out[m] = int(n.replace(",", ""))
    return out


def dot_months_from_xlsx(data: bytes) -> dict[str, int]:
    from openpyxl import load_workbook

    out: dict[str, int] = {}
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            cells = [c for c in row if c is not None and str(c).strip() != ""]
            if len(cells) < 2:
                continue
            first = cells[0]
            if isinstance(first, dt.datetime):
                key, star = f"{first.year:04d}-{first.month:02d}", False
            else:
                s = str(first).strip()
                if not (m := re.fullmatch(r"(\d{4}-\d{2})(\*?)", s)):
                    continue
                key, star = m[1], bool(m[2])
            n = cells[1]
            if isinstance(n, str):
                n = n.replace(",", "").strip()
            try:
                n = int(float(n))
            except (TypeError, ValueError):
                continue
            if not star and key not in out:
                out[key] = n
    return out


# --- Chicago ---------------------------------------------------------------------------------------------


def soql(dataset: str, tries: int = 3, timeout: int = 180, **params: str) -> list[dict]:
    q = urllib.parse.urlencode({f"${k}": v for k, v in params.items()})
    return json.loads(get(f"{PORTAL}/resource/{dataset}.json?{q}", tries=tries, timeout=timeout, backoff=10))


def log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.UTC):%H:%M:%S} {msg}", flush=True)


def month_range(first: str, last: str) -> list[str]:
    y, m = map(int, first.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def next_month(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y + (m == 12):04d}-{m % 12 + 1:02d}"


def count_month(ds: str, m: str) -> int:
    lo, hi = f"{m}-01T00:00:00", f"{next_month(m)}-01T00:00:00"
    where = f"trip_start_timestamp >= '{lo}' AND trip_start_timestamp < '{hi}'"
    rows = soql(ds, tries=1, timeout=900, select="count(*) AS n", where=where)
    return int(rows[0]["n"]) if rows else 0


def count_dataset(ds: str) -> dict[str, int]:
    """All months of a dataset in one grouped query (one scan of the table instead of one per month)."""
    rows = soql(
        ds,
        tries=1,
        timeout=3600,
        select="date_trunc_ym(trip_start_timestamp) AS m, count(*) AS n",
        group="m",
        order="m",
        limit="5000",
    )
    return {r["m"][:7]: int(r["n"]) for r in rows if r.get("m")}


def chicago_trips(today: str, progress: Path | None = None, passes: int = 3, workers: int = 2) -> dict[str, dict]:
    """Trips by month of trip start (PLAN.md, changes of 12 November).

    Every answer is saved to ``progress`` at once, so a rerun picks up from there. Months not yet counted are
    first asked for with one grouped query per dataset (one scan of the table, allowed an hour); the months a
    grouped query does not cover are then counted one at a time, two at once, in up to ``passes`` passes. A
    month still missing after the last pass is left out and logged.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    done: dict[str, int] = json.loads(progress.read_text(encoding="utf-8")) if progress and progress.exists() else {}
    tasks = [(ds, m) for ds, (first, last) in TRIPS.items() for m in month_range(first, last or today)]

    def save() -> None:
        if progress:
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text(json.dumps(done, indent=1, sort_keys=True), encoding="utf-8")

    pending = sorted({ds for ds, m in tasks if f"{ds} {m}" not in done})
    if pending:
        log(f"chicago trips, grouped queries for {' '.join(pending)}")
        with ThreadPoolExecutor(len(pending)) as ex:
            futures = {ex.submit(count_dataset, ds): ds for ds in pending}
            for f in as_completed(futures):
                ds = futures[f]
                try:
                    got = f.result()
                except (RuntimeError, ValueError, KeyError):
                    log(f"chicago {ds}: grouped query did not answer")
                    continue
                for ds2, m in tasks:
                    if ds2 == ds and f"{ds} {m}" not in done:
                        done[f"{ds} {m}"] = got.get(m, 0)
                save()
                log(f"chicago {ds}: grouped query answered")
    for p in range(passes):
        todo = [t for t in tasks if f"{t[0]} {t[1]}" not in done]
        if not todo:
            break
        log(f"chicago trips, pass {p + 1}: {len(todo)} months to count")
        with ThreadPoolExecutor(workers) as ex:
            futures = {ex.submit(count_month, ds, m): (ds, m) for ds, m in todo}
            for f in as_completed(futures):
                ds, m = futures[f]
                try:
                    done[f"{ds} {m}"] = f.result()
                except (RuntimeError, ValueError, KeyError, IndexError):
                    log(f"chicago {ds} {m}: no answer in pass {p + 1}")
                    continue
                save()
    out: dict[str, dict] = {}
    for ds, m in tasks:
        n = done.get(f"{ds} {m}")
        if n is None:
            log(f"chicago {ds} {m}: left out, no answer after {passes} passes")
        elif n > 0:
            out.setdefault(m, {"trips": 0, "datasets": []})
            out[m]["trips"] += n
            out[m]["datasets"].append(ds)
    return dict(sorted(out.items()))


def chicago_per_unit(dataset: str) -> dict[str, dict]:
    rows = soql(
        dataset,
        tries=4,
        timeout=600,
        select="month_reported, sum(number_of_trips) AS trips, count(*) AS records, "
        "count(number_of_trips) AS with_trips",
        group="month_reported",
        order="month_reported",
        limit="5000",
    )
    out = {}
    for r in rows:
        m = str(r.get("month_reported", ""))[:7]
        if re.fullmatch(r"\d{4}-\d{2}", m):
            out[m] = {
                "trips": int(float(r.get("trips") or 0)),
                "records": int(r["records"]),
                "with_trips": int(r["with_trips"]),
            }
    return out


def view_meta(dataset: str) -> dict:
    v = json.loads(get(f"{PORTAL}/api/views/{dataset}.json"))
    return {
        "id": dataset,
        "name": v.get("name"),
        "rowsUpdatedAt": v.get("rowsUpdatedAt"),
        "columns": [[c.get("fieldName"), c.get("dataTypeName")] for c in v.get("columns", [])],
    }


# --- snapshot --------------------------------------------------------------------------------------------


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def snapshot_citibike(out: Path, metadata_only: bool) -> None:
    meta: dict = {"retrieved": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "documents": {}}
    log("bucket")
    keys = bucket_keys()
    (out / "citibike_bucket.json").write_text(json.dumps(keys, indent=1), encoding="utf-8")
    html = get(REPORTS_PAGE).decode("utf-8", "replace")
    links = report_links(html)
    meta["documents"]["operating_reports_page"] = {"url": REPORTS_PAGE, "sha256": sha(html.encode()), "links": links}
    meta["dot_candidates"] = [status(u) for u in DOT_CANDIDATES]
    if metadata_only:
        (out / "METADATA.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        log(f"metadata only: {len(keys)} bucket keys, {len(links)} report links")
        return

    log("operator reports")
    rows = []
    for month, url in links.items():
        try:
            data = get(url if url.startswith("http") else urllib.parse.urljoin(REPORTS_PAGE, url))
        except Exception as e:  # noqa: BLE001 - a missing report is recorded, not fatal
            rows.append([month, "", "", url, "", f"download failed: {str(e)[:80]}"])
            continue
        n, pat = report_total(pdf_text(data, pages=6))
        rows.append([month, n if n is not None else "", pat if pat is not None else "", url, sha(data), ""])
    write_csv(out / "citibike_operator_reports.csv", ["month", "trips", "pattern", "url", "sha256", "note"], rows)

    log("DOT table")
    for u in DOT_CANDIDATES:
        try:
            data = get(u)
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
        table = dot_months_from_xlsx(data) if data[:2] == b"PK" else dot_months_from_text(pdf_text(data))
        if len(table) >= 24:
            ext = "xlsx" if data[:2] == b"PK" else "pdf"
            (out / f"citibike_dot_report.{ext}").write_bytes(data)
            meta["documents"]["dot_report"] = {"url": u, "sha256": sha(data), "months": len(table)}
            write_csv(out / "citibike_dot.csv", ["month", "trips"], [[m, n] for m, n in sorted(table.items())])
            break
    (out / "SOURCES.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"values: {len(rows)} operating reports")


def snapshot_chicago(out: Path, metadata_only: bool, progress: Path | None = None) -> None:
    meta = {
        "retrieved": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "datasets": {ds: view_meta(ds) for ds in [*TRIPS, *PER_UNIT.values()]},
    }
    if metadata_only:
        (out / "METADATA_chicago.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        log("metadata only: Chicago datasets")
        return
    for name, ds in PER_UNIT.items():
        log(f"chicago {name}")
        per = chicago_per_unit(ds)
        write_csv(
            out / f"chicago_{name}.csv",
            ["month", "trips", "records", "records_with_trips"],
            [[m, v["trips"], v["records"], v["with_trips"]] for m, v in sorted(per.items())],
        )
    log("chicago trips")
    trips = chicago_trips(dt.date.today().isoformat()[:7], progress)
    write_csv(
        out / "chicago_trips.csv",
        ["month", "trips", "datasets"],
        [[m, v["trips"], " ".join(v["datasets"])] for m, v in trips.items()],
    )
    (out / "SOURCES_chicago.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"values: {len(trips)} Chicago months")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("sources"))
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--only", choices=["citibike", "chicago"])
    ap.add_argument("--progress", type=Path, help="file of Chicago month counts already fetched (resumes)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.only != "chicago":
        snapshot_citibike(a.out, a.metadata_only)
    if a.only != "citibike":
        snapshot_chicago(a.out, a.metadata_only, a.progress)


if __name__ == "__main__":
    main()
