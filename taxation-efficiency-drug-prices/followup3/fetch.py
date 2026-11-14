"""Download and reduce the public files used in follow-up 3 (whether the fall in the US share of established
medicines comes from the United States or from abroad).

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits the reduced files in data/
together with data/fetch_log.txt. Each source is fetched independently; a failure is logged and the others
still run.

The revenue of each medicine in the United States and in total, and the FDA approval files, come from the
snapshot of follow-up 2 (followup2/data) and are not downloaded again.

Sources
  partd   CMS, Medicare Part D Spending by Drug: every release in the CMS data catalogue. Each release covers
          five calendar years; for each brand and year the value kept is the one from the latest release that
          covers the year. Spending is gross of manufacturer rebates; dosage units and claims measure volume.
  partb   CMS, Medicare Part B Spending by Drug (medicines administered in a doctor's office or hospital
          outpatient department), by HCPCS code, the same way.

    python fetch.py --out data
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import time
import traceback
from datetime import date
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)"}
CATALOGUE = "https://data.cms.gov/data.json"
TITLES = {"partd": "Medicare Part D Spending by Drug", "partb": "Medicare Part B Spending by Drug"}
MEASURES = {"spndng": "spend", "dsg_unts": "units", "clms": "claims", "benes": "benes"}

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def get(url: str, tries: int = 4, timeout: int = 300, wait: float = 5.0, **kw) -> requests.Response:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"status {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(wait * (k + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def save(df: pd.DataFrame, out: Path, name: str) -> None:
    path = out / f"{name}.csv.gz"
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode("utf-8"))
    log(f"[{name}] wrote {path.name}: {len(df)} rows, columns {list(df.columns)}")


_catalogue: dict | None = None


def catalogue() -> dict:
    global _catalogue
    if _catalogue is None:
        _catalogue = get(CATALOGUE).json()
        log(f"[catalogue] {len(_catalogue.get('dataset', []))} datasets")
    return _catalogue


def txt(v) -> str:
    if isinstance(v, list):
        return ";".join(txt(x) for x in v)
    if isinstance(v, dict):
        return txt(v.get("@id") or v.get("url") or "")
    return "" if v is None else str(v)


def releases(title: str) -> list[dict]:
    """The CSV distributions of the dataset with this title, one per release, newest first."""
    # each release is its own catalogue entry, titled "<title> : <last day covered>"
    ds = [d for d in catalogue()["dataset"]
          if re.fullmatch(re.escape(title.lower()) + r"(\s*:\s*[\d-]+)?", d.get("title", "").strip().lower())]
    if not ds:
        near = [d.get("title") for d in catalogue()["dataset"] if "spending by drug" in d.get("title", "").lower()]
        raise RuntimeError(f"no dataset titled {title!r}; similar: {near}")
    out = []
    for d in ds:
        for dist in d.get("distribution", []):
            url = txt(dist.get("downloadURL"))
            api = txt(dist.get("accessURL"))
            kind = (txt(dist.get("mediaType")) or txt(dist.get("format"))).lower()
            out.append({"title": txt(dist.get("title")) or txt(d.get("title")),
                        "temporal": txt(dist.get("temporal")) or txt(d.get("temporal")),
                        "modified": txt(dist.get("modified")),
                        "csv": url if (url.lower().endswith(".csv") or "csv" in kind) else "",
                        "api": api if "data-api" in api else ""})
    for r in out:
        log(f"  [{title}] {r['title']!r} temporal {r['temporal']} csv={bool(r['csv'])} api={bool(r['api'])}")
    # one entry per release: prefer the CSV file, else the API
    by: dict[str, dict] = {}
    for r in out:
        key = r["temporal"] or r["title"]
        cur = by.setdefault(key, {"title": r["title"], "temporal": r["temporal"], "csv": "", "api": ""})
        cur["csv"] = cur["csv"] or r["csv"]
        cur["api"] = cur["api"] or r["api"]
    rel = [r for r in by.values() if r["csv"] or r["api"]]
    return sorted(rel, key=lambda r: r["temporal"] or r["title"], reverse=True)


def read_release(r: dict) -> pd.DataFrame:
    if r["csv"]:
        raw = get(r["csv"]).content
        return pd.read_csv(io.BytesIO(raw), low_memory=False, dtype=str)
    rows, off = [], 0
    while True:
        page = get(r["api"], params={"size": 5000, "offset": off}).json()
        if not page:
            break
        rows += page
        off += len(page)
    return pd.DataFrame(rows, dtype=str)


def to_long(df: pd.DataFrame, ids: list[str], release: str) -> pd.DataFrame:
    cols = {}
    for c in df.columns:
        m = re.fullmatch(r"(?i)tot_(spndng|dsg_unts|clms|benes)_(\d{4})", c.strip())
        if m:
            cols[c] = (MEASURES[m.group(1).lower()], int(m.group(2)))
    if not cols:
        raise RuntimeError(f"no yearly measure columns in {list(df.columns)[:30]}")
    keep = [c for c in ids if c in df.columns]
    parts = []
    for year in sorted({y for _, y in cols.values()}):
        sub = df[keep].copy()
        for c, (meas, y) in cols.items():
            if y == year:
                sub[meas] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
        sub["year"] = year
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    out["release"] = release
    return out


def fetch_program(name: str, out: Path) -> None:
    title = TITLES[name]
    rel = releases(title)
    log(f"[{name}] {len(rel)} releases")
    ids = ["Brnd_Name", "Gnrc_Name", "Mftr_Name", "Tot_Mftr"] if name == "partd" else \
          ["HCPCS_Cd", "HCPCS_Desc", "Brnd_Name", "Gnrc_Name", "Tot_Mftr"]
    frames = []
    for r in rel:
        try:
            df = read_release(r)
        except Exception as exc:  # noqa: BLE001
            log(f"[{name}] release {r['temporal']} failed: {exc}")
            continue
        df.columns = [c.strip() for c in df.columns]
        log(f"[{name}] release {r['temporal']}: {len(df)} rows, columns {list(df.columns)[:12]} ...")
        if "Mftr_Name" in df.columns:
            df = df[df["Mftr_Name"].str.strip().str.lower() == "overall"]
        long = to_long(df, ids, r["temporal"] or r["title"])
        frames.append(long)
    if not frames:
        raise RuntimeError(f"no release of {title} could be read")
    allr = pd.concat(frames, ignore_index=True)
    allr = allr[allr[["spend", "units"]].notna().any(axis=1)]
    # for each brand (and code) and year, the value of the latest release that covers the year
    key = [c for c in ["HCPCS_Cd", "Brnd_Name", "Gnrc_Name"] if c in allr.columns]
    allr = allr.sort_values("release").groupby(key + ["year"], dropna=False).tail(1)
    allr = allr.drop(columns=[c for c in ["Mftr_Name"] if c in allr.columns])
    log(f"[{name}] years {sorted(allr['year'].unique())}; rows {len(allr)}; brands {allr['Brnd_Name'].nunique()}")
    save(allr.sort_values(key + ["year"]), out, f"cms_{name}_by_drug")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"fetch.py run on {date.today().isoformat()}")
    failed = []
    for name in ["partd", "partb"]:
        if a.only and name not in a.only:
            continue
        try:
            fetch_program(name, out)
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            log(f"[{name}] FAILED: {exc}")
            log(traceback.format_exc())
    (out / "fetch_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
