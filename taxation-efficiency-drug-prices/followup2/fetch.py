"""Download and reduce the public files used in follow-up 2 (where the rise in the US share of drugmakers'
revenue comes from: new medicines or wider gaps on old ones).

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits the reduced files in data/
together with data/fetch_log.txt. Each source is fetched independently; a failure is logged and the others
still run.

Sources
  sec     SEC EDGAR: the annual reports (10-K, 20-F) of the seventeen companies of Part 6, fiscal years
          2015-2025, read from the XBRL instance of each filing. Every annual revenue fact is kept with its
          dimensions (product, geography, segment), for the year of the report and the comparative years it
          restates, so that revenue by product and by country can be rebuilt.
  fda     openFDA, Drugs@FDA bulk file: application number, brand names, active ingredients, sponsor and the
          date of the original approval of each NDA and BLA reviewed by CDER.
  purple  FDA Purple Book: licensed biological products of CDER and CBER (vaccines, blood products, gene
          therapies) with their licence dates.

    python fetch.py --out data
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

SEC_UA = {"User-Agent": "Diogo Ribeiro dfr@esmad.ipp.pt", "Accept-Encoding": "gzip, deflate"}
UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)"}

COMPANIES = {  # name: (CIK, headquarters), as in Part 6
    "Pfizer": (78003, "US"), "Merck & Co": (310158, "US"), "Johnson & Johnson": (200406, "US"),
    "AbbVie": (1551152, "US"), "Bristol-Myers Squibb": (14272, "US"), "Eli Lilly": (59478, "US"),
    "Amgen": (318154, "US"), "Gilead": (882095, "US"), "Regeneron": (872589, "US"), "Vertex": (875320, "US"),
    "Biogen": (875045, "US"),
    "Novartis": (1114448, "CH"), "AstraZeneca": (901832, "GB"), "GSK": (1131399, "GB"), "Sanofi": (1121404, "FR"),
    "Novo Nordisk": (353278, "DK"), "Takeda": (1395064, "JP"),
}
FIRST_FY, LAST_FY = 2015, 2025

# revenue concepts: the standard ones, and any company concept whose name says revenue or sales but is not a
# cost, an asset, a liability, a deferral or a ratio
REVENUE = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
           "SalesRevenueGoodsNet", "Revenue", "RevenueFromSaleOfGoods",
           "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenueNet"}
REV_NAME = re.compile(r"(Revenue|Sales)")
NOT_REV = re.compile(r"(CostOf|Cost|Deferred|Receivable|Unearned|Unbilled|RemainingPerformance|Allowance|Reserve|"
                     r"Proceeds|Gain|Loss|Liabilit|Asset|Percent|Ratio|Selling|Marketing|Expense|Accrual|"
                     r"Payable|Provision|Impairment|Investment|Tax(es)?Payable|Days|Number|Count)")

URLS = {
    "fda": "https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip",
    "purple_page": "https://purplebooksearch.fda.gov/downloads",
    "purple_file": "https://purplebooksearch.fda.gov/files/{y}/purplebook-search-{m}-data-download.csv",
}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def get(url: str, headers=None, tries: int = 4, timeout: int = 300, wait: float = 5.0) -> requests.Response:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=headers or UA, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"status {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(wait * (k + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def sec(url: str) -> requests.Response:
    time.sleep(0.12)  # SEC asks for at most ten requests a second
    return get(url, headers=SEC_UA, wait=2.0)


def save(df: pd.DataFrame, out: Path, name: str) -> None:
    path = out / f"{name}.csv.gz"
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode("utf-8"))
    log(f"[{name}] wrote {path.name}: {len(df)} rows, columns {list(df.columns)}")


# ---------------------------------------------------------------------------------------------- SEC
def local(tag: str) -> str:
    return tag.split("}")[-1]


def is_revenue(name: str) -> bool:
    return name in REVENUE or (bool(REV_NAME.search(name)) and not NOT_REV.search(name))


def parse_instance(xml: bytes) -> list[dict]:
    """Annual-duration monetary facts of revenue concepts, with every explicit dimension of their context."""
    root = ET.fromstring(xml)
    ctx, units = {}, {}
    for c in root.iter():
        t = local(c.tag)
        if t == "context":
            start = end = None
            dims = []
            for e in c.iter():
                lt = local(e.tag)
                if lt == "startDate":
                    start = (e.text or "").strip()
                elif lt == "endDate":
                    end = (e.text or "").strip()
                elif lt == "explicitMember":
                    dims.append(f"{(e.get('dimension') or '').split(':')[-1]}={(e.text or '').strip()}")
                elif lt == "typedMember":
                    dims.append(f"{(e.get('dimension') or '').split(':')[-1]}=typed")
            ctx[c.get("id")] = (start, end, ";".join(sorted(dims)))
        elif t == "unit":
            ms = [(m.text or "").strip() for m in c.iter() if local(m.tag) == "measure"]
            units[c.get("id")] = "/".join(ms)
    facts = []
    for e in root:
        name = local(e.tag)
        if e.get("contextRef") is None or not (e.text or "").strip() or not is_revenue(name):
            continue
        unit = units.get(e.get("unitRef"), "")
        if "iso4217" not in unit.lower() or "/" in unit:
            continue
        try:
            value = float(e.text.strip())
        except ValueError:
            continue
        start, end, dims = ctx.get(e.get("contextRef"), (None, None, ""))
        if not start or not end:
            continue
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days
        if not 330 <= days <= 380:
            continue
        facts.append({"concept": name, "prefix": e.tag.split("}")[0].strip("{"), "start": start, "end": end,
                      "dims": dims, "value": value, "currency": unit.split(":")[-1].upper()})
    return facts


def fiscal_year(end: str) -> int:
    d = date.fromisoformat(end)
    return d.year if d.month >= 7 else d.year - 1


def annual_filings(cik: int) -> list[dict]:
    sub = sec(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    rec = sub["filings"]["recent"]
    rows = [dict(zip(rec.keys(), vals)) for vals in zip(*rec.values())]
    for pg in sub["filings"].get("files", []):
        more = sec(f"https://data.sec.gov/submissions/{pg['name']}").json()
        rows += [dict(zip(more.keys(), vals)) for vals in zip(*more.values())]
    out = {}
    for r in rows:
        if r["form"] not in ("10-K", "20-F") or not r.get("reportDate"):
            continue
        fy = fiscal_year(r["reportDate"])
        if FIRST_FY <= fy <= LAST_FY:
            if fy not in out or r["filingDate"] < out[fy]["filingDate"]:
                out[fy] = r
    return [out[k] for k in sorted(out)]


def instance_name(files: list[str]) -> str | None:
    cand = [x for x in files if x.endswith("_htm.xml")]
    if not cand:
        cand = [x for x in files if x.endswith(".xml")
                and not re.search(r"(_cal|_def|_lab|_pre|FilingSummary|MetaLinks|^R\d+)", x)]
    return cand[0] if cand else None


PRODUCT_AXIS = re.compile(r"(Product|Brand|Franchise|Medicine|Drug|TherapeuticArea)", re.I)
GEO_AXIS = re.compile(r"(Geograph|Area|Region|Countr|Market)", re.I)


def fetch_sec(out: Path) -> None:
    rows, used = [], []
    for name, (cik, hq) in COMPANIES.items():
        try:
            filings = annual_filings(cik)
        except Exception as exc:  # noqa: BLE001
            log(f"[sec] {name}: submissions failed: {exc}")
            continue
        for f in filings:
            fy = fiscal_year(f["reportDate"])
            acc = f["accessionNumber"].replace("-", "")
            try:
                idx = sec(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json").json()
                files = [i["name"] for i in idx["directory"]["item"]]
                inst = instance_name(files)
                if inst is None:
                    log(f"[sec] {name} FY{fy} {f['form']}: no XBRL instance ({len(files)} files)")
                    continue
                facts = parse_instance(sec(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{inst}").content)
            except Exception as exc:  # noqa: BLE001
                log(f"[sec] {name} FY{fy}: failed: {exc}")
                continue
            used.append({"company": name, "hq": hq, "fiscal_year": fy, "form": f["form"],
                         "period_end": f["reportDate"], "filing_date": f["filingDate"],
                         "accession": f["accessionNumber"], "instance": inst})
            seen = set()
            for x in facts:
                key = (x["concept"], x["end"], x["dims"], x["value"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"company": name, "fiscal_year": fy, "form": f["form"], "accession": f["accessionNumber"],
                             **x})
            cur = [x for x in facts if abs((date.fromisoformat(x["end"]) - date.fromisoformat(f["reportDate"])).days) <= 7]
            axes = {}
            for x in cur:
                ax = tuple(sorted(d.split("=")[0] for d in x["dims"].split(";") if d))
                axes[ax] = axes.get(ax, 0) + 1
            both = sum(n for ax, n in axes.items() if any(PRODUCT_AXIS.search(a) for a in ax)
                       and any(GEO_AXIS.search(a) for a in ax))
            prod = sum(n for ax, n in axes.items() if any(PRODUCT_AXIS.search(a) for a in ax))
            top = sorted(axes.items(), key=lambda t: -t[1])[:8]
            log(f"[sec] {name} FY{fy} {f['form']}: {len(facts)} annual revenue facts ({len(cur)} for the year); "
                f"product axis {prod}, product and geography {both}; axes {top}")
    save(pd.DataFrame(rows), out, "sec_revenue_facts")
    save(pd.DataFrame(used), out, "sec_filings")


# ---------------------------------------------------------------------------------------------- FDA
def fetch_fda(out: Path) -> None:
    raw = get(URLS["fda"]).content
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        js = json.loads(z.read(z.namelist()[0]))
    rows = []
    for app in js["results"]:
        subs = app.get("submissions") or []
        orig = [s for s in subs if s.get("submission_type") == "ORIG" and s.get("submission_status") == "AP"
                and s.get("submission_status_date")]
        first = min((s["submission_status_date"] for s in orig), default=None)
        for p in app.get("products") or []:
            ings = "; ".join(sorted({(i.get("name") or "").strip() for i in p.get("active_ingredients") or []}))
            rows.append({"application_number": app.get("application_number"), "sponsor": app.get("sponsor_name"),
                         "brand_name": p.get("brand_name"), "active_ingredients": ings,
                         "marketing_status": p.get("marketing_status"), "dosage_form": p.get("dosage_form"),
                         "route": p.get("route"), "first_approval": first})
    df = pd.DataFrame(rows).drop_duplicates()
    save(df, out, "fda_drugsfda")


def fetch_purple(out: Path) -> None:
    urls = []
    try:
        html = get(URLS["purple_page"]).text
        urls = sorted(set(re.findall(r"https?://[^\"'\s>]+\.csv|/files/[^\"'\s>]+\.csv", html)))
        log(f"[purple] download page: {len(urls)} CSV links {urls[:5]}")
    except Exception as exc:  # noqa: BLE001
        log(f"[purple] download page failed: {exc}")
    today = date.today()
    tries = [u if u.startswith("http") else "https://purplebooksearch.fda.gov" + u for u in urls]
    for k in range(0, 14):
        y, m = today.year, today.month - k
        while m <= 0:
            y, m = y - 1, m + 12
        tries.append(URLS["purple_file"].format(y=y, m=MONTHS[m - 1]))
    for u in tries:
        try:
            r = get(u, tries=1, timeout=120)
        except Exception as exc:  # noqa: BLE001
            log(f"[purple] {u}: {exc}")
            continue
        text = r.content.decode("utf-8", errors="replace")
        lines = text.splitlines()
        head = next((i for i, l in enumerate(lines[:40]) if "Proprietary Name" in l or "BLA Number" in l), None)
        if head is None:
            log(f"[purple] {u}: no header row ({lines[:3]})")
            continue
        df = pd.read_csv(io.StringIO("\n".join(lines[head:])), dtype=str)
        log(f"[purple] {u}: {len(df)} rows")
        save(df, out, "fda_purplebook")
        return
    raise RuntimeError("no Purple Book file found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, fn in (("fda", fetch_fda), ("purple", fetch_purple), ("sec", fetch_sec)):
        t0 = time.time()
        try:
            fn(out)
            log(f"== {name} done in {time.time() - t0:.0f} s")
        except Exception:  # noqa: BLE001
            failed.append(name)
            log(f"== {name} FAILED\n{traceback.format_exc()}")
    log(f"failed: {failed}")
    (out / "fetch_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
