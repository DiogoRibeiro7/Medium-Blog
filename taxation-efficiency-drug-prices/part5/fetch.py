"""Download and reduce the public files used in Part 5 (parallel export and export restrictions).

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits the reduced files in
data/ together with data/fetch_log.txt. Each source is fetched independently; a failure is logged and
the others still run.

Sources
  Portugal
    pt_lists            Infarmed's export-control lists under the regulation on transactions of medicines
                        to other countries (Deliberação 391/2021, then 234/2025): the prior-notification
                        list (LNP, quarterly), the list of medicines whose export is temporarily suspended
                        (monthly) and, since late 2025, the list of LNP medicines whose export is prohibited
                        in the month, with Infarmed's indicators. Infarmed's page shows only the current
                        lists; earlier ones are recovered from the versions of that page kept by the
                        Internet Archive, and every document is then downloaded from Infarmed itself.
    pt_reports          Infarmed's annual reports on the availability of medicines (2021, 2024, 2025) and
                        the 2021 regulation, kept as text.
  Europe
    ema_pd              EMA's public register of parallel distribution notices (centrally authorised
                        medicines): product, holder, parallel distributor, status, dates, member states of
                        origin and of destination.
    comext              Eurostat Comext: Portugal's exports and imports of medicaments in measured doses
                        (HS 3004) by partner country, 2015-2025, value and quantity.

    python fetch.py --out data
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import traceback
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urljoin

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)",
      "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8"}

INF = "https://www.infarmed.pt"
AVAIL_PAGE = INF + "/web/infarmed/gestao-da-disponibilidade-do-medicamento"
URLS = {
    "cdx": "https://web.archive.org/cdx/search/cdx?url=infarmed.pt/web/infarmed/gestao-da-disponibilidade-do-medicamento"
           "&output=json&from=2021&filter=statuscode:200&collapse=digest",
    "snapshot": "https://web.archive.org/web/{ts}id_/https://www.infarmed.pt/web/infarmed/gestao-da-disponibilidade-do-medicamento",
    "iris_page": "https://iris.ema.europa.eu/registerpd/",
    "iris_token": "https://iris.ema.europa.eu/_layout/tokenhtml",
    "comext": "https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/ds-045409?format=JSON&lang=EN"
              "&freq=A&reporter=PT&product=3004&flow={flow}&indicators={ind}&sinceTimePeriod=2015",
}
REPORTS = {
    "pt_availability_2021": INF + "/documents/15786/4326055/Relat%C3%B3rio+anual+sobre+a+gest%C3%A3o+da+disponibilidade+de+medicamentos+de+2021/d6729605-96aa-7b0c-e702-d59e0e7e863e",
    "pt_availability_2024": INF + "/documents/15786/10901776/Relat%C3%B3rio+2024/2f02f60e-ef04-a5b1-7962-be7d5ef353a2",
    "pt_availability_2025": INF + "/documents/15786/10901776/Relat%C3%B3rio2025/a2d40b5d-60ea-4297-6bc8-fa146f1a5c70",
    "pt_regulation_2021": "https://files.diariodarepublica.pt/2s/2021/04/078000000/0020900218.pdf",
}

MONTHS = {"janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4, "maio": 5, "junho": 6, "julho": 7,
          "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12}
HREF = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
TAGS = re.compile(r"<[^>]+>")

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def get(url: str, tries: int = 4, timeout: int = 180, session=None, wait: float = 5.0) -> requests.Response:
    last = None
    for k in range(tries):
        try:
            r = (session or requests).get(url, headers=UA, timeout=timeout)
            if r.status_code in (429, 502, 503, 504):
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


def save_text(text: str, out: Path, name: str) -> None:
    path = out / f"{name}.txt.gz"
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(text.encode("utf-8"))
    log(f"[{name}] wrote {path.name}: {len(text)} characters")


def pdf_text(data: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "in.pdf"
        pdf.write_bytes(data)
        subprocess.run(["pdftotext", "-layout", str(pdf), str(pdf.with_suffix(".txt"))], check=True)
        return pdf.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")


def clean(x: str) -> str:
    return re.sub(r"\s+", " ", unescape(TAGS.sub(" ", x))).strip()


# ------------------------------------------------------------------------------------- Infarmed lists
def list_links(html: str, source: str) -> dict[str, dict]:
    """Links to export-control documents on a version of Infarmed's availability page."""
    found = {}
    for href, text in HREF.findall(html):
        label = clean(text)
        href = unescape(href).strip()
        full = urljoin(INF + "/", href)
        full = re.sub(r"^https?://web\.archive\.org/web/\d+[a-z_]*/", "", full)
        full = full.replace("http://", "https://").split("?")[0]
        if "/documents/15786/" not in full:
            continue
        key = unquote(full).lower()
        if not re.search(r"suspens|notifica|lnp|proibi", key):
            continue
        if re.search(r"aue|faltas|web.?service|infografia|chnm", key):
            continue
        found[full] = {"label": label, "source": source}
    return found


def classify(url: str, label: str) -> tuple[str, str]:
    key = unquote(url).lower().replace("+", " ")
    kind = "deliberation" if "delibera" in key else "list"
    if "proibi" in key:
        typ = "prohibited"
    elif "suspens" in key:
        typ = "suspended"
    elif "notifica" in key or "lnp" in key:
        typ = "lnp"
    else:
        typ = "other"
    return typ, kind


def effective_date(url: str, label: str, snap: str) -> str:
    t = (label + " " + unquote(url).replace("+", " ")).lower()
    m = re.search(r"de (\d{1,2}) de ([a-zç]+)\b(?! de \d{4})", label.lower())
    yr = re.search(r"(20\d{2})", unquote(url)) or re.search(r"/(20\d{2})", label)
    if "delibera" in t and m and m.group(2) in MONTHS and yr:
        return f"{yr.group(1)}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
    m = re.search(r"a partir de (\d{1,2}) de ([a-zç]+)(?: de)? (\d{4})", t)
    if m and m.group(2) in MONTHS:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(\d{1,2}) de ([a-zç]+)(?: de)? (\d{4})", t)
    if m and m.group(2) in MONTHS:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
    for mm in re.finditer(r"([a-zç]+)(?: de)? ?(20\d{2})", t):
        if mm.group(1) in MONTHS:
            return f"{mm.group(2)}-{MONTHS[mm.group(1)]:02d}-01"
        for name, num in MONTHS.items():
            if mm.group(1).endswith(name):
                return f"{mm.group(2)}-{num:02d}-01"
    return f"{snap[:4]}-{snap[4:6]}-{snap[6:8]}" if snap else ""


XLSX_COLS = [
    (r"registo", "reg"), (r"^motivo", "motive"), (r"nome", "name"), (r"dci|subst", "dci"), (r"dosagem", "strength"),
    (r"forma", "form"), (r"apresenta|tamanho", "pack"), (r"^cft", "cft"), (r"titular", "mah"),
    (r"satisfa\w* consumo", "sat_consumption"), (r"satisfa\w* prescri", "sat_prescription"),
    (r"notifica\w* de faltas", "n_shortage_reports"), (r"faltas\s*/", "shortage_rate"),
    (r"inten\w* exporta", "intent_rate"), (r"export\s*/", "export_rate"), (r"mesmo cnpem", "n_same_cnpem"),
    (r"essencial", "essential"),
]


def parse_xlsx(data: bytes) -> list[dict]:
    rows = []
    book = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str)
    for sheet, df in book.items():
        head = None
        for i in range(min(15, len(df))):
            if any(isinstance(v, str) and "registo" in v.lower() for v in df.iloc[i].tolist()):
                head = i
                break
        if head is None:
            continue
        names = {}
        for j, v in enumerate(df.iloc[head].tolist()):
            v = (v or "").strip().lower() if isinstance(v, str) else ""
            for pat, key in XLSX_COLS:
                if key not in names.values() and re.search(pat, v):
                    names[j] = key
                    break
        for i in range(head + 1, len(df)):
            rec = {"sheet": sheet}
            for j, key in names.items():
                v = df.iat[i, j]
                rec[key] = None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v).strip()
            reg = re.sub(r"\D", "", rec.get("reg") or "")
            if len(reg) == 7:
                rec["reg"] = reg
                rows.append(rec)
    return rows


def parse_pdf(text: str) -> list[dict]:
    """Rows of a list printed as a table: a seven-digit registration number at the start of a line,
    then the cells separated by runs of spaces (name, then INN). For the 2021 regulation, rows after the
    heading of annex II belong to the suspended list."""
    rows = []
    annex2 = text.find("ANEXO II")
    pos = 0
    for line in text.splitlines(keepends=True):
        m = re.match(r"^\s{0,20}(\d{7})\s{2,}(\S.*)$", line.rstrip("\n"))
        if m:
            cells = []
            for c in re.split(r"\s{2,}", m.group(2).strip()):
                # The 2021 regulation fills the name cell with leader dots before the INN.
                cells.extend(x.strip(" .") for x in re.split(r"(?:\s?\.){3,}", c))
            cells = [c for c in cells if c]
            rows.append({"sheet": "pdf", "reg": m.group(1), "name": cells[0] if cells else None,
                         "dci": cells[1] if len(cells) > 1 else None,
                         "annex": ("II" if 0 <= annex2 <= pos else "I")})
        pos += len(line)
    return rows


def fetch_lists(out: Path) -> None:
    docs: dict[str, dict] = {}
    # Versions of the page kept by the Internet Archive (one per distinct content).
    try:
        cdx = get(URLS["cdx"], timeout=240).json()[1:]
    except Exception as exc:  # noqa: BLE001
        log(f"[pt_lists] CDX failed: {exc}")
        cdx = []
    log(f"[pt_lists] archived versions of the page: {len(cdx)}")
    failed = []
    for row in cdx:
        ts = row[1]
        try:
            html = get(URLS["snapshot"].format(ts=ts), tries=5, timeout=120, wait=10).text
        except Exception as exc:  # noqa: BLE001
            failed.append(ts)
            log(f"  {ts} failed: {exc}")
            continue
        for url, meta in list_links(html, ts).items():
            docs.setdefault(url, meta)
        time.sleep(2.0)
    log(f"[pt_lists] versions not retrieved: {len(failed)} {failed}")
    # The live page.
    live = get(AVAIL_PAGE).text
    for url, meta in list_links(live, "live").items():
        docs.setdefault(url, meta)
    log(f"[pt_lists] documents found: {len(docs)}")
    # The 2021 regulation carries the first lists of that regime in its annexes.
    docs[REPORTS["pt_regulation_2021"]] = {"label": "Deliberação n.º 391/2021 (anexos I e II)", "source": "fixed"}

    meta_rows, rows = [], []
    for url, meta in sorted(docs.items()):
        typ, kind = classify(url, meta["label"])
        if url == REPORTS["pt_regulation_2021"]:
            typ, kind = "regulation", "deliberation"
        elif kind == "deliberation":
            # The deliberations approve the lists but do not reproduce them; keep them as metadata only.
            meta_rows.append({"url": url, "label": meta["label"], "first_seen": meta["source"], "type": typ, "kind": kind,
                              "effective": effective_date(url, meta["label"], meta["source"] if meta["source"][:2] == "20" else ""),
                              "format": "", "n_rows": 0, "error": "not downloaded"})
            continue
        eff = effective_date(url, meta["label"], meta["source"] if meta["source"][:2] == "20" else "")
        rec = {"url": url, "label": meta["label"], "first_seen": meta["source"], "type": typ, "kind": kind,
               "effective": eff, "format": "", "n_rows": 0, "error": ""}
        try:
            r = get(url, tries=3, timeout=180)
            data = r.content
            if data[:4] == b"%PDF":
                rec["format"] = "pdf"
                text = pdf_text(data)
                parsed = parse_pdf(text)
                if typ != "regulation":
                    for p in parsed:
                        p["annex"] = None
            elif data[:2] == b"PK":
                rec["format"] = "xlsx"
                parsed = parse_xlsx(data)
            else:
                rec["format"] = r.headers.get("content-type", "")[:40]
                parsed = []
            for p in parsed:
                p.update({"url": url, "type": typ, "kind": kind, "effective": eff})
            rows.extend(parsed)
            rec["n_rows"] = len(parsed)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)[:200]
        meta_rows.append(rec)
        log(f"  {typ:10s} {kind:12s} {eff:10s} {rec['format']:5s} rows {rec['n_rows']:4d}  {meta['label'][:60]}")
        time.sleep(0.5)
    save(pd.DataFrame(meta_rows), out, "pt_list_documents")
    cols = ["url", "type", "kind", "effective", "sheet", "annex", "reg", "motive", "name", "dci", "strength", "form",
            "pack", "cft", "mah", "sat_consumption", "sat_prescription", "n_shortage_reports", "shortage_rate",
            "intent_rate", "export_rate", "n_same_cnpem", "essential"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    save(df[cols], out, "pt_list_rows")


def fetch_reports(out: Path) -> None:
    for name, url in REPORTS.items():
        r = get(url)
        if not r.content.startswith(b"%PDF"):
            raise RuntimeError(f"{name}: not a PDF")
        save_text(pdf_text(r.content), out, name)


# ----------------------------------------------------------------------------------- EMA register
IRIS_FIELDS = ["ema_name", "ema_relatedproductname", "ema_relatedproductstrength", "ema_relatedproductpacksize",
               "ema_relatedmarketingauthorisationholder", "ema_relatedsponsorname", "ema_entitlementstatus",
               "ema_relatedproducteunumber", "ema_dateofnotification", "ema_dateofecdecision", "ema_repackagers",
               "ema_memberstatesoforigin", "ema_memberstatesofdestination"]


def fetch_ema_pd(out: Path) -> None:
    s = requests.Session()
    html = get(URLS["iris_page"], session=s).text
    lay = json.loads(base64.b64decode(unescape(re.search(r"data-view-layouts=\"([^\"]+)\"", html).group(1))))
    conf = lay[0]["Base64SecureConfiguration"]
    url = urljoin(URLS["iris_page"], unescape(re.search(r"data-get-url=\"([^\"]+)\"", html).group(1)))
    token = re.search(r"value=\"([^\"]+)\"", get(URLS["iris_token"], session=s).text).group(1)
    hdr = {**UA, "Content-Type": "application/json; charset=utf-8",
           "Accept": "application/json, text/javascript, */*; q=0.01", "__RequestVerificationToken": token,
           "X-Requested-With": "XMLHttpRequest", "Referer": URLS["iris_page"], "Origin": "https://iris.ema.europa.eu"}
    records: list[dict] = []
    seen: set[str] = set()
    cookie = ""
    page = 1
    size = 500
    while page <= 400:
        body = {"base64SecureConfiguration": conf, "sortExpression": "ema_name ASC", "search": None, "page": page,
                "pageSize": size, "pagingCookie": cookie, "filter": None, "metaFilter": None, "nlSearchFilter": None,
                "timezoneOffset": 0, "customParameters": [], "entityName": None, "entityId": None}
        for k in range(4):
            try:
                rr = s.post(url, data=json.dumps(body), headers=hdr, timeout=180)
                js = rr.json()
                break
            except Exception as exc:  # noqa: BLE001
                if k == 3:
                    raise RuntimeError(f"page {page}: {exc}") from exc
                time.sleep(10 * (k + 1))
        recs = js.get("Records") or []
        new = 0
        for rec in recs:
            d = {a["Name"]: a.get("DisplayValue") for a in rec["Attributes"]}
            key = d.get("ema_name") or rec.get("Id")
            if key in seen:
                continue
            seen.add(key)
            records.append({f: d.get(f) for f in IRIS_FIELDS})
            new += 1
        if page == 1 or page % 10 == 0:
            log(f"[ema_pd] page {page}: {len(recs)} records ({new} new), total {len(records)}, "
                f"ItemCount {js.get('ItemCount')}, MoreRecords {js.get('MoreRecords')}")
        if not recs or not js.get("MoreRecords") or new == 0:
            break
        cookie = js.get("NextPagePagingCookie") or ""
        page += 1
        time.sleep(0.5)
    df = pd.DataFrame(records, columns=IRIS_FIELDS)
    df.columns = ["notice", "product", "strength", "pack", "mah", "distributor", "status", "eu_number",
                  "date_notice", "date_ec_decision", "repackagers", "origins", "destinations"]
    df = df.sort_values("notice").reset_index(drop=True)
    save(df, out, "ema_pd_notices")


# ------------------------------------------------------------------------------------------- Comext
def fetch_comext(out: Path) -> None:
    frames = []
    for flow, flow_name in (("1", "import"), ("2", "export")):
        for ind in ("VALUE_IN_EUROS", "QUANTITY_IN_100KG"):
            js = get(URLS["comext"].format(flow=flow, ind=ind)).json()
            ids, sizes = js["id"], js["size"]
            cats = {k: js["dimension"][k]["category"] for k in ids}
            codes = {k: sorted(c["index"], key=c["index"].get) for k, c in cats.items()}
            strides = {}
            acc = 1
            for k, n in reversed(list(zip(ids, sizes))):
                strides[k] = acc
                acc *= n
            for flat, v in js["value"].items():
                flat = int(flat)
                p = codes["partner"][(flat // strides["partner"]) % len(codes["partner"])]
                t = codes["time"][(flat // strides["time"]) % len(codes["time"])]
                frames.append({"flow": flow_name, "indicator": ind, "partner": p,
                               "partner_name": cats["partner"]["label"].get(p), "year": int(t), "value": v})
    df = pd.DataFrame(frames).sort_values(["flow", "indicator", "partner", "year"]).reset_index(drop=True)
    save(df, out, "comext_pt_3004")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {"lists": fetch_lists, "reports": fetch_reports, "comext": fetch_comext, "ema_pd": fetch_ema_pd}
    failed = []
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        log(f"===== {name}")
        try:
            fn(out)
        except Exception:  # noqa: BLE001
            failed.append(name)
            log(f"[{name}] FAILED\n{traceback.format_exc()}")
    log(f"failed: {failed}")
    (out / "fetch_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
