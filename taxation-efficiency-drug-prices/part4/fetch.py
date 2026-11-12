"""Download and reduce the public files used in Part 4 (time to access).

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits the reduced files in
data/ together with data/fetch_log.txt. Each source is fetched independently; a failure is logged and
the others still run.

Sources
  Portugal
    pt_siats            Infarmed's public process tracker (SIATS, "Pesquisa de Processos"): every
                        reimbursement or hospital prior-evaluation process for a new active substance
                        (and, separately, for a new indication), with the dates of marketing
                        authorisation, submission, start and end of the assessment and access in the SNS,
                        and the days attributed to Infarmed, to the company and to negotiation. The
                        search returns five processes a page; the script pages through all of them.
  Europe
    ema_medicines       EMA, list of centrally authorised medicines (authorisation date, orphan status,
                        ATC code, holder, conditional approval, PRIME, advanced therapy).
    wait_2025, 2024     EFPIA / IQVIA, Patients W.A.I.T. Indicator 2025 and 2024: number of medicines
                        available and time from authorisation to availability by country, read from
                        the charts of the published PDF by the position of each label.
    euripid_pli_2017    EURIPID / Eurostat study (2018): price level indices of pharmaceutical products
                        by country, all medicines and by ATC group, 2015-2017.
    eurostat_*          Eurostat: population on 1 January, GDP per head and actual individual
                        consumption per head in purchasing power standards (EU27 = 100), and the price
                        level index of actual individual consumption.

    python fetch.py --out data
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile  # noqa: F401  (kept for symmetry with the other parts)
from html import unescape
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)",
      "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8"}

URLS = {
    "siats": "https://extranet.infarmed.pt/siats/Publico/Processos/InfoDetalhesProcesso.aspx",
    "ema_medicines": "https://www.ema.europa.eu/en/documents/report/medicines-output-medicines-report_en.xlsx",
    "wait_2025": "https://via.tt.se/files/3237315/4386931/339076/sv",
    "wait_2024": "https://www.iqvia.com/-/media/iqvia/pdfs/library/publications/efpia-patients-wait-indicator-2024.pdf",
    "euripid_2017": "https://jasmin.goeg.at/755/1/report_PPPsPharmaceuticalProducts.pdf",
    "eurostat": "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{code}?format=JSON&lang=EN&{query}",
}
SIATS_TYPES = {"66": "new active substance", "67": "new indication"}

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def get(url: str, tries: int = 4, timeout: int = 300, session=None) -> requests.Response:
    last = None
    for k in range(tries):
        try:
            r = (session or requests).get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def save(df: pd.DataFrame, out: Path, name: str) -> None:
    path = out / f"{name}.csv.gz"
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode("utf-8"))
    log(f"[{name}] wrote {path.name}: {len(df)} rows, columns {list(df.columns)}")


TAGS = re.compile(r"<[^>]+>")


def clean(x: str) -> str:
    return re.sub(r"\s+", " ", unescape(TAGS.sub(" ", x))).strip()


# ------------------------------------------------------------------------------------------- SIATS
def hidden_fields(html: str) -> dict:
    out = {}
    for m in re.finditer(r'<input[^>]*type="hidden"[^>]*>', html):
        t = m.group(0)
        n = re.search(r'name="([^"]+)"', t)
        v = re.search(r'value="([^"]*)"', t)
        if n:
            out[n.group(1)] = unescape(v.group(1)) if v else ""
    return out


def siats_records(html: str) -> list[dict]:
    """One dictionary per process card: the title and every label/value pair under it."""
    parts = re.split(r'<h3 class="ribbon-title">', html)[1:]
    rows = []
    for p in parts:
        name = clean(p.split("</h3>", 1)[0])
        rec = {"name": name}
        for lab, val in re.findall(r'<div class="label">(.*?)</div>\s*<div class="value">(.*?)</div>', p, flags=re.S):
            rec[clean(lab).rstrip(":")] = clean(val)
        rows.append(rec)
    return rows


def pager(html: str) -> list[tuple[str, str]]:
    return re.findall(r"__doPostBack\(&#39;(ctl00\$ContentPlaceHolder1\$pagerCounter\$ctl\d+\$lnkPagerChange)&#39;"
                      r",&#39;&#39;\)\"?>([^<]*)</a>", html)


def fetch_siats(out: Path) -> None:
    frames = []
    for code, label in SIATS_TYPES.items():
        s = requests.Session()
        r = get(URLS["siats"], session=s)
        base = {"ctl00$ContentPlaceHolder1$ddlTipoMedicamento": code,
                "ctl00$ContentPlaceHolder1$ddlTipoProcesso": "-1",
                "ctl00$ContentPlaceHolder1$txtNomeMedicamentoProcesso": "",
                "ctl00$ContentPlaceHolder1$txtDciProcesso": "",
                "ctl00$ContentPlaceHolder1$txtAreaTerapProcesso": ""}
        data = {**hidden_fields(r.text), **base, "ctl00$ContentPlaceHolder1$btnPesquisarProcessos": "Pesquisar"}
        p = s.post(URLS["siats"], data=data, headers=UA, timeout=180)
        page, rows, seen = 1, [], set()
        while True:
            recs = siats_records(p.text)
            key = tuple((x.get("name"), x.get("Submissão do Processo pelo Titular de AIM")) for x in recs)
            if key in seen:  # a "..." that led back to an earlier block
                break
            seen.add(key)
            for x in recs:
                x["page"] = page
            rows.extend(recs)
            links = pager(p.text)
            nxt = next((t for t, txt in links if txt.strip() == str(page + 1)), None)
            if nxt is None:
                dots = [t for t, txt in links if txt.strip() == "..."]
                # The last "..." opens the next block of page numbers.
                if dots and (len(dots) > 1 or page % 10 == 0):
                    nxt = dots[-1]
            if nxt is None or page >= 600:
                break
            data = {**hidden_fields(p.text), **base, "__EVENTTARGET": nxt, "__EVENTARGUMENT": ""}
            time.sleep(0.8)
            p = s.post(URLS["siats"], data=data, headers=UA, timeout=180)
            page += 1
            if page % 20 == 0:
                log(f"[siats {label}] page {page}, {len(rows)} records")
        df = pd.DataFrame(rows)
        df["search"] = label
        log(f"[siats {label}] {len(df)} records over {page} pages; columns {list(df.columns)}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    stamp = re.search(r"Última atualização:\s*([\d/]+ às [\d:]+)", clean(p.text))
    log(f"[siats] tracker last updated: {stamp.group(1) if stamp else 'unknown'}")
    save(df, out, "pt_siats_processes")


# --------------------------------------------------------------------------------------------- EMA
def fetch_ema(out: Path) -> None:
    raw = get(URLS["ema_medicines"]).content
    df = pd.read_excel(io.BytesIO(raw), header=8, dtype=str)
    df = df[df["Category"] == "Human"]
    keep = ["Name of medicine", "EMA product number", "Medicine status",
            "International non-proprietary name (INN) / common name", "Active substance", "Therapeutic area (MeSH)",
            "ATC code (human)", "Accelerated assessment", "Advanced therapy", "Biosimilar", "Conditional approval",
            "Exceptional circumstances", "Generic", "Orphan medicine", "PRIME: priority medicine",
            "Marketing authorisation developer / applicant / holder", "European Commission decision date",
            "Marketing authorisation date", "Withdrawal / expiry / revocation / lapse of marketing authorisation date"]
    df = df[[c for c in keep if c in df.columns]]
    save(df, out, "ema_medicines")


# ------------------------------------------------------------------------------------------- W.A.I.T.
def pdf_words(data: bytes) -> list[list[tuple]]:
    """Words of each page of a PDF with their boxes, from poppler's pdftotext -bbox-layout."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "in.pdf"
        pdf.write_bytes(data)
        subprocess.run(["pdftotext", "-bbox-layout", str(pdf), str(pdf.with_suffix(".html"))], check=True)
        html = pdf.with_suffix(".html").read_text(encoding="utf-8", errors="replace")
    pages = re.findall(r"<page[^>]*>.*?</page>", html, flags=re.S)
    word = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>')
    return [[(float(a), float(b), float(c), float(d), unescape(t)) for a, b, c, d, t in word.findall(pg)]
            for pg in pages]


COUNTRIES = ["Germany", "Switzerland", "Macedonia", "Serbia", "England", "Scotland", "Austria", "Denmark", "Turkey",
             "Bosnia", "Italy", "Finland", "Sweden", "Netherlands", "Luxemburg", "Luxembourg", "Hungary", "Cyprus",
             "Iceland", "Spain", "Norway", "Belgium", "Slovenia", "Latvia", "France", "Greece", "Bulgaria", "Czechia",
             "Ireland", "Estonia", "Poland", "Croatia", "Slovakia", "Lithuania", "Portugal", "Romania", "Malta",
             "Albania", "Montenegro", "Kosovo"]


def chart_values(words: list[tuple], ylo: float, yhi: float) -> dict:
    """Country label (rotated, under the bar) -> the number printed above the bar, matched by x position;
    plus the 'Available medicines' and 'Dates submitted' rows printed under the labels."""
    labels = [w for w in words if w[4].lstrip("*") in COUNTRIES]
    nums = [w for w in words if re.fullmatch(r"\d{1,2},\d{3}|\d{2,3}", w[4]) and ylo < w[1] < yhi]
    rows = {}
    for key in ("Available", "Dates"):
        anchor = next((w for w in words if w[4] == key), None)
        if anchor:
            rows[key] = [w for w in words if re.fullmatch(r"\d+", w[4]) and abs(w[1] - anchor[1]) < 3
                         and w[0] > anchor[2] + 10]
    out = {}
    for lab in labels:
        cx = (lab[0] + lab[2]) / 2
        name = lab[4].lstrip("*")
        rec = {}
        if nums:
            n = min(nums, key=lambda w: abs((w[0] + w[2]) / 2 - cx))
            if abs((n[0] + n[2]) / 2 - cx) < 8:
                rec["days"] = float(n[4].replace(",", ""))
        for key, ws in rows.items():
            if ws:
                n = min(ws, key=lambda w: abs((w[0] + w[2]) / 2 - cx))
                if abs((n[0] + n[2]) / 2 - cx) < 8:
                    rec[key] = int(n[4])
        out[name] = rec
    return out


def fetch_wait(out: Path) -> None:
    for name, cohort in (("wait_2025", "2021-2024"), ("wait_2024", "2020-2023")):
        pages = pdf_words(get(URLS[name]).content)
        frames = {}
        for title, key in ((f"Time to availability ({cohort})", "mean_days"),
                           (f"Median time to availability ({cohort})", "median_days")):
            for pg in pages:
                text = " ".join(w[4] for w in pg[:60])
                if text.startswith(title) or (" " + title) in text[:200]:
                    if key == "mean_days" and "Median" in text[:40]:
                        continue
                    vals = chart_values(pg, 150, 420)
                    frames[key] = vals
                    break
        rows = []
        for c in sorted(set().union(*[set(v) for v in frames.values()])):
            mean = frames.get("mean_days", {}).get(c, {})
            med = frames.get("median_days", {}).get(c, {})
            rows.append({"country": c.replace("Luxemburg", "Luxembourg"), "mean_days": mean.get("days"),
                         "median_days": med.get("days"), "available": mean.get("Available", med.get("Available")),
                         "dates_submitted": mean.get("Dates")})
        df = pd.DataFrame(rows)
        log(f"[{name}] pages {len(pages)}; charts found {list(frames)}")
        log(df.to_string())
        save(df, out, name)


# -------------------------------------------------------------------------------------------- EURIPID
def fetch_euripid(out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "in.pdf"
        pdf.write_bytes(get(URLS["euripid_2017"]).content)
        subprocess.run(["pdftotext", "-layout", str(pdf), str(pdf.with_suffix(".txt"))], check=True)
        text = pdf.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")
    # Table 7 (all pharmaceutical products, 2015-2017) and Table 8 (2017 by ATC main group).
    t7 = text.split("Table 7: PLIs for pharmaceutical products")[-1].split("Data source")[0]
    rows = []
    for line in t7.splitlines():
        m = re.match(r"^\s+([A-Z]{2})\s+([\d.\s]+)$", line)
        if m:
            vals = [float(v) for v in m.group(2).split()]
            rows.append({"geo": m.group(1), "pli_last": vals[-1], "n_years": len(vals)})
    t8 = re.split(r"Data source|Figure 6|Table 9", text.split("Table 8: PLIs by ATC subheadings, 2017")[-1])[0]
    head = None
    atc = {}
    for line in t8.splitlines():
        if re.match(r"^\s+A\s+B\s+C\s", line):
            head = line.split()
            continue
        m = re.match(r"^([A-Z]{2})\s+([\d.\s]+)$", line.strip())
        if m and head:
            vals = [float(v) for v in m.group(2).split()]
            # Missing cells collapse the columns; keep the group only when the row is complete.
            if len(vals) == len(head):
                atc[m.group(1)] = dict(zip(head, vals))
    df = pd.DataFrame(rows)
    df["pli_L"] = df["geo"].map(lambda g: atc.get(g, {}).get("L"))
    log(df.to_string())
    save(df, out, "euripid_pli_2017")


# -------------------------------------------------------------------------------------------- Eurostat
def jsonstat(js: dict) -> pd.DataFrame:
    """JSON-stat 2.0 (Eurostat) to a long table with one column per dimension of size > 1."""
    ids, sizes = js["id"], js["size"]
    cats = [js["dimension"][k]["category"] for k in ids]
    codes = [sorted(c["index"], key=c["index"].get) for c in cats]
    strides = [1] * len(ids)
    for i in range(len(ids) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    rows = []
    for flat, v in js["value"].items():
        flat = int(flat)
        rec = {}
        for k, cs, st, n in zip(ids, codes, strides, sizes):
            code = cs[(flat // st) % n]
            if n > 1 or k in ("geo", "time"):
                rec[k] = code
        rec["value"] = v
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["name"] = df["geo"].map(js["dimension"]["geo"]["category"]["label"])
    return df.sort_values(list(df.columns.drop(["value", "name"]))).reset_index(drop=True)


def fetch_eurostat(out: Path) -> None:
    # Population on 1 January; GDP per head in PPS (EU27 = 100); actual individual consumption (AIC) per
    # head in PPS and the AIC price level index (both EU27 = 100). AIC per head is less distorted than
    # GDP per head by multinational accounting in Ireland and Luxembourg.
    for name, code, query in (
            ("eurostat_population", "tps00001", "sinceTimePeriod=2023"),
            ("eurostat_gdp_pps", "tec00114", "sinceTimePeriod=2022"),
            ("eurostat_aic_pps", "prc_ppp_ind",
             "sinceTimePeriod=2023&ppp_cat=A01&na_item=VI_PPS_EU27_2020_HAB&na_item=PLI_EU27_2020")):
        df = jsonstat(get(URLS["eurostat"].format(code=code, query=query)).json())
        log(f"[{name}] {len(df)} rows; {df.drop(columns=['value', 'name']).nunique().to_dict()}")
        save(df, out, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {"eurostat": fetch_eurostat, "euripid": fetch_euripid, "wait": fetch_wait, "ema": fetch_ema,
            "siats": fetch_siats}
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
