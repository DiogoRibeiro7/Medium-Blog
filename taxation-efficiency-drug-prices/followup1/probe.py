"""Survey the sources for follow-up 1 (do Portugal's confidential returns from industry grow, and how do
they compare with the rebates of the countries it references?). Run by the data workflow; writes
data/probe/probe_report.txt and the text of the documents found. Deleted once fetch.py exists.

    python probe.py --out data
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)"}
REPORT: list[str] = []
OUT = Path("data")

KEYWORDS = re.compile(r"nota[s]? de cr[eé]dito|devolu|contribui[cç][aã]o|acordo|l[ií]quid|remise|payback|pay-back|"
                      r"note di credito|accordi negoziali|milh[oõ]es|M€|Md€|milliards", re.I)

DOCS = {
    # Belgium: INAMI/RIZIV MORSE reports (reimbursement expenditure and article 81/111 compensations)
    "be_morse_2024_fr": "https://www.inami.fgov.be/SiteCollectionDocuments/MORSE_rapport_2024_FR.pdf",
    "be_morse_2022_2023_fr": "https://www.riziv.fgov.be/SiteCollectionDocuments/morse_rapport_donnees_2022_2023.pdf",
    # Portugal: magazine supplement quoting the APIFARMA study on returns 2018-2022
    "pt_executivedigest_2024": "https://executivedigest.sapo.pt/wp-content/uploads/2024/12/O-futuro-da-saude.pdf",
    "apifarma_newsletter_2024_10": "https://apifarma.pt/wp-content/uploads/2024/11/apifarma_newsletter_outubro_2024.pdf",
}


def log(msg: str) -> None:
    print(msg, flush=True)
    REPORT.append(msg)
    (OUT / "probe_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")


def get(url: str, timeout: int = 120) -> requests.Response:
    last = None
    for k in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(4 * (k + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def pdf_text(data: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "in.pdf"
        pdf.write_bytes(data)
        subprocess.run(["pdftotext", "-layout", str(pdf), str(pdf.with_suffix(".txt"))], check=True)
        return pdf.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")


def save_text(text: str, name: str) -> None:
    with open(OUT / f"{name}.txt.gz", "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(text.encode("utf-8"))


def scan(name: str, text: str, limit: int = 80) -> None:
    n = 0
    for line in text.splitlines():
        s = re.sub(r"\s+", " ", line).strip()
        if s and KEYWORDS.search(s) and re.search(r"\d", s):
            log(f"      {s[:240]}")
            n += 1
            if n >= limit:
                log("      ...")
                break


def probe_documents() -> None:
    for name, url in DOCS.items():
        try:
            text = pdf_text(get(url, timeout=180).content)
            save_text(text, name)
            log(f"[doc] {name}: {len(text)} characters")
            scan(name, text)
        except Exception as exc:  # noqa: BLE001
            log(f"[doc] {name} failed: {exc}")


def probe_apifarma_media() -> None:
    seen = set()
    for term in ("investimento", "Investimento", "analise", "estudo", "modelos", "financiamento", "devolucoes"):
        try:
            js = get(f"https://apifarma.pt/wp-json/wp/v2/media?search={term}&per_page=100").json()
        except Exception as exc:  # noqa: BLE001
            log(f"[apifarma media] {term} failed: {exc}")
            continue
        for m in js:
            url = m.get("source_url", "")
            if url and url not in seen:
                seen.add(url)
                log(f"[apifarma media] {term}: {m.get('date', '')[:10]} {url}")
    for url in sorted(u for u in seen if re.search(r"investiment|analise|an%c3%a1lise|estudo|modelo|financiament|devolu", u, re.I)
                      and u.lower().endswith(".pdf")):
        name = "apifarma_" + re.sub(r"[^a-z0-9]+", "_", url.rsplit("/", 1)[-1].lower())[:60]
        try:
            text = pdf_text(get(url, timeout=180).content)
            save_text(text, name)
            log(f"[doc] {name}: {len(text)} characters")
            scan(name, text, limit=25)
        except Exception as exc:  # noqa: BLE001
            log(f"[doc] {name} failed: {exc}")


def probe_transparencia() -> None:
    base = "https://transparencia.sns.gov.pt/api/explore/v2.1/catalog/datasets"
    rows, offset = [], 0
    while True:
        js = get(f"{base}?limit=100&offset={offset}").json()
        for r in js["results"]:
            m = r["metas"]["default"]
            rows.append((r["dataset_id"], m.get("title", ""), m.get("records_count")))
        offset += 100
        if offset >= js.get("total_count", 0):
            break
    log(f"[transparencia] {len(rows)} datasets")
    for ds, title, n in rows:
        log(f"  {ds} | {title} | {n}")
    pat = re.compile(r"custo|gasto|consum|financ|demonstra|conta|nota|receit|balan|execu", re.I)
    for ds, title, n in rows:
        if pat.search(ds) or pat.search(title):
            try:
                rec = get(f"{base}/{ds}/records?limit=2").json()
                for x in rec.get("results", [])[:2]:
                    log(f"      {ds}: {json.dumps(x, ensure_ascii=False)[:500]}")
            except Exception as exc:  # noqa: BLE001
                log(f"      {ds} records failed: {exc}")


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    OUT = Path(ap.parse_args().out) / "probe"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("documents", probe_documents), ("apifarma_media", probe_apifarma_media)):
        t0 = time.time()
        try:
            fn()
            log(f"== {name} done in {time.time() - t0:.0f} s")
        except Exception:  # noqa: BLE001
            log(f"== {name} FAILED\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
