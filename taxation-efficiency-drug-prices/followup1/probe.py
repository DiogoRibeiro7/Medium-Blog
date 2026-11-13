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
    # Conselho das Finanças Públicas, reports on the SNS
    "cfp_sns_2013_2019": "https://www.cfp.pt/uploads/publicacoes_ficheiros/cfp-rel-10-2020%5B1%5D.pdf",
    "cfp_sns_2020": "https://www.cfp.pt/uploads/publicacoes_ficheiros/cfp-rel-06-2021.pdf",
    "cfp_sns_2021": "https://www.cfp.pt/uploads/publicacoes_ficheiros/cfp-rel-07-2022.pdf",
    "cfp_sns_2022": "https://www.cfp.pt/uploads/publicacoes_ficheiros/cfp-rel-07-2023.pdf",
    "cfp_sns_2023": "https://www.cfp.pt/uploads/publicacoes_ficheiros/cfp-rel-07-2024.pdf",
    "cfp_sns_2024": "https://www.cfp.pt/uploads/publicacoes_ficheiros/cfp-rel-05-2025.pdf",
    "cfp_sns_2025": "https://www.cfp.pt/uploads/publicacoes_ficheiros/cfp_rel-05-2026.pdf",
    # Tribunal de Contas, follow-up audit of the SNS consolidated accounts
    "tc_rel009_2022": "https://www.tcontas.pt/pt-pt/ProdutosTC/Relatorios/RelatoriosAuditoria/Documents/2022/rel009-2022-2s.pdf",
    # APIFARMA
    "apifarma_newsletter_2024_09": "https://apifarma.pt/wp-content/uploads/2024/10/apifarma_newsletter_setembro_2024.pdf",
    "apifarma_newsletter_2025_09": "https://apifarma.pt/wp-content/uploads/2025/10/apifarma_newsletter_setembro_2025.pdf",
    "apifarma_agreement_2016": "https://apifarma.pt/wp-content/uploads/2021/04/Acordo-2016.pdf",
    # France: DREES, reimbursable medicines and rebates 2016-2024
    "drees_dd139": "https://drees.solidarites-sante.gouv.fr/sites/default/files/2026-06/DD139_M%C3%A9dicaments.pdf",
    "drees_es2026_fiche23": "https://drees.solidarites-sante.gouv.fr/sites/default/files/2026-07/ES%202026%20-%20Fiche%2023%20-%20Les%20m%C3%A9dicaments%20et%20dispositifs%20m%C3%A9dicaux%20on%C3%A9reux.pdf",
    # Italy: AIFA, spending monitoring for 2024 (final)
    "aifa_monitoraggio_2024": "https://www.aifa.gov.it/documents/20142/2505785/Monitoraggio_Spesa_gennaio-dicembre-2024_consultivo.pdf",
    "aifa_sintesi_2024": "https://www.aifa.gov.it/documents/20142/2505785/Sintesi_monitoraggio_Spesa_gen-dic-2024.pdf",
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
    for term in ("acordo", "Acordo", "newsletter", "devolu", "despesa"):
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
    for url in sorted(u for u in seen if re.search(r"acordo", u, re.I) and u.lower().endswith(".pdf")):
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
    for name, fn in (("documents", probe_documents), ("apifarma_media", probe_apifarma_media),
                     ("transparencia", probe_transparencia)):
        t0 = time.time()
        try:
            fn()
            log(f"== {name} done in {time.time() - t0:.0f} s")
        except Exception:  # noqa: BLE001
            log(f"== {name} FAILED\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
