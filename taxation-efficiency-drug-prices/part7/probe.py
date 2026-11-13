"""Survey the sources for Part 7 before writing fetch.py. Run by the data workflow; writes
data/probe_report.txt and a few small files. Deleted once fetch.py exists.

    python probe.py --out data
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)"}
REPORT: list[str] = []
OUT = Path("data")


def log(msg: str) -> None:
    print(msg, flush=True)
    REPORT.append(msg)
    (OUT / "probe_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")


def get(url: str, headers=None, timeout: int = 120) -> requests.Response:
    last = None
    for k in range(3):
        try:
            r = requests.get(url, headers=headers or UA, timeout=timeout)
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


def probe_transparencia() -> None:
    base = "https://transparencia.sns.gov.pt/api/explore/v2.1/catalog/datasets"
    rows, offset = [], 0
    while True:
        js = get(f"{base}?limit=100&offset={offset}").json()
        for r in js["results"]:
            m = r["metas"]["default"]
            rows.append((r["dataset_id"], m.get("title", ""), m.get("records_count"), m.get("modified", "")))
        offset += 100
        if offset >= js.get("total_count", 0):
            break
    log(f"[transparencia] {len(rows)} datasets")
    pat = re.compile(r"medic|farm|hospit|despesa|encargo|compra|aquisi|contrat", re.I)
    for ds, title, n, mod in rows:
        if pat.search(ds) or pat.search(title):
            log(f"  {ds} | {title} | records {n} | modified {mod[:10]}")
            try:
                rec = get(f"{base}/{ds}/records?limit=3").json()
                for x in rec.get("results", [])[:2]:
                    log(f"      {json.dumps(x, ensure_ascii=False)[:600]}")
            except Exception as exc:  # noqa: BLE001
                log(f"      records failed: {exc}")


def jsonstat_dims(js: dict) -> str:
    out = []
    for d in js["id"]:
        cats = js["dimension"][d]["category"]
        labels = cats.get("label", {})
        keys = list(cats["index"].keys())
        out.append(f"{d} ({len(keys)}): " + ", ".join(f"{k}={labels.get(k, '')}" for k in keys[:40]))
    return "\n      ".join(out)


def probe_eurostat() -> None:
    base = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{ds}?format=JSON&lang=EN&sinceTimePeriod=2018{flt}"
    for ds, flt in (("hlth_sha11_hchf", "&geo=PT&geo=FR"),
                    ("hlth_sha11_hc", "&geo=PT"),
                    ("hlth_sha11_hf", "&geo=PT")):
        try:
            js = get(base.format(ds=ds, flt=flt), timeout=300).json()
            log(f"[eurostat] {ds}: size {js['size']}\n      {jsonstat_dims(js)}")
        except Exception as exc:  # noqa: BLE001
            log(f"[eurostat] {ds} failed: {exc}")


def probe_oecd() -> None:
    js = get("https://sdmx.oecd.org/public/rest/dataflow/all",
             headers={**UA, "Accept": "application/vnd.sdmx.structure+json;version=1.0"}).json()
    flows = js["data"]["dataflows"]
    hits = [f for f in flows if re.search(r"SHA|HEALTH_EXP|HEALTH_PHMC|PHARMA", f["id"])
            or re.search(r"health expenditure|pharmaceutical", json.dumps(f.get("names", f.get("name", ""))), re.I)]
    log(f"[oecd] {len(flows)} flows; {len(hits)} health-expenditure-like")
    for f in hits[:30]:
        log(f"  {f['agencyID']} {f['id']} {f['version']} | {f.get('name') or f.get('names', {}).get('en')}")
    for f in hits:
        if re.fullmatch(r"DSD_SHA@DF_SHA", f["id"]):
            url = f"https://sdmx.oecd.org/public/rest/dataflow/{f['agencyID']}/{f['id']}/{f['version']}?references=all"
            st = get(url, headers={**UA, "Accept": "application/vnd.sdmx.structure+json;version=1.0"}, timeout=300).json()["data"]
            dims = st["dataStructures"][0]["dataStructureComponents"]["dimensionList"]["dimensions"]
            lists = {cl["id"]: cl.get("codes", []) for cl in st.get("codelists", [])}
            for d in sorted(dims, key=lambda d: d.get("position", 0)):
                enum = d.get("localRepresentation", {}).get("enumeration", "")
                cl_id = re.sub(r"\(.*\)$", "", enum).split(".")[-1].split(":")[-1]
                codes = lists.get(cl_id, [])
                sample = ", ".join(f"{c['id']}={c.get('name') or c.get('names', {}).get('en', '')}" for c in codes[:60])
                log(f"    {d['id']} [{cl_id}, {len(codes)} codes]: {sample[:2500]}")
            break


def probe_pdfs() -> None:
    docs = {
        "infarmed_note_2025": "https://www.infarmed.pt/documents/15786/10634545/Nota+Informativa+-+Refor%C3%A7o+do+acesso+dos+cidad%C3%A3os+e+impacto+da+inova%C3%A7%C3%A3o%C2%A0/f7f43298-6e88-1b69-b292-8ba9edb51f5f",
        "apifarma_boletim_2026_05": "https://apifarma.pt/wp-content/uploads/2026/07/Boletim_Conjuntura_Mai-2026.pdf",
        "apifarma_boletim_2025_12": "https://apifarma.pt/wp-content/uploads/2026/01/Boletim_Conjuntura-Dez-2025.pdf",
        "apifarma_agreement_2025": "https://apifarma.pt/wp-content/uploads/2025/07/Acordo-Governo-APIFARMA-2025.pdf",
    }
    for name, url in docs.items():
        try:
            text = pdf_text(get(url, timeout=180).content)
            save_text(text, name)
            log(f"[pdf] {name}: {len(text)} characters")
            for line in text.splitlines():
                if re.search(r"milh|M€|contribui|devolu|concorrencial|quota|encargos|despesa", line, re.I):
                    s = re.sub(r"\s+", " ", line).strip()
                    if s:
                        log(f"      {s[:220]}")
        except Exception as exc:  # noqa: BLE001
            log(f"[pdf] {name} failed: {exc}")


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    OUT = Path(ap.parse_args().out) / "probe"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("pdfs", probe_pdfs), ("eurostat", probe_eurostat), ("oecd", probe_oecd),
                     ("transparencia", probe_transparencia)):
        t0 = time.time()
        try:
            fn()
            log(f"== {name} done in {time.time() - t0:.0f} s")
        except Exception:  # noqa: BLE001
            log(f"== {name} FAILED\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
