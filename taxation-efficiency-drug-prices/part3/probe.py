"""Survey the public sources that could hold Portuguese and reference-country drug prices.

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits data/probe_report.txt.
For every page it records the HTTP status, the content type and size and, for HTML pages, the links
that point to data files or downloads; for catalogue APIs it records the datasets found. Nothing is
reduced or analysed here: the report is used to write fetch.py.

    python probe.py --out data
"""

from __future__ import annotations

import argparse
import json
import re
import time
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)",
      "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8"}

PAGES = {
    # Portugal: the public medicine search (Infomed) and the page that links to it
    "pt_infomed_fo": "https://extranet.infarmed.pt/INFOMED-fo/",
    "pt_infomed_pesquisa": "https://extranet.infarmed.pt/INFOMED-fo/pesquisa-avancada.xhtml",
    "pt_infarmed_pesquisa_medicamento": "https://www.infarmed.pt/web/infarmed/servicos-on-line/pesquisa-do-medicamento",
    # Spain: Ministerio de Sanidad, Nomenclator de facturacion
    "es_nomenclator": "https://www.sanidad.gob.es/profesionales/nomenclator.do",
    "be_sam_civics": "https://www.vas.ehealth.fgov.be/websamcivics/samcivics/",
}

# Documents saved as they are, to be read before fetch.py is written.
SAVE = {
    "pt_pvp_margins_2026": "https://www.infarmed.pt/documents/15786/1647560/Informa%C3%A7%C3%B5es+para+c%C3%A1lculo+de+PVP+de+"
                           "Medicamentos+%28margens%2C+factores+e+sites%29_2025/3708f280-6919-4cf0-0604-923ab03b0cc1",
    "pt_price_form_non_generic_2026": "https://www.infarmed.pt/documents/15786/1647560/Formul%C3%A1rio+Aprova%C3%A7%C3%A3o+"
                                      "Pre%C3%A7os_Medicamentos+N%C3%A3o+Gen%C3%A9ricos_Ambulat%C3%B3rio_2026/"
                                      "38576fa5-8c2e-4733-7a70-00139512229b",
    "pt_spr_deliberacao_2026q4_nov": "https://www.infarmed.pt/documents/15786/12277919/Sistema+de+Pre%C3%A7os+de+Ref%C3%AArencia"
                                     "+-+4%C2%BA+Trimestre+de+2026+%28novembro%29/6abc207f-1c49-d98b-a2c3-ba468e4d9f61",
    "pt_spr_circular_2026q4_nov": "https://www.infarmed.pt/documents/15786/12277909/Sistema+de+Pre%C3%A7os+de+Ref%C3%AArencia"
                                  "+-+4%C2%BA+Trimestre+de+2026+%28Novembro%29/2d0c4318-06bb-d67e-bd4c-b654ab5dbf1c",
    "pt_portaria_394_2025": "https://www.infarmed.pt/documents/15786/1065780/Portaria+n%C2%BA+394_2025_1%2C+de+14+de+novembro"
                            "%C2%A0/7a3319dd-d244-80c8-0c0b-7ee7a8c409d3",
}

# Strings searched for in the raw HTML of pages whose download links are built by scripts or forms.
RAW_GREP = {
    "es_nomenclator": r"[^\"'<>\s]*(?:csv|CSV|xls|rdf)[^\"'<>\s]*",
    "be_sam_civics": r"[^\"'<>\s]*(?:zip|xml|download|export)[^\"'<>\s]*",
    "pt_infomed_fo": r"[^\"'<>\s]*(?:xhtml|pesquisa|export|excel)[^\"'<>\s]*",
    "pt_infomed_pesquisa": r"(?:<form[^>]*>|<input[^>]*>|<select[^>]*>|[^\"'<>\s]*(?:xhtml|export|excel)[^\"'<>\s]*)",
}

APIS = {
    "pt_transparencia_medicament": "https://transparencia.sns.gov.pt/api/explore/v2.1/catalog/datasets?limit=100"
                                   "&where=search(%22medicament%22)",
    "pt_transparencia_farmac": "https://transparencia.sns.gov.pt/api/explore/v2.1/catalog/datasets?limit=100"
                               "&where=search(%22farm%C3%A1cia%22)",
    "pt_dados_gov_infarmed": "https://dados.gov.pt/api/1/organizations/infarmed/datasets/?page_size=100",
    "pt_dados_gov_medicamentos": "https://dados.gov.pt/api/1/datasets/?q=medicamentos&page_size=100",
}

FILE_LINK = re.compile(r"\.(csv|xls|xlsx|zip|ods|xml|json|txt|accdb|mdb|pdf)(\?|$|#)|/documents/|download|descarga|export",
                       re.IGNORECASE)
HREF = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
TAGS = re.compile(r"<[^>]+>")


def get(url: str, tries: int = 3, timeout: int = 120):
    last = None
    for i in range(tries):
        try:
            return requests.get(url, headers=UA, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (i + 1))
    raise last


def survey_page(name: str, url: str, out: list[str]) -> None:
    out.append(f"===== {name}\nURL {url}")
    try:
        r = get(url)
    except Exception as e:  # noqa: BLE001
        out.append(f"ERROR {type(e).__name__}: {e}")
        return
    ctype = r.headers.get("content-type", "")
    out.append(f"status {r.status_code}  final {r.url}  type {ctype}  bytes {len(r.content)}")
    if "html" not in ctype:
        out.append(r.content[:400].decode("utf-8", "replace"))
        return
    html = r.text
    title = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    out.append("title " + (unescape(title.group(1)).strip()[:200] if title else ""))
    seen = set()
    n = 0
    for href, text in HREF.findall(html):
        href = unescape(href).strip()
        if not FILE_LINK.search(href) or href.startswith("javascript"):
            continue
        full = urljoin(r.url, href)
        if full in seen:
            continue
        seen.add(full)
        label = re.sub(r"\s+", " ", unescape(TAGS.sub(" ", text))).strip()[:160]
        out.append(f"  LINK {label} -> {full}")
        n += 1
        if n >= 250:
            out.append("  (more links not listed)")
            break
    if n == 0:
        out.append("  no file links; first 1500 characters of text:")
        out.append(re.sub(r"\s+", " ", unescape(TAGS.sub(" ", html)))[:1500])
    if name in RAW_GREP:
        hits = list(dict.fromkeys(re.findall(RAW_GREP[name], html)))[:120]
        out.append(f"  raw matches ({len(hits)}):")
        out.extend("    " + h[:300] for h in hits)


def survey_api(name: str, url: str, out: list[str]) -> None:
    out.append(f"===== {name}\nURL {url}")
    try:
        r = get(url)
    except Exception as e:  # noqa: BLE001
        out.append(f"ERROR {type(e).__name__}: {e}")
        return
    out.append(f"status {r.status_code}  type {r.headers.get('content-type', '')}  bytes {len(r.content)}")
    try:
        js = r.json()
    except ValueError:
        out.append(r.text[:1000])
        return
    items = js.get("results") or js.get("data") or []
    out.append(f"total {js.get('total_count', js.get('total', len(items)))}")
    for d in items:
        if "dataset_id" in d:  # Opendatasoft
            meta = d.get("metas", {}).get("default", {})
            desc = re.sub(r"\s+", " ", TAGS.sub(" ", meta.get("description") or ""))[:300]
            fields = [f["name"] for f in d.get("fields", [])]
            out.append(f"  DATASET {d['dataset_id']} | {meta.get('title')} | modified {meta.get('modified')} | "
                       f"records {meta.get('records_count')}\n    fields {fields}\n    {desc}")
        else:  # udata (dados.gov.pt)
            res = d.get("resources", [])
            out.append(f"  DATASET {d.get('slug')} | {d.get('title')} | modified {d.get('last_modified')}")
            for x in res[:12]:
                out.append(f"    RES {x.get('format')} | {x.get('title')} -> {x.get('url')}")


def save(name: str, url: str, out_dir: Path, out: list[str]) -> None:
    out.append(f"===== save {name}\nURL {url}")
    try:
        r = get(url)
    except Exception as e:  # noqa: BLE001
        out.append(f"ERROR {type(e).__name__}: {e}")
        return
    ctype = r.headers.get("content-type", "")
    disp = r.headers.get("content-disposition", "")
    out.append(f"status {r.status_code}  type {ctype}  bytes {len(r.content)}  disposition {disp}")
    if r.status_code != 200 or len(r.content) > 15_000_000:
        return
    ext = {"pdf": ".pdf", "spreadsheetml": ".xlsx", "ms-excel": ".xls", "wordprocessingml": ".docx",
           "msword": ".doc", "zip": ".zip", "html": ".html", "csv": ".csv"}
    suffix = next((v for k, v in ext.items() if k in ctype), "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', disp)
    if m and "." in m.group(1):
        suffix = "." + m.group(1).rsplit(".", 1)[1].lower()
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)
    (raw / f"{name}{suffix}").write_bytes(r.content)
    out.append(f"saved raw/{name}{suffix}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    out_dir = Path(ap.parse_args().out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for k, v in SAVE.items():
        save(k, v, out_dir, out)
    for k, v in PAGES.items():
        survey_page(k, v, out)
    if "--apis" in __import__("sys").argv:
        for k, v in APIS.items():
            survey_api(k, v, out)
    (out_dir / "probe_report.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
