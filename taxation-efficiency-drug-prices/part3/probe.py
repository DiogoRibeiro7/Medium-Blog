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
    "es_nomenclator": "https://www.sanidad.gob.es/profesionales/nomenclator.do",
}

# Documents saved as they are, to be read before fetch.py is written.
SAVE = {
    "es_nomenclator_csv_try": "https://www.sanidad.gob.es/profesionales/nomenclator.csv",
    "es_nomenclator_xls_try": "https://www.sanidad.gob.es/profesionales/nomenclator.xls",
    "be_sam_last_version": "https://www.vas.ehealth.fgov.be/websamcivics/samcivics/download/samv2-full-getLastVersion?xsd=6",
    "be_sam_last_datetime": "https://www.vas.ehealth.fgov.be/websamcivics/samcivics/download/samv2-full-getLastDateTime?xsd=6",
    "it_classe_a_principio_attivo": "https://www.aifa.gov.it/documents/20142/3815901/Classe-A_per_principio-attivo_30-6-2026.csv",
}

# Strings searched for in the raw HTML of pages whose download links are built by scripts or forms.
RAW_GREP = {
    "es_nomenclator": r".{0,200}nomenclator\.(?:csv|xls|rdf).{0,200}",
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


def infomed_attempt(out_dir: Path, out: list[str], dci: str = "Apixabano") -> None:
    """Open the advanced search page in a session and submit one search, saving what comes back."""
    import gzip
    out.append(f"===== infomed search attempt for {dci}")
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)
    s = requests.Session()
    s.headers.update(UA)
    try:
        r = s.get("https://extranet.infarmed.pt/INFOMED-fo/pesquisa-avancada.xhtml", timeout=120)
    except Exception as e:  # noqa: BLE001
        out.append(f"ERROR {e}")
        return
    html = r.text
    (raw / "pt_infomed_pesquisa.html.gz").write_bytes(gzip.compress(r.content, mtime=0))
    action = re.search(r'<form id="mainForm"[^>]*action="([^"]+)"', html)
    vs = re.findall(r'name="javax.faces.ViewState"[^>]*value="([^"]+)"', html)
    buttons = sorted(set(re.findall(r'id="(mainForm:[^"]*(?:[Pp]esquis|[Ss]earch|btn)[^"]*)"', html)))
    out.append(f"form action {action.group(1) if action else None}; view states {len(vs)}; buttons {buttons[:20]}")
    if not action or not vs:
        return
    url = urljoin(r.url, unescape(action.group(1)))
    for i, b in enumerate([x for x in buttons if not x.endswith("_s")][:4]):
        data = {"mainForm": "mainForm", "mainForm:dci_input": dci, "javax.faces.ViewState": vs[-1],
                "javax.faces.partial.ajax": "true", "javax.faces.source": b, "javax.faces.partial.execute": "@all",
                "javax.faces.partial.render": "@all", b: b}
        try:
            p = s.post(url, data=data, timeout=120, headers={"Faces-Request": "partial/ajax",
                                                              "X-Requested-With": "XMLHttpRequest"})
        except Exception as e:  # noqa: BLE001
            out.append(f"  POST {b}: ERROR {e}")
            continue
        (raw / f"pt_infomed_result_{i}.xml.gz").write_bytes(gzip.compress(p.content, mtime=0))
        txt = re.sub(r"\s+", " ", unescape(TAGS.sub(" ", p.text)))
        out.append(f"  POST {b}: status {p.status_code} bytes {len(p.content)}; mentions of {dci[:6]}: "
                   f"{p.text.lower().count(dci[:6].lower())}; redirect {'redirect' in p.text[:300]}")
        k = txt.lower().find(dci[:6].lower())
        out.append("    " + txt[max(0, k - 300):k + 900])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    out_dir = Path(ap.parse_args().out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for k, v in SAVE.items():
        save(k, v, out_dir, out)
    infomed_attempt(out_dir, out)
    for k, v in PAGES.items():
        survey_page(k, v, out)
    if "--apis" in __import__("sys").argv:
        for k, v in APIS.items():
            survey_api(k, v, out)
    (out_dir / "probe_report.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
