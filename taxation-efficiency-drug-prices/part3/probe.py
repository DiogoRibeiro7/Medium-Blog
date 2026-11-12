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
    # Portugal: Infarmed pages that list price files
    "infarmed_guia_genericos": "https://www.infarmed.pt/web/infarmed/entidades/medicamentos-uso-humano/genericos/guia_dos_genericos",
    "infarmed_spr": "https://www.infarmed.pt/web/infarmed/entidades/medicamentos-uso-humano/avaliacao-tecnologias-saude/"
                    "avaliacao-terapeutica-e-economica/sistema-de-precos-de-referencia",
    "infarmed_atribuicao_precos": "https://www.infarmed.pt/web/infarmed/entidades/medicamentos-uso-humano/avaliacao-economica/"
                                  "regulamentacao-preco-medicamentos/atribuicao_precos",
    "infarmed_regulamentacao_precos": "https://www.infarmed.pt/web/infarmed/entidades/medicamentos-uso-humano/"
                                      "avaliacao-tecnologias-saude/regulamentacao-de-precos",
    "infarmed_monitorizacao": "https://www.infarmed.pt/web/infarmed/entidades/medicamentos-uso-humano/monitorizacao-mercado",
    "infarmed_monitorizacao_ambulatorio": "https://www.infarmed.pt/web/infarmed/entidades/medicamentos-uso-humano/"
                                          "monitorizacao-mercado/relatorios/ambulatorio",
    "infarmed_estatistica": "https://www.infarmed.pt/web/infarmed/entidades/medicamentos-uso-humano/monitorizacao-mercado/"
                            "estatistica-anual",
    # Spain: Ministerio de Sanidad, Nomenclator de facturacion
    "es_nomenclator": "https://www.sanidad.gob.es/profesionales/nomenclator.do",
    # Italy: AIFA lists of class A and H medicines with prices
    "it_aifa_liste_ah": "https://www.aifa.gov.it/liste-farmaci-a-h",
    "it_aifa_liste": "https://www.aifa.gov.it/liste-dei-farmaci",
    # Belgium: INAMI reimbursable specialities and the SAM database
    "be_inami_specialites": "https://www.inami.fgov.be/fr/themes/soins-de-sante-cout-et-remboursement/"
                            "les-prestations-de-sante-que-vous-rembourse-votre-mutualite/medicaments/"
                            "remboursement-d-un-medicament/specialites-pharmaceutiques-remboursables",
    "be_sam_civics": "https://www.vas.ehealth.fgov.be/websamcivics/samcivics/",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    out_dir = Path(ap.parse_args().out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for k, v in PAGES.items():
        survey_page(k, v, out)
    for k, v in APIS.items():
        survey_api(k, v, out)
    (out_dir / "probe_report.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
