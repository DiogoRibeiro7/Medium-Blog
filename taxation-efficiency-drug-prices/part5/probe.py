"""Survey the public sources on parallel export and export restrictions, for Part 5.

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits data/probe_report.txt
and, under data/raw/, the text of the documents (PDFs as layout text, gzipped) and small spreadsheets.
Nothing is analysed here: the report is used to write fetch.py.

    python probe.py --out data
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import subprocess
import tempfile
import time
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)",
      "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8"}

PAGES = {
    "pt_disponibilidade": "https://www.infarmed.pt/web/infarmed/gestao-da-disponibilidade-do-medicamento",
    "pt_ruturas": "https://www.infarmed.pt/web/infarmed/ruturas",
    "pt_exportacao": "https://www.infarmed.pt/web/infarmed/exportacao-de-medicamentos",
    "ema_pd": "https://www.ema.europa.eu/en/human-regulatory-overview/post-authorisation/parallel-distribution",
    "iris_pd": "https://iris.ema.europa.eu/registerpd/",
}

SAVE = {
    "pt_disponibilidade_2024": "https://www.infarmed.pt/documents/15786/10901776/Relat%C3%B3rio+2024/2f02f60e-ef04-a5b1-7962-be7d5ef353a2",
    "pt_disponibilidade_2021": "https://www.infarmed.pt/documents/15786/4326055/Relat%C3%B3rio+anual+sobre+a+gest%C3%A3o+da+disponibilidade+de+medicamentos+de+2021/d6729605-96aa-7b0c-e702-d59e0e7e863e",
    "pt_lnp_aug_2026": "https://www.infarmed.pt/documents/15786/9423561/Delibera%C3%A7%C3%A3o+86+-+LNP+%28agosto+2026%29/087086db-c4d1-2a15-3c45-0cea642169e3",
    "pt_circ_oct_2025": "https://www.infarmed.pt/documents/15786/10773025/Atualiza%C3%A7%C3%A3o+das+listas+previstas+no+Regulamento+sobre+o+controlo+de+transa%C3%A7%C3%B5es+de+medicamentos+para+o+exterior+do+pa%C3%ADs/db92c49c-92b6-13c7-364d-c9758a3b70e6",
    "pt_circ_jun_2026": "https://www.infarmed.pt/documents/15786/12277909/Atualiza%C3%A7%C3%A3o+da+lista+de+medicamentos+cuja+exporta%C3%A7%C3%A3o+%C3%A9+temporariamente+suspensa+-+junho+2026/5c4742ff-efec-47b6-fdf0-99dc4db20498",
    "pt_regulamento_2021": "https://files.diariodarepublica.pt/2s/2021/04/078000000/0020900218.pdf",
}

FILE_LINK = re.compile(r"\.(csv|xls|xlsx|zip|ods|xml|json|txt|pdf)(\?|$|#)|/documents/|download|export|lista|relat",
                       re.IGNORECASE)
HREF = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
TAGS = re.compile(r"<[^>]+>")


def get(url: str, tries: int = 3, timeout: int = 120, session=None, **kw):
    last = None
    for i in range(tries):
        try:
            return (session or requests).get(url, headers=UA, timeout=timeout, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (i + 1))
    raise last


def pdf_text(data: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "in.pdf"
        p.write_bytes(data)
        subprocess.run(["pdftotext", "-layout", str(p), str(p.with_suffix(".txt"))], check=True)
        return p.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")


def survey_page(name: str, url: str, out: list[str], out_dir: Path) -> str:
    out.append(f"===== {name}\nURL {url}")
    try:
        r = get(url)
    except Exception as e:  # noqa: BLE001
        out.append(f"ERROR {type(e).__name__}: {e}")
        return ""
    ctype = r.headers.get("content-type", "")
    out.append(f"status {r.status_code}  final {r.url}  type {ctype}  bytes {len(r.content)}")
    if "html" not in ctype:
        out.append(r.content[:400].decode("utf-8", "replace"))
        return ""
    html = r.text
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)
    (raw / f"{name}.html.gz").write_bytes(gzip.compress(html.encode("utf-8"), mtime=0))
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
        if n >= 300:
            out.append("  (more links not listed)")
            break
    if n == 0:
        out.append("  no file links; first 2000 characters of text:")
        out.append(re.sub(r"\s+", " ", unescape(TAGS.sub(" ", html)))[:2000])
    return html


def save(name: str, url: str, out_dir: Path, out: list[str], session=None) -> None:
    out.append(f"===== save {name}\nURL {url}")
    try:
        r = get(url, timeout=300, session=session)
    except Exception as e:  # noqa: BLE001
        out.append(f"ERROR {type(e).__name__}: {e}")
        return
    ctype = r.headers.get("content-type", "")
    out.append(f"status {r.status_code}  final {r.url}  type {ctype}  bytes {len(r.content)}")
    if r.status_code != 200:
        return
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)
    if r.content[:4] == b"%PDF":
        text = pdf_text(r.content)
        (raw / f"{name}.txt.gz").write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        out.append(f"saved raw/{name}.txt.gz ({text.count(chr(12)) + 1} pages)")
        out.append("  first lines: " + re.sub(r"\s+", " ", text[:600]))
    elif r.content[:2] == b"PK" and len(r.content) < 20_000_000:
        (raw / f"{name}.xlsx").write_bytes(r.content)
        out.append(f"saved raw/{name}.xlsx")
    else:
        out.append(r.content[:600].decode("utf-8", "replace"))


def iris(out: list[str], out_dir: Path) -> None:
    """EMA's register of parallel distribution notices is a Power Pages portal. Find the entity grid on the
    page and ask its data service for the first pages of records."""
    out.append("===== iris register (entity grid)")
    s = requests.Session()
    try:
        r = get("https://iris.ema.europa.eu/registerpd/", session=s)
    except Exception as e:  # noqa: BLE001
        out.append(f"ERROR {e}")
        return
    html = r.text
    out.append(f"status {r.status_code} bytes {len(html)}")
    for pat in (r"data-view-layouts=\"([^\"]+)\"", r"data-get-url=\"([^\"]+)\"", r"entity-grid[^\"']{0,80}",
                r"_services/[^\"'<>\s]+", r"data-[a-z-]+=\"[^\"]{0,120}\"", r"/_api/[^\"'<>\s]+",
                r"<script[^>]+src=\"([^\"]+)\""):
        hits = list(dict.fromkeys(re.findall(pat, html)))[:40]
        out.append(f"  {pat}: {len(hits)}")
        for h in hits:
            out.append("    " + h[:400])
    layouts = re.findall(r"data-view-layouts=\"([^\"]+)\"", html)
    get_url = re.findall(r"data-get-url=\"([^\"]+)\"", html)
    try:
        tok = get("https://iris.ema.europa.eu/_layout/tokenhtml", session=s).text
        token = re.search(r"value=\"([^\"]+)\"", tok)
        token = token.group(1) if token else ""
        out.append(f"  token {'found' if token else 'missing'}")
    except Exception as e:  # noqa: BLE001
        out.append(f"  token ERROR {e}")
        token = ""
    if layouts and get_url:
        try:
            lay = json.loads(base64.b64decode(unescape(layouts[0])))
        except Exception as e:  # noqa: BLE001
            out.append(f"  layouts not base64 JSON: {e}")
            lay = json.loads(unescape(layouts[0])) if layouts[0].strip().startswith("[") else []
        out.append("  layout keys: " + json.dumps(lay)[:1500])
        if lay:
            conf = lay[0].get("Base64SecureConfiguration") or lay[0].get("base64SecureConfiguration")
            body = {"base64SecureConfiguration": conf, "sortExpression": "", "search": "", "page": 1,
                    "pageSize": 50, "pagingCookie": "", "filter": None, "metaFilter": None, "timezoneOffset": 0,
                    "customParameters": [], "entityName": lay[0].get("EntityName"),
                    "entityId": None}
            url = urljoin("https://iris.ema.europa.eu/", unescape(get_url[0]))
            try:
                rr = s.post(url, json=body, headers={**UA, "__RequestVerificationToken": token,
                                                     "Content-Type": "application/json"}, timeout=120)
                out.append(f"  POST {url} -> {rr.status_code} bytes {len(rr.content)}")
                out.append("  " + rr.text[:3000])
                raw = out_dir / "raw"
                raw.mkdir(exist_ok=True)
                (raw / "iris_page1.json.gz").write_bytes(gzip.compress(rr.content, mtime=0))
            except Exception as e:  # noqa: BLE001
                out.append(f"  POST ERROR {e}")
    # Power Pages Web API, in case it is enabled for anonymous reads.
    for path in ("/_api/ema_paralleldistributionnotices?$top=3", "/_api/$metadata"):
        try:
            rr = s.get("https://iris.ema.europa.eu" + path, headers={**UA, "__RequestVerificationToken": token},
                       timeout=60)
            out.append(f"  GET {path} -> {rr.status_code} {rr.text[:600]}")
        except Exception as e:  # noqa: BLE001
            out.append(f"  GET {path} ERROR {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for name, url in PAGES.items():
        survey_page(name, url, out, out_dir)
    for name, url in SAVE.items():
        save(name, url, out_dir, out)
    iris(out, out_dir)
    (out_dir / "probe_report.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
