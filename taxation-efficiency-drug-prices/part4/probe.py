"""Survey the public sources on time to access for new medicines, for Part 4.

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits data/probe_report.txt
and, under data/raw/, the text of the documents (PDFs are kept as layout text, gzipped) and small
spreadsheets. Nothing is analysed here: the report is used to write fetch.py.

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
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)",
      "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8"}

PAGES: dict = {}
RAW_GREP: dict = {}
SAVE: dict = {}
KEEP_HTML: dict = {}
APIS: dict = {}
SIATS = "https://extranet.infarmed.pt/siats/Publico/Processos/InfoDetalhesProcesso.aspx"

FILE_LINK = re.compile(r"\.(csv|xls|xlsx|zip|ods|xml|json|txt|pdf)(\?|$|#)|/documents/|download|export|tracker",
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


def pdf_text(data: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "in.pdf"
        p.write_bytes(data)
        subprocess.run(["pdftotext", "-layout", str(p), str(p.with_suffix(".txt"))], check=True)
        return p.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")


def save(name: str, url: str, out_dir: Path, out: list[str]) -> None:
    out.append(f"===== save {name}\nURL {url}")
    try:
        r = get(url, timeout=300)
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
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "in.pdf"
            pdf.write_bytes(r.content)
            subprocess.run(["pdftotext", "-bbox-layout", str(pdf), str(pdf.with_suffix(".html"))], check=True)
            (raw / f"{name}.bbox.html.gz").write_bytes(gzip.compress(pdf.with_suffix(".html").read_bytes(), mtime=0))
        out.append(f"saved raw/{name}.txt.gz ({text.count(chr(12)) + 1} pages)")
    elif r.content[:2] == b"PK" and len(r.content) < 20_000_000:
        (raw / f"{name}.xlsx").write_bytes(r.content)
        out.append(f"saved raw/{name}.xlsx")
    else:
        out.append(r.content[:600].decode("utf-8", "replace"))


def save_api(name: str, url: str, out_dir: Path, out: list[str]) -> None:
    out.append(f"===== api {name}\nURL {url}")
    try:
        r = get(url)
    except Exception as e:  # noqa: BLE001
        out.append(f"ERROR {e}")
        return
    out.append(f"status {r.status_code} bytes {len(r.content)}")
    try:
        js = r.json()
    except ValueError:
        out.append(r.text[:800])
        return
    for dim, d in js.get("dimension", {}).items():
        labels = d.get("category", {}).get("label", {})
        out.append(f"  DIM {dim}: {len(labels)} categories")
        for k, v in list(labels.items())[:120]:
            out.append(f"    {k}: {v}")
    out.append("  values: " + json.dumps(js.get("value", {}))[:2000])


def form_fields(html: str) -> dict:
    out = {}
    for m in re.finditer(r'<input[^>]*type="hidden"[^>]*>', html):
        t = m.group(0)
        n = re.search(r'name="([^"]+)"', t)
        v = re.search(r'value="([^"]*)"', t)
        if n:
            out[n.group(1)] = unescape(v.group(1)) if v else ""
    return out


def table_text(html: str, limit: int = 60) -> list[str]:
    rows = re.findall(r"<tr\b.*?</tr>", html, flags=re.S | re.I)
    out = []
    for r in rows:
        cells = [re.sub(r"\s+", " ", unescape(TAGS.sub(" ", c))).strip()
                 for c in re.findall(r"<t[dh]\b.*?</t[dh]>", r, flags=re.S | re.I)]
        cells = [c for c in cells if c]
        if cells:
            out.append(" | ".join(cells)[:400])
    return out[:limit]


def siats(out_dir: Path, out: list[str]) -> None:
    out.append("===== SIATS search")
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)
    s = requests.Session()
    s.headers.update(UA)
    r = s.get(SIATS, timeout=120)
    f = form_fields(r.text)
    data = dict(f)
    data.update({"ctl00$ContentPlaceHolder1$ddlTipoMedicamento": "66",
                 "ctl00$ContentPlaceHolder1$ddlTipoProcesso": "-1",
                 "ctl00$ContentPlaceHolder1$txtNomeMedicamentoProcesso": "",
                 "ctl00$ContentPlaceHolder1$txtDciProcesso": "",
                 "ctl00$ContentPlaceHolder1$txtAreaTerapProcesso": "",
                 "ctl00$ContentPlaceHolder1$btnPesquisarProcessos": "Pesquisar"})
    p = s.post(SIATS, data=data, timeout=180)
    (raw / "siats_search_p1.html.gz").write_bytes(gzip.compress(p.content, mtime=0))
    out.append(f"POST search: status {p.status_code} bytes {len(p.content)}")
    out.extend("  ROW " + x for x in table_text(p.text))
    pager = sorted(set(re.findall(r"__doPostBack\(&#39;([^&]+)&#39;,&#39;(Page\$[^&]+)&#39;\)", p.text)))
    out.append(f"  pager targets {pager[:30]}")
    links = sorted(set(re.findall(r'href="([^"]*(?:aspx|Processo|processo)[^"]*)"', p.text)))
    out.append(f"  links {links[:40]}")
    onclick = sorted(set(re.findall(r'onclick="([^"]{0,200})"', p.text)))
    out.append(f"  onclick samples {onclick[:20]}")
    counts = re.findall(r"(\d+)\s*(?:registos|resultados|processos)", TAGS.sub(" ", p.text), flags=re.I)
    out.append(f"  count mentions {counts[:10]}")
    if pager:
        f2 = form_fields(p.text)
        tgt, arg = pager[0]
        d2 = dict(f2)
        d2.update({"__EVENTTARGET": unescape(tgt).replace("&#39;", "'"), "__EVENTARGUMENT": arg,
                   "ctl00$ContentPlaceHolder1$ddlTipoMedicamento": "66", "ctl00$ContentPlaceHolder1$ddlTipoProcesso": "-1"})
        time.sleep(1)
        p2 = s.post(SIATS, data=d2, timeout=180)
        (raw / "siats_search_p2.html.gz").write_bytes(gzip.compress(p2.content, mtime=0))
        out.append(f"POST page {arg}: status {p2.status_code} bytes {len(p2.content)}")
        out.extend("  ROW " + x for x in table_text(p2.text, 15))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for name, url in PAGES.items():
        survey_page(name, url, out)
    for name, url in SAVE.items():
        save(name, url, out_dir, out)
    for name, url in APIS.items():
        save_api(name, url, out_dir, out)
    try:
        siats(out_dir, out)
    except Exception as e:  # noqa: BLE001
        import traceback
        out.append("SIATS ERROR " + traceback.format_exc())
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)
    for name, url in KEEP_HTML.items():
        try:
            r = get(url)
            (raw / f"{name}.html.gz").write_bytes(gzip.compress(r.content, mtime=0))
            out.append(f"===== kept {name}: status {r.status_code} bytes {len(r.content)}")
        except Exception as e:  # noqa: BLE001
            out.append(f"===== kept {name}: ERROR {e}")
    (out_dir / "probe_report.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
