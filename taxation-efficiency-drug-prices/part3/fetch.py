"""Download and reduce the public price files used in Part 3.

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits the reduced files in
data/ together with data/fetch_log.txt. Each source is fetched independently; a failure is logged and
the others still run. France reuses the Part 2 snapshot (BDPM and Open Medic) in ../part2/data.

Sources
  Spain
    es_nomenclator      Nomenclator de facturacion (Ministerio de Sanidad): every presentation billed to
                        the national health system with its retail price including VAT (PVP IVA).
  Italy
    it_classe_a         AIFA list of class A medicines by active substance, 30 June 2026: retail price
                        for every reimbursed presentation.
  Belgium
    be_sam              SAM v2 full export (FPS Public Health / RIZIV-INAMI): every authorised pack with
                        its ex-factory price. The export is a zip of XML files; only the current version
                        of each record is kept.
  Portugal
    pt_gh_2026q3        Infarmed, Deliberacao 071/CD/2026: homogeneous groups and reference prices for
                        the third quarter of 2026, with the retail price of every presentation in a group.
                        Published as a PDF; kept as layout text.
    pt_rep_*            Infarmed, exceptional price revisions granted by the Secretary of State for
                        Health (ambulatory market), 2025 and 2026 tables.
    pt_rap_form_2026    Infarmed, calculation form for the 2026 annual price revision (non-generics).
    pt_margins_2026     Infarmed, margins and factors used to compute retail prices, 2026.
    pt_transparencia_*  SNS Transparency portal: monthly ambulatory spending, packs and average price
                        per pack.

    python fetch.py --out data
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)"}

INF = "https://www.infarmed.pt/documents/15786/"
URLS = {
    "es_nomenclator": [
        "https://www.sanidad.gob.es/profesionales/nomenclator.do?metodo=buscarProductos&especialidad=%25%25%25"
        "&d-4015021-e=1&6578706f7274=1",
        "https://www.mscbs.gob.es/profesionales/nomenclator.do?metodo=buscarProductos&especialidad=%25%25%25"
        "&d-4015021-e=1&6578706f7274=1",
    ],
    "it_classe_a": "https://www.aifa.gov.it/documents/20142/3815901/Classe-A_per_principio-attivo_30-6-2026.csv",
    "be_sam_version": "https://www.vas.ehealth.fgov.be/websamcivics/samcivics/download/samv2-full-getLastVersion?xsd=6",
    "be_sam_full": "https://www.vas.ehealth.fgov.be/websamcivics/samcivics/download/samv2-download?type=FULL&xsd=6"
                   "&version={version}",
    "pt_gh_2026q3": INF + "12277919/Sistema+de+Pre%C3%A7os+de+Ref%C3%AArencia+-+3%C2%BA+Trimestre+de+2026%28julho%29/"
                          "3d57d852-aaa6-009c-0231-80736411ba02",
    "pt_rep_2026": INF + "3368817/tabela+2026+ambulat%C3%B3rio.pdf/1420c0b8-4e94-8b7a-8bb0-f15f5a751af9",
    "pt_rep_2025": INF + "1647560/REP+2025/69e7a27a-5565-7e56-24c9-3d6b15757be8",
    "pt_rap_form_2026": INF + "1647560/Formul%C3%A1rio_+RAP+%28mercado+ambulat%C3%B3rio+e+hospitalar%29+2026%C2%A0/"
                              "4481fcbe-c20e-4cb8-4449-ac0230f573f0",
    "pt_margins_2026": INF + "1647560/Informa%C3%A7%C3%B5es+para+c%C3%A1lculo+de+PVP+de+Medicamentos+"
                             "%28margens%2C+factores+e+sites%29_2025/3708f280-6919-4cf0-0604-923ab03b0cc1",
    "pt_transparencia": "https://transparencia.sns.gov.pt/api/explore/v2.1/catalog/datasets/{id}/exports/csv"
                        "?delimiter=%3B",
}
TRANSPARENCIA = ["preco-medio-por-embalagem", "despesa-com-medicamentos-no-ambulatorio-sns"]

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def get(url: str, tries: int = 4, timeout: int = 300) -> requests.Response:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def write_gz(data: bytes, path: Path) -> None:
    # mtime=0 keeps the gzip header free of a timestamp, so an unchanged file gives identical bytes.
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(data)
    log(f"wrote {path.name}: {path.stat().st_size / 1e6:.2f} MB")


def save(df: pd.DataFrame, out: Path, name: str) -> None:
    write_gz(df.to_csv(index=False).encode("utf-8"), out / f"{name}.csv.gz")
    log(f"[{name}] {len(df)} rows, columns {list(df.columns)}")


def pdf_text(data: bytes) -> str:
    """Layout-preserving text of a PDF, from poppler's pdftotext (installed by the workflow)."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "in.pdf"
        pdf.write_bytes(data)
        txt = Path(tmp) / "out.txt"
        subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=True)
        return txt.read_text(encoding="utf-8", errors="replace")


def kind(data: bytes) -> str:
    if data[:4] == b"%PDF":
        return "pdf"
    if data[:2] == b"PK":
        return "zip"
    return "text"


def save_document(data: bytes, out: Path, name: str) -> None:
    """A PDF is kept as layout text; a spreadsheet (zip) as it is; anything else as gzipped bytes."""
    k = kind(data)
    log(f"[{name}] {len(data) / 1e6:.2f} MB, type {k}")
    if k == "pdf":
        text = pdf_text(data)
        log(f"[{name}] {text.count(chr(12)) + 1} pages, first lines:\n" + "\n".join(text.splitlines()[:25]))
        write_gz(text.encode("utf-8"), out / f"{name}.txt.gz")
    elif k == "zip":
        (out / f"{name}.xlsx").write_bytes(data)
        log(f"wrote {name}.xlsx")
    else:
        write_gz(data, out / f"{name}.bin.gz")


# -------------------------------------------------------------------------------------------- Spain
def fetch_es(out: Path) -> None:
    for url in URLS["es_nomenclator"]:
        try:
            r = get(url, timeout=600)
        except Exception as exc:  # noqa: BLE001
            log(f"[es_nomenclator] {url}: {exc}")
            continue
        ctype = r.headers.get("content-type", "")
        log(f"[es_nomenclator] {url}\n  status {r.status_code} type {ctype} bytes {len(r.content)} "
            f"disposition {r.headers.get('content-disposition', '')}")
        head = r.content[:2000].decode("latin-1", errors="replace")
        log("  first lines:\n" + "\n".join(head.splitlines()[:6]))
        if "html" in ctype.lower() and "<html" in head.lower():
            continue
        write_gz(r.content, out / "es_nomenclator.csv.gz")
        return
    raise RuntimeError("no nomenclator export")


# -------------------------------------------------------------------------------------------- Italy
def fetch_it(out: Path) -> None:
    r = get(URLS["it_classe_a"])
    write_gz(r.content, out / "it_classe_a.csv.gz")
    head = r.content[:1500].decode("latin-1", errors="replace")
    log("[it_classe_a] first lines:\n" + "\n".join(head.splitlines()[:4]))


# ------------------------------------------------------------------------------------------ Belgium
def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def current(el: ET.Element, today: str) -> bool:
    """SAM keeps every version of a record in Data blocks with from/to dates; keep the one in force."""
    f, t = el.get("from"), el.get("to")
    return (f is None or f <= today) and (t is None or t >= today)


def flatten(el: ET.Element, today: str, skip: set[str], prefix: str = "", acc: dict | None = None) -> dict:
    """Leaf values of a SAM record keyed by their path, Data levels dropped and repeats joined."""
    acc = {} if acc is None else acc
    for a, v in el.attrib.items():
        if a not in ("from", "to"):
            acc.setdefault(f"{prefix}@{a}", []).append(v)
    for ch in el:
        name = local(ch.tag)
        if name in skip:
            continue
        if name == "Data":
            if current(ch, today):
                flatten(ch, today, skip, prefix, acc)
            continue
        key = f"{prefix}{name}"
        if len(ch) == 0:
            if ch.text and ch.text.strip():
                acc.setdefault(key, []).append(ch.text.strip())
            for a, v in ch.attrib.items():
                if a not in ("from", "to"):
                    acc.setdefault(f"{key}@{a}", []).append(v)
        else:
            flatten(ch, today, skip, key + ".", acc)
    return acc


def joined(d: dict) -> dict:
    return {k: " | ".join(dict.fromkeys(v)) for k, v in d.items()}


def sample(el: ET.Element, lines: int = 140) -> str:
    el2 = ET.fromstring(ET.tostring(el))
    ET.indent(el2)
    return "\n".join(ET.tostring(el2, encoding="unicode").splitlines()[:lines])


def parse_sam_member(fh, today: str, structure: list[str], member: str) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    depth = 0
    root = None
    sampled: set[str] = set()
    for event, el in ET.iterparse(fh, events=("start", "end")):
        if event == "start":
            depth += 1
            if depth == 1:
                root = el
            continue
        depth -= 1
        if depth != 1:
            continue
        name = local(el.tag)
        if name not in sampled and len(sampled) < 6:
            sampled.add(name)
            structure.append(f"===== {member}: {name}\n{sample(el)}")
        if name == "Amp":
            base = joined(flatten(el, today, skip={"Ampp"}))
            for ampp in el.iter():
                if local(ampp.tag) != "Ampp":
                    continue
                row = dict(base)
                row.update({f"ampp.{k}": v for k, v in joined(flatten(ampp, today, skip=set())).items()})
                rows.setdefault("amp", []).append(row)
        else:
            rows.setdefault(name, []).append(joined(flatten(el, today, skip=set())))
        el.clear()
        if root is not None:
            root.clear()
    return rows


def fetch_be(out: Path) -> None:
    version = get(URLS["be_sam_version"]).text.strip()
    url = URLS["be_sam_full"].format(version=version)
    log(f"[be_sam] version {version}: {url}")
    today = dt.date.today().isoformat()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sam.zip"
        with requests.get(url, headers=UA, timeout=1800, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as fh:
                shutil.copyfileobj(r.raw, fh, length=1 << 20)
        log(f"[be_sam] downloaded {path.stat().st_size / 1e6:.1f} MB")
        structure: list[str] = []
        tables: dict[str, list[dict]] = {}
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                log(f"[be_sam] member {info.filename}: {info.file_size / 1e6:.1f} MB")
            for info in z.infolist():
                prefix = info.filename.split("-")[0].upper()
                if prefix not in ("AMP", "VMP", "REF"):
                    continue
                with z.open(info) as fh:
                    for name, rows in parse_sam_member(fh, today, structure, info.filename).items():
                        tables.setdefault(name, []).extend(rows)
    write_gz("\n\n".join(structure).encode("utf-8"), out / "be_sam_structure.txt.gz")
    for name, rows in tables.items():
        df = pd.DataFrame(rows)
        log(f"[be_sam] {name}: {len(df)} rows, {df.shape[1]} columns")
        if name == "amp" or len(df) > 50:
            # Keep the fields that identify the product and its pack and price; drop the leaflets,
            # contacts and the multilingual copies other than French and English.
            keep = [c for c in df.columns
                    if not any(s in c for s in (".Nl", ".De", "Url", "Leaflet", "Spc", "Contact"))]
            df = df[keep]
            save(df, out, f"be_sam_{name.lower()}")


# ----------------------------------------------------------------------------------------- Portugal
def fetch_pt_docs(out: Path) -> None:
    failed = []
    for name in ("pt_gh_2026q3", "pt_rep_2026", "pt_rep_2025", "pt_rap_form_2026", "pt_margins_2026"):
        try:
            save_document(get(URLS[name]).content, out, name)
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            log(f"[{name}] FAILED: {exc}")
    if failed:
        raise RuntimeError(f"failed: {failed}")


def fetch_pt_transparencia(out: Path) -> None:
    for ds in TRANSPARENCIA:
        r = get(URLS["pt_transparencia"].format(id=ds))
        df = pd.read_csv(io.BytesIO(r.content), sep=";", dtype=str)
        df = df.sort_values(list(df.columns)).reset_index(drop=True)
        save(df, out, "pt_transparencia_" + ds.replace("-", "_"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {"es": fetch_es, "it": fetch_it, "pt_docs": fetch_pt_docs,
            "pt_transparencia": fetch_pt_transparencia, "be": fetch_be}
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
