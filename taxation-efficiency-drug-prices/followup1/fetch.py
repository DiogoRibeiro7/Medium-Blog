"""Download the documents used in follow-up 1 (confidential rebates in Portugal, France and Belgium, and
what Italy and Spain publish), keep their text, and check that every figure the analysis reads is in it.

Run by .github/workflows/taxation-efficiency-drug-prices-data.yml, which commits the reduced files in
data/ together with data/fetch_log.txt. Each document is fetched independently; a failure is logged and
the others still run. The SNS spending series and Infarmed's note of November 2025 are read from
../part7/data and are not downloaded again.

Documents
  Portugal  The agreements between the State and the pharmaceutical industry, 2013-2025 (APIFARMA; the
            English versions where the Portuguese scan has no text layer); APIFARMA newsletters of September
            2024 and September 2025 (returns as a share of the SNS hospital market, 2018 and 2022; returns
            in 2024); the Conselho das Finanças Públicas reports on the SNS, 2020-2025 (revenue of the
            extraordinary contribution on the pharmaceutical industry); ECO, 29 May 2025, on Infarmed's
            report of SNS spending in 2024 (gross and net of returns).
  France    DREES, Les Dossiers de la Drees n.º 139 (reimbursable medicines and rebates, 2016-2024);
            DREES, Les établissements de santé 2026, fiche 23 (expensive hospital medicines, gross and net);
            IGAS-IGF 2026 mission report on the regulation of medicine spending (rebates 2019 and 2023), from the
            vie-publique.fr copy.
  Belgium   INAMI/RIZIV, MORSE reports (data 2022-2023 and data 2024): gross and net expenditure on
            reimbursable specialties, and the compensations paid under article 81/111 conventions.
  Italy     AIFA, monitoring of pharmaceutical spending, January-December 2024 (final) and its summary:
            paybacks and credit notes under negotiated agreements.

    python fetch.py --out data
"""

from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import requests

UA = {"User-Agent": "Mozilla/5.0 (research script; contact dfr@esmad.ipp.pt)"}
LOG: list[str] = []

APIFARMA = "https://apifarma.pt/wp-content/uploads/"
CFP = "https://www.cfp.pt/uploads/publicacoes_ficheiros/"
# name: (url, kind, patterns that must be in the text, whitespace collapsed)
DOCS = {
    "pt_agreement_2013": (APIFARMA + "2021/04/Acordo-2013-Versao-Inglesa.pdf", "pdf",
                          [r"contribution in the amount of 122 million euros"]),
    "pt_agreement_2014": (APIFARMA + "2021/04/Acordo-2014-Versao-Inglesa.pdf", "pdf",
                          [r"contribution in the amount of 160 million euro"]),
    "pt_agreement_2015": (APIFARMA + "2021/04/Acordo-2015-Versao-Inglesa.pdf", "pdf",
                          [r"contribution in the amount of 180 million euro"]),
    "pt_agreement_2016": (APIFARMA + "2021/04/Acordo-2016-Versao-Inglesa.pdf", "pdf",
                          [r"financial contribution amounting to 200 million euro", r"minimum value of 150 million euro"]),
    "pt_agreement_2017": (APIFARMA + "2021/04/Acordo-2017-Versao-Inglesa.pdf", "pdf",
                          [r"minimum amount of 200 million euros, which will be quarterly updated"]),
    "pt_agreement_2024": (APIFARMA + "2024/03/Acordo-Governo-APIFARMA-2024.pdf", "pdf",
                          [r"replicado anualmente, entre 2019 e 2023"]),
    "pt_agreement_2025": (APIFARMA + "2025/07/Acordo-Governo-APIFARMA-2025.pdf", "pdf",
                          [r"Despesa L.quida Medicamentos = Despesa Total"]),
    "pt_apifarma_newsletter_2024_09": (APIFARMA + "2024/10/apifarma_newsletter_setembro_2024.pdf", "pdf",
                                       [r"aumentaram de 11% para 18%", r"aos 30% do mercado SNS"]),
    "pt_apifarma_newsletter_2025_09": (APIFARMA + "2025/10/apifarma_newsletter_setembro_2025.pdf", "pdf",
                                       [r"na ordem dos 600 milh.es de euros, em 2024"]),
    "pt_cfp_sns_2020": (CFP + "cfp-rel-06-2021.pdf", "pdf",
                        [r"representou 14 M€ em 2018, cerca de 13,3 M€ em 2019 e de 13,9 M€ no ano de 2020"]),
    "pt_cfp_sns_2022": (CFP + "cfp-rel-07-2023.pdf", "pdf", [r"farmac.utica \(17,9 M€, com um aumento de 1,2 M€"]),
    "pt_cfp_sns_2023": (CFP + "cfp-rel-07-2024.pdf", "pdf", [r"farmac.utica \(18,4 M€"]),
    "pt_cfp_sns_2024": (CFP + "cfp-rel-05-2025.pdf", "pdf", [r"farmac.utica \(17,2 M€"]),
    "pt_cfp_sns_2025": (CFP + "cfp_rel-05-2026.pdf", "pdf", [r"farmac.utica \(26,2 M€"]),
    "pt_eco_2025_05_29": ("https://eco.sapo.pt/2025/05/29/despesa-do-sns-com-medicamentos-subiu-para-3-960-milhoes-de-euros-em-2024/",
                          "html", [r"3\.960 milh", r"3,5 mil milh"]),
    "fr_drees_dd139": ("https://drees.solidarites-sante.gouv.fr/sites/default/files/2026-06/DD139_M%C3%A9dicaments.pdf", "pdf",
                       [r"passant de 1,2 à 9,0 milliards d'euros entre 2016 et 2024",
                        r"augmenté de 12,1 milliards d'euros, pour atteindre 41,1 milliards d'euros en 2024",
                        r"dépense nette totale à 32,0 milliards", r"prix facial, dont le niveau sert de référence"]),
    "fr_drees_es2026_fiche23": ("https://drees.solidarites-sante.gouv.fr/sites/default/files/2026-07/ES%202026%20-%20Fiche%2023%20-%20Les%20m%C3%A9dicaments%20et%20dispositifs%20m%C3%A9dicaux%20on%C3%A9reux.pdf",
                                "pdf", [r"11,4 milliards d’euros, et 7,6 milliards après déduction"]),
    "fr_igf_2026": ("https://www.vie-publique.fr/files/rapport/pdf/303194.pdf",
                    "pdf", [r"remises ayant atteint 8,3 Mds€ en 2023", r"36,5 Mds", r"3,2 Mds€ en 2019"]),
    "be_morse_2022_2023": ("https://www.riziv.fgov.be/SiteCollectionDocuments/morse_rapport_donnees_2022_2023.pdf", "pdf",
                           [r"chiffre d.affaires de 11,7 milliards", r"4,5 milliards à l’Institut, soit 38,5%"]),
    "be_morse_2024": ("https://www.inami.fgov.be/SiteCollectionDocuments/MORSE_rapport_2024_FR.pdf", "pdf",
                      [r"dépenses brutes de l’INAMI atteignent 7,9 Md€", r"dépenses nettes s’élèvent à 5,6 Md€",
                       r"1,8 Md€ de rétrocessions", r"17,3 milliards d’euros", r"7,8 milliards à l’Institut, soit 45,4%"]),
    "it_aifa_sintesi_2024": ("https://www.aifa.gov.it/documents/20142/2505785/Sintesi_monitoraggio_Spesa_gen-dic-2024.pdf", "pdf",
                             [r"accordi negoziali \(impatto Gennaio - Dicembre 2024 pari a 136,0 mln"]),
    "it_aifa_monitoraggio_2024": ("https://www.aifa.gov.it/documents/20142/2505785/Monitoraggio_Spesa_gennaio-dicembre-2024_consultivo.pdf",
                                  "pdf", [r"Payback 1,83"]),
}


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def get(url: str, tries: int = 4, timeout: int = 180, wait: float = 5.0) -> requests.Response:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"status {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(wait * (k + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def pdf_text(data: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "in.pdf"
        pdf.write_bytes(data)
        subprocess.run(["pdftotext", "-layout", str(pdf), str(pdf.with_suffix(".txt"))], check=True)
        return pdf.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")


def html_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    return re.sub(r"<[^>]+>", " ", html)


def save_text(text: str, out: Path, name: str) -> None:
    path = out / f"{name}.txt.gz"
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(text.encode("utf-8"))
    log(f"[{name}] wrote {path.name}: {len(text)} characters")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, (url, kind, patterns) in DOCS.items():
        try:
            r = get(url)
            text = pdf_text(r.content) if kind == "pdf" else html_text(r.text)
            flat = re.sub(r"\s+", " ", text)
            missing = [p for p in patterns if not re.search(p, flat)]
            if missing:
                raise RuntimeError(f"expected text not found: {missing}")
            save_text(text, out, name)
        except Exception:  # noqa: BLE001
            failed.append(name)
            log(f"== {name} FAILED\n{traceback.format_exc()}")
    log(f"failed: {failed}")
    (out / "fetch_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
