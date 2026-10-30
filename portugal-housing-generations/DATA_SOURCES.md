# Data sources, reuse terms and redistribution

Every file in `data/` is listed in [`DATA_SOURCES.csv`](DATA_SOURCES.csv), which `python data_sources.py`
builds from the retrieval records stored with the snapshots. For each file it gives:
- the provider, dataset or indicator and title;
- the request URL, the Eurostat DOI where there is one, and the retrieval time;
- the provider's last-update stamp;
- the reuse terms, the attribution required and the redistribution status.

Provider terms were read from the providers' own pages on **30 October 2026**. They can change: check them again
before any new deposit.

## Licences: what covers what

- **Original code and documentation** in this folder (scripts, tests, reports, figures and generated results) are
  covered by the repository's Apache License 2.0.
- **Third-party data** in `data/` are **not** relicensed under Apache-2.0. They are redistributed, where
  permitted, under each provider's own terms, listed below. Generated results and figures are derived from these
  data and carry the attributions given here.

## Providers

| Provider | Files | Terms (checked 2026-10-30) | Attribution required | Redistribution |
|---|---|---|---|---|
| **INE** (Statistics Portugal) | 35 API containers (`ine_*.jsonl.gz`) | Statistical information on the INE Portal is free and may be used and reused under **CC BY 4.0**, including commercial use; INE keeps its rights. Modifications must be stated. ([Termos e condições de utilização do Portal do INE](https://www.ine.pt/xportal/xmain?xpid=INE&xpgid=ine_cont_inst&INST=3585857&xlang=pt); the [API page](https://www.ine.pt/xportal/xmain?xpid=INE&xpgid=ine_api_v2&xlang=pt) states that API data are public and links to CC BY 4.0.) | "Fonte: Instituto Nacional de Estatística, IP – Portugal", with the statistic and reference period | **Permitted.** Files are the unmodified API responses, gzip-compressed, with per-request SHA-256. |
| **Eurostat** | 18 JSON snapshots (`eurostat_*.json`) and 7 retrieval records | Reuse for commercial or non-commercial purposes is authorised provided the source is acknowledged; no written licence is needed; changes must be stated (Commission Decision 2011/833/EU). The exceptions (third-party material, non-EU trade data) do not apply to these Portuguese series. ([Eurostat copyright notice](https://ec.europa.eu/eurostat/help/copyright-notice)) | "Source: Eurostat", the dataset DOI (`https://doi.org/10.2908/<DATASET>`) and the access date | **Permitted.** Files are unmodified API responses. |
| **European Central Bank** | 1 CSV and its retrieval record (`ecb_mir_*`) | Publicly available ESCB statistics "may be reused free of charge on the condition that the source is quoted". Statistics and metadata must not be modified. ([ECB statistics usage policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html)) The series is compiled by Banco de Portugal within the ESCB. | "Source: ECB statistics", with the series key `MIR.M.PT.B.A2C.AM.R.A.2250.EUR.N` | **Permitted.** The file is the unmodified API response. |
| **PORDATA** (Fundação Francisco Manuel dos Santos) | 1 CSV (`ine_pordata_completed_new_dwellings_PT_2026-10-29.csv`) | The [PORDATA legal terms](https://www.pordata.pt/Termos+Legais) (updated 19.07.2024) state that the service is protected by copyright and related rights, which users must respect, including for reproduction and making content available to the public. They name no reuse licence and note that the data come from INE, Eurostat and other official sources. | "Fonte: PORDATA", with INE as the original producer | **Not established.** See below. |

## Blocker: the PORDATA transcription

`ine_pordata_completed_new_dwellings_PT_2026-10-29.csv` holds 31 annual values: dwellings completed in new
construction for family housing in Portugal, 1995–2025. They were transcribed by hand from PORDATA's
republication of INE data, because the INE endpoint was unavailable at the time. The PORDATA terms grant no reuse
licence. The values originate with INE, whose data are under CC BY 4.0, but whether that covers a transcription
from PORDATA's service has not been established.

**Consequence.** The `release-archive` workflow task runs `python data_sources.py --check-redistribution` and
stops before building an archive while any file is "not established". The file is therefore not included,
silently or otherwise, in a public archive. It **is** present in the public GitHub repository, where it was
committed in an earlier pull request. Whether to keep it there is a decision for the maintainer.

**Where the file is used.**
- `analysis.py`: national completions 1995–2025 and completions per 1,000 residents.
- `regional.py`: a cross-check against INE's municipal series.
- `integrated.py`: the national growth-rate correlations, 2009–2024.

For 2011–2025 the values equal the sum over municipalities of INE indicator `0013248`, which is already pinned
under CC BY 4.0. Only 1995–2010 have no pinned INE equivalent.

**Routes to clear it** (one is needed before a Zenodo deposit):
1. *Replace it with INE data.* Identify INE's national annual series for completed dwellings in new construction
   for family housing back to 1995 and add it to `fetch_regional.py`. Fetch it through the manual workflow task,
   which commits the raw response with its retrieval record. Then point the three scripts at it. For 2011–2025,
   the municipal sum of `0013248` can be used directly.
2. *Obtain written permission* from PORDATA (pordata@ffms.pt) to redistribute the 31 values with attribution.
   Record the permission in this file.
3. *Shorten the national series to 2011–2025*, built from `0013248`. This would drop the 1995–2010 figures
   quoted in the report (such as the 2002 peak) and change the 2009–2024 national correlations.

After any of these, set the file's status in `data_sources.py`, rebuild `DATA_SOURCES.csv`, and rerun
`python reproduce.py`.

## Figures quoted from publications (not data files)

| Statement | Source | Verified |
|---|---|---|
| Share of households owning their home when the reference person is aged 25–34: about 70% (born 1967–1976), 60% (1977–1986), 40% (1987–1996) | Costa, Farinha, Lourenço and Mesquita (Banco de Portugal), *Boletim Económico*, October 2023, and [slides of 21 November 2023](https://www.bportugal.pt/sites/default/files/forum_economia_habitacao_portugal.pdf) (slides 10 and 12; censuses 1981–2021) | Rounded values verified on slide 12. Press reports of the bulletin give 61% and 42%; the bulletin text could not be retrieved to confirm them. |
| 2021 dwelling stock: 18.5% secondary residences, 12.1% vacant (5.8% for sale or rent, 6.3% for other reasons) | INE, Censos 2021, indicator 0011495, read through INE's JSON API on 2026-10-30 | Verified against the counts: 1,104,881 and 723,215 of 5,970,677. The OECD's *Economic Surveys: Portugal 2026*, quoted in an earlier version, was not needed and is no longer cited. |
