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
| **INE** (Statistics Portugal) | 37 API containers (`ine_*.jsonl.gz`) | Statistical information on the INE Portal is free and may be used and reused under **CC BY 4.0**, including commercial use; INE keeps its rights. Modifications must be stated. ([Termos e condições de utilização do Portal do INE](https://www.ine.pt/xportal/xmain?xpid=INE&xpgid=ine_cont_inst&INST=3585857&xlang=pt); the [API page](https://www.ine.pt/xportal/xmain?xpid=INE&xpgid=ine_api_v2&xlang=pt) states that API data are public and links to CC BY 4.0.) | "Fonte: Instituto Nacional de Estatística, IP – Portugal", with the statistic and reference period | **Permitted.** Files are the unmodified API responses, gzip-compressed, with per-request SHA-256. |
| **Eurostat** | 18 JSON snapshots (`eurostat_*.json`) and 7 retrieval records | Reuse for commercial or non-commercial purposes is authorised provided the source is acknowledged; no written licence is needed; changes must be stated (Commission Decision 2011/833/EU). The exceptions (third-party material, non-EU trade data) do not apply to these Portuguese series. ([Eurostat copyright notice](https://ec.europa.eu/eurostat/help/copyright-notice)) | "Source: Eurostat", the dataset DOI (`https://doi.org/10.2908/<DATASET>`) and the access date | **Permitted.** Files are unmodified API responses. |
| **European Central Bank** | 1 CSV and its retrieval record (`ecb_mir_*`) | Publicly available ESCB statistics "may be reused free of charge on the condition that the source is quoted". Statistics and metadata must not be modified. ([ECB statistics usage policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html)) The series is compiled by Banco de Portugal within the ESCB. | "Source: ECB statistics", with the series key `MIR.M.PT.B.A2C.AM.R.A.2250.EUR.N` | **Permitted.** The file is the unmodified API response. |

## Completed dwellings, 1995–2025: INE replaces the PORDATA transcription

Earlier versions used `ine_pordata_completed_new_dwellings_PT_2026-10-29.csv`. It held 31 annual values transcribed
by hand from PORDATA's republication of INE data, because the INE endpoint was unavailable at the time. PORDATA's
[legal terms](https://www.pordata.pt/Termos+Legais) grant no reuse licence, so its redistribution could not be
established, and the archive task refused to run. The file was removed on 30 October 2026. It remains in the
repository's git history.

**Replacement.** INE indicator [`0000076`](https://www.ine.pt/xurl/indx/0000076/PT): dwellings completed in new
construction for family housing, by NUTS 2002 and dwelling typology, annual, 1995–2025 (*Estatísticas das obras
concluídas*, last updated 17 July 2026). It is fetched for Portugal only by `python fetch_regional.py --set
national` (the workflow task `national`), and pinned unmodified with its retrieval record under CC BY 4.0.
- The Portugal totals equal the PORDATA transcription in all 31 years, so no result changes.
- For 2011–2025 they also equal INE's municipal series `0013248` summed to Portugal (`results_regional.json`).
- INE's note states that data are final up to 2020 and preliminary from 2021, and that 2024 and 2025 are based on
  the *Estimativas de Obras Concluídas*. The preliminary years 2021–2025 are now read from that note.
- The same note says that responses were incomplete for Lisboa and Seia in 2002–2005, and for Amadora in
  2011–2017, so the figures for those municipalities are understated.

**INE's long series.** INE also publishes a national long series of the same statistic,
[`0011725`](https://www.ine.pt/xurl/indx/0011725/PT) (1975–2025), attributed to the *Inquérito aos projetos de
obras de edificação e de demolição de edifícios*. It is pinned as a cross-check (`completions_long_series_check` in
`results.json`). It is higher than `0000076` in every year:
- by 2.0–4.9% in 1995–2002;
- by 24–37% in 2011–2017;
- by 11.4% in 2025 (30,425 against 27,301).

INE's published metadata does not explain the difference. The analysis keeps `0000076`, the series it has used
throughout and the national counterpart of the municipal data in the regional panel.

**Consequence.** No pinned file has an unestablished status. `python data_sources.py --check-redistribution`
passes, and the `release-archive` task keeps the check, so a future file that is not cleared stops the archive.

## Figures quoted from publications (not data files)

| Statement | Source | Verified |
|---|---|---|
| Share of households owning their home when the reference person is aged 25–34: about 70% (born 1967–1976), 60% (1977–1986), 40% (1987–1996) | Costa, Farinha, Lourenço and Mesquita (Banco de Portugal), *Boletim Económico*, October 2023, and [slides of 21 November 2023](https://www.bportugal.pt/sites/default/files/forum_economia_habitacao_portugal.pdf) (slides 10 and 12; censuses 1981–2021) | Rounded values verified on slide 12. Press reports of the bulletin give 61% and 42%; the bulletin text could not be retrieved to confirm them. |
| 2021 dwelling stock: 18.5% secondary residences, 12.1% vacant (5.8% for sale or rent, 6.3% for other reasons) | INE, Censos 2021, indicator 0011495, read through INE's JSON API on 2026-10-30 | Verified against the counts: 1,104,881 and 723,215 of 5,970,677. The OECD's *Economic Surveys: Portugal 2026*, quoted in an earlier version, was not needed and is no longer cited. |
