# Portugal's housing market, 2008–2025: a reproducible descriptive assessment

**Research release v1.0.0.** This folder holds pinned official data, standard-library Python code, tests,
figures and a research report on house prices, incomes, construction, population and rents in Portugal.

- **Report:** [`REPORT.md`](REPORT.md) is the consolidated research report: questions, institutional context,
  data, methods, results, statistical analysis, discussion, limitations and references.
- **Review log:** [`REVIEW.md`](REVIEW.md) records the scientific and technical review of this release.
- **Release notes:** [`RELEASE_NOTES_v1.0.0.md`](RELEASE_NOTES_v1.0.0.md).

The work began as a descriptive check of statistics quoted in José Jordão's 28 October 2026 article,
**Habitação, Reforma ou Revolução**. It does not establish that housing conditions cause any voting pattern or
political instability, and none of its associations is causal. The sections below document every source,
definition, check and result in detail.

## Reproducing the release

Requirements: Python 3.10 or later (CI tests 3.10 and 3.13). The analysis uses the standard library only.
Figures need matplotlib 3.11.2.

```bash
cd portugal-housing-generations
python reproduce.py                                    # data checksums, tests, rebuild, byte comparison
pip install matplotlib==3.11.2 && python reproduce.py  # also redraws the figures
```

`reproduce.py` prints `REPRODUCED` when:
1. every file in `data/` matches [`data/SHA256SUMS`](data/SHA256SUMS);
2. the unit tests pass;
3. the five analysis scripts rebuild their outputs byte for byte.

Figures are redrawn too, but byte differences in figures are reported as a warning only, because font
rendering differs between systems. Individual steps:

| Step | Command | Writes |
|---|---|---|
| National series | `python analysis.py` | `results.json` |
| Regional population and construction | `python regional.py` | `results_regional.json`, `regional_panel.csv` |
| Regional purchase affordability | `python affordability.py` | `results_affordability.json`, `affordability_panel.csv` |
| Rental market | `python rental.py` | `results_rental.json`, `rental_panel.csv` |
| Integrated assessment | `python integrated.py` | `results_integrated.json`, `integrated_panel.csv`, `data_audit.csv` |
| Figures | `python regional_figures.py`, `affordability_figures.py`, `rental_figures.py`, `integrated_figures.py`, `report_figures.py`, `article_figures.py` | `figures/*.png` |
| Tests | `python -m unittest discover -p 'test_*.py'` | — |

`test_report.py` recomputes the numbers quoted in `REPORT.md` from the result files, so the report cannot drift
from the code.

### Continuous integration and data refreshes

The workflow `.github/workflows/portugal-housing-generations.yml` runs on every pull request that touches this
folder:
- the tests, on Python 3.10 and 3.13;
- a rebuild of every result file, which fails if a committed file differs;
- the figure scripts.

Data are never downloaded while computing. The workflow's manual jobs (`fetch-mortgage-rates`, `fetch-regional`,
`fetch-affordability`, `fetch-rental`) download dated snapshots with retrieval records. They are dispatched on a
branch other than `main` and commit to that branch for review. After a reviewed refresh, run
`python reproduce.py --write-manifest` to update the checksums. The manual `release-archive` job builds the
archive for a Zenodo upload (see `RELEASE_CHECKLIST_v1.0.0.md`); it publishes nothing, and it stops while any
data file's redistribution has not been established.

Each Eurostat and INE snapshot is pinned to avoid silent changes when providers revise historical values. To
refresh a source, replace the dated snapshot, update the file name and the script and test defaults, and
review the change.

## Repository map

| Path | Contents |
|---|---|
| `data/` | 63 pinned source files (INE `.jsonl.gz` containers, Eurostat and ECB JSON with `.meta.json` retrieval records, one PORDATA transcription) and `SHA256SUMS` |
| `analysis.py`, `regional.py`, `affordability.py`, `rental.py`, `integrated.py` | Analysis pipelines (one per release stage) |
| `regional_core.py`, `affordability_core.py`, `rental_core.py`, `panel_stats.py` | Parsers, geography, index and decomposition helpers, and the panel estimators (OLS, CR1, wild cluster bootstrap) |
| `fetch_*.py` | Downloaders used only by the manual workflow jobs |
| `results*.json`, `*_panel.csv`, `data_audit.csv` | Generated results, committed and checked in CI |
| `*_figures.py`, `figures/` | Figure scripts and the 21 figures (two drawn for the Medium article) |
| `test_*.py` | Unit, snapshot, pinned-result and report-claim tests |
| `reproduce.py` | One-command reproduction |
| `REPORT.md`, `REVIEW.md`, `RELEASE_NOTES_v1.0.0.md` | Report, review log, release notes |
| `article.md` | Medium article based on the release |
| `CITATION.cff`, `.zenodo.json` | Citation and archive metadata |
| `DATA_SOURCES.md`, `DATA_SOURCES.csv`, `data_sources.py` | Data inventory, provider terms and redistribution status |
| `RELEASE_CHECKLIST_v1.0.0.md` | Release-readiness checklist and manual release steps |

## Citing and licences

Cite the release with the metadata in [`CITATION.cff`](CITATION.cff):

> Ribeiro, D. (2026). *Prices, incomes, construction and rents in Portugal's housing market, 2008–2025: a
> reproducible descriptive assessment* (Version 1.0.0) [Research release]. https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/portugal-housing-generations

No DOI exists yet. One will be added if and when the release is deposited on Zenodo.

- **Code and documentation** (scripts, tests, reports, figures and generated results) are covered by the
  repository's Apache License 2.0.
- **Third-party data** in `data/` are **not** relicensed under Apache-2.0. Each file keeps its provider's terms,
  and redistribution follows those terms. [`DATA_SOURCES.md`](DATA_SOURCES.md) gives the terms as checked on
  2026-10-30, and [`DATA_SOURCES.csv`](DATA_SOURCES.csv) gives the provider, source, retrieval time, attribution
  and status of every file:
  - INE: CC BY 4.0; cite "Fonte: Instituto Nacional de Estatística, IP – Portugal".
  - Eurostat: reuse authorised with acknowledgement of the source and the dataset DOI.
  - European Central Bank: free reuse with the source quoted, without modification.
  - PORDATA: **redistribution not established.** One transcribed file, the national completions series for
    1995–2025, is blocked from any public archive until it is replaced with INE data or permission is obtained
    (see `DATA_SOURCES.md`).

## Sources and definitions

1. Eurostat `prc_hpi_a`: Portugal, purchase `TOTAL`, unit `I15_A_AVG`,
   frequency `A`, annual house price index, 2015=100. Source JSON:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hpi_a?geo=PT&purchase=TOTAL&unit=I15_A_AVG
   Accessed 2026-10-29; source `updated` timestamp is preserved in JSON.
2. Costa, Farinha, Lourenço and Mesquita (Banco de Portugal), "Habitação em
   Portugal nos últimos 40 anos: regime de ocupação e recurso ao crédito".
   - Published in *Boletim Económico*, October 2023:
     https://www.bportugal.pt/sites/default/files/anexos/pdf-boletim/be_out2023_p.pdf
   - Slides of 21 November 2023:
     https://www.bportugal.pt/sites/default/files/forum_economia_habitacao_portugal.pdf
   - Measure: the share of households that own their home when the reference
     person is aged 25–34, from the censuses of 1981–2021.
   - Values: about 70% for the 1967–1976 cohort, 60% for 1977–1986 and 40%
     for 1987–1996. These are the rounded values on slide 12; press reports
     of the bulletin give 61% and 42%.
   - These are household shares, **not individual homeownership rates**.
3. INE, Censos 2021, indicator 0011495 (conventional dwellings by type of
   occupancy), accessed 2026-10-30 through INE's JSON API and quoted, not
   pinned:
   - 5,970,677 dwellings in total;
   - 1,104,881 (18.5%) secondary residences;
   - 723,215 (12.1%) vacant: 348,097 for sale or rent and 375,118 for other
     reasons.

   These categories do **not** indicate immediately available market supply.
   An earlier version quoted the OECD's *Economic Surveys: Portugal 2026* (12%
   and 19%); the census is the primary source.
4. Eurostat `ilc_di03`: Portugal, `statinfo=MED_EI`, `unit=EUR`,
   `age=TOTAL`, `sex=T`; annual median equivalised net income of the overall
   population, not young people's earnings:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ilc_di03?geo=PT&age=TOTAL&sex=T&statinfo=MED_EI&unit=EUR
   Accessed 2026-10-29; dataset update timestamp preserved.
5. Eurostat `ilc_di03` with identical filters but age `Y18-24` and `Y25-49`,
   pinned as two additional JSON-stat source responses. The 25–49 bracket is
   wider than millennials, and neither series isolates wage-earners or homebuyers.
   Both are repeated age cross-sections, **not longitudinal birth cohorts**.
6. Eurostat `ilc_lvho07a`, `geo=PT`, `sex=T`, `rskpovth=TOTAL`,
   `unit=PC`, ages TOTAL, Y18-24 and Y25-29:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ilc_lvho07a?geo=PT&sex=T&rskpovth=TOTAL&unit=PC
   This is the percentage of people in a household where net housing costs
   exceed 40% of disposable household income (net of housing allowances).
   Accessed 2026-10-29; raw age-category JSON-stat snapshot and metadata pinned.
7. Eurostat `ilc_lvps08`, people aged 18–34 living with parents, with
   selected groups 18–24, 18–34 and 25–34, sex total, percent:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ilc_lvps08?geo=PT&sex=T&unit=PC
8. Eurostat `yth_demo_030`, estimated age when 50% have left the parental
   home, sex total:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/yth_demo_030?geo=PT&sex=T
   Both responses accessed 2026-10-29 and pinned with their source timestamps.
9. Eurostat `ilc_lvho02`, population shares by housing tenure, selected
   `geo=PT`, `hhcomp=TOTAL`, `rskpovth=TOTAL`, `unit=PC`:
   https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/ilc_lvho02?geo=PT&rskpovth=TOTAL&hhcomp=TOTAL&unit=PC
   The source includes owner with/without outstanding housing loan and
   tenants paying market rent or reduced rent/occupying free. The percentages
   are of **people**, not dwellings or individual ownership titles.
10. Eurostat `sts_cobp_a`, annual unadjusted building permits for the *number
    of dwellings* (`indic_bt=BPRM_DW`) in residential buildings excluding
    residences for communities (`cpa2_1=CPA_F41001_X_410014`), measured in
    thousands (`unit=THS`):
    https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_cobp_a?geo=PT&indic_bt=BPRM_DW&cpa2_1=CPA_F41001_X_410014&s_adj=NSA&unit=THS
    Accessed 2026-10-29. The snapshot contains Eurostat preliminary (`p`)
    flags for 2022–2025; these are preserved in `results.json`.
11. INE, *Estatísticas das obras concluídas*, as republished by PORDATA,
    *Fogos concluídos em construções novas para habitação familiar: total e por
    tipologia do fogo*, nationwide annual total:
    https://www.pordata.pt/portugal/fogos+concluidos+em+construcoes+novas+para+habitacao+familiar+total+e+por+tipologia+do+fogo-189
    The 1995–2025 published total column was manually transcribed to the pinned
    CSV on 2026-10-29 because the INE JSON endpoint returned HTTP 503. This
    is not an original machine-generated INE export. PORDATA identifies 2021–2025
    as preliminary; those flags are retained in the CSV and output. INE
    metadata: https://ine.pt/bddXplorer/htdocs/minfo.jsp?lingua=PT&var_cd=0000076
12. Eurostat `lfst_hhnhtych`: total private households in Portugal,
    `phhcomp=TOTAL`, `agechild=TOTAL`, `n_child=TOTAL`, `unit=THS_HH`:
    https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/lfst_hhnhtych?geo=PT&unit=THS_HH&agechild=TOTAL&n_child=TOTAL&phhcomp=TOTAL
    These are household estimates from the Labour Force Survey, not an
    administrative register of dwelling addresses. Eurostat marks structural
    breaks in 2011 and 2021 (`b`). The dated source JSON retains these flags.
13. Eurostat `demo_pjan`: total usually resident population on 1 January,
    Portugal, all ages and both sexes, number of persons:
    https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/demo_pjan?geo=PT&age=TOTAL&sex=T
    This is a date-specific demographic stock, not a calendar-year average.
    Both sources accessed 2026-10-29, with source update timestamps pinned.
14. ECB MFI interest rate statistics (MIR), series
    `MIR.M.PT.B.A2C.AM.R.A.2250.EUR.N`, *Cost of borrowing for households for
    house purchase*, Portugal, monthly, percent per annum (`UNIT=PCPA`), source
    agency code `PT2` (Banco de Portugal), January 2003 to August 2026:
    https://data-api.ecb.europa.eu/service/data/MIR/M.PT.B.A2C.AM.R.A.2250.EUR.N?format=csvdata
    Retrieved 2026-10-29T14:49:43Z by `fetch_mortgage_rates.py` on a GitHub
    Actions runner (response `Last-Modified: Thu, 01 Oct 2026 08:00:00 GMT`).
    The raw SDMX-CSV is stored byte for byte, with its original CRLF line
    endings (`.gitattributes` marks it `-text`). Next to it,
    `*.meta.json` records the URL, retrieval time, HTTP status, headers and
    SHA-256; `analysis.py` refuses to run if the file no longer matches.
    Definition, from the response's `TITLE_COMPL`: annualised agreed rate
    (AAR) / narrowly defined effective rate (NDER) on *new business* lending to
    households and NPISH for house purchase, in euro, by monetary financial
    institutions; the total across rate-fixation periods is weighted by new
    business volumes with a moving average. Every observation has status `A`
    (normal); the response carries no break flags. The ECB documents one change
    in content: from reference period December 2014 renegotiated loans are also
    reported separately (Regulation ECB/2013/34), and "new business" has always
    included renegotiations of existing loans, not only first-time borrowing:
    https://data.ecb.europa.eu/methodology/bank-interest-rates
15. Banco de Portugal, macroprudential recommendation on new consumer and
    housing credit (in force since July 2018; amended version applies to
    agreements from 1 April 2022): maximum original maturity of 40 years for
    borrowers aged 30 or under, 37 years from 31 to 35 and 35 years above 35, with
    average maturity converging to 30 years by the end of 2022. Used only to
    choose the loan terms below:
    https://www.bportugal.pt/sites/default/files/documents/2024-01/31012022_Recomendacao_versao_alteradora_EN.pdf
16. INE, EU-SILC quality report for Portugal (Eurostat metadata `ilc_simsilc_pt`):
    income refers to the calendar year *before* the survey; the 2024 survey
    introduced administrative tax data for contributory old-age pensions, a break
    that lowers pension values. The Eurostat `ilc_di03` snapshots flag only a
    2004 break (`b`):
    https://ec.europa.eu/eurostat/cache/metadata/en/ilc_simsilc_pt.htm

## Relative price/income index

For each year with both series, compute

`100 * (HPI[year] / HPI[2015]) / (income[year] / income[2015])`.

This yields a *national relative price/income growth proxy* (2015=100), not
the number of median-income years required to purchase a typical home. It uses
the house-price transaction index, not a fixed dwelling's sale price, and
income from an equivalised household survey population, not borrower income.
The European Union Statistics on Income and Living Conditions income reference
period and the index's calendar year may differ; nominal index comparisons are
descriptive and should not be construed as an exact affordability measure.

`results.json` also records this index separately for the Eurostat 18–24 and
25–49 income groups, with each age group's own 2015 baseline. These index
levels should **not** be used to compare absolute affordability across ages:
the normalization erases their initial income levels, while housing needs,
household sizes, wealth, and borrowing access vary by group.

## Direct housing-cost stress measure

The `housing_overburden_pct_by_age` result adds a direct but different outcome.
It measures the share above a 40% housing-cost threshold *among all people*,
including young adults in their parents' homes. It is not a measure of the
financial barrier to leaving home or buying a first dwelling. Such selection
effects mean falling observed overburden rates can coexist with worsening
entry affordability and rising house prices. Do not equate the two trends or
infer a causal relationship from their aggregate difference.

## Leaving the parental home

`parental_coresidence_pct_by_age` records the repeated cross-sectional share
living with parents. `estimated_leaving_parental_home_age` is a model-derived
threshold (age when 50% no longer co-reside), not the mean age of individual
departure events. The 18–34 share was 62.9% in 2015, 69.8% in 2020, and
61.0% in 2025: it did not rise monotonically alongside house prices.

Neither measure identifies *why* someone remains with parents. Family norms,
education, partnership, employment and housing conditions can all matter.
These observations must not be used as proof that high prices caused delayed
emancipation or that housing access has become unproblematic.

## Tenure composition and aggregation

`population_tenure_pct` provides population shares by tenure status, with
internal sum checks allowing for published rounding. The 2015→2025 owner
share moved from 74.8% to 71.2%, while market-price renters increased from
12.7% to 16.3%. Owners with an outstanding mortgage fell from 36.5% to 31.5%
of the population; owners without one rose from 38.3% to 39.7%.

These are population *stock* categories: they cannot identify first-time
buyers, entry prices, or wealth transfers. Do not add parent categories to
their subcategories or assume they are a cohort transition matrix.

## New residential building permits

`dwellings_permitted_thousands` tracks dwellings covered by new-building
permits, not the number of permit documents or the number of homes completed.
The published series drops from 73.9 thousand dwellings in 2005 to 8.6
thousand in 2015, then rises to 40.9 thousand in 2025 (preliminary). The
2015–2025 change is about +376%, but 2015 is a depressed comparison year;
2025 remains well below the 2005 observation. Permits are neither a measure
of net additions to habitable stock nor of dwelling affordability. Annual
national permits must not be causally compared with house-price inflation:
there are completion lags, geographic differences, replacement and
demolition, and unobserved demand and financing conditions.

## Completed new family dwellings

`completed_new_family_dwellings` adds completed dwellings in *new construction*
from INE's annual works-completion statistics. Counts are units, not buildings;
they exclude rehabilitation and do not measure the net change in occupied
housing stock. They peaked at 125,708 in 2002, reached 7,148 in 2015,
and recovered to 27,301 in 2025 (preliminary). The 2015 trough makes the
2015–2025 percentage increase unusually sensitive to the baseline.

Permits and completions refer to different stages of construction, may be
revised independently, and do **not** form matched cohorts. Consequently,
dividing completions in year *t* by permits in year *t* is *not* a permit
conversion rate: a dwelling completed this year may have been permitted in
an earlier year. We report the two annual series separately and make no
causal or conversion-rate inference from their difference.

## Population, households and construction intensity

`completed_new_dwellings_construction_intensity` reports annual completed
new-family dwellings divided by two independently observed stocks: population
on 1 January and number of private households (both standardized per 1,000).
In 2015 there were 0.688 completions per 1,000 residents and 1.743 per
1,000 households; in 2025 the figures were 2.540 and 5.984, respectively.

The denominator measures need *imperfectly*. Household formation itself can
be suppressed by high housing costs, while additional demand can arise from
internal migration, location, second homes, the condition of existing homes,
and changes in household size. The private-household series is survey-based
and has Eurostat break flags in 2011 and 2021; its apparent jumps across
those years must not be interpreted as pure demographic changes. Nor should
the change in annual household stock be subtracted mechanically from annual
completions to declare a national housing shortage: the former is a stock
estimate and the latter a selected gross construction flow, and geographic
matching, replacement, vacancy, and conversion are unobserved.

## Mortgage financing cost and the initial payment

`initial_payment_to_income_index_2015_100` turns the price/income index into the
first monthly instalment on a new loan relative to income. For calendar year
*t* and base year 2015:

`100 * (HPI[t] / HPI[2015]) * (a(r[t]) / a(r[2015])) / (Y[t+k] / Y[2015+k])`

where `a(r)` is the level-payment annuity factor for a loan at annual rate *r*
(compounded monthly) over 30 years, *r* is the unweighted mean of the twelve
monthly cost-of-borrowing values (2026 has eight months and gets no annual
value), and *Y* is median equivalised net income. Two alignments are reported:
`survey_year_alignment` (*k* = 0) pairs prices with the EU-SILC survey of the
same year, as the price/income index above does; and
`income_reference_year_alignment` (*k* = 1) pairs them with the next survey,
whose income refers to year *t*, so the last year it covers is 2024. With a
constant rate the index reduces exactly to the price/income index.
`initial_payment_to_income_index_term_sensitivity_2015_100` repeats the
income-reference-year version with 25-, 35- and 40-year terms.
`initial_payment_to_income_change_log_points` splits the change between two
years into price, financing-cost and income terms (100 × natural-log ratios,
which add up exactly: total = price + financing cost − income).

What the numbers show (2015 = 100; survey-year alignment first, then
income-reference-year alignment):

- The annual mean cost of borrowing fell from 2.38% in 2015 to 0.83% in 2021,
  rose to 4.13% in 2023 and was 3.94% in 2024 and 2.99% in 2025. It was 5.43%
  in 2008.
- From 2015 to 2021 the price/income index rose to 128.4. The initial-payment
  index ended at 103.7 (108.7): lower rates offset 21.4 log points of a 52.4
  log-point price rise.
- From 2021 to 2023 the payment index rose to 183.1 (178.2), +56.9 (+49.5) log
  points. The financing-cost term, +43.5, was larger than the price term, +19.8.
- 2023 is the highest year of the series under both alignments and every term
  tested, above 2008 (161.4; 165.2), the first year of the price index and the
  year with the highest annual cost of borrowing (5.43%). In 2025 the index was
  166.5, still 66% above 2015, with prices up 24.9 log points since 2023,
  financing cost down 14.2 and income up 20.2.
- Longer terms make the payment more sensitive to the rate. With a 40-year
  term the 2023 index is 188.4; with 25 years, 172.8. Whether 2024 lies above
  2008 depends on the term: it does at 25 and 30 years (161.6 against 157.1;
  166.2 against 165.2) and does not at 35 and 40 years (170.6 against 172.9;
  174.8 against 180.3).

What it is **not**:

- Not a repayment burden of any household. The loan is assumed proportional to
  the house price index (constant loan-to-value and dwelling quality); fees,
  insurance, taxes and the deposit are left out; the rate is an AAR/NDER, not the
  annual percentage rate of charge.
- Not the payment path of a cohort. On a variable-rate or short-fixation loan
  the instalment moves with market rates after the first reset; the index
  describes the cost of entry only, at the average rate of the year of entry.
- The rate covers renegotiated as well as first-time loans and is a national
  average over borrowers who obtained credit; it says nothing about those who
  were refused, and the 2018 macroprudential limits changed who qualifies.
- The decomposition is an accounting identity, not a counterfactual. Rates and
  house prices are jointly determined, so "prices without the rate fall" is not
  identified by these series; nor does the index measure an effect of interest
  rates on household formation, ownership or voting.
- Income is one national median for the whole population. The INE-documented
  2024 break in pension values affects it, the base years differ between the
  two alignments, and young buyers' incomes are not observed here.

## Regional panel: population and new dwellings by NUTS 2024 region, 2011–2025

`regional.py` builds a region-year panel for the 26 NUTS 3 and 9 NUTS 2 regions of
the NUTS 2024 classification, the union Grande Lisboa + Península de Setúbal
(code `AML`, identical to the former Área Metropolitana de Lisboa) and Portugal.
Lisbon is reported as Grande Lisboa (`1A0`), Península de Setúbal (`1B0`) and
their union; Porto as Área Metropolitana do Porto (`11A`).

### Sources

All from INE's indicator API, downloaded on 2026-10-29 between 17:38 and 17:47
UTC by `fetch_regional.py` on a GitHub Actions runner (the `fetch-regional` job,
dispatched on a branch; this workspace cannot reach INE). Each file
`data/ine_<code>_2026-10-29.jsonl.gz` holds INE's metadata response and its data
responses byte for byte, each with URL, retrieval time, HTTP status, size and
SHA-256; `regional.py` re-checks every digest before reading. Metadata pages:
`https://www.ine.pt/bddXplorer/htdocs/minfo.jsp?var_cd=<code>&lingua=PT`.

- `0012917` Resident population, annual estimates, NUTS 2024, 2021–2025, persons,
  both sexes, all ages (INE update 2026-06-22).
- `0008273` The same estimates published on NUTS 2013, 2011–2023 (update
  2026-07-17); used for 2011–2020.
- `0013248` Dwellings completed in new construction for family housing, NUTS 2024,
  2011–2025, all typologies (*Estatísticas das obras concluídas*). INE's note:
  Amadora under-reported 2011–2017; 2024–2025 from *Estimativas de Obras
  Concluídas*; final to 2020, preliminary from 2021.
- `0013235` Dwellings licensed in new construction for family housing, NUTS 2024,
  2011–2025 (*Inquérito aos projetos de obras de edificação e de demolição*);
  preliminary from 2021 (INE's note on the same survey's NUTS 2002 table).
- `0014406` Private households / classical families, censuses, municipal series
  1864–2021 (2011 and 2021 used). 2011 counted *famílias clássicas* and 2021
  *agregados domésticos privados*; INE publishes them as one series.
- `0014627` Private households, Census 2021, NUTS 2024 (used only as a check).

Annual private households are not published below NUTS 2 (Labour Force Survey),
so households enter the panel only in the census years 2011 and 2021.

### Geography and the NUTS mapping

Every NUTS version is a grouping of the same 308 municipalities, identified by
their four-digit DICO code. The panel uses INE's NUTS 2024 hierarchy (parents are
read from INE's category order); series published on another version are
regrouped through municipalities. Checks, all in `results_regional.json`:

- Each source's municipal values sum exactly to its own published regional and
  national totals (zero mismatches in every source).
- Linking NUTS 2013 to NUTS 2024 through shared municipalities finds exactly two
  changed groups: Área Metropolitana de Lisboa = Grande Lisboa + Península de
  Setúbal, and Médio Tejo + Beira Baixa (Sertã and Vila de Rei moved between
  them); the other 22 NUTS 3 regions map one to one. No series is compared across
  the boundary change without regrouping.
- The NUTS 2013 population regrouped to NUTS 2024 is identical to INE's NUTS 2024
  figures in all 111 overlapping region-years (2021–2023), and the census household
  series regrouped to NUTS 2024 is identical to INE's 2021 NUTS 2024 table.
- National totals: completions equal the PORDATA/INE series used above in every
  year 2011–2025; population for year *t* equals Eurostat's 1 January *t*+1 for
  2011–2020 (so INE's estimates refer to 31 December).

### Panel, coverage and flags

`regional_panel.csv` has one row per region and year, 2011–2025: population,
completed and permitted dwellings, census households (2011, 2021 only), both
dwellings per 1,000 residents, and a `flags` column. A region-year with any member
municipality missing is left empty, never partially summed; in this snapshot no
cell is missing (390 of 390 NUTS 3 region-years for each annual series), so
2011–2025 is the longest common complete span. Flags record INE's notes without
changing values: `completions_underreported_amadora` (2011–2017; affects
Grande Lisboa, its NUTS 2 region, the Lisbon union and Portugal),
`construction_preliminary` (2021+), `completions_estimated` (2024–2025) and
`population_administrative_basis` (2021+).

**The population series has a method break between 2020 and 2021.** Until 2020
INE's estimates are census-based; from 2021 they are administrative-based. INE's
2021 figure for Portugal (10,599,117) is 2.0% above its 2020 figure, and from 2021
INE's series exceeds Eurostat's (1 January of the following year) by 1.7% in 2021,
3.9% in 2022, 5.3% in 2023 and 5.9% in 2024. Growth is therefore reported
separately for 2011–2020 and 2021–2025; the 2011–2025 figure in
`results_regional.json` crosses the break and is labelled so. Rates per 1,000
residents from 2021 use the higher administrative-based population.

### What the panel shows

Population (average annual change):

- 2011–2020: only five NUTS 3 regions grew: Algarve (+0.42%), Península de
  Setúbal (+0.36%), Grande Lisboa (+0.19%), Cávado (+0.19%) and Oeste (+0.02%).
  Portugal fell by 0.17% a year and Área Metropolitana do Porto by 0.11%. The
  largest falls were in Alto Alentejo (−1.24%), Beiras e Serra da Estrela
  (−1.15%), Alto Tâmega e Barroso (−1.04%) and Douro (−1.03%).
- 2021–2025: every region grew. Fastest: Algarve (+3.29%), Alentejo Litoral
  (+3.09%), Península de Setúbal (+3.06%), Oeste (+2.60%), Grande Lisboa (+2.54%)
  and Lezíria do Tejo (+2.40%); Portugal +1.89%, Área Metropolitana do Porto
  +1.37%; slowest Açores (+0.44%), Alto Tâmega e Barroso (+0.45%) and Alto Alentejo
  (+0.46%).
- Census households 2011–2021: Portugal +2.6%, Grande Lisboa +2.9%, Península de
  Setúbal +6.8%, Área Metropolitana do Porto +4.4%, Algarve +6.2%.

Construction (dwellings per 1,000 residents; Portugal first):

- Completions fell from 2.50 in 2011 to 0.69 in 2015 and rose to 2.39 in 2025;
  permits fell from 1.67 to 0.67 (2014) and rose to 3.68. Nationally, permits
  have exceeded completions every year since 2015.
- Mean 2021–2025: completions 2.07 in Portugal, 1.26 in Grande Lisboa, 2.15 in
  Península de Setúbal (1.51 for the union) and 2.71 in Área Metropolitana do
  Porto; permits 3.11, 2.03, 2.67 (2.20) and 4.52. Relative to Portugal, Grande
  Lisboa's completion rate was 0.58 in 2011–2015, 0.44 in 2016–2020 (both lowered
  by the Amadora gap) and 0.61 in 2021–2025; Porto's was 0.79, 0.85 and 1.31.
- Highest completions per 1,000 in 2021–2025: Cávado (3.53), Ave (2.84), Oeste
  (2.73), Área Metropolitana do Porto (2.71); lowest: Baixo Alentejo (0.91),
  Alentejo Central (1.02), Alto Alentejo (1.07). Dispersion across NUTS 3
  (coefficient of variation) stayed between 0.28 and 0.38 for completions and 0.28
  and 0.44 for permits, without a trend.
- The regions that grew fastest in 2021–2025 are not those with the most
  construction per resident: Grande Lisboa grew faster than Portugal and built
  well below the national rate, while Cávado, Ave and Porto built the most.

Figures: `figures/fig_regional_completions_per_1000.png`,
`fig_regional_permits_and_completions.png`, `fig_regional_population_growth.png`
and `fig_regional_growth_and_construction.png`.

### What the panel does not show

- **Construction per resident is not a measure of housing shortage or need.** It
  ignores the existing stock, vacancy, second homes, demolitions, conversions,
  household size, prices, incomes and where people would live if they could; a low
  rate can reflect scarce land, regulation or weak demand alike.
- Associations between regional population growth and construction are
  descriptive. Both can respond to employment, migration, prices, credit and land,
  and causality can run either way.
- Permits and completions are different stages, not matched cohorts; a permit
  ratio above completions does not say how many permitted dwellings will be built.
- Rates use the population of the same year (31 December). Before and after 2021
  they rest on different estimation methods, and regional growth after 2021 depends
  on how administrative records place recent migrants.
- NUTS 3 averages hide large differences between municipalities; the municipal
  values are in the snapshots for finer work.

## Regional affordability: transaction prices and declared income, 2019–2024

`affordability.py` builds a region-year panel of house prices and incomes for
Portugal, the 9 NUTS 2 and 26 NUTS 3 regions of NUTS 2024 and the 308
municipalities, and compares their growth. It writes `results_affordability.json`
and `affordability_panel.csv`; `affordability_figures.py` draws the figures.

### Sources

From INE's indicator API, downloaded on 2026-10-29 between 19:22 and 19:38 UTC by
`fetch_regional.py --set affordability` on a GitHub Actions runner. They are stored
and checked like the regional panel's sources: one container per indicator, every
response with its SHA-256. INE's last updates were 2026-10-23 (prices), 2026-07-24
(income) and 2026-04-11 (appraisals). The 2018-method table was last updated in
2022 and is closed.

| Code | Statistic | Geography | Periods used |
|---|---|---|---|
| `0012234` | Median value per m² of family dwellings sold in the last 12 months, by dwelling category (total, new, existing); 2022 method | NUTS 2024 to municipality | Q4 2019–Q4 2025 |
| `0009490` | The same statistic, 2018 method | NUTS 2013 to municipality | Q4 2016–Q4 2020 |
| `0012256` | Median bank appraisal value per m², by construction type (total used) | Municipality 2024 | 2019–2025 |
| `0012712` | Median declared gross income per tax household | NUTS 2024 to municipality | 2019–2024 |
| `0012741` | Median declared gross income net of IRS per tax household | NUTS 2024 to municipality | 2019–2024 |
| `0012749` | Median declared gross income per taxpayer | NUTS 2024 to municipality | 2019–2024 |
| `0009763` | Median declared gross income per tax household | NUTS 2013 to municipality | 2016–2021 |

**Which price is this?** Three different things are called a house price:

- *Transaction prices* (used here): INE's local housing-price statistics are
  built from the tax authority's administrative records of dwelling sales. They
  give the median value per m² of the family dwellings actually sold, over the 12
  months ending in the reference quarter. The fourth-quarter value therefore
  covers the calendar year, which is how it is aligned with annual income. Where
  INE publishes a cell without a value, it stays empty here.
- *Bank appraisal values* (sensitivity check only): the median per m² that banks'
  valuers assign to dwellings offered as mortgage collateral. They are neither
  prices paid nor asking prices, cover only mortgage-financed purchases and follow
  the valuers' methods.
- *Advertised asking prices*: INE publishes none. They come only from private
  property portals, with their own coverage, duplicates and listing biases, and
  are not used.

**Which income?** INE's local income statistics come from personal income-tax
returns (IRS), as produced by the tax authority, for the year. The headline uses
the median *declared gross income per tax household* (agregado fiscal: the
household covered by one return). It is the pre-tax income declared on the
return, and it excludes anyone who does not file. It is not disposable income, not
equivalised and not the income of buyers. The net-of-IRS household median and the
gross median per taxpayer are used as checks.

### Measures and the longest reliable panel

The 2022-method prices start with the 12 months to 2019 Q4 and income ends in
2024, so **2019–2024 is the longest window in which both are published on one
method and one geography**. Price-only figures go to 2025. Nothing is spliced
across the 2018 and 2022 price methods or the 2013 and 2024 NUTS versions; the
earlier window is reported separately.

- *Square metres per year of income* = median income / median price per m². It
  reads as the floor area that one year of the median household's gross income
  would pay for at the median price per m². Lower values mean prices are high
  relative to incomes.
- *Price/income index* = (P*t*/P*base*) / (Y*t*/Y*base*) × 100, with base years
  2019, 2020 and 2021. It measures a relative change, not a level.
- *Price-minus-income gap* = price growth minus income growth in log points. It
  equals the log change of the ratio, so a positive gap means prices rose faster
  than incomes.

**Neither measure is the number of years needed to buy a dwelling.** Both divide
one median by another median from a different population (dwellings sold vs. tax
households). They ignore dwelling size, deposits, mortgage rates and terms, taxes
and transaction costs, savings and wealth, and the fact that buyers are not a
random sample of tax households. The index is normalised, so 110 means the
price/income ratio rose 10%, not that a dwelling costs 110% of anything.

Medians are never summed or averaged across areas. Every value is INE's own
median for that area. So the former Área Metropolitana de Lisboa exists only in
the NUTS 2013 series, and Lisbon is reported as Grande Lisboa (`1A0`) and
Península de Setúbal (`1B0`), with Porto as Área Metropolitana do Porto (`11A`).

### Checks

All recorded in `results_affordability.json`; the build stops if any fails.

- The price and the three NUTS 2024 income tables assign the 308 municipalities to
  the same 26 NUTS 3 and 9 NUTS 2 regions. The five NUTS 2 regions made of a single
  NUTS 3 region (Grande Lisboa, Península de Setúbal, Algarve, Açores, Madeira)
  carry identical medians at both levels.
- Portugal and every NUTS 2 and NUTS 3 region have a price and an income in every
  year 2019–2024. Gaps occur only in municipalities.
- **Price method, 2018 vs 2022.** The two methods overlap in the 12 months to Q4
  2019 and Q4 2020. In the 22 NUTS 3 regions that NUTS 2024 left unchanged, the
  median absolute difference is 1.2% and all 44 cells are within 5% (largest
  −3.0%). Across municipalities, matched by DICO, the median difference is 1.9% and
  80% of cells are within 5%. Small municipalities can differ much more (+75% in
  one Douro municipality in 2019), which is why the two methods are not spliced.
- **Income, NUTS 2013 vs 2024.** For 2018–2021, 1,183 of 1,190 municipal cells are
  identical and the other 7 differ by €1. All 88 cells of the unchanged NUTS 3
  regions are identical. Only the grouping changed between the two tables.
- **Coverage.** A municipal price is published for 286 of 308 municipalities in
  2019, 282 in 2020 and 298–305 from 2021. Municipal income is published for
  297–299. Both are available for 283 municipalities in 2019 and 299 in 2024, and
  for 272 in every year 2019–2024. The cells without a value carry INE's sign
  "Dado nulo ou não aplicável" and are left empty. In 2024 the five municipalities
  without a price were Barrancos, Corvo, Lajes das Flores, Penedono and Santa Cruz
  das Flores.

### What the panel shows (2019–2024, nominal)

**Portugal.** The median price per m² of dwellings sold rose from €1,094 to €1,777
(+10.2% a year), reaching €2,076 in 2025. Median gross household income rose from
€12,499 to €15,573 (+4.5% a year). Prices grew 26.5 log points faster than
incomes, so the price/income ratio rose 30%. One year of median income paid for
11.4 m² at the median price in 2019 and 8.8 m² in 2024.

**Prices outpaced incomes almost everywhere.** This held in all 9 NUTS 2 regions,
21 of the 26 NUTS 3 regions and 220 of 283 municipalities (215 of the 272 with
every year published). The five NUTS 3 exceptions are low-price interior regions,
where prices barely moved: Beira Baixa (gap −21.6 log points), Douro (−18.8), Viseu
Dão Lafões (−12.0), Beiras e Serra da Estrela (−11.1) and Terras de Trás-os-Montes
(−10.2).

**Price growth was not accompanied by matching income growth.** Across NUTS 3
regions the correlation between price and income growth is negative: −0.34
(Pearson; Spearman −0.29), and −0.21 across municipalities. Regional income growth
was compressed between 2.8% and 6.1% a year, while price growth ranged from 0.4%
to 14.6%. Year by year, median NUTS 3 income growth rose from 1.5% (2019–20) to 7.1%
(2023–24). The number of regions where prices outpaced incomes fell from 24 in 2019–20 to 13 in 2022–23 and
16 in 2023–24.

**Lisbon and Porto.**

- Grande Lisboa: prices +10.1% a year, but income +2.8%, the slowest of the 26
  regions. The gap is 34.3 log points (ratio +41%), and 5.7 m² per year of income
  in 2024 (8.0 in 2019), 0.65 times the national value.
- Península de Setúbal: prices +13.1% and income +3.3%, the largest gap of any
  NUTS 3 region (45.5 log points; ratio +58%). It fell from 12.3 to 7.8 m².
- Área Metropolitana do Porto: prices +12.4% and income +5.0%, a gap of 34.0 log
  points (ratio +40%). It fell from 11.3 to 8.0 m².
- The lowest values in 2024 were Algarve (5.0 m²), Grande Lisboa (5.7), Madeira
  (6.4), Alentejo Litoral (7.5), Península de Setúbal (7.8) and Porto (8.0). The
  highest were Beira Baixa (25.7) and Alto Alentejo (25.4).

**Disparities widened through prices, not incomes.** Across NUTS 3 regions, the
coefficient of variation of prices rose from 0.39 in 2019 to 0.50 in 2024 (0.48 in
2025), and the highest-to-lowest ratio rose from 4.2 to 5.0. Income dispersion fell
(0.088 to 0.068; ratio 1.48 to 1.28). The m²-per-income-year proxy therefore
diverged, with the coefficient of variation going from 0.29 to 0.43 and the ratio
from 4.0 to 5.1.

### Sensitivity

- **Base year.** At 2024 the price/income index is above 100 in 21 of 26 NUTS 3
  regions with base 2019, 18 with base 2020 and 20 with base 2021. The ranking of
  regions barely changes (Spearman 0.99 and 0.97 against base 2019), but the level
  does: Grande Lisboa is 141, 128 or 122 and Portugal 130, 121 or 115. The jump in
  2020, when price growth was strong and income growth weak, is what a later base
  removes. These numbers are relative changes and depend on the base; they are not
  levels.
- **Definitions** (price minus income growth, NUTS 3):
  - Net-of-IRS household income gives the same 21 of 26 regions (rank correlation
    with the headline 0.999).
  - Gross income per taxpayer gives 19 of 26 (0.997).
  - Existing-dwelling prices give 20 of 26 (0.98), and new-dwelling prices also 20
    of 26 but with a different ranking (0.83), since new-build sales are few and
    vary in location and quality.
  - Under all nine combinations, the gap for Grande Lisboa, Península de Setúbal and
    Porto stays between 27 and 46 log points, and for Portugal between 21 and 29.
- **Transactions vs bank appraisals.** Across NUTS 3, appraisal values rose by a
  median 7.9% a year against 8.5% for transaction prices (correlation of regional
  growth rates 0.82). In 2024 the median appraisal was 1.02 times the transaction
  median. With appraisal values in place of prices, prices outpace incomes in 24 of
  26 regions. In Lisbon, Setúbal and Porto, appraisals rose 1.4–2.4 points a year
  less than transactions (Grande Lisboa 8.7% vs 10.1%), but still far faster than
  incomes.
- **Earlier window, 2016–2020** (2018-method prices with NUTS 2013 income, never
  joined to the main window):
  - Prices outpaced incomes in 24 of 25 NUTS 3 regions.
  - Portugal: prices +8.2% a year and income +2.9%.
  - The former Área Metropolitana de Lisboa: +8.4% and +2.1%, with m² per year of
    income falling from 11.2 (2016) to 8.8 (2020).
  - Porto: +11.0% and +3.0%.
  - The divergence did not start in 2019.

### Limitations

- **The m² per year of income measure and the index are proxies, not the years
  needed to buy a dwelling.** They divide medians of two different populations
  (dwellings sold, tax households). They take no account of dwelling size,
  deposits, interest rates, loan terms, taxes, transaction costs, savings or
  wealth.
- **The composition of both populations changes.** Sales shift between locations,
  sizes and ages of dwelling within a region. Tax households change with migration,
  ageing and filing behaviour. A change in a median is not a change for a typical
  household that stayed put.
- **The income is not that of the buyers.** Buyers include non-residents,
  companies and households from other regions, and they buy where they choose. The
  median local tax household need not be a potential buyer.
- **Declared gross income is pre-tax and not equivalised.** It covers only
  households that file an IRS return. The net and per-taxpayer tables also exclude
  households whose income net of IRS is negative (INE's note). It is not disposable
  income or the income concept of the national accounts.
- **All values are nominal.** Ratios are unaffected by inflation, but growth rates
  are not real growth.
- **Medians for small municipalities rest on few sales** and are noisy (see the
  method comparison). The five NUTS 3 regions where incomes outpaced prices are
  thin markets.
- **Renters and financing costs are outside this panel.** Rents of new lease
  contracts and the cost of borrowing (national, above) would be needed for a fuller
  picture.
- **The panel is descriptive.** It does not identify why prices rose (demand from
  residents, non-residents or tourism, credit, supply, regulation), and the
  negative regional correlation between price and income growth is not a causal
  relationship.

Figures: `figures/fig_affordability_price_vs_income_growth.png`,
`fig_affordability_m2_per_income_year.png`, `fig_affordability_index_base_years.png`
and `fig_affordability_definitions.png`.

## Rental market: rents, tenure and housing-cost burden

`rental.py` sets three rent measures side by side, plus tenure and housing-cost
burden. It writes `results_rental.json` and `rental_panel.csv`;
`rental_figures.py` draws the figures. Nothing is spliced across measures,
methods or geographies, and nothing here identifies a cause. Immigration,
tourism, investment demand and regulation are not tested.

### Three rent measures and the populations behind them

| Measure | What it covers | Population | Source |
|---|---|---|---|
| Median rent per m² of **new lease contracts** | Contracts for family dwellings registered with the tax authority in the year | Dwellings let in that year (movers) | INE, *Estatísticas de rendas da habitação ao nível local* |
| **Mean monthly rent** at the census | All rented dwellings of usual residence, old and new contracts, social rents included | Every tenancy on census day | INE, Censos 2001, 2011, 2021 |
| **HICP actual rentals** (COICOP 04.1) | Average rent change for a sample of rented dwellings; new and existing contracts; private, public and social units (HICP Methodological Manual 2024, §12.4) | Rented dwellings in INE's rent-survey panel (an index, not a level) | Eurostat `prc_hicp_aind` |
| Rent share, tenure, overburden | Rent and housing costs as reported in a household survey | Persons in private households | Eurostat EU-SILC |

**Advertised asking rents** are published by no official source. Property-portal
listings have their own coverage, duplicates and listing biases, and are not
used. The contracted rents above are what tenants agreed to pay, as registered.
The HICP is a price index with no rent level. In Portugal it comes from INE's
monthly rent survey, a panel of rented dwellings stratified by region, rent class
and contract date. It follows dwellings, not households, so a re-let dwelling's new
rent enters it. It cannot describe the rents or rent burden of a fixed group of
sitting tenants (see `REPORT.md`, Section 6.4).

### Sources

All series were downloaded on 2026-10-29 by `fetch_regional.py --set rental` on
a GitHub Actions runner: Eurostat at 21:04 UTC, INE between 21:13 and 22:06 UTC.
Every INE response is stored with its SHA-256, and every Eurostat body with a
`.meta.json` retrieval record. `rental.py` checks all of them before reading.

- **INE, new leases** (latest update 2026-09-29):
  - `0014698` median rent per m², 2026 method, NUTS 2024, 2020–2025;
  - `0014697` number of contracts;
  - `0014711` quartiles;
  - `0014700`/`0014699` contracts whose tenant is resident in Portugal, regions
    and municipalities over 100,000 inhabitants, 12 months to Q4;
  - `0012600`/`0012601` the 2021 method on NUTS 2024, 2020–2024;
  - `0009631`/`0009632` the 2021 method on NUTS 2013, 2017–2023.
- **INE, censuses:**
  - `0012521`, `0006993`, `0000977` share of dwellings rented, 1991–2021;
  - `0012522`, `0006995`, `0000777` mean monthly rent, 2001–2021;
  - Census 2021 tenancies by contract date (`0012508`), rental support
    (`0012529`; `0011522` for unsupported tenancies by contract date), years in
    the dwelling (`0012528`) and rent band by contract date for Portugal
    (`0011501`).
  - The 1991/2001 tables carry 3 rows for parish codes absent from INE's own
    metadata; they are counted and skipped.
- **INE, overburden by NUTS 2** (EU-SILC as published by INE): `0006260` on
  NUTS 2013, 2004–2025, and `0014472` on NUTS 2024, 2024–2025.
- **Eurostat EU-SILC** (update 2026-10-20):
  - `ilc_lvho07c` overburden by tenure;
  - `ilc_lvho02` tenure by poverty status;
  - `ilc_mded02` share of rent in disposable income;
  - `ilc_di09` median income by tenure;
  - `ilc_lvho25` overburden by citizenship (adults);
  - `ilc_lvho07d` overburden by degree of urbanisation.
- **Eurostat HICP:** `prc_hicp_aind`, CP041 and CP00, Portugal and the euro
  area, 1996–2025 (update 2026-02-06).
- **Already pinned:** INE IRS income (`0012712`) and EU-SILC median income
  (`ilc_di03`). EU-SILC income refers to the year before the survey, and both
  alignments are reported.

### Checks

- New-lease medians are complete for Portugal and all 9 NUTS 2 and 26 NUTS 3
  regions in 2020–2025. Municipal medians are published for 222 of 308
  municipalities in 2020 and 247 in 2025, and 217 have both rent and income in
  every year 2020–2024. INE's second quartile equals its published median in
  every cell.
- **Method 2021 vs 2026** (2020–2024): the median NUTS 3 difference is 2.1%, and
  75% of cells are within 5%. For Portugal the 2026 method is 3.2–6.3% higher, and
  annual growth over 2020–2024 is 10.0% against 9.2% under the 2021 method.
  Only the 2026 method is used for 2020–2025.
- The 2021-method tables on NUTS 2013 and NUTS 2024 are identical in all 88
  cells of the 22 regions that did not change.
- The EU-SILC overburden rates by tenure, weighted by the tenure shares,
  reproduce the published total within 0.1 points in every year 2005–2025.
- Census 2021 tenancy counts add up across contract periods and support
  categories in every region.

### What the data show

**New leases rose faster than incomes almost everywhere (2020–2024).**

- *Portugal:* the median rent per m² of new contracts rose from €5.79 to €8.47
  (+10.0% a year) and reached €9.29 in 2025. Median gross household income (IRS)
  rose 5.5% a year, so rents outpaced income by 16.6 log points (+18% in the
  ratio).
- *Regions:* rents outpaced income in 8 of 9 NUTS 2 regions, 24 of 26 NUTS 3
  regions and 190 of 217 municipalities. The exceptions were the Açores and
  Alto Tâmega e Barroso.
- *Growth rates were unrelated:* across NUTS 3 regions, rent growth and income
  growth were weakly negatively correlated (−0.24), and uncorrelated across
  municipalities (−0.05).
- *Disparities were broadly stable:*
  - the spread of new-lease rents across NUTS 3 regions changed little
    (coefficient of variation 0.34 in 2020, 0.36 in 2025);
  - within Portugal, the third-to-first quartile ratio moved from 2.34 to 2.40.
- *The same pattern appears earlier:* under the 2021 method, Portugal's new-lease
  rents rose 10.1% a year in 2017–2019, the former Área Metropolitana de Lisboa's
  15.4% and Porto's 12.1%.

**Lisbon and Porto.**

| Area | Rent €/m², 2020 → 2025 | Rent growth 2020–24 | Income growth 2020–24 | m² per month of income, 2020 → 2024 |
|---|---|---|---|---|
| Grande Lisboa | 9.38 → 14.19 | 9.2% | 3.6% | 129 → 104 |
| Lisboa (municipality) | 11.54 → 16.88 | 9.0% | 2.9% | 114 → 90 |
| Península de Setúbal | 6.63 → 11.05 | 11.3% | 4.1% | 176 → 135 |
| Área Metropolitana do Porto | 6.14 → 10.17 | 10.5% | 6.2% | 171 → 146 |
| Porto (municipality) | 8.97 → 14.13 | 10.2% | 3.1% | 128 → 98 |
| Portugal | 5.79 → 9.29 | 10.0% | 5.5% | 181 → 153 |

*m² per month of income* is the median gross household income for one month
divided by the median new-lease rent per m². It is a ratio of medians from two
different populations, not the rent any household pays.

**The HICP rent index and the stock of tenancies tell a different story.**

- *HICP index:* actual rentals rose 2.5% a year in 2001–2011, 2.1% in 2011–2021
  and 4.9% in 2021–2025 (euro area 2.3% in 2020–2025). Measured against EU-SILC
  median equivalised income (income-year alignment), this index fell to 79 in
  2024 (2015 = 100), or 90 with base 2019. This compares two indices with
  different populations. It does not show that rent burdens fell for tenants who
  kept their contracts. EU-SILC's direct measure, the share of rent in tenant
  households' disposable income, was 23% in 2015 and 24% in 2025.
- *Census mean rent:* it rose faster than the index, from €123 (2001) to €235
  (2011) and €334 (2021), i.e. +6.7% then +3.6% a year. In the Lisboa
  municipality it went from €118 to €269 and €471; in Porto from €94 to €191 and
  €318. A stock average and a price index can diverge because dwellings enter
  and leave the rental stock, because location, size and quality shift, and
  because old contracts are replaced by new ones in proportions the index's
  weights need not follow. These data cannot separate those sources.
- *Contract dates in 2021:*
  - 45% of the 922,810 tenancies had been signed in 2017–2021, 20% in 2012–2016
    and 16% before 1991;
  - in the Lisboa and Porto municipalities, 24% and 26% of unsupported tenancies
    were pre-1991 contracts;
  - pre-1987 contracts had a median rent band of €100–150, with 49% below €100;
    contracts from 2017–2021 had a median band of €300–400, with 30% at €500 or
    more.
- *Rent share:* EU-SILC's share of rent in disposable household income
  (`ilc_mded02`) rose from 16% (2004) to 21% (2019) and 24% (2025); below the
  poverty threshold it was 33%.

**Who pays market rents.**

- *EU-SILC:* tenants at market rent were 10.8% of the population in 2004, 13.2%
  in 2019, 10.9% in 2021 (pandemic-year surveys) and 16.3% in 2025. Tenants at a
  reduced rent or free fell from 14.6% to 12.5%. Market-rent tenants went from
  43% to 57% of all tenants. Among people below the poverty threshold, the
  market-rent share rose from 12.6% to 23.9%.
- *Censuses:* the share of dwellings rented fell from 27.5% (1991) to 19.9%
  (2011) and rose to 22.3% (2021). In the Lisboa municipality it fell from 60% to
  42%, and in Porto from 59% to 44%.
- *Support in 2021:* 93.1% of tenancies had no rental support. 4.3% paid a
  social or supported rent and 2.5% received a rent subsidy. "Without support"
  is not the same as "at market rent": contracts from before 1990 were signed
  under earlier rent laws, and the census does not say how far their rents are
  from market levels.

**Housing-cost overburden** (housing costs above 40% of disposable income):

- *Trend:* 4.3% of the population in 2005, 9.2% at the 2014 peak, 4.1% in 2020
  and 6.2% in 2025.
- *By tenure:* market-rent tenants are by far the most exposed: 22.5% in 2005,
  35.8% in 2012, 26.3% in 2019 and 27.2% in 2025. Owners with a mortgage were at
  3.0% in 2025.
- *Regions (INE, NUTS 2):* the former Área Metropolitana de Lisboa went from 6.2%
  (2018) to 9.0% (2025). On NUTS 2024 in 2025, the highest rates were Grande
  Lisboa (10.0%) and Algarve (8.6%), the lowest Alentejo (3.7%). Cities were at
  7.7% and rural areas at 3.3%.
- *Composition:* between 2019 and 2025 the national rate rose 0.53 points. The
  shift towards market-rent tenure accounts for +0.72 points, and rates within
  each tenure group for −0.19. Over 2015–2025 the rate fell 2.84 points: −3.75
  within groups, +0.92 composition.

### Composition and selection

Several features of the data can move these measures without any change in what
a given household pays:

- **Movers vs stayers.** New-lease medians describe the dwellings let that year
  and the people who moved. A change in the mix of locations, sizes or quality
  of the dwellings let moves the median.
- **Turnover in the stock.** The census mean rises when old, low-rent contracts
  end, even if no existing rent changes. The HICP tries to hold the tenancy
  constant, which is why it rises much less.
- **Tenure composition.** More people in market-rent tenure arithmetically raises
  the national overburden rate even when each group's rate is flat. In the
  shift-share decomposition it accounts for all of the 2019–2025 rise; this is an
  accounting identity, not an explanation of why tenure changed.
- **Who rents.** Renters' median equivalised income stayed between 0.74 and 0.86
  times owners' since 2004 (0.75 in 2025), so the gap between the two groups
  shows no trend.
- **Non-resident tenants.** About 3% of new contracts in 2025 had a tenant not
  resident in Portugal. Excluding them changes the 2025 medians by a median 0.3%,
  and at most 1.2% (Lisboa municipality).
- **Citizenship.** Adults with foreign citizenship had a far higher overburden
  rate than nationals (35% vs 4% in 2025). That series rests on small samples:
  it fell to 10% in 2018, and Eurostat flags a break in 2021. A growing group
  with high overburden can raise the national rate by composition, but the
  shares needed to measure this are not in these tables. No conclusion about
  the effect of immigration follows.
- **Survey changes.** EU-SILC 2020–2021 was collected during the pandemic, and
  the 2024 survey changed how pensions are recorded.

### Limitations

- The three rent measures describe different populations and are not
  interchangeable. A rent per m² cannot be converted into a monthly rent without
  dwelling sizes.
- Medians of new contracts are published only where enough contracts were
  registered. The 61–86 municipalities without a value are left empty.
- IRS income is gross income per tax household, not tenants' income or
  disposable income. EU-SILC income is equivalised disposable income of all
  persons, with a one-year lag.
- EU-SILC is a sample survey; the regional, citizenship and tenure breakdowns
  have wide margins of error.
- Informal tenancies that are never registered are missing from the new-lease
  statistics.
- Everything here is descriptive. Changes in rents, tenure or burden are not
  attributed to immigration, tourism, investment demand, short-term letting or
  regulation: none of these is identified by these data.

Figures:
- `figures/fig_rental_three_measures.png`
- `figures/fig_rental_tenure.png`
- `figures/fig_rental_overburden.png`
- `figures/fig_rental_regional_growth.png`
- `figures/fig_rental_census_contracts.png`

## Integrated assessment: prices, income, construction, population and rents

`integrated.py` joins the series pinned by the earlier analyses into one
municipality-year panel and asks how house prices relate to income, construction,
population and rents. It downloads nothing. It writes:
- `data_audit.csv`, the audit of every variable;
- `integrated_panel.csv`, 308 municipalities × 2011–2025, NUTS 2024 geography,
  municipalities matched by DICO code;
- `results_integrated.json`, every estimate, interval and diagnostic.

`integrated_figures.py` draws the figures. The estimators are in
`panel_stats.py`, written in the standard library so every step can be checked.
The run is deterministic: the bootstrap seeds are fixed, and CI rebuilds the
outputs and compares them.

Findings come in three layers, kept apart below:
- **robust descriptive** facts;
- **model-dependent** associations;
- **questions that need a causal design.** These data cannot answer them.

No correlation or coefficient here is a causal effect.

### Audit

| Variable | Source | Years | Municipalities published | Alignment | Breaks and notes |
|---|---|---|---|---|---|
| population | INE 0008273 + 0012917 | 2011–2025 | 308 every year | 31 December | Census-based to 2020, administrative-based from 2021 (+2.0% for Portugal in one step) |
| completions, permits | INE 0013248, 0013235 | 2011–2025 | 308 | calendar year | Amadora under-reported 2011–2017; 2021+ preliminary; completions 2024–2025 estimated |
| price per m² (sales) | INE 0012234 | 2019–2025 | 282–305 | 12 months to Q4 | 2022 method; existing dwellings 272–302 |
| price per m² (2018 method) | INE 0009490 | 2016–2020 | 303–306 | 12 months to Q4 | closed series, NUTS 2013 table |
| bank appraisal per m² | INE 0012256 | 2011–2025 | 141–261 | calendar year | suppressed where appraisals are few: selected, larger municipalities |
| income per tax household | INE 0012712 + 0009763 | 2015–2024 | 297–299 | tax year | two geography tables of one statistic; 1,183 of 1,190 overlapping cells identical, the rest differ by €1 |
| new-lease rent per m² | INE 0014698 | 2020–2025 | 222–247 | contracts of the year | 2026 method; the 2021 method (2017–2023, 193–215) is kept separate |

All values are positive except completions and permits, where zeros are real.
Units are not mixed:
- prices and appraisals are € per m²;
- rents are € per m² per month;
- income is € per year;
- population is persons;
- construction is dwellings, and becomes a rate per 1,000 residents of a fixed
  base year, so the population break cannot leak into it.

All calendar alignments are annual. Two exceptions are handled explicitly:
- EU-SILC income refers to the year before the survey and is shifted.
- Eurostat's 1 January population is assigned to 31 December of the year before.

### Analytical samples

Each sample uses one price method, one population basis and one geography, and is
balanced: a municipality enters only if every variable is published in every
year. A value that is published, even an imprecise one, is never dropped.

| Sample | Years | Outcome | Regressors | Municipalities |
|---|---|---|---|---|
| A | 2019–2024 | sale price (2022 method) | income | 272 |
| A2 | 2021–2024 | sale price | income, completions per 1,000, population (administrative basis only) | 289 |
| B | 2016–2020 | sale price (2018 method) | income, completions per 1,000, population (census basis only) | 295 |
| C | 2015–2024 | bank appraisal | income, completions per 1,000 | 179 |
| D | 2020–2024 | sale price | new-lease rent | 215 |

**Methods.**

- *Estimand.* The within-municipality association between log price and each
  regressor, conditional on municipality effects (time-invariant differences)
  and year effects (shocks common to all municipalities). It is a conditional
  association, not a causal effect.
- *Specifications compared:* two-way fixed effects; first differences with year
  effects; NUTS 2 × year effects (regional shocks); long differences across the
  whole window with NUTS 2 effects; existing-dwelling prices; net income; and
  sub-periods.
- *Inference:*
  - cluster-robust (CR1) standard errors by municipality;
  - a wild cluster bootstrap-t by NUTS 3 region (26 clusters, Rademacher
    weights, 1,999 replications) to allow for spatial correlation within
    regions. The intervals quoted below come from the unrestricted bootstrap
    (WCU) unless stated otherwise. P-values come from the restricted bootstrap
    (WCR), which imposes the null hypothesis when resampling and is more
    reliable with few clusters (MacKinnon and Webb, 2017). The two p-values
    differ by at most 0.024, and none falls on the other side of 0.05.
- *Diagnostics:*
  - *Within variation:* the share of each regressor's variance left after the
    fixed effects.
  - *Serial correlation:* Wooldridge's test on first-difference residuals. A
    slope near −0.5 supports fixed effects; near 0, first differences.
  - *Stability:*
    - fixed effects against first differences;
    - leaving out each NUTS 3 region in turn;
    - dropping the Lisbon and Porto metropolitan municipalities;
    - dropping the 5% most influential municipalities (one-step DFBETA).
- *Verdicts.* An association is called **robust** only if all of these hold:
  - the regressor keeps at least 1% of its variance after the fixed effects;
  - the bootstrap interval excludes zero and the restricted-bootstrap p-value is
    below 0.05;
  - the sign holds in first differences, without any single NUTS 3 region and
    without the metropolitan areas;
  - the sign is shared by at least 80% of all specifications of the same
    regressor.

  It is **fragile** if the bootstrap test rejects zero but some other check
  fails, and **not supported** otherwise. The rule is mechanical: the completions
  association passes it, but it is absent for existing dwellings and is not
  identified as a price effect (see below).

### Layer 1: robust descriptive findings

These are correlations across municipalities, as Spearman rank correlations with
95% intervals from a bootstrap that resamples whole NUTS 3 regions.

**Levels in 2024.** Municipalities with higher sale prices per m² have:
- higher new-lease rents per m² (0.92 [0.85, 0.95]);
- faster population growth in 2021–2024 (0.72 [0.57, 0.82]);
- higher completions per 1,000 residents (0.46 [0.30, 0.57]);
- higher median income (0.45 [0.18, 0.66]).

**Changes.**

| Price growth against | 2019–2024 (2022 method) | 2016–2020 (2018 method) | 2015–2024 (appraisals) | NUTS 3, 2019–2024 |
|---|---|---|---|---|
| income growth | −0.23 [−0.38, −0.07] | −0.10 [−0.28, 0.11] | −0.34 [−0.57, −0.04] | −0.29 |
| population growth | +0.37 [0.19, 0.55] (2021–24) | +0.15 [−0.01, 0.28] | +0.62 [0.42, 0.74] (2015–20) | +0.57 |
| completions per 1,000 | +0.19 [0.04, 0.33] | −0.03 [−0.15, 0.08] | +0.22 [−0.01, 0.42] | +0.42 |
| new-lease rent growth | +0.38 [0.21, 0.52] (2020–24) | +0.37 [0.11, 0.54] (2017–19) | | +0.71 |

- *Price and rent:* prices and rents move together across places and over time.
  This holds in every window and at both geographic levels.
- *Price and population:* prices rose more where population grew more after
  2015. The relationship is clearest after 2021 and in the appraisal series.
- *Price and income:* prices did **not** rise more where incomes rose more. If
  anything the opposite holds: the correlation is negative in two of three
  windows and at NUTS 3.
- *Price and construction:* the link with completions is weak and changes with
  the window.

**Rental yield.** The gross rental yield (12 × new-lease rent ÷ sale price, both
medians per m²) fell from 2020 to 2025 in 16 of 26 NUTS 3 regions. It fell from
5.8% to 5.4% for Portugal, from 5.6% to 5.0% in Grande Lisboa and from 5.9% to
5.3% in Porto.

**National time series, 2009–2024.**
- *Same-direction movements:* annual house-price growth moved with growth in
  completions (r = 0.81), median income (0.63) and population (0.61). It moved
  less with the HICP rent index (0.33) and with changes in the mortgage rate
  (0.17).
- *None is precise:* the series are strongly autocorrelated, so the effective
  samples are 3–12 years. No interval excludes zero except completions in
  2013–2024 (0.83 [0.15, 0.98], effective n ≈ 7).
- *Nothing more is attempted:* sixteen annual observations with breaks
  (2008–2014 crisis, pandemic, the 2024 EU-SILC pension change) do not support
  cointegration or dynamic models.

### Layer 2: model-dependent associations

These are within-municipality estimates.

**Income: not identified.**
- *Range:* the income elasticity of price ranges from −0.74 to +0.79 across 18
  specifications. 13 are negative and 5 positive. Four exclude zero below it
  and two above it.
- *Why the sign moves:* log income keeps only 0.9–1.9% of its variance once
  municipality and year effects are removed. Municipal income medians move
  almost in parallel, and what is left is mostly noise. The sign changes with
  the window, with first differences and with the controls: −0.63 in sample A,
  −0.09 in first differences, +0.06 with NUTS 2 × year effects, and +0.79 once
  population is added in A2.
- *Conclusion:* these data cannot say how prices respond to local income growth
  within a municipality.

**Completions: consistent with the sale mix, mechanism not identified.**
- *All sales:* more completions per 1,000 residents go with a higher median price
  of all sales, about +1.4% per extra dwelling per 1,000 residents in A2
  (CR1 interval [0.7%, 2.1%]). This passes every check.
- *Existing dwellings only:* for the same years and controls (279 of the 289
  municipalities) the estimate is 0.0% [−0.7%, 0.8%].
- *Interpretation:* new dwellings are dearer (27% above existing dwellings per
  m² in 2024), so more completions can raise the median of all sales without any
  change in the price of a given dwelling. The near-zero coefficient for
  existing dwellings is consistent with that, but it does not identify it.
  Reverse causality, prices of new dwellings, offsetting effects on existing
  dwellings and the different samples remain possible. Counts of new and
  existing sales, a joint model on a common sample, or transaction microdata
  would be needed (see `REPORT.md`, Section 7.2). The result must not be read as
  "construction raises prices" or the reverse.

**Population: consistent in sign, fragile.**
- *Estimates:* the elasticity of price with respect to population is 0.9–1.6,
  positive in all four specifications (A2 and B, fixed effects and first
  differences), with intervals above zero.
- *Why fragile:* log population keeps only 0.01–0.02% of its variance after the
  fixed effects. Official population estimates are smooth trends, and the
  coefficient is identified from tiny deviations from them. Those deviations
  include the administrative re-estimation of migrants after 2021.

**Rents: robust co-movement.**
- *Estimates:*
  - two-way effects 0.44 [0.29, 0.61];
  - first differences 0.31 [0.15, 0.48];
  - long difference 2020–2024 with NUTS 2 effects 0.50 [0.26, 0.76].
- *Stability:*
  - 0.41–0.49 leaving out any NUTS 3 region;
  - 0.43 without the metropolitan areas;
  - 0.43 without the most influential municipalities.
- *Serial correlation:* the Wooldridge slope (−0.46) supports fixed effects.
- *Reading:* both prices and rents are likely to respond to local demand and supply, so this
  is a co-movement, not one causing the other. Little within variation remains
  in rents (1.8%), so measurement error in the medians probably biases the
  estimate towards zero.

**Diagnostics.**
- *Errors:*
  - Cluster-robust standard errors are 1.0–2.3 times the conventional ones.
  - First-difference residuals in the sale-price samples are consistent with
    fixed effects (slopes −0.42 to −0.46).
  - In the appraisal sample the errors are persistent (slope −0.19). There the
    first-difference estimates are preferred: income −0.13 [−0.39, 0.13],
    completions +0.6% per dwelling per 1,000.
- *Influence:* leaving out single regions or the most influential municipalities
  changes no sign except where the cross-specification check already fails.

### Layer 3: questions that need a causal design

These questions cannot be answered with these data:
- Does local income growth raise prices or rents?
- Does more construction lower them?
- Does population growth (migration) raise them?
- Do immigration, tourism, investment demand, short-term letting or regulation
  explain the divergence between prices, rents and income?

**Why the data do not support a credible design.**
- *Short panels:* 4–6 years of sale prices and 10 of appraisals, with a method
  break in prices, rents and population.
- *Little variation:* income and population barely vary once municipality and
  year effects are removed.
- *Changing composition:* medians shift with the mix of what is sold or let, as
  the completions result shows.
- *Simultaneity:* prices, construction and migration determine one another.
- *Spillovers:* effects cross municipal borders.
- *No exogenous source of variation:* nothing here, no instrument and no
  policy change with documented timing and geography, separates cause from
  response.

**Evidence that would be needed.**
- *Composition-free prices and rents:* transaction-level or repeat-sales data
  (hedonic or repeat-sales indices by municipality).
- *Policy timing and geography*, for event studies or difference-in-differences
  with pre-trend tests:
  - short-term rental restrictions (containment areas in Lisbon and Porto; the
    2023 *Mais Habitação* licensing changes);
  - the end of the Golden Visa for housing and of the NHR regime;
  - the 2023 cap on rent updates.
- *Demand shifters with pre-determined exposure* (shift-share designs with
  balance checks), each by municipality and year:
  - foreign buyers (INE publishes prices by buyer domicile);
  - tourism capacity and short-term rental registrations (RNAL);
  - residence permits (AIMA).
- *Supply:* zoning and land constraints, building costs and permit processing
  times, to estimate supply responses.
- *Household data:* income of buyers and tenants, rather than all tax
  households, from linked tax and property records.

### Limitations

- Municipal medians are noisy where few sales or contracts occur. The balanced
  samples drop municipalities with any gap, which leaves the smallest ones out.
  The appraisal sample (179) is particularly selected.
- Population after 2021 includes INE's administrative re-estimation. Growth
  across 2020/2021 is never used, and models with population stay on one side of
  the break.
- Construction data for 2021–2025 are preliminary or estimated, and Amadora's
  completions are under-reported in 2011–2017. This touches samples B and C.
- With 26 NUTS 3 clusters, the wild bootstrap is the main source of inference.
  Spatial correlation across regions is not modelled.
- Effect sizes are associations at the municipal level. Applying them to
  households would commit the ecological fallacy.

Figures:
- `figures/fig_integrated_coverage.png`
- `figures/fig_integrated_cross_section.png`
- `figures/fig_integrated_specifications.png`
- `figures/fig_integrated_influence.png`
- `figures/fig_integrated_national.png`

## What the calculations support

- Nominal house-price-index changes, a relative house-price/income-growth
  proxy for overlapping years, and an initial-mortgage-payment/income index
  that adds the cost of borrowing on new loans; neither includes the deposit.
- An ownership gap of about 20 percentage points between two cohorts at the same age
  (rounded figures quoted from Banco de Portugal economists, not recomputed).
- Regional transaction prices per m² rising faster than declared household income
  in 2019–2024 (and 2016–2020) in almost every region, as a ratio of medians.
- New-lease rents per m² rising faster than declared household income in
  2020–2024 in 24 of 26 NUTS 3 regions, while the HICP rent index (a price index
  for a sample of rented dwellings) rose more slowly than median income; a
  larger share of the population renting
  at market prices.
- Across municipalities: prices and rents moving together; price growth higher
  where population grew more and not higher where income grew more.
- The stated breakdown of non-primary-residence dwellings in 2021.

## What they do not support

- A causal effect of speculation, immigration, or regulation on housing prices.
- A fall in the rent burden of tenants who kept their contracts, inferred from
  the HICP rent index relative to median income.
- A mechanism behind the completions association: it is consistent with the
  mix of dwellings sold, but that is not identified.
- Any within-municipality elasticity of prices with respect to income, construction
  or population: these change sign or rest on almost no variation.
- A claim that young people reject democracy because of housing costs.
- The assertion that non-votes for PS/PSD indicate opposition to democracy.
- Aggregating these different vintages and populations into a single causal model.

Next research step: the regional panels above cover population, construction,
census households, transaction prices, declared income and rents, and the
integrated assessment shows what they can and cannot support. A causal design
needs the additional evidence listed there (composition-free prices, policy
timing and geography, pre-determined demand exposure, supply constraints). Electoral analysis additionally
requires individual-level evidence to avoid ecological inference.
