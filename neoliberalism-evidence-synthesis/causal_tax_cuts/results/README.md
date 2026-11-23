# First live-data matching audit — 13 November 2026

This is an auditable *exploratory result* from the code in
`causal_tax_cuts/`, **not** a replication of Hope and Limberg (2022),
a causal proof, or an estimate with validated sampling uncertainty.
The numbers below were obtained from the successful source/estimation
[GitHub Actions run](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/43445919545)
at commit `bc61537f00a865f942e3b477358f80a0a47f93f2`.

## Sources and sample

- Policy exposure: [Hope and Limberg public author data](https://www.taxrich.uk/data),
  filename `hope_limberg_reforms_11_2020.csv`, original SHA-256
  `3f1bc22a526d3beff2409de30e5644c6dc696b78614b46bede148cb80f7ad653`.
  The file contains 969 rows for 19 countries during 1965–2015. The paper's
  18-country sample excludes South Korea (`KOR`), leaving **30** major events
  under the `taxcut_2SD` definition.
- Income outcome: [World Inequality Database](https://wid.world/data/) series
  `sptinc992j` for `p99p100`: the top 1% share of pre-tax national income,
  equal-split adults, converted from a 0–1 fraction to **percentage points**.
  The source manifest records 18 original bulk-file SHA-256 checksums and
  country-level missing years. WID's own estimation and interpolation
  methodology has **not** been independently audited here.
- Potential panel: 18 countries × 51 years = 918 country-years. The current
  WID extraction supplies **853**, with **65 missing (7.1%)**. The code
  neither fills nor interpolates missing outcome cells; one event-year is
  absent, leaving **29** reform observations in the joined outcome panel.
- Actual matching: of these 29 event observations, 22 meet the historical
  reform-path and outcome-availability eligibility rules; **21** are matched
  to at least one eligible donor, spanning **14 treated countries**.

## Observed event-onset contrasts

Outcome: change in the top 1% income share, **percentage points**. Horizon
0 compares the reform year with the previous year; horizon 5 compares five
years after the reform with the previous year. All horizons use 21 matched
reform events, but donors may change across horizons. No p-values or confidence
intervals were calculated.

| Years after reform | Mean matched contrast | Median contrast | Mean pre-event trend gap |
| ---: | ---: | ---: | ---: |
| 0 | +0.221 | +0.072 | −0.060 |
| 1 | +0.296 | +0.246 | −0.054 |
| 2 | +0.395 | +0.658 | −0.065 |
| 3 | +0.547 | +0.431 | −0.094 |
| 4 | +0.656 | +0.637 | −0.110 |
| 5 | +0.795 | +0.611 | −0.131 |

Machine-readable exact summaries are in
[`att_by_horizon_2026-11-13.csv`](att_by_horizon_2026-11-13.csv).
Source metadata is in `author_event_probe_2026-11-13.json` and
`wid_top1_provenance_2026-11-13.json`. These contain *metadata and hashes*,
not the authors' raw CSV or the WID raw country files. The generated joined
panel and event-level contrasts were uploaded as **seven-day CI artifacts**;
the scripts can regenerate them against live source vintages.

## Design and interpretation limits

1. The comparison matches five years of **past treatment history**, not
   pre-treatment income paths, structural factors, or macroeconomic trends.
   With no supplied covariates, all eligible donor countries are averaged;
   the `max_donors` argument does not select a smaller set.
2. A donor must have no new major tax cut between the reform year and the
   evaluated horizon. This explicitly conditions on future donor exposure.
   The treated countries can undergo additional subsequent reforms.
3. Mean pre-event trend gaps are not zero and become more negative at later
   horizons. Those diagnostics do not establish parallel counterfactual trends.
4. Country dependence, serial correlation, selection into reform, anticipation,
   confounding reforms, WID data revisions and missingness are all unresolved.
   No country-cluster or block-bootstrap inference has been implemented.
5. The paper used PanelMatch and additional controls. A similar numerical
   trajectory in this simple comparison does **not** validate the causal
   identification assumptions or reproduce the original estimate.

The observed series is a **positive descriptive matched contrast**, worthy
of further robustness work but not an isolated causal effect of neoliberalism
as a whole. It addresses one specific major-tax-cut channel, not the much
broader claim that wage stagnation and housing unaffordability were caused by
Reagan–Thatcher-era policy shifts.

## Reproduce the latest data vintage

From `neoliberalism-evidence-synthesis/`, with Python 3.12 and NumPy 2.3.5:

```bash
python -m causal_tax_cuts.event_source --output-dir data/tax-events
python -m causal_tax_cuts.wid --output-dir data/wid-source
python -m causal_tax_cuts.prepare \
  --events data/tax-events/hope_limberg_reforms_11_2020.csv \
  --outcomes data/wid-source/wid_top1_pretax.csv \
  --event-mode binary-panel --event-country iso3 --event-year year \
  --event-treatment taxcut_2SD --fields top1_share_pct \
  --output data/tax-study/panel.csv
python -m causal_tax_cuts.analysis \
  --panel data/tax-study/panel.csv --outcome top1_share_pct \
  --lags 5 --horizon 5 --min-events 5 \
  --output data/tax-study/results
```

The commands query *live* external sources. Future values may differ; compare
download checksums to the frozen manifest before making temporal comparisons.

## Further data-availability audits

The [official World Bank macroeconomic-controls coverage report](WDI_MACRO_COVERAGE_RESULTS_2026-11-14.md)
freezes four 14 November 2026 WDI archive checksums and their overlap with
the existing 853-row outcome panel. It does not revise any historical ATT.

The [pre-reform macro matching report](MACRO_MATCHING_RESULTS_2026-11-14.md)
and [machine-readable audit](macro_balance_2026-11-14.json) use those same
official WDI/WID observations to report 29 reform-event attrition, actual
donor support and three-control balance, with no adjusted ATT or inference.
Event-level rows and the complete joined panel are available in the seven-day
`pre-reform-macro-balance` and `world-bank-macro-coverage` CI artifacts.

The [macro matching sensitivity report](MACRO_SENSITIVITY_RESULTS_2026-11-14.md)
and [frozen JSON](macro_sensitivity_2026-11-14.json) compare nine pre-reform
lag/donor-cap rules. Their **19-event common cohort** and fixed three-year
balance window permit like-for-like descriptive comparisons; they are not
adjusted causal estimates. Live reproduction includes event-level donor sets
in the seven-day `pre-reform-macro-sensitivity` workflow artifact.

The [future donor reform and macro support report](DONOR_SUPPORT_RESULTS_2026-11-15.md)
and [frozen JSON](donor_support_2026-11-15.json) quantify donor–reform
attrition due to later tax cuts, with a fixed outcome window and pre-reform
macro distances. They show **65 of 239** eligible pairs removed by year five,
and only **8 of 20** supported reforms inside all three separate macro
donor ranges. These diagnostics do **not** estimate an adjusted ATT or
establish joint covariate overlap. The `tax-cut-donor-support` seven-day
workflow artifact provides the event-, donor-, and horizon-level source
records when the live data probe succeeds.

The [dynamic policy-path results](DYNAMIC_EXPOSURE_RESULTS_2026-11-16.md)
and [frozen JSON](dynamic_exposure_2026-11-16.json) show that **5 of 20**
treated reform episodes experience another major cut during five-year
follow-up. Treating the hypothetical strategies of *one coded major
cut* and *no coded major cuts* symmetrically leaves **15 treated
episodes** and **144 compatible donor–reform pairs** at year five;
these are not independent observations, an estimated policy effect, or
evidence of sequential exchangeability. The
`tax-cut-dynamic-exposure` artifact contains annual policy paths and
pre-decision WDI coverage diagnostics.

The [joint three-variable donor support report](JOINT_SUPPORT_RESULTS_2026-11-18.md)
and [frozen audit JSON](joint_support_2026-11-18.json) add a more stringent
pre-treatment interpolation check on that *identical* 20-event cohort.
Only **2 of 20** reforms lie inside the surviving donors' three-dimensional
convex hull at horizon five, whereas **8 of 20** pass all three separate
macro-variable ranges. Six marginal passes therefore do **not** establish
joint support. This does not measure causal positivity or estimate an
effect. The seven-day `tax-cut-joint-macro-support` artifact contains all
event-level witness combinations, eligibility and numeric tolerances.

The [political and institutional prehistory report](POLITICAL_INSTITUTIONS_RESULTS_2026-11-18.md)
and [frozen source manifest](political_institutions_2026-11-18.json) audit
the published IDB DPI2023 dataset for exactly the same 20 eligible reforms,
239 initial donor–event pairs and 174 year-five pairs. Executive economic
orientation remains a **nominal** right/center/left class, with unknown and
not-applicable values kept separate. All three five-year histories exist
for **17/20** treated reforms, **206/239** initial donor pairs and
**143/174** year-five donor pairs. None of these counts is an adjusted
causal estimate; the `tax-cut-political-institutions` artifact has the
year-by-year controls and event/donor completeness details.


The [political comparability audit](../POLITICAL_COMPARABILITY.md) and
[21 November frozen JSON](political_comparability_2026-11-21.json) use the
same 20 reforms and unchanged donor exclusions. At the pre-censor stage,
80/206 complete pairs have the same executive-party orientation at t-1;
at year five, 54/143 do. Only 20/143 have identical five-year orientation
histories. Almost all have the same EIEC code (142/143 at year five), so
that field offers little discrimination. **These are descriptive comparisons,
not causal estimates.** The live source workflow publishes event-level
records and verifies the frozen snapshot on matching hashes.


The [annual policy timing audit](../TAX_TIMING.md) distinguishes
Hope–Limberg's **30 coded country-year tax-cut flags** from independently
verified legislative enactment or implementation dates. Its
[22 November archived-panel summary](TAX_TIMING_RESULTS_2026-11-22.md)
documents the 29 WID-observed flag-years. The subsequent live source run
verified all 30 author flag-years across 25 consecutive runs, with
Italy 1972 the only WID-missing flag. The source-probe publishes a short-lived
`tax-cut-annual-timing` artifact. This audit does not shift
treatment years or estimate new policy effects.


The [reform onset and statutory-period audit](../ONSET_PHASEIN.md) and
[23 November frozen result](ONSET_PHASEIN_RESULTS_2026-11-23.md)
reconcile 30 author flags, 25 run starts, 20 previously donor-eligible
onsets and 5 additional five-year follow-up flags. Four are contiguous
extensions; the fifth is USA 1982 followed by USA 1987 in a different
flagged run. The three provision-period examples are **not** verified
dates for the composite tax-cut indicator. No original estimate changes.
