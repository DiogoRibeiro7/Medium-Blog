# Major tax cuts for the rich: an auditable causal-analysis scaffold

## Research question and current status

Did major reductions in taxes on top incomes and capital raise the pre-tax
national income share of the richest 1%, and did they improve growth or
employment? This is a **falsifiable fiscal-policy hypothesis**, not a causal
estimate for the entire political movement called "neoliberalism".

**No new empirical effect has been estimated in this PR.** The
[authors' data catalogue](https://www.taxrich.uk/data) links their original
[`hope_limberg_reforms_11_2020.csv`](https://drive.google.com/file/d/1FgV9wfFl7FG5cl7PZ7UMzQQ24Wye0VT3/view?usp=sharing),
but its Google Drive preview did not expose the records to our execution
environment. The actual file headings and layout are therefore unknown.
We supply a source-auditing importer and a tested estimator, **not** fake
historical rows or invented causal results. Automated tests use clearly
synthetic countries and known effects.

The source catalogue labels its data as **19 OECD economies, 1965–2015**.
The published paper uses **18 economies and 30 major-cut country-years**.
`prepare.py` explicitly checks the 18-country sample and the 30-event count;
it will fail on an unresolved version discrepancy.

## Published benchmark and our different estimator

[Hope and Limberg (2022)](https://doi.org/10.1093/ser/mwab061) use lagged
treatment-history and covariate matching with a panel difference-in-differences
design. They report a roughly **0.6 percentage-point increase** in the top
1% share three years after a major cut and **more than 0.8 points** after five
years; their GDP and unemployment estimates are not statistically distinct
from zero. Those are the **authors' published estimates**, not ours.

Our code is an **exploratory adaptation**, with these choices:

| Dimension | Operational definition |
| --- | --- |
| Treatment | Annual author-coded tax-cut event, 0/1 (not absorbing treatment) |
| Eligibility | Reform at year `t`, no reform at `t-1`, 5 prior years observed |
| Exact matching | Same treatment-event path from `t-1` through `t-5` |
| Covariate matching | Ridge-regularised Mahalanobis distance on **pre-treatment** trajectories; up to five neighbours |
| No-covariate variant | All eligible control countries, with no arbitrary alphabetic K-selection |
| Donor rule | No new tax cut from `t` to `t+h` at each horizon |
| Treated rule | Later treated-country reforms retained in the observed policy pathway |
| Effect | `(treated[t+h] - treated[t-1]) - mean(donor[t+h] - donor[t-1])` |
| Aggregation | Equal-weight average over successfully matched reform events |
| Audit | Matched donors, event counts, exclusions, distances, two-year pretrend gap |
| Inference | **Not implemented:** no standard errors, p-values or confidence intervals |

The donor rule may select on *future political decisions*, which can bias a
counterfactual. Our shrinkage, matching and weighting rules are **not
claimed to reproduce** the authors' PanelMatch specification. A small
pre-period gap is not proof of parallel trends. These issues require explicit
sensitivity checks before publication.

## Real-data workflow

Work from the parent `neoliberalism-evidence-synthesis/` folder.

### Source retrieval now implemented

The repository includes two independent, network-enabled source commands:

```bash
python -m causal_tax_cuts.event_source --output-dir /tmp/hope-limberg-events
python -m causal_tax_cuts.wid --output-dir /tmp/wid-top1
```

The first checks whether the *original authors' public CSV* is actually
downloadable, validates the CSV format and writes its unmodified bytes only
to the chosen output directory. `author_event_probe.json` records the **true
headings**, observed examples, row count and SHA-256; on a sign-in wall it
records `status=unavailable` and never creates fictional events.

The WID downloader streams the official semicolon-delimited bulk CSV for
each of the paper's 18 countries, selects `sptinc992j` / `p99p100`
(top 1% of pre-tax national income among equal-split adults), and multiplies
WID's 0–1 fractions by 100. It emits `wid_top1_pretax.csv` and
`wid_top1_provenance.json`, with each source's original SHA-256, actual
country-year coverage and missing years. **It neither extrapolates missing
years nor claims to reproduce the WID vintage used in the 2022 paper.**

The separate [source-probe GitHub Action](../../.github/workflows/causal-tax-source-probe.yml)
tests public access and processes the full 18-country WID sample. The live
author file was verified to have **969 rows** (19 countries × 51 years) and
the actual fields `iso3`, `year`, `taxcut_2SD` and `taxcut_1SD`, plus an
unnamed original row-index column. The runner configures the **2-SD** series
without guessing the layout. All 18 country files downloaded successfully
in the extended live probe; coverage differs across countries, as quantified
below. The live source audit must be rerun for each new data vintage.

The Action uploads only author-source **metadata**, not the authors' raw CSV.
It attempts to produce a short-lived, audited joined panel and **exploratory
treatment-history-only** matching output. These are not licensed as published
paper replication and cannot be treated as a full causal result. A failed
live download, paper-sample audit or estimation step is recorded in CI and
must not be described as empirical verification.

### First observed coverage audit (13 November 2026)

The networked workflow successfully downloaded all 18 WID country files and
the authors' 19-country event panel. Its source and join audit found:

| Check | Observed result |
| --- | ---: |
| Original author CSV | 969 country-year rows (19 countries × 51 years) |
| Paper-sample tax-cut events (`taxcut_2SD`) | 30 (as published) |
| Extra author-file country excluded from paper sample | South Korea (`KOR`, 51 rows) |
| Possible paper-sample outcome country-years | 918 (18 × 51) |
| Current-vintage WID top-share country-years present | 853 |
| Missing current-vintage WID values | 65 |
| Tax-cut years lost to outcome missingness | 1 |

The current WID extraction has an **approximately 7.1% missing country-year
rate** in the 1965–2015 paper window. This is not automatically comparable
to the authors' 2022-vintage dataset; the working paper used a documented
interpolation procedure for some top-income observations. Our pipeline does
**not** impute them. Treat this as a real-coverage finding, not an ATT.

The fully completed live run also produced **exploratory** matched contrasts.
The frozen source-provenance manifest, six observed horizon means and limits
of the identification strategy are documented in
[`results/README.md`](results/README.md). The matched comparison is **not**
a replication of the published PanelMatch method or proof of causality.

### Balanced pre-reform placebo diagnostics

[`PRE_TREND_DIAGNOSTICS.md`](PRE_TREND_DIAGNOSTICS.md) describes a separate
event-time audit implemented in `event_study.py`. It holds the reform cohort
and the donor sets fixed across relative times −5 through +5, excluding the
baseline −1. It calculates pre-event contrasts at −5 through −2 relative to
the year before treatment, plus the matching post-event series. Run it on
the same joined panel as above:

```bash
python -m causal_tax_cuts.event_study \
  --panel data/tax-study/panel.csv --outcome top1_share_pct \
  --lags 5 --horizon 5 --min-events 5 \
  --output data/tax-study/placebo
```

These are **descriptive diagnostics**, not an acceptance test for causal
identification. Missing pre-trends do not establish parallel counterfactual
trends, and outcome-window completeness can reduce the matched sample.
The audited first live run and fixed-cohort event-time table are available in
[`results/PRE_TREND_RESULTS_2026-11-13.md`](results/PRE_TREND_RESULTS_2026-11-13.md).

### Whole-country deletion influence

The next diagnostic reruns the entire balanced-cohort event study after
removing **each country in turn**, from both potential treatment and donor
pools. It records the effect of country removal on the matched cohort,
pre-reform leads and post-reform contrasts. A deletion that leaves too few
events is reported as unavailable, never given a zero effect.

[`COUNTRY_INFLUENCE.md`](COUNTRY_INFLUENCE.md) describes the estimand,
outputs and assumptions. From the same joined real-data panel:

```bash
python -m causal_tax_cuts.country_influence \
  --panel data/tax-study/panel.csv --outcome top1_share_pct \
  --lags 5 --horizon 5 --min-events 5 \
  --output data/tax-study/country-influence
```

Leave-one-country-out ranges are **descriptive sensitivity checks**, not
bootstrap confidence intervals or a substitute for economic controls.

1. Download the authors' file via the link above. Inspect the true headings:

   ```bash
   python -c "import csv; p='data/raw/hope_limberg_reforms_11_2020.csv'; print(next(csv.reader(open(p, encoding='utf-8-sig'))))"
   ```

2. Obtain a verified CSV of annual WID **top-1% shares of pre-tax national
   income** for the same countries and years, either using the downloader
   above or a documented WID export. The result needs a `country`, `year`,
   `top1_share_pct` heading. For example, 12.5% must be `12.5`, not `0.125`.
   Record the WID source vintage, interpolation and series metadata.

3. After inspecting the actual author headings, prepare the panel (replace
   heading placeholders; no guesses about author column names):

   ```bash
   python -m causal_tax_cuts.prepare \
     --events data/raw/hope_limberg_reforms_11_2020.csv \
     --outcomes data/raw/wid_top1_audited.csv \
     --event-mode binary-panel \
     --event-country COUNTRY_HEADING --event-year YEAR_HEADING \
     --event-treatment TREATMENT_HEADING \
     --fields top1_share_pct \
     --output data/derived/tax_cut_panel.csv
   ```

   If the authors' source is an **exhaustive positive-event list**, choose
   `--event-mode positive-events` instead. That explicit mode interprets
   absent country-years as non-events only **after** the event-count check.
   Binary-panel mode does not fabricate missing 0 flags. Study the generated
   `tax_cut_panel.audit.json` for dropped country labels, event coverage,
   missing outcomes and event-year attrition before proceeding.

4. Obtain and join documented, pre-treatment covariates from the sources in
   [SOURCES.md](SOURCES.md). A run without covariates is only a sensitivity
   baseline; it is **not** the published adjusted model.

5. Run the exploratory analysis:

   ```bash
   python -m causal_tax_cuts.analysis \
     --panel data/derived/tax_cut_panel.csv \
     --outcome top1_share_pct --lags 5 --horizon 5 \
     --min-events 5 --output results/causal_tax_cuts/history_only
   ```

   For a covariate-adjusted variant, provide an audited panel containing the
   columns listed below, then append:

   ```bash
   --covariates real_gdp_pc capital_open trade_open govt_expenditure left_vote_share
   ```

The results contain `event_effects.csv` (treated country/year, donors,
distances, pretrend and effect), `att_by_horizon.csv` (point estimate and
counts) and `run_metadata.json` (declared settings). Every horizon must have
the minimum number of matched events; otherwise the run fails closed.

## Validation

```bash
python -m unittest discover -s causal_tax_cuts/tests -v
ruff check causal_tax_cuts
ruff format --check causal_tax_cuts
```

The synthetic tests recover known effects, reject duplicate or non-finite
data, enforce the 18-country/30-event audit, and check that missing outcomes
are **not** silently set to zero. No historical estimate, CI or empirical
figure is checked in or asserted.

## What must be resolved before a causal claim

- Endogenous reform timing, anticipation, reporting responses and spillovers.
- A fully source-audited country panel and preregistered measurement units.
- Multiple event-study leads, placebo reforms, covariate balance and donor
  attrition, especially under future-policy censoring.
- Appropriate country-level/block matching inference. Naive country-year
  standard errors would be misleading.
- Direct comparison with the published PanelMatch rules and results, including
  robustness to later reforms and to different country inclusion criteria.

Until these steps are complete, cite the authors' results **as published
research**, not as a new causal replication. A tax-reform ATT must also not be
presented as proof about wages, privatisation or housing financialisation.
