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

1. Download the authors' file via the link above. Inspect the true headings:

   ```bash
   python -c "import csv; p='data/raw/hope_limberg_reforms_11_2020.csv'; print(next(csv.reader(open(p, encoding='utf-8-sig'))))"
   ```

2. Obtain a verified CSV of annual WID **top-1% shares of pre-tax national
   income** for the same countries and years, with a `country`, `year`,
   `top1_share_pct` heading. For example, a share of 12.5% is `12.5`, not
   `0.125`. Log the WID indicator, version, extraction date and transformations.

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
