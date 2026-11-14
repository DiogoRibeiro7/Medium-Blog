# Country-deletion sensitivity results — 13 November 2026

This audit applies complete leave-one-country-out **re-matching** to the
same 18-country tax-cut panel and WID top-1% pre-tax national-income share
used in the earlier studies. The live run succeeded in
[GitHub Actions](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/43516748403).
The author's original event-file SHA-256 and **all 18 WID country-file
SHA-256 hashes matched the earlier frozen provenance** recorded in
[`README.md`](README.md).

## Sample and headline outcome

- The baseline balanced event-time estimate includes **21 matched tax-cut
  episodes in 14 treated countries**.
- Deleting each of the **18 countries** in turn leaves **18–21** matched
  reform events. **All 18 refits succeeded**. Each refit removes the country
  from both the treatment-event and eligible-donor pools, regardless of its
  role in the full sample.
- At **five years after the reform**, the full matched mean is
  **+0.795 percentage points** of the top-1% income share.
- Across the 18 complete refits, the five-year mean ranges from
  **+0.577 pp** (excluding Japan) to **+0.960 pp** (excluding Finland).
  Thus, **no single-country deletion reverses the sign at five years**.
- Deleting the **United Kingdom** gives +0.789 pp; deleting the **United
  States** gives +0.782 pp. These are *separate* omissions, not a test
  excluding both countries together.

The following are **descriptive country-influence ranges, never confidence
intervals**. Event time `-1` is the normalized pre-reform baseline.
All effects are in percentage points of the top-1% income share.

| Years from reform | Full estimate | Min deletion | Max deletion | Deletions reversing sign |
| ---: | ---: | ---: | ---: | ---: |
| −5 | +0.162 | −0.039 | +0.244 | 1 |
| −4 | +0.060 | −0.131 | +0.129 | 2 |
| −3 | +0.131 | −0.023 | +0.201 | 1 |
| −2 | +0.119 | +0.069 | +0.163 | 0 |
| 0 | +0.197 | +0.102 | +0.301 | 0 |
| 1 | +0.276 | +0.137 | +0.382 | 0 |
| 2 | +0.367 | +0.261 | +0.457 | 0 |
| 3 | +0.512 | +0.395 | +0.586 | 0 |
| 4 | +0.642 | +0.493 | +0.759 | 0 |
| 5 | +0.795 | +0.577 | +0.960 | 0 |

Machine-readable results:

- [`country_influence_summary_2026-11-13.csv`](country_influence_summary_2026-11-13.csv):
  all 10 event-time means, deletion extrema, country of maximum influence,
  sample attrition and sign-change counts, at full precision.
- [`country_deletions_2026-11-13.csv`](country_deletions_2026-11-13.csv):
  all **180** country-by-event-time refits, each with its changed matched
  event count, treated-country count and pretrend slope diagnostic.

These are derived diagnostic outputs. The authors' original reform CSV and
WID bulk country files are not redistributed in this repository. The exact
source SHA-256 manifest and the scripts to download and re-create the joined
panel remain in the same research package.

## What this does and does not imply

The **five-year sign is robust to deleting any one country**, which is
stronger descriptive evidence than reporting only the full-sample mean.
Nevertheless, the point estimate moves by up to approximately **0.218 pp**,
and country deletion can change the underlying treated event composition.

The **pre-reform placebo leads are less stable**: some turn negative after a
country is removed. This remains a warning against treating observed
pre-trend diagnostics as proof of the parallel counterfactual assumption.

There is no valid sampling-uncertainty calculation here. The min–max interval
is not a 95% confidence interval, and the exercise does not resolve
unobserved confounding, anticipation, endogenous reform timing, overlapping
reforms, or dependence induced by shared donor pools.

Crucially, this study addresses **major top-income tax cuts and income
concentration**. It is not a causal estimate of the wider Reagan–Thatcher
neoliberal policy package, nor of its downstream effect on wages or housing.
The next separate step should add verified pre-treatment macroeconomic
covariates, with a transparent sample-attrition audit.
