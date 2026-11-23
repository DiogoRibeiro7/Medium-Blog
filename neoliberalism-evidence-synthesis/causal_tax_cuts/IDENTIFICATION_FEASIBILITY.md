# Policy-specific identification feasibility gate

**Scope:** compare necessary evidence for each of the **same 20 eligible**
Hope–Limberg tax-cut onsets, using unchanged, source-hashed WID/WDI/DPI
and legislative registers. This is a **diagnostic of the assembled
research design**, not an effect estimate or a declaration that causal
inference about these policies is intrinsically impossible.

## Why this step is needed

The original treatment is a binary country-year threshold crossing in a
multi-component latent tax index. The expanded statutory register
documents seven flag-years; only **five** are eligible initiating reforms
under the earlier study's WID outcome and WDI eligibility restrictions.
Their legal references establish particular legislation and tax-period
rules, *not a verified one-to-one mapping* between any statute and a
specific author-coded composite index change.

Other limitations arise separately: country-level policy anticipation
and effects that spill across years, future donor reforms, complete
post-baseline outcome selection, observed pre-treatment trend differences,
omitted confounding and the small number of independent countries.

The existing three-dimensional donor convex-hull test is an exact
**interpolation feasibility diagnostic** for three pre-reform WDI
means. It is not the probability of treatment (causal positivity), and
its failure alone does not invalidate every possible estimator. Its
passing alone does not identify an effect, either. For methodological
background, see [Abadie (2021)](https://doi.org/10.1257/jel.20191450)
on synthetic-control requirements, and
[Callaway & Sant'Anna (2021)](https://doi.org/10.1016/j.jeconom.2020.12.001)
on time-varying difference-in-differences identification conditions.
Neither should be transplanted mechanically to this **recurrent,
nonabsorbing** binary cut indicator.

## What the audit joins, without re-estimating anything

The deterministic, typed
[identification_feasibility.py](identification_feasibility.py) ingests
source outputs from the previously merged workflow:

| Upstream artifact | Evidence carried into this gate |
|---|---|
| Legislative expansion | Original 30 flags, 25 run starts, 20 matched-eligible initiators, five reviewed eligible statutes; author-composite attribution and effective-date verification still absent |
| Joint WDI support | Strict three-variable donor-hull membership and marginal ranges at horizons minus-one through five, with changing original donor risk sets |
| Dynamic policy pathways | Additional major cuts during follow-up, and donor reform-censoring at each horizon |
| DPI2023 political coverage | Whether treated and donor histories have all three pre-reform political variables; **completeness is not balance** |
| Pre-reform macro matching | The observed slope gap in top-1% pre-tax income shares relative to the earlier macro-selected donors; **not a parallel-trends acceptance test** |

All five upstream manifest files must describe **the same joined-panel
SHA-256**. Exact event/horizon keys, donor counts and prior aggregate
results are reconciled. Unexpected schema, nonfinite slope, missing
horizons, changed sample or inconsistent identities fail closed.
The algorithm never uses WDI or DPI data after reform to construct a
pre-reform profile. The earlier future-reform donor exclusion remains
unchanged and explicitly disclosed.

Outputs:

- \`policy_identification_events.csv\`: exactly 20 rows with independent
  binary/missingness indicators and the observed pretrend slope gap.
- \`policy_identification_event_horizons.csv\`: 140 rows, seven
  horizons per eligible reform, with separate joint/marginal overlap,
  donor counts and known recurrent exposure.
- \`policy_identification_audit.json\`: source hashes, counts, transparent
  limitations and the decision gate.

A legal-context row is **not** a verified policy intervention, and the
finite-sample overlap flag is not a causal probability. The audit makes
**no arbitrary weighted feasibility score**, chooses no new donor
weights, and estimates no standard errors, ATT or p-value.

## Findings on the frozen WID/WDI vintage

The [complete results note](results/IDENTIFICATION_FEASIBILITY_RESULTS_2026-11-23.md)
uses the original 20 supported onsets, without changing the historical
research sample.

| Criterion | Observed count |
|---|---:|
| Initiating reform onsets in previous donor cohort | 20 |
| With a provision-level legal context | **5/20** |
| Reviewed onsets with strict joint WDI hull at horizon 0 | **1/5** |
| Reviewed onsets with strict joint WDI hull at horizon 3 | **1/5** |
| Reviewed onsets with strict joint WDI hull at horizon 5 | **0/5** |
| All onsets with strict joint WDI hull at horizon 5 | **2/20** |
| Reviewed onsets with all five years of three DPI fields observed | **4/5** |
| Reviewed onsets followed by another major author cut by year five | **1/5** |
| With a verified flag-to-law assignment and composite effective date | **0/20** |
| Demonstrably identified policy-specific causal estimands | **0/20** |

The only two cases passing the **existing** strict joint WDI donor-hull
criterion at year five are **Finland 1986 and Germany 2008**. Neither
has statute-level context in the reviewed legal register. **UK 1988**
passes this WDI hull criterion at horizons 0–3 and loses it by
horizons 4–5; that does *not* establish causal identification over
a three-year horizon.

## Individual legally reviewed cases

| Eligible author onset | Statutory context | WDI strict hull at h=0 / h=5 | Other observed limitations |
|---|---|---|---|
| UK 1979 | Finance (No. 2) Act | No / No | DPI five-year history incomplete (DPI starts 1975); UK tax year predates assent |
| US 1982 | Economic Recovery Tax Act 1981 | No / No | Another major cut at h=5 (US 1987) |
| UK 1988 | Finance Act 1988 | Yes / No | Donor overlap erodes after horizon 3; tax year predates assent |
| Germany 2001 | Steuersenkungsgesetz 2000 | No / No | Existing donor pool cannot interpolate three WDI pre-means |
| Netherlands 2001 | Wet inkomstenbelasting 2001 | No / No | Existing donor pool cannot interpolate three WDI pre-means |

The sources for these acts are linked individually in
[LEGISLATIVE_EXPANSION.md](LEGISLATIVE_EXPANSION.md).
Their enacted dates and provision start dates remain those previously
verified. They **cannot** be inserted as substitute intervention dates
for the composite variable without additional component-level evidence.

## Reproduction

Run from \`neoliberalism-evidence-synthesis/\` after the full historical
source-probe CI job; paths below are its established output paths.

~~~bash
python -m causal_tax_cuts.identification_feasibility \
  --legal data/tax-study/legislative-expansion/legislative_review_all_flag_years.csv \
  --hull data/tax-study/joint-support/joint_support_event_horizons.csv \
  --dynamic data/tax-study/dynamic-exposure/dynamic_event_horizons.csv \
  --political data/tax-study/political-institutions/political_reform_coverage.csv \
  --macro data/tax-study/macro-balance/macro_event_eligibility.csv \
  --legal-audit data/tax-study/legislative-expansion/legislative_coverage_audit.json \
  --hull-audit data/tax-study/joint-support/joint_support_audit.json \
  --dynamic-audit data/tax-study/dynamic-exposure/dynamic_exposure_audit.json \
  --political-audit data/tax-study/political-institutions/political_coverage_audit.json \
  --macro-audit data/tax-study/macro-balance/macro_balance_audit.json \
  --output-dir data/tax-study/identification-feasibility
~~~

The CI reproduces all event rows and both audit files using *live*
original sources and a seven-day artifact. It checks the frozen
[JSON manifest](results/identification_feasibility_2026-11-23.json)
byte-for-byte when all source hashes agree, or reports a new source
vintage honestly.

## Decision and next research gate

**Current decision: no credible policy-specific ATT identified by
the existing 18-country, five-year donor framework.** Do not publish
a point estimate as a causal treatment effect on the basis of these
audits or equate the result to an effect of "neoliberalism" generally.

Two different research routes are worth exploring separately:

1. **Source-first policy-specific design.** Locate statute texts,
   component-specific enactment and actual implementation schedules
   for Finland 1986 and Germany 2008, which currently have the strongest
   *limited* macro-hull support but no reviewed statute context.
   Then test exposure consistency, anticipation, candidate controls,
   parallel trends, political and fiscal confounding and sample size
   before precommitting an estimator.
2. **Different identifying variation.** For a narrower, accurately
   measured tax-policy intervention, consider a within-country
   bracket/rate threshold, discontinuity or other exogenous eligibility
   variation *if* suitable administrative tax data and institutional
   evidence can be obtained. This is a research direction, not a design
   asserted to be available in the present data.

No further effect estimation is authorized by this feasibility audit.
