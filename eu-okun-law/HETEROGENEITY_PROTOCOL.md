# Country heterogeneity: Portugal, Spain, the Netherlands and Luxembourg

Pre-registered protocol of the comparative case study. It was written, committed and
pushed before any of the quantities below were computed. Changes after the first
computation are listed in the change log at the end, with reasons.

**Data work before this protocol was written.**

- Eurostat: availability, units, adjustment, country coverage and break flags of the
  datasets in section 2.
- OECD: the structure of the employment-protection dataset.
- The committed results that this study starts from (section 1).

No regression, correlation or decomposition involving the new series was computed.

## 1. Starting point, and the rule against retrofitting

These are the committed estimates: two-lag sums of the unemployment-rate change on GDP
growth, with 2020-2021 excluded. Sources: `docs/review/grid.csv` (classical) and
`docs/bayes-robustness/country_summaries.csv`, `pandemic_excluded` (Bayesian).

| Country | Classical (K = 2) | Bayesian posterior median | Known measurement issue |
| --- | ---: | ---: | --- |
| ES | −0.79 | −0.50 | LFS "definition differs" from 2021-Q1; ages 16-74 |
| PT | −0.50 | −0.21 | — |
| NL | −0.28 | −0.09 | 2021 survey phased in without a flag |
| LU | +0.17, sign-unstable | −0.08 | GDP covers cross-border workers; unemployment covers residents only |

**The rule against retrofitting.**

- Every hypothesis below follows from definitions, accounting identities or standard
  mechanisms, and makes **predictions that can fail**.
- Each prediction is evaluated by the same rule in the comparison countries, so it cannot
  be tailored to one country.
- An explanation is not accepted because a coefficient is unusually large or small, and
  no new hypothesis is added after the results are seen. Any later addition would go in
  the change log as exploratory.

## 2. Data

All series are official and harmonised, and they are frozen in
`snapshots/heterogeneity-<date>/`: raw responses, a SHA-256 manifest, processed CSVs and
an audit that re-derives them. There is no interpolation and no imputation. Every ratio
takes its numerator and denominator from the same concept: domestic with domestic,
resident with resident.

| Indicator | Source | Filters | Unit, frequency | Coverage | Concept and limits |
| --- | --- | --- | --- | --- | --- |
| GDP growth, unemployment, participation | `snapshots/eurostat-2026-10-15` | as in the study | %, pp; quarterly, SA | 27 countries | GDP is domestic production; unemployment and participation cover residents |
| Domestic employment `E_dc` | `namq_10_pe` | `EMP_DC`, `THS_PER`, `SCA` | thousand persons | 20 countries (no BE, CZ, EL, FR, MT, PL, PT) | Persons working in resident production units, including non-resident commuters |
| Resident employment `E_nc` | `namq_10_pe` | `EMP_NC`, `THS_PER`, `SCA` | thousand persons | 21 countries (no BE, CZ, FR, MT, PL, PT) | Employed residents, wherever they work |
| Domestic employment, persons | `namq_10_a10_e` | `TOTAL`, `EMP_DC`, `THS_PER`, `SCA` | thousand persons | 26 countries (no BE) | As `E_dc`, the denominator of hours per worker |
| Hours worked `H` | `namq_10_a10_e` | `TOTAL`, `EMP_DC`, `THS_HW`, `SCA` | thousand hours | 26 countries (no BE) | Total hours worked in resident units |
| Temporary employees | `lfsq_etgaed` | `T`, `Y15-74`, `ISCED TOTAL`, `THS_PER` | thousand; quarterly, **not seasonally adjusted** | 27 | Breaks flagged at 2014-Q1 and 2021-Q1 in every country, and in some countries other years |
| Employees | `lfsq_eegaed` | `T`, `Y15-74`, `ISCED TOTAL`, `THS_PER` | as above | 27 | Permanent = employees − temporary |
| Temporary share | `lfsq_etpga` | `T`, `Y15-74`, `PC` | % of employees, NSA | 27 | Structural indicator: mean over 2009-2019 |
| Part-time share | `lfsq_eppga` | `T`, `Y15-74`, `PC` | % of employment, NSA | 27 | Structural indicator: mean over 2009-2019 |
| Employment protection | OECD `DSD_EPL@DF_EPL` | `EPL_R` (regular), `EPL_T` (temporary), version 3, 2008 | index 0-6 | 20 EU OECD members | 2008 precedes the sample; the 2012 reforms in ES and PT are not reflected |

**National statistical offices.** These sources were identified but are not used in
estimation:

| Office | Series | Why not used |
| --- | --- | --- |
| STATEC (LU) | Cross-border workers (frontaliers) from social-security registers | Already enter the national accounts' domestic employment, which this study uses in a harmonised form |
| CBS (NL) | Flexible employment relationships, hours | Definitions differ from the EU-LFS; harmonised LFS temporary employment is used instead |
| INE (ES) | EPA, Social Security affiliations | As for CBS |
| INE (PT) | Inquérito ao Emprego | As for CBS |

## 3. Measures and the margin identity

All measures are quarter on quarter, ×100 log changes, except that GDP growth `g` is the
published % change.

```text
h      = 100 Δ ln H                (hours worked, domestic)
e_dom  = 100 Δ ln E_dc             (domestic employment, persons; namq_10_a10_e)
hpw    = h − e_dom                 (hours per worker)
prod   = g − h                     (output per hour, by construction)
e_res  = 100 Δ ln E_nc             (resident employment; namq_10_pe)
comm   = 100 Δ ln(E_dc / E_nc)     (commuting margin; both from namq_10_pe)
du, dp = changes in the unemployment and participation rates (pp)
```

**The identity.** `g = prod + hpw + e_dom` holds exactly, so the elasticities of the
three margins to GDP sum to one. This is the accounting decomposition of METHODS.md and
IDENTIFICATION_STUDY.md, section 3: output fluctuations are absorbed by productivity,
hours per worker or employment.

## 4. Estimator and uncertainty

**Elasticities.** For each country and measure m, OLS of `m[t]` on a constant and
`g[t], g[t−1], g[t−2]`. The elasticity β is the sum of the three slopes, which is the
study's two-lag headline.

**Primary sample.** Quarters 2009-Q3 to 2026-Q2 whose window [t−2, t] avoids
2020-Q1..2021-Q4. This is the explicit, pre-registered pandemic choice, consistent with
the classical and Bayesian `pandemic_excluded` estimates.

**Temporary and permanent employees.** These are not seasonally adjusted, so they enter
as year-on-year changes.

- `100 Δ4 ln X[t]` is regressed on a constant and `sum_{j=0..3} g[t−j]`.
- An observation is dropped if a quarter in [t−4, t] carries a break or low-reliability
  flag (`b`, `u`) in either series, or touches 2020-Q1..2021-Q4.

**Uncertainty: a synchronised circular block bootstrap over quarters.**

- Whole quarters are resampled, keeping every country and every measure of a quarter
  together, so the dependence between countries and between measures is preserved.
- Block length 4, B = 1999, seed 20261018.
- Report: 95% percentile intervals of β and of differences, and the two-sided bootstrap
  p-value `2 min(share ≤ 0, share ≥ 0)` of a difference.

## 5. Hypotheses and predictions

### H1, Luxembourg: measurement mismatch

GDP includes production by cross-border commuters; unemployment and participation cover
residents only. If part of labour-input adjustment falls on non-residents, three things
should hold:

- **P1.1** In LU, the domestic-employment elasticity exceeds the resident-employment
  elasticity: `β_e_dom − β_e_res > 0`. Both come from `namq_10_pe` here, so the
  comparison is like for like.
- **P1.2** In LU, the commuting margin responds to GDP: `β_comm > 0`.
- **P1.3** The LU gap `β_e_dom − β_e_res` is larger than the median gap of the other
  countries that have both series.

**Descriptive companion.** Where LU's domestic-employment elasticity lies within the
distribution of the other countries: inside or outside their interquartile range.

### H2, the Netherlands: other margins

NL's smaller unemployment response coincides with more adjustment through other margins.

- **P2.1** `β_hpw(NL) > β_hpw(ES)` and `β_hpw(NL) > β_hpw(PT)` (hours per worker).
- **P2.2** `β_prod(NL) > β_prod(ES)` (productivity per hour, a hoarding-type margin).
- **P2.3** `β_dp(NL) > β_dp(ES)` and `β_dp(NL) > β_dp(PT)` (participation rises more
  with GDP).

### H3, the Netherlands: timing

**P3.1** The share of NL's K = 4 unemployment response coming from lags 1-4 exceeds that
of ES and of PT.

- The share is reported only where the interval of the K = 4 sum excludes zero.
- **This is timing only.** A delayed response is not evidence about temporary
  contracts.

### H4, Spain, Portugal and the Netherlands: temporary jobs as a buffer

**P4.1** In each of ES, PT and NL, temporary employment is more GDP-elastic than
permanent employment: `β_temp − β_perm > 0`. This is year on year, with LFS data.

### H5, Portugal against Spain: employment structure

- **P5.1** `β_e_dom(ES) > β_e_dom(PT)`: Spain adjusts output fluctuations more through
  employment.
- **P5.2** `β_prod(PT) > β_prod(ES)`: Portugal adjusts more through productivity.

**Descriptive companion: the crisis.** Cumulative changes from 2008-Q1 to 2013-Q1 in GDP
and in each margin, for ES and PT. This is not tested.

### H6, institutions across countries

Spearman correlations between the Bayesian `pandemic_excluded` posterior median θ and six
indicators, over the countries with data. θ is negative, so "less negative" is a positive
correlation.

| Indicator | Predicted sign | Reason |
| --- | --- | --- |
| Temporary share (2009-2019) | − | Temporary jobs make employment, and so unemployment, more responsive |
| Part-time share (2009-2019) | + | More hours margin |
| `EPL_R` (2008) | + | Stricter dismissal rules, less employment adjustment |
| `EPL_T` (2008) | + | Stricter temporary-contract rules, fewer buffer jobs |
| Hours-per-worker elasticity `β_hpw` | + | Adjustment through hours |
| Commuting ratio `E_dc / E_nc` (2009-2019 mean) | + | Mismatch between production and residents |

- Each correlation gets a permutation p-value (9999 permutations, seed 20261018). The
  six are Holm-adjusted.
- With at most 27 countries these correlations are weak tests, and they describe
  countries, not mechanisms.

## 6. Do the country differences remain under other measures?

For the six pairs (ES-PT, ES-NL, ES-LU, PT-NL, PT-LU, NL-LU), the elasticity difference is
computed under four labour-input measures:

1. `−du`: unemployment, the study's measure;
2. `e_res`: resident employment;
3. `e_dom`: domestic employment;
4. `h`: total hours.

**Rule.**

- A difference **remains** under a measure if its 95% interval excludes zero with the
  same sign as under `−du`.
- It **disappears** if the interval includes zero.
- It **reverses** if the interval excludes zero with the opposite sign.

The units differ across measures, so only the persistence of the ordering is compared, not
magnitudes.

## 7. Sensitivity (fixed in advance)

- **Lag order:** K = 1 and K = 4 instead of 2.
- **Sample:** pre-2020 only (2009-Q3 to 2019-Q4). This also removes Spain's
  post-2021 definition flag.
- **Bootstrap:** block length 8.
- **H6:**
  - classical K = 2 `pandemic_excluded` sums instead of Bayesian medians;
  - without LU and IE, whose GDP is dominated respectively by commuters and by
    multinational accounting.

## 8. Verdicts and labels

**Prediction verdicts.**

- **Supported:** the Holm-adjusted bootstrap p-value is below 0.05 and the sign is the
  predicted one. The Holm family is all predictions P1.1-P5.2.
- **Contradicted:** p < 0.05 with the opposite sign.
- **Not determined:** otherwise.
- H6 has its own Holm family.

**What a verdict means.**

- "Supported" means the data show the predicted **pattern**, an association or margin of
  adjustment. It does not mean that the mechanism causes the country's Okun coefficient.
- No result is labelled causal. Identification is unavailable (IDENTIFICATION_STUDY.md).

**Evidence classes in the report.**

- **Correlation:** H6 and the descriptive companions.
- **Pattern consistent with a mechanism:** H1-H5.
- **Identified causal effect:** none.

## 9. Hypotheses that cannot be tested reliably (documented before any result)

- **Temporary contracts cause** the Dutch or Spanish response. There is no exogenous
  variation in contract regulation, and H4 shows only where adjustment falls.
- **The 2012 labour-market reforms** in Spain and Portugal. Each is a single event that
  coincides with the sovereign-debt crisis and adjustment programmes, so there is no
  credible counterfactual in this sample.
- **Short-time work** (the Dutch deeltijd-WW of 2009-2011, the Spanish ERTE). There is
  no harmonised quarterly take-up data before the pandemic.
- **Labour hoarding as opposed to unmeasured effort or capacity utilisation.** Measured
  productivity per hour cannot separate them.
- **Collective bargaining, minimum wages and benefit generosity.** These indicators are
  slow-moving, and there are too few countries to separate them from each other.
- **The unemployment of Luxembourg's cross-border workers.** It is recorded in France,
  Belgium and Germany, which LU data do not observe.
- **Sectoral composition.** Not attempted.

## 10. Outputs

- **Report:** `docs/HETEROGENEITY_CASE_STUDY.md`, generated; every number and verdict is
  computed.
- **Tables:** `docs/heterogeneity/`.
- **Figures:**
  - an evidence map (every prediction, with its interval and verdict);
  - the margin decomposition of the four countries;
  - Luxembourg's domestic and resident employment;
  - the crisis decomposition for ES and PT;
  - the cross-country correlations.

## Change log

- **During implementation, before any computation on real data: clarifications.**
  - *P1.2 is P1.1.* The commuting margin is defined as `e_dom_pe − e_res`, so its
    elasticity equals `β_e_dom_pe − β_e_res`. P1.2 is the same test as P1.1: it is
    evaluated once, under P1.1, and counted once in the Holm family, which therefore has
    14 predictions.
  - *Crisis companion.* The LFS series of most countries start in 2009, so the
    unemployment and participation columns of the crisis table begin at the first quarter
    with data, which is reported. GDP and the national-accounts margins cover the whole
    window from 2008-Q2 to 2013-Q1.
  - *H6.* An indicator without variation, or one with fewer than four countries, gets p = 1
    and "not determined" instead of a correlation.
  - *Vintages.* The national-accounts employment and hours series come from Eurostat's
    2026-10-15 update, the same day as the frozen GDP series but a later release. Any
    revisions between them are not reconciled.
  - *Snapshot.* `snapshots/heterogeneity-2026-10-18`, audited.
