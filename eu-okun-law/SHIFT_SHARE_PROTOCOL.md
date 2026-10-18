# Country-specific GDP shocks from foreign demand: feasibility protocol

Written, committed and pushed before any value of the instrument and any power number
was computed. Changes after the first computation are listed in the change log at the
end, with reasons.

**Data work before this protocol was written.** Only availability and structure were
checked. No regression was run and no instrument was computed. The checks covered:

- **Comext** (`DS-045409`): access to bilateral annual exports.
- **OECD quarterly national accounts:** which candidate partners have seasonally adjusted
  quarterly real GDP growth from 2006. China starts in 2011-Q1, Saudi Arabia in 2010-Q2,
  and Russia ends in 2021-Q3.
- **OECD TiVA 2025 edition, "origin of value added in final demand"** (`FDVA`):
  - its structure and the codes of every area used below;
  - the 2006 shares of total value added absorbed by the United States, China, Japan, the
    United Kingdom and domestic final demand, printed for AT, DE, IE, NL and LU only.

## 1. Question

**Why this question.** The identification study
([docs/IDENTIFICATION_STUDY.md](docs/IDENTIFICATION_STUDY.md), sections 4.4 and 8)
concluded that country-specific GDP shocks need an instrument with two properties:

- it moves one country's GDP;
- it affects that country's unemployment only through GDP.

It named a shift-share foreign-demand instrument as the candidate.

**What comes first.** Before any estimate, this protocol settles two questions:

1. For which set of trading partners is the exclusion restriction defensible?
2. Can the instrument built on that set be strong enough to be informative in this
   panel?

A shift-share IV estimate is run, under a further protocol, only if both answers are
positive.

## 2. The instrument

```text
Z[i,t] = sum_{p in P} e[i,p] g[p,t]
```

- **Exposure `e[i,p]`.** The value added of country i absorbed in the final demand of
  partner p, divided by i's total value added, averaged over 2005-2007.
  - Source: OECD TiVA 2025 edition, measure `FDVA`, all activities, in current USD. The
    shares do not depend on the currency.
  - The years precede the 2009 start of the study period, so the shares are not affected
    by the outcomes in the sample.
- **Shift `g[p,t]`.** Partner p's quarterly real GDP growth: seasonally adjusted,
  quarter on quarter, in % (OECD quarterly national accounts).
- **Reading.** `Z[i,t]` is the predicted contribution of foreign demand to i's GDP growth,
  in pp, if p's final demand for i's value added grows with p's GDP.
- **Why value added, not gross exports.**
  - It includes services, which matter for LU, MT, CY and IE.
  - It includes indirect exposure through supply chains, e.g. Slovak value added in
    German cars sold in China.
  - It excludes the re-exports and quasi-transit that inflate Dutch and Belgian gross
    exports.

## 3. Partner sets and the exclusion restriction

The quarter effects of every specification absorb shocks common to all EU countries.
What remains is the interaction of different exposures with partner growth. Exclusion
fails if partner growth affects a country's unemployment other than through its GDP, or
if a shock correlated with partner growth hits more exposed countries through other
channels.

| Set | Partners | Main threats to exclusion | Judgement |
| --- | --- | --- | --- |
| **P1, distant demand (primary)** | USA, CAN, MEX, BRA, ARG, CHN, JPN, KOR, IND, IDN, AUS, ZAF | Global financial and commodity cycles whose non-trade effects differ with the same characteristics that raise exposure (manufacturing share, US financial links) | **Defensible with caveats.** No common labour market, no large migration or commuting flows, no energy dependence. Partly testable (section 5) |
| P2, near extra-EU | P1 + GBR, CHE, NOR, TUR | Migration and cross-border commuting (CH: FR, DE, IT, AT; UK: PL, LT, LV, RO; NO: SE, PL), so partner growth moves resident unemployment directly; Brexit; customs union with TR | Not defensible. Computed only to show what relevance the exclusion costs |
| P3, intra-EU leave-one-out | The other 26 EU countries (Eurostat quarterly snapshot) | Common EU shocks with country-specific loadings; spillovers from i to its partners (reflection); free labour mobility | Not defensible. Computed only for the same comparison |

**Excluded from every set.**

- Saudi Arabia and Russia: their growth tracks oil and gas supply, which affects EU
  countries directly as an energy shock. Russia's series also ends in 2021-Q3.
- Other European economies outside these sets: they are small, and the same mobility
  arguments as for P2 apply.

## 4. Sample

- **Countries:** the 27.
- **Origins:** 2012-Q1 to 2026-Q2. The instrument and its four lags need every P1 partner
  growth rate, and China's starts in 2011-Q1.
- **Pandemic:** as in IDENTIFICATION_PROTOCOL.md, an origin t is used only if no quarter
  from t-4 to t lies in 2020-Q1..2021-Q4.

## 5. Feasibility computation

No outcome enters the instrument. Unemployment is not used at all. The only domestic
quantity used is the residual standard deviation of GDP growth, which has been used
throughout the study.

**Identifying variation.**

- `Z~` is the residual of `Z[i,t]` on country effects, quarter effects and four lags of
  `Z`. The lags are the lag augmentation of the local projections.
- The report gives `S = sum Z~^2` over the sample.

**First stage.** `g[i,t] = beta Z[i,t] + a_i + d_t + controls + u[i,t]`.

- **Benchmark expected first-stage t:** `beta sqrt(S) / sigma_u`, with independent,
  homoskedastic errors.
- **Approximate probability that a 90% Anderson-Rubin (Fieller) set is bounded:**
  `P(|t + N(0,1)| > 1.645)`. A Fieller set is bounded exactly when the first-stage
  |t| exceeds its critical value.

**Calibration.**

- `beta = 1` (central): a 1 pp rise in `Z` raises GDP growth by 1 pp. This holds by
  construction of value-added shares, without multipliers. Also `beta = 0.5` and
  `beta = 2`.
- `sigma_u` (central): the residual standard deviation of quarterly GDP growth after
  country effects, quarter effects and four lags each of GDP growth, the unemployment
  change and the participation change. It is computed over the same origins, from the
  frozen snapshot `snapshots/eurostat-2026-10-15`. Also `sigma_u / 2`.

**Diagnostics computed without outcomes.**

- **Concentration:** the shares of `S` contributed by each partner (Rotemberg-type
  weights of the shifts) and by each country.
- **Balance:** the correlation of each country's P1 exposure with its 2005-2007 shares
  of industry (TiVA `VALU`, B-E) and of total exports in value added. High correlation
  means the exclusion restriction depends on sector-specific common shocks being absent.

## 6. Decision rule

- **Go.** A shift-share IV is estimated, under a further pre-registered protocol, only
  if the benchmark expected first-stage |t| for **P1** at the central calibration is at
  least 3.16 (F of at least 10).
- **No go.** Otherwise the design is recorded as infeasible for this panel, and the
  report gives what would be needed:
  - the pass-through `beta` that would reach |t| = 3.16;
  - the number of quarters needed at `beta = 1`, with `S` scaled in proportion.
- **Comparison sets.** P2 and P3 are reported for the trade-off between relevance and
  exclusion. They cannot be adopted, because their exclusion restrictions are judged not
  defensible in section 3.

## 7. Data

- **Exposure.** OECD TiVA 2025 edition, `DSD_TIVA_FDVA@DF_FDVA`, annual 2005-2007:
  - the 27 EU countries as sources;
  - the P1 and P2 partners, the 26 other EU countries and the world (`W`) as
    final-demand areas.
- **Balance.** TiVA `DSD_TIVA_MAINLV@DF_MAINLV` for industry value added and exports.
- **Shifts.** OECD quarterly national accounts `DSD_NAMAIN1@DF_QNA`: `B1GQ`, percentage
  change on the previous quarter, chain-linked volumes, seasonally adjusted, from
  2010-Q1, for the P1 and P2 partners.
- **P3 and `sigma_u`.** The frozen Eurostat snapshot `snapshots/eurostat-2026-10-15`.
- **Freezing.** All downloads are frozen in `snapshots/shift-share-<download date>/`,
  with the raw responses, a SHA-256 manifest and processed CSVs. OECD data are CC BY 4.0
  and are attributed.

## 8. Not covered

- The IV estimates themselves. They are run only after a "go", under a further protocol.
- Shocks other than foreign demand.

## Change log

- **During implementation, before any instrument value or power number: clarifications.**
  - *Exposure.* `e[i,p]` is the mean of the three annual ratios `FDVA(i,p) / FDVA(i,W)`.
    The world total `W` equals TiVA's total value added of i.
  - *Shifts.* The intra-EU set P3 takes its shifts from the GDP growth of the frozen
    Eurostat snapshot (`namq_10_gdp`, `CLV_PCH_PRE`, `SCA`), the same concept as the OECD
    shifts.
  - *`sigma_u`.* It is computed with a degrees-of-freedom correction: country effects,
    quarter effects and 12 lag controls.
  - *Snapshot.* `snapshots/shift-share-2026-10-18` was downloaded on 2026-10-18, and its
    audit passes.
