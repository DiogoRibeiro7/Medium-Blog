# Finland 1986 and Germany 2008 component audit — 23 November 2026

This is a **documentary treatment-definition audit**, not a new
estimate of inequality, wages, economic growth or a causal tax-cut effect.

The author panel still has **30 country-year tax-cut flags**, of which
**20** previously passed the baseline WID/WDI donor eligibility rule.
Only **two** passed strict joint pre-reform three-WDI-mean donor-hull
support through year five: Finland 1986 and Germany 2008.

## Historical sources verified

| Dimension | Finland 1986 | Germany 2008 |
|---|---|---|
| Annual statute | Finlex 997/1985, original 1986 tax-scale legislation | Unternehmensteuerreformgesetz 2008, BGBl. I 2007 p. 1912 |
| Supporting official account | Ministry of Finance author in Bank of Finland Bulletin, April 1986 | Bundestag legislative archive and BMF Monthly Report, May 2008 |
| Exact statute signing date in this audit | Not verified for 997/1985; separate Finlex 1038/1985 dated 20 Dec 1985 | **14 August 2007** |
| Main corporate tax rate | **43% → 33%**, from 1986 | **25% → 15%**, from January 2008 |
| Other changes | Wealth-tax threshold **FIM 275,000 → 800,000**; capital-gains base broadened | Trade-tax factor **3.5%**; interest deduction restricted |
| Collection / phase-in caveat | Corporate advances stayed at **43% in 1986**, excess refunded in 1987 | Private capital-income withholding **25%**, effective January **2009** |
| Law-to-composite treatment assignment verified | **No** | **No** |

Primary links:
[Finlex 997/1985](https://www.finlex.fi/fi/lainsaadanto/1985/997),
[Finnish government HE 114/1985](https://www.edilex.fi/he/fi19850114.pdf),
[Bank of Finland April 1986](https://publications.bof.fi/items/005ef29d-d4bc-405a-879d-b3611dde7332),
[Finlex amendment 1038/1985](https://www.finlex.fi/fi/lainsaadanto/1968/360),
[Bundestag XVI/0221](https://dserver.bundestag.de/gm/16/160221.pdf),
[BMF May 2008](https://www.bundesfinanzministerium.de/Monatsberichte/2001-2016/Inhalte/Monatsbericht-Archiv-Downloads/2008/Monatsbericht_Mai_2008.pdf?__blob=publicationFile&v=1).

## Original empirical evidence, copied rather than re-estimated

| Original 20-event eligibility gate | Finland 1986 | Germany 2008 |
|---|---:|---:|
| Three-WDI strict donor hull through h=5 | True | True |
| Donors surviving original policy-censoring through h=5 | **8** | **16** |
| With three complete pre-five-year DPI covariates | **7** | **15** |
| Later major author tax-cut flag through h=5 | False | False |
| Top-1% income-share slope gap, pp/year | **−0.135** | **+0.496** |
| Composite-flag implementation date identified | False | False |
| Causal estimand identified | False | False |

Both examples have multiple tax-policy components. The Finnish
corporate rate was legally reduced for the assessment year, but
collection was delayed by retained advance payments. The German
package includes a 2009 phase that follows the *2008* coded event.
These are not one-day, one-instrument treatment assignments.

## Research decision

The evidence is sufficient to identify **historically plausible tax
components and statutory-period scopes for further study**. It is
insufficient to attribute either latent-index threshold crossing
to a particular component or to assert a counterfactual comparison
on annual pre-tax income shares.

The new audit validates eight original-law component records, checks
the unchanged two original event gates against the entire 20-event
study, and writes source-hashed CSV/JSON outputs. The CI job runs from
the same official WID/WDI source data and produces a seven-day artifact.

**Status:** no newly identified effect, no ATT, no p-values, no
confidence intervals, and no change to prior treatment, donor or
outcome inputs. See the
[methods and source-level qualifications](../POLICY_COMPONENT_CASE_STUDIES.md).

The frozen machine-readable audit will be checked against the exact
live GitHub Actions artifact on matching SHA-256 inputs. Original
source changes will not be presented as a successful replay.
