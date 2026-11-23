# Annual author flag timing — preliminary source audit, 22 November 2026

This record intentionally distinguishes **two levels of verification**.
It is not a causal estimate, policy-date reconstruction, or a claim
that we have verified the implementation date of every reform.

## Verified on the existing archived WID/WDI panel

The already archived, 18-country WID/WDI join has 853 observed
country-years and **29 of the original authors' 30** 2-SD
tax-cut flag-years. The source also contains a separate, out-of-sample
South Korean series that the existing study excludes.

Grouping *those 29 WID-observed flags* into maximal consecutive
country-year sequences yields **24 descriptive runs**. There are five
two-year runs:

| Country | Consecutive 2-SD flag-years in the WID-observed subset |
| --- | --- |
| Australia | 1987, 1988 |
| Canada | 1972, 1973 |
| New Zealand | 1987, 1988 |
| Sweden | 1990, 1991 |
| United States | 1987, 1988 |

The **one author flag missing from the WID outcome join** is not yet
identified in this preliminary analysis; therefore **24 is not being
presented as the run count for all 30 original author flags**.
The live source job will validate the full source, identify that
missing country-year, and calculate all runs without imputing it.

These runs are an indicator-series property, **not necessarily
a count of separate policy acts** or a new treatment definition.

## Primary-law context verified for three flag-years

- **USA 1987 and 1988**: [Tax Reform Act of 1986](https://www.govinfo.gov/content/pkg/STATUTE-100/pdf/STATUTE-100-Pg2085.pdf)
  is Public Law 99-514, enacted **22 October 1986**.
  The [Congressional Research Service summary](https://www.everycrsreport.com/files/19861022_IP267T_c566df03603c63888edd6a88ef3fddcf60435355.pdf)
  discusses rate transition in 1987 and 1988.
- **GBR 1988**: [Finance Act 1988](https://www.legislation.gov.uk/ukpga/1988/39/enacted/data.xht?view=snippet&wrap=true)
  received Royal Assent **29 July 1988**; section 24 provides
  the 40% higher individual income-tax rate for 1988–89.

The **source law and enactment date** are documented. The exact
component in the authors' latent tax index responsible for each
flag, the dates those components became effective, and the
flag-to-law mapping have **not** been independently verified.
No exact implementation date is asserted.

## Reproduction and validation status

The PR adds a typed command, deterministic outputs, synthetic tests,
and a new GitHub Actions artifact, `tax-cut-annual-timing`.
The command uses the original author CSV downloaded by the existing
source-probe workflow, reconciles all country-year flags against the
WID/WDI outcome panel, and rejects missing policy values rather
than assuming they are zero. It preserves the original treatment
years and reports each source SHA-256.

**Status at authorship:** The new PR workflow had not yet started
on GitHub. Therefore the all-30-country-year numerical audit
and live result artifact are *pending*, and no frozen JSON
reproduction is claimed. Review the PR's latest Actions run
before merging, and only freeze its live JSON after verifying
the source SHA-256 and treatment counts.

The methodology is detailed in
[TAX_TIMING.md](../TAX_TIMING.md). Neither the earlier
+0.795-percentage-point exploratory contrast nor its
identification assumptions have been changed.
