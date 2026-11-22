# Tax-cut flag-years versus legislative timing

This is an audit of **measurement and timing**, not a revised causal model.
The Hope–Limberg indicator reports a 2-standard-deviation annual decline
in a latent index of taxes on top incomes, capital and assets. In the
original 18-country, 1965–2015 panel, there are **30 coded country-years**.
These are not necessarily **30 independent legislative acts** with
implementation dates in those same years. The source CSV contains no
announcement, enactment or effective-date columns.

References: [Hope and Limberg (2022)](https://doi.org/10.1093/ser/mwab061),
[authors' data](https://www.taxrich.uk/data).

## Source contract and method

The typed audit code is in [tax_timing.py](tax_timing.py). It loads the
original author CSV and requires all 18 × 51 country-year indicator values,
the original binary field and exactly 30 positive flags. Only the
out-of-sample South Korean country series is excluded. **Unobserved
country-year indicators are never treated as zero.**

We compare each flag against the existing WID/WDI joined outcome panel,
checking that the joined treatment indicator matches the original author
value. An unavailable WID outcome does not erase or invent an author flag.

For every country, the code groups maximal consecutive annual flags into
**descriptive runs**. A two-year run can represent phased implementation,
two separate laws, or shifts in several underlying tax components. It
must not be collapsed or counted as one unique law without documentary
evidence. The original flagged year is never moved.

The outputs include an author-year register, a run table and a JSON audit
with original-source SHA-256 hashes, coverage and missing-event IDs.
A live GitHub Actions job reproduces the results from the original authors'
CSV and the same joined source. No raw author data are committed.

## Selected primary-legislation context

A small, separately reviewed
[statute context register](statute_context.csv) records two **illustrative
primary statutes**, affecting three author-coded rows.

| Flag-year | Statute and established enactment | Limitation |
| --- | --- | --- |
| USA 1987 and 1988 | [US Tax Reform Act of 1986](https://www.govinfo.gov/content/pkg/STATUTE-100/pdf/STATUTE-100-Pg2085.pdf), **22 October 1986** | The US statute predates both coded years. Provisions were phased in; which component(s) made each latent-index threshold cross is not independently verified. |
| GBR 1988 | [UK Finance Act 1988](https://www.legislation.gov.uk/ukpga/1988/39/enacted/data.xht?view=snippet&wrap=true), **29 July 1988** | Section 24 applies a 40% top income tax rate to tax year 1988–89; this is not a verified daily implementation date for the composite author flag. |

The [Congressional Research Service 1986 Act summary](https://www.everycrsreport.com/files/19861022_IP267T_c566df03603c63888edd6a88ef3fddcf60435355.pdf)
identifies 1987 as transitional and 1988 as the full-rate year.
This illustrates why enactment, tax year, rate phase-in and the author
flag-year must be kept distinct. The evidence register explicitly marks
the *attribution between a statute and a composite flag as unverified*.
**Zero flag-specific effective dates have been verified** in this audit.

## Reproduce

From the neoliberalism-evidence-synthesis directory:

~~~bash
python -m causal_tax_cuts.tax_timing \
  --author-csv data/tax-events/hope_limberg_reforms_11_2020.csv \
  --joined-panel data/wdi-source/outcome_panel_with_macro.csv \
  --statute-ledger causal_tax_cuts/statute_context.csv \
  --frozen-reference causal_tax_cuts/results/dynamic_exposure_2026-11-16.json \
  --output-dir data/tax-study/timing
~~~

The source-probe CI already downloads the original author file and
rebuilds the panel. This additional step produces:

- author_flag_years.csv — 30 flags; source year, run position and WID availability;
- author_consecutive_flag_runs.csv — descriptive contiguous country-year runs;
- author_timing_audit.json — exact counts, missing cases, source hashes and caveats.

The first live summary is in
[results/TAX_TIMING_RESULTS_2026-11-22.md](results/TAX_TIMING_RESULTS_2026-11-22.md).
Its machine-readable provenance is in
[results/author_timing_2026-11-22.json](results/author_timing_2026-11-22.json).

## Identification implications

The previous event study compared WID outcome changes at calendar
t+h against t−1. The **within-year** effect of a legislative change
is not identified by one annual composite flag. Anticipation, delayed
commencement, fiscal-year boundaries and reforms affecting multiple
components might blur the timing.

Furthermore, the earlier sensitivity work found five of twenty eligible
reform episodes followed by another major cut within five years, while
only two of twenty reforms remain within the three-variable joint macro
donor hull at year five. These are independent concerns about dynamic
treatment and support, not remedied by shifting event years arbitrarily.

**Do not** interpret the number of coded years as the number of
independent laws; change event dates without primary evidence; or publish
the previously exploratory +0.795pp contrast as a causally identified ATT.
Next, investigate policy-specific announcement, enactment and
component-specific effective dates, and specify a feasible estimand
before any new causal effect estimation.
