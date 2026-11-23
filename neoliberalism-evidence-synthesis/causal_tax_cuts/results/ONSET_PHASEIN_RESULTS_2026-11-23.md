# Source-verified onset and legal-period audit — 23 November 2026

This is **descriptive treatment-measurement evidence**, not a new
counterfactual or ATT. Reproduced from the archived successful PR #747
GitHub Actions artifacts, using the identical 18-country Hope–Limberg
flag and WID/WDI donor-support source vintage.

## Original-source reconciliation

| Measurement | Verified count |
| --- | ---: |
| Original author 2-SD flag-years | **30** |
| Distinct contiguous runs, not necessarily independent laws | **25** |
| Continuation flag-years within those runs | 5 |
| Flag-years with observed WID outcome | 29 |
| Run starts with WID outcome | 24 |
| Run starts eligible for the previous WID/WDI donor model | **20** |
| Eligible starts with another major 2-SD cut at t+1..t+5 | **5** |
| Of those, next flag in the same contiguous run | 4 |
| Of those, next flag in a separate later run | 1 |
| Eligible starts without additional flag through t+5 | **15** |

The five eligible repeated-exposure paths are:

| Original onset | Next flagged year | Relation |
| --- | ---: | --- |
| Australia 1987 | 1988 | Same consecutive run |
| Canada 1972 | 1973 | Same consecutive run |
| New Zealand 1987 | 1988 | Same consecutive run |
| Sweden 1990 | 1991 | Same consecutive run |
| United States 1982 | 1987 | New, separate flagged run |

The **five** run starts not previously eligible: Denmark 1968 and Ireland
1977 (invalid treatment histories), Italy 1972 (missing WID outcome), New
Zealand 1972 (missing pre-treatment WDI history), and USA 1987 (no eligible
pre-treatment macro donors). All **five** continuation flags had previously
been classified as invalid treatment histories by the existing matcher.
Thus the new run-first-year criterion **does not revise** the prior
20-event eligible cohort.

## Documented legal periods

| Context | Statute enactment | Relevant statutory-period start | What can be inferred? |
| --- | --- | --- | --- |
| US 1987, individual rates | 22 Oct 1986 | 1 Jan 1987 (earliest start for calendar-year taxpayers) | US Title I applies generally after 1986; not exact composite-flag effective date |
| US 1988, individual rates | 22 Oct 1986 | 1 Jan 1988 (calendar-year schedule boundary) | A later rate schedule of the 1986 act; not independent proof of 1988 flag attribution |
| UK 1988, 40% higher rate | 29 Jul 1988 | 6 Apr 1988 (assessment year 1988–89) | The statute addresses a tax year beginning before enactment; anticipation/implementation remains unresolved |

Exactly **3 of 30** author flag-years have reviewed statutory-period
examples, corresponding to two statutes. One statute period starts before
enactment. **0 of 30** have verified effective days attributable to the
composite author index. The remaining **27 flags** are not documented in
this limited legal-period register.

## Source provenance

- Original author file SHA-256: `3f1bc22a526d3beff2409de30e5644c6dc696b78614b46bede148cb80f7ad653`.
- Existing WID/WDI joined panel SHA-256: `8dfcdd2bb7ea8c378436d5415a70ae0a83f8f1b431f4e46c70039bf578072321`.
- Original verified flag-year CSV: `4f0baf1afaabe39f6461ec7ac1cb5ae1f00e8d32c7e1dd84eb50a9ad3dc89c61`.
- Previous donor-eligibility CSV: `a338900e2c2de76568bf37704fba75010f9df1c3a9c4a651be5dcc3ce0d55623`.
- Official period context and all other input hashes are recorded in
  [the machine-readable frozen audit](onset_phasein_2026-11-23.json).

Methodology, limits, reproduction command and official legal references:
[ONSET_PHASEIN.md](../ONSET_PHASEIN.md). The workflow regenerates the
same summary from official source inputs and publishes a new ephemeral
source-hashed data artifact; source-vintage drift is detectable.
