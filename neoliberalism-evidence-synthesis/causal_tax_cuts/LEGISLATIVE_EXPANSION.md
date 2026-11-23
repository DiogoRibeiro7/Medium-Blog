# Additional primary-legislation evidence and feasibility gate

This audit follows the [annual flag timing](TAX_TIMING.md) and
[onset/phase-in](ONSET_PHASEIN.md) studies. It adds **four**
independently sourced statutory/provision examples in a *new*
ledger rather than altering the three existing cases or their frozen
source hashes. It describes documentary evidence coverage, **not**
identification of a causal tax-cut effect.

## Source-backed cases added

| Original composite flag | Independently verified statute/date | Relevant legal period | Evidence and limits |
|---|---|---|---|
| UK 1979 | Finance (No. 2) Act 1979, **26 July 1979** | Section 5: income-tax rates for **1979–80**, whose UK tax year began **6 April 1979** | The tax period predates assent; the Act says PAYE deductions need not change before **6 October 1979**. [Original enacted statute](https://www.legislation.gov.uk/ukpga/1979/47/pdfs/ukpga_19790047_en.pdf) |
| US 1982 | Economic Recovery Tax Act, **13 August 1981**, Pub. L. 97-34 | Title I section 101: individual rate schedule for tax years beginning in **1982**; earliest calendar-year taxpayer boundary **1 January 1982** | The rate reductions were phased. This provision is *not* proven to cause the composite 1982 flag. [Original enacted statute](https://www.govinfo.gov/content/pkg/STATUTE-95/pdf/STATUTE-95-Pg172.pdf) |
| Germany 2001 | Steuersenkungsgesetz, **23 October 2000** | Reduction of corporation tax to **25%**, effective **1 January 2001** | Multicomponent reform with other schedules. [Original Bundesgesetzblatt](https://www.bgbl.de/xaver/bgbl/start.xav?start=%2F%2F%2A%5B%40attr_id%3D%27bgbl100s1433.pdf%27%5D) and [Federal Ministry of Finance, February 2002](https://www.bundesfinanzministerium.de/Monatsberichte/2001-2016/Inhalte/Monatsbericht-Archiv-Downloads/2002/Monatsbericht_Feb_2002.pdf?__blob=publicationFile&v=1) |
| Netherlands 2001 | Wet inkomstenbelasting 2001 and implementing law, both **11 May 2000** | General commencement **1 January 2001**, subject to exceptions | Covers a broad tax reform, not an identified cause of the composite flag. [Original act (Stb. 215)](https://zoek.officielebekendmakingen.nl/stb-2000-215.html), [commencement (Stb. 216, Art. II)](https://zoek.officielebekendmakingen.nl/stb-2000-216.html) |

The previous cases remain unchanged: UK 1988 and US 1987/1988, representing
two additional statutes and three author flag-years.

**A statute date or provision-applicable tax period is not a verified
implementation date for a latent index of several tax instruments.**
The author-flag-to-provision mapping is unverified for all seven
reviewed flag-years. It would be wrong to align outcomes directly to
these dates or call them seven independent interventions.

## Fixed cohort and audit

The original Hope–Limberg 2-SD series remains **30 annual flags** and
25 contiguous flag-run starts. The WID/WDI donor cohort remains
**20 eligible onsets**, with its previously documented complete-outcome
and later-donor-censoring restrictions. Neither treatment nor matching
is altered.

The typed module `legislative_expansion.py` reads the original GitHub
Actions author flag register, the onset eligibility table, the
**unchanged original three-entry provision ledger** and its hashed
onset audit, then adds `legislative_expansion.csv`. It verifies that
all new references point to observed author flags, date and tax-period
fields are valid, sources use reviewed official domains, and no
causal attribution is claimed.

| Measurement | Source-verified result |
|---|---:|
| Author flags / original onsets / eligible onsets | **30 / 25 / 20** |
| Previously reviewed flag-years / additional flag-years | **3 / 4** |
| Flag-years with provision context | **7 / 30** |
| Run starts with legal context | **6 / 25** |
| Donor-eligible onsets with legal context | **5 / 20** |
| Distinct statutes | **6** |
| Reviewed provision periods beginning before enactment | **2** (UK 1979 and UK 1988) |
| Verified composite-flag-effective dates | **0 / 30** |
| Causal effects identified by this documentary audit | **0 / 20** |

Unreviewed means **not researched in this register**, not absence of
tax changes. Matching donors recur across multiple episodes. The earlier
joint WDI macro donor-hull check found only 2 of 20 eligible reforms
with strict joint support at year five; documentary evidence alone
does not resolve that problem.

## Reproduce

From the repository's `neoliberalism-evidence-synthesis/` directory,
after the existing official-source workflow:

~~~bash
python -m causal_tax_cuts.legislative_expansion \
  --flags data/tax-study/timing/author_flag_years.csv \
  --onsets data/tax-study/onset-phasein/tax_cut_onset_eligibility.csv \
  --legacy causal_tax_cuts/provision_periods.csv \
  --expansion causal_tax_cuts/legislative_expansion.csv \
  --previous data/tax-study/onset-phasein/tax_cut_onset_audit.json \
  --output-dir data/tax-study/legislative-expansion
~~~

The output is `legislative_review_all_flag_years.csv` (all 30,
including unreviewed cases) and `legislative_coverage_audit.json`
(source hashes, all seven context records, sample checks, and
explicit limits). CI rebuilds the audit from official-source
artifacts, compares the [frozen snapshot](results/legislative_coverage_2026-11-23.json)
**only when all input hashes match**, and uploads a seven-day
research artifact.

The next identification gate must be **policy-specific**: establish
the actual affected tax component, announcement/implementation
timing, treated taxpayer population, plausible untreated countries,
historical pretrends, institutional and fiscal confounders, and real
multivariate support. **Do not estimate an ATT solely because a
legal tax-period date exists.**
