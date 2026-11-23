# Tax-cut onset, repeated exposure and statutory tax periods

This audit follows the verified annual tax-cut timing audit in PR #747. It
answers two separate measurement questions without changing the original
Hope–Limberg treatment coding: **which flagged country-years are the first
of a consecutive run**, and **what legal tax-period information is genuinely
available for the selected US and UK examples?** A run is not a unique law.

## Fixed sample and definitions

The [original Hope–Limberg author dataset](https://www.taxrich.uk/data)
contains 30 major 2-SD declines in a composite tax index in 18 countries
between 1965 and 2015. It is an **annual** indicator, not a legislative
register. The prior workflow provides its verified 30-row flag register,
including 25 contiguous flag runs, and the unmodified 29-row WID-observed
reform cohort. One original flag, Italy 1972, lacks the WID outcome.

For a flagged country-year t, `first_flag_year` means that the same
country was **not flagged at t−1**. Consecutive flagged years beginning at
t form a descriptive *run*. This is compatible with the prior matching
requirement that treatment at t−1 was zero. It does not identify a unique
statute, tax-law dosage, announcement or effective date. There are 25 run
starts and 5 continuation flags.

The input `support_reform_eligibility.csv` is **not recomputed**. It supplies
the donor-eligibility status and year-five donor count from the earlier
WID/WDI analysis, preserving its sample-selection caveats. On the frozen
vintage, only **20 of 25** run starts satisfy those previous eligibility
rules. The other five comprise two with invalid treatment history, one
incomplete pre-reform macro record, one with no pre-reform macro donors,
and one missing the WID outcome. These are statuses inherited from the
original workflow, **not new exclusion decisions**.

For each of the 20 eligible run starts, the audit searches all author-coded
flags from t+1 through t+5, without requiring their WID outcome to be
observed. It records whether the **next** flag is a continuation of the
original consecutive run or belongs to a separate later run. A flag in a
separate run is *not necessarily* an independent statute. The result is
15 eligible starts with no further 2-SD major-cut flag during follow-up,
4 with a continuation the next year, and 1 with a later separate run.
This audit does not remove or censor any observation from the main estimator.

## Statutory period examples (not flag-specific effective dates)

The source-linked [provision period register](provision_periods.csv) holds
**three** entries, related to two historically documented statutes. It
is deliberately distinct from [the earlier enactment-date
register](statute_context.csv). Each row is marked
`provision_period_verified_author_flag_attribution_unverified`:

| Author flag | Primary legal source and tax-period evidence | Meaning and limitation |
| --- | --- | --- |
| USA 1987 | [Tax Reform Act of 1986](https://www.govinfo.gov/content/pkg/USCODE-2023-title26/html/USCODE-2023-title26-subtitleA-chap1-subchapA-sec1.htm), Title I §151(a), applies generally to taxable years **beginning after 31 December 1986** | The register records 1 January 1987 as the **earliest possible start for a calendar-year taxpayer**. Other taxpayers' periods and exceptions may differ. It is not a proven effective date for the 1987 composite flag. |
| USA 1988 | [Tax Reform Act of 1986](https://www.govinfo.gov/content/pkg/STATUTE-100/pdf/STATUTE-100-Pg2085.pdf), §101 contains the restructured individual tax rate schedule for taxable years beginning in 1988 | 1 January 1988 is the calendar-year period boundary only. The 1988 flag may reflect multiple components; assigning this one provision as its cause is unsupported. |
| GBR 1988 | [Finance Act 1988 §24](https://www.legislation.gov.uk/ukpga/1988/39/section/24/enacted), 40% higher income tax rate for **1988–89**, with UK assessment year beginning **6 April 1988** | The act received Royal Assent **29 July 1988**, after the statutory tax period began. This does not establish when people anticipated or first experienced the new rate, or which source variables crossed the author threshold. |

The tax-year boundary for GBR 1988 follows the [Income and Corporation
Taxes Act 1988 definition](https://www.legislation.gov.uk/ukpga/1988/1/part/XIX/1993-07-27)
of 1988–89 as the year beginning 6 April 1988. The US Title I §151(a)
general effective-date rule is preserved in the [US Code historical
notes](https://www.govinfo.gov/content/pkg/USCODE-2023-title26/html/USCODE-2023-title26-subtitleA-chap1.htm).

All three examples have documented **statutory period rules**, and one
(GBR 1988) has a statutory period beginning before enactment. **None**
establishes the precise implementation date of the authors' composite
binary flag, and 27 of 30 flag-years have no provision-period entry.
No date is imputed for those years.

## Reproduction

From `neoliberalism-evidence-synthesis/`, after the existing WID/WDI,
donor-support and annual timing workflow steps:

~~~bash
python -m causal_tax_cuts.onset_phasein \
  --flags data/tax-study/timing/author_flag_years.csv \
  --support data/tax-study/donor-support/support_reform_eligibility.csv \
  --provisions causal_tax_cuts/provision_periods.csv \
  --timing-audit data/tax-study/timing/author_timing_audit.json \
  --donor-audit data/tax-study/donor-support/donor_support_audit.json \
  --output-dir data/tax-study/onset-phasein
~~~

The command produces `tax_cut_onset_eligibility.csv` (25 rows),
`tax_provision_periods.csv` (3 entries) and
`tax_cut_onset_audit.json` (source SHA-256 manifest, count reconciliation,
complete follow-up classifications and explicit no-causal-inference status).
If the two previous audits do not agree on the joined source vintage, or
if any inherited event count fails reconciliation, the command **fails
closed**. The GitHub Actions workflow generates a seven-day
`tax-cut-onset-phasein` artifact and verifies the frozen JSON **exactly on
matching input-source hashes**; a changed vintage is labelled as changed,
not silently presented as replication.

## Identification limits

A series of contiguous author flags could arise from a phased law or
multiple separate reforms. The number of **one-cut-through-five-year paths
(15)** refers only to the 2-SD composite indicator. It does not mean there
were no contemporaneous fiscal, labour or capital-market reforms in those
countries. The existing dynamic donor selection remains asymmetric;
source selection still demands complete WID windows out to year five.

The earlier joint macroeconomic donor-hull audit found only **2 of 20**
reform episodes with strict joint macro support at year five. These
limitations rule out treating this diagnostic as an adjusted causal
estimate. To estimate a policy effect, future work must validate a
specific intervention and effective period, its treated population,
anticipation and time-varying policies, then reevaluate exchangeability
and positivity on the **actual policy intervention**.
