# Forbes 2026 historical-vintage check: March publication vs November API

## Outcome: the live API is not an exact March country distribution

Forbes published its 2026 billionaires list in March, with wealth values
measured as of **1 March 2026**. Its [11 March country ranking][forbes-ranking]
independently reports ten country/area counts and 3,428 billionaires across
80 countries and territories. The [10 March Forbes release][forbes-release]
also corroborates the total, and the counts for the United States, China and
India.

The previously archived [5 November API evidence][frozen-api] returns exactly
3,428 `countryOfCitizenship` values across 80 labels. **Matching those totals
does not establish that the geography values are historically identical.**
Three of the ten published March country counts contradict the November API.

| Forbes country/area | March published | November API | Secondary transcription | November minus March |
| --- | ---: | ---: | ---: | ---: |
| United States | 989 | 991 | 989 | +2 |
| China | 539 | 539 | 539 | 0 |
| India | 229 | 229 | 229 | 0 |
| Germany | 212 | 212 | 212 | 0 |
| Russia | 147 | 149 | 147 | +2 |
| Italy | 89 | 90 | 89 | +1 |
| Canada | 82 | 82 | 82 | 0 |
| Hong Kong | 71 | 71 | 71 | 0 |
| Brazil | 70 | 70 | 70 | 0 |
| Taiwan | 66 | 66 | 66 | 0 |

The published March top ten contains **2,494 people**. The November API
agrees for **seven** of those countries/areas and disagrees for **three**.
The incomplete secondary table agrees with **all ten** published counts.
This is evidence that the two extractions are not interchangeable. It does
**not** establish how or why the discrepancy arose, or authenticate the
remaining 70 country counts in the secondary table.

## Remaining country reconciliation

The November API contains 3,428 people, while the secondary transcription
contains 3,359, a gap of 69. After the Eswatini name alias is reconciled:

- 76 labels have counts in both series; 35 count pairs differ, with +64
  people net (+92 absolute counts) in the November API.
- Four November labels are absent from the secondary table: Afghanistan,
  Pakistan, Peru and St. Kitts and Nevis (five people combined).
- The 70 countries/areas not in the independent published top ten **have
  no independently established March count in this check**. Empty values
  in the report mean *not verified*, not zero.

Forbes' [March ranking][forbes-ranking] additionally mentions first-time
billionaires from Afghanistan and Pakistan. Their missing secondary entries
must not be imputed as zero or interpreted as proof of absence in March.
The published top-ten comparison rules out exact full-list equivalence for
the November API, but the cause of differences within other countries remains
unresolved.

## Scientific and publication gate

The evidence supports a **source-vintage discrepancy**, not a conclusion about
country residence, billionaires' effect on poverty, or a 64-country study
sample. `countryOfCitizenship` identifies a citizenship label, not verified
residence. UN-reviewed ISO3 codes are geographic mappings, not evidence that
wealth, population and poverty datasets use the same territory definitions.

Do not publish a refreshed chart, claim to reproduce the original claimed
`R²`, or rerun models as though the November API were a frozen March Forbes
release. Before new inferential work, recover an independently archived
complete March country list and the original chart's 64-country sample, and
align welfare measures by geography, survey year and PPP definition. The
historical list **remains unapproved for regression**.

## Reproduce using frozen aggregate-only files

The [March top-ten CSV](references/forbes_2026_march_top10_official.csv) is a
manually transcribed, SHA-256-pinned subset of the cited Forbes article. The
comparison program checks this reference against the SHA-verified November
country aggregates and the reviewed UN ISO3 ledger without fetching people.
It emits a diagnostic report, not an input to the existing regressions.

```bash
cd billionaires-poverty-audit
python -m unittest discover -s tests -p 'test_forbes*.py' -v
python forbes_march_vintage.py --out-dir results/forbes_march_vintage
```

Output files:

- `summary.json`: citations, dates, frozen-source hashes, the three
  contradictions, totals and explicit publication veto.
- `country_comparison.csv`: all 80 November citizenship labels, sourced March
  counts for ten rows only, explicit unknowns for the other 70, secondary
  count differences and `approved_for_regression=false` in every row.

Generation fails on altered source bytes or attempts to overwrite output.
No person-level Forbes details are persisted, and no article manuscripts are
modified by this audit.

[forbes-ranking]: https://www.forbes.com/sites/samellefson/2026/03/11/the-countries-with-the-most-billionaires/
[forbes-release]: https://www.forbes.com/sites/pr/2026/03/10/forbes-40th-annual-worlds-billionaires-list-elon-musk-is-worlds-richest-person-ever-recorded/
[frozen-api]: evidence/forbes-2026-run-42099896845/README.md
