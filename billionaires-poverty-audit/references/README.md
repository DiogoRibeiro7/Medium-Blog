# UN geographical-code reference for Forbes citizenship labels

This is a separately sourced, reviewable crosswalk for the **80 exact labels**
in the 5 November 2026 Forbes `countryOfCitizenship` aggregate. It does not
modify the frozen [primary audit](../evidence/forbes-2026-run-42099896845/README.md).

The CSV `forbes_2026_un_country_code_review.csv` records the original Forbes
label, ISO-alpha3 code, three-digit M49 area code, official UN area name,
evidence URL, name-review classification and explicit territory-review flag.
The reference was compiled against official live UN metadata on **6 November
2026**; it is a reviewed code mapping, **not** a frozen copy of the UN website.

## Authoritative references

- [UN Statistics Division, M49 country or area codes (English)](https://unstats.un.org/unsd/methodology/m49/overview/): 79 of the 80 area-code records. This list covers both states and other statistical areas; its codes do **not** establish sovereignty.
- [UN Population Division, Taiwan area metadata](https://population.un.org/dataportal/data/indicators/19/locations/158/start/1980/end/2026/metadata/metadatalocations): separately documents `TWN`, M49 ID `158`, and its own official area-name convention. The UNSD overview does not enumerate this entry separately.

There are **66 exact name matches**, **13 independently checked name variants**
and **one separately referenced UN Population Division area entry**. Every
ISO3 code agrees with the previously proposed candidate, including the five
labels absent as exact names in the secondary transcription. This validates
the **area-code assignments** against the cited UN entries; it does not
verify Forbes's historical March vintage or the meaning of each citizenship
attribution.

Examples of nonidentical names, checked against the source records:

| Forbes label | UN area name | ISO3 | M49 |
| --- | --- | --- | --- |
| `Eswatini (Swaziland)` | Eswatini | `SWZ` | `748` |
| `St. Kitts and Nevis` | Saint Kitts and Nevis | `KNA` | `659` |
| `Czech Republic` | Czechia | `CZE` | `203` |
| `South Korea` | Republic of Korea | `KOR` | `410` |
| `Hong Kong` | China, Hong Kong Special Administrative Region | `HKG` | `344` |
| `Taiwan` | China, Taiwan Province of China (UN Population Division metadata) | `TWN` | `158` |

Guernsey, Hong Kong and Taiwan are additionally flagged for an explicit
**territory/area compatibility review** before any statistical join. Country
and area codes do not imply that population, GDP, poverty or sovereignty
definitions are interchangeable between sources.

## What remains unresolved

The secondary citizenship transcription contains **3,359**, while the
primary Forbes person-list capture contains **3,428** people. After
reconciling `Eswatini` as a name alias, **76** country labels have secondary
counts, **35** of those differ, and their net difference is **+64**.
Four primary labels absent from the secondary table (Afghanistan, Pakistan,
Peru and St. Kitts and Nevis) contain **five** more people, so the total
discrepancy remains **+69**. This explains the arithmetic, **not the cause**:
we have not independently established whether the secondary transcription
uses a different list vintage, record selection, or counting policy.

Even with the ISO3 references established, the following remain blockers:

1. Independently establish Forbes's **March 2026 historical list vintage**
   relative to the live November snapshot.
2. Verify geographical semantics and territory-to-population comparability;
   `countryOfCitizenship` is not automatically residence.
3. Recover the original chart's **64-country selection** and align GDP and
   poverty observations by survey year and purchasing-power standard.
4. Investigate the 35 nonzero primary-versus-secondary differences without
   changing counts merely to match an expected result.

The generated `forbes_iso3_review.py` ledger keeps
`approval_status=not_approved_for_regression` for every row. The existing
research models still use their separately labelled secondary exploratory
input; **no regression has been rerun on this crosswalk**.

## Reproduce offline

```bash
cd billionaires-poverty-audit
python -m unittest discover -s tests -p 'test_forbes_iso3_review.py' -v
python forbes_iso3_review.py --output results/forbes_iso3_review.csv
```

The generator refuses altered primary, secondary or reference CSV bytes and
will not overwrite an existing output. A changed UN code reference requires
creating a new documented crosswalk revision, not silently replacing the
2026-11-06 review.
