# The Right to Vote Was Only the Beginning

Reproducible companion to **[The Right to Vote Was Only the Beginning: Civil Rights, the Great Society, and the Limits of Reform](article.md)**, a long-form Medium-ready historical essay. It follows earlier articles on the Gilded Age, Progressive Era, New Deal, and political civility.

The manuscript treats civil rights and the Great Society as an interacting set of institutions—not a single intervention or a simple before-and-after experiment. It distinguishes **legal eligibility**, **enforcement and realized access**, **public-service or insurance coverage**, and **changes in observed outcomes**. No chart identifies the causal effect of a law from temporal adjacency alone.

## Reproduce locally

Python 3.10 or later, from this folder:

```bash
python -m pip install -r requirements.txt
python analysis.py
python -m unittest discover -s tests -v
```

For an isolated destination:

```bash
python analysis.py --output /tmp/great-society-figures
```

The script generates **eight original publication figures** under [figures/](figures/) and **three machine-readable result files** under [results/](results/), including a typed evidence-class column. The [Jupyter walkthrough](model_walkthrough.ipynb) imports the same functions from [analysis.py](analysis.py); it does not reproduce logic separately. A [scoped GitHub Actions workflow](../.github/workflows/great-society-article.yml) tests the sources, model contracts, manuscript, notebook structure, and regenerated figures.

Only the checked-in CSV transcriptions are required to regenerate visuals: **no API token, network download, random seed, hidden data file, or interpolated observation** is needed.

## Figure evidence map

| Figure | Classification | Basis | Scope and limitation |
| --- | --- | --- | --- |
| 1. National poverty | Historical descriptive | Census historical Table 2 | All-person official cash-income poverty rate, 1959–1975; not a legislative effect |
| 2. Racial poverty | Historical descriptive | Census historical Table 2 | White annual 1959–75; Black 1959 and 1966–75 only, **1960–65 unavailable** |
| 3. Age poverty | Historical descriptive | Census historical Table 3 | Under-18 annual 1959–75; age 18–64 and 65+ **1960–65 unavailable** |
| 4. Medicare enrollment | Historical administrative | CMS Hospital Insurance historical enrollment | Enrolled persons as of July 1, 1966–75; disabled category starts 1973 |
| 5. Chronology | Selected dated original events | King Institute, LOC, NARA, EEOC, CMS, LBJ Library, HUD | Not a full chronology, causal sequence, or estimate of program exposure |
| 6. Registration-access friction | **Hypothetical** | Invented eligible population and probabilities | Not an estimate of registration, turnout, or VRA effectiveness |
| 7. Benefit cliff and phaseout | **Hypothetical** | Invented $9,000 benefit, $18,000 cutoff, 35% taper | Not any real benefit schedule or a labor-supply estimate |
| 8. Cash versus noncash resources | **Hypothetical** | Invented cash income, threshold, assigned noncash value | Not the official or Supplemental Poverty Measure |

## Historical source transcriptions

### Census historical Table 2

Original source: [Historical Poverty Tables: People](https://www.census.gov/data/tables/time-series/demo/income-poverty/historical-poverty-people.html), U.S. Census Bureau, [source workbook `hstpov2.xlsx`](https://www2.census.gov/programs-surveys/cps/tables/time-series/historical-poverty-people/hstpov2.xlsx), worksheet `pov02`. The files were accessed in October 2026; the site exposed the current historical tables incorporating its 2025 vintage.

[census_poverty_all.csv](data/census_poverty_all.csv) transcribes the 1959–1975 entries in the workbook's *All Races* section; the relevant original column is the **percent below poverty**, not count in thousands. [census_poverty_race.csv](data/census_poverty_race.csv) transcribes the same statistic for the *White* and *Black* sections. Rates are percentages of persons, not percentages of households. Years that carry original worksheet footnote markers (e.g., “1974 (21)”) are recorded by their calendar year only; the source workbook's methodological footnotes remain relevant if extending or combining the historical series.

The Black section reports 1959 and 1966–1975 for this interval, but **no Black 1960–1965 entries**. The transcription writes empty cells, the reader produces `np.nan`, and matplotlib does not connect the Black line across these unavailable years. In particular, the visual is not a straight-line reconstruction of an unobserved 1960s trend.

| Check | Official rate |
| --- | ---: |
| All people, 1959 | 22.4% |
| All people, 1963 | 19.5% |
| All people, 1969 | 12.1% |
| All people, 1973 | 11.1% |
| White, 1959 | 18.1% |
| Black, 1959 | 55.1% |
| White, 1975 | 9.7% |
| Black, 1975 | 31.3% |

The 1959 Black–White rate difference is 37.0 **percentage points**, and the 1975 difference 21.6 percentage points. These are descriptive gaps; neither one estimates a legal intervention's effect.

### Census historical Table 3

Original source: the same [Census historical poverty tables page](https://www.census.gov/data/tables/time-series/demo/income-poverty/historical-poverty-people.html), [`hstpov3.xlsx`](https://www2.census.gov/programs-surveys/cps/tables/time-series/historical-poverty-people/hstpov3.xlsx), worksheet `pov03`, *All Races* section. [census_poverty_age.csv](data/census_poverty_age.csv) contains poverty-rate columns for **under 18**, **18–64**, and **65 or older**.

For **1960–1965**, the original workbook explicitly gives `N` for the 18–64 and 65+ rate fields; this transcription records these as empty cells, not zero and not an interpolated estimate. The under-18 series exists in every year of the displayed interval. Test assertions protect these gaps.

| Check | Official rate |
| --- | ---: |
| Age 65+, 1959 | 35.2% |
| Age 65+, 1966 | 28.5% |
| Age 65+, 1975 | 15.3% |
| Under 18, 1959 | 27.3% |
| Under 18, 1969 | 14.0% |
| Under 18, 1975 | 17.1% |

For additional independent reading, Census's [1979 historical compilation](https://www2.census.gov/library/publications/1979/demographics/p23-080.pdf) provides separate racial-and-age comparisons for selected years. Its cross-tabulations are **not** appended to these time series, as mixed source constructions could create artificial trends.

The Census [poverty-measures guidance](https://www.census.gov/topics/income-poverty/poverty/guidance/poverty-measures.html) explains that the official measure counts pre-tax cash income, including cash Social Security, while generally excluding noncash medical insurance, housing and nutrition assistance, and tax-credit effects. None of the figures should be interpreted as a comprehensive measure of all Great Society benefits.

### CMS hospital-insurance enrollment

[cms_medicare_enrollment.csv](data/cms_medicare_enrollment.csv) transcribes the **1966–1975 July 1** annual counts of total, aged, and disabled Hospital Insurance enrollees from the [CMS original historical HI table, 1966–2010](https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/MedicareEnrpts/downloads/HI2010.pdf). The numeric columns report **persons**, not thousands or millions; Figure 4 explicitly converts counts to millions.

The aged count in 1966 is 19,082,454 and total in 1975 is 24,640,497. Disability enrollment enters the series in 1973 at 1,730,538, reflecting the entitlement extension rather than a sudden discovery of an older missing measure. The disabled column is left empty in 1966–1972; during those years total equals aged enrollment. From 1973 onward, the reported total equals the sum of aged and disabled categories. The tests check both conditions.

Enrolled-person counts are not coverage percentages, utilization, spending, quality, treatment access, mortality, or money income. The essay relies on the [CMS 1985 historical account](https://www.cms.gov/files/document/hcfr-85-supp-013pdf) for legislative and operational dates and the [CMS Medicare retrospective](https://www.cms.gov/research-statistics-data-and-systems/research/healthcarefinancingreview/downloads/cms1191479dl.pdf) plus [original presidential statements](https://www.cms.gov/History/downloads/CMSPresidentsSpeeches.pdf) for how Title VI certification was connected to participation. A contemporary reported bed-compliance percentage is not a direct patient-level estimate of equity.

### Dated institutional events

[institutional_events.csv](data/institutional_events.csv) transcribes **ten selected dates** from the [Stanford King Institute](https://kinginstitute.stanford.edu/letter-birmingham-jail), [Library of Congress](https://www.loc.gov/item/today-in-history/august-28/), [National Archives](https://www.archives.gov/milestone-documents/civil-rights-act), [LBJ Presidential Library](https://www.lbjlibrary.org/object/text/remarks-upon-signing-economic-opportunity-act-08-20-1964), [CMS](https://www.cms.gov/files/document/hcfr-85-supp-013pdf), [HUD](https://archives.hud.gov/local/ia/goodstories/2012-04-25.cfm), and [EEOC](https://www.eeoc.gov/history/equal-employment-opportunity-act-1972). Each row includes an explicit primary-source URL and ISO date. The event list is selective and should never be read as a complete timeline of organizing, legislation, litigation, or implementation.

## Hypothetical mathematical contracts

**Registration access.** In an invented population `N`, with hypothetical legal eligibility share `e` and administrative barrier `b`:

\[
A(N,e,b)=N e(1-b),\quad 0\leq e,b\leq1.
\]

The figure uses `N=100000`, eligibility scenarios 0.65, 0.85, and 1.00, and barrier levels 0–1. It omits registration effort, turnout, preference formation, intimidation and endogenous responses. The 303 exported rows are tagged `hypothetical`.

**Cash-benefit cliff.** An invented annual maximum benefit `B=9000`, cutoff `T=18000`, and hypothetical phaseout rate `r=0.35` produce two alternatives:

\[
B_{cliff}(y)=B\mathbf{1}(y\leq T),\qquad
B_{taper}(y)=\max\{0,B-r\max(0,y-T)\}.
\]

Resources equal `y+B(y)`. The figure illustrates a discontinuity under one **fictional** schedule; the other is a continuous gradual phaseout. It is not a historical or modern program parameterization. The 601 rows are tagged `hypothetical`.

**Cash versus assigned noncash value.** The intentionally naive illustration stipulates cash income `C=18000`, a threshold `T=25000`, and varied assigned noncash value `v`. It contrasts `1(C<T)` with `1(C+v<T)`. It neither estimates the welfare value of medical insurance nor implements the Census Supplemental Poverty Measure. The source-based result table never contains these invented figures.

## Generated result files

- [source_based_summary.csv](results/source_based_summary.csv): 132 source table/column-year entries, each with year, source-series label, value, and `historical_descriptive` or `missing` evidence class. The **25 missing** entries are retained for audit but never plotted as numerical zero.
- [hypothetical_voter_access.csv](results/hypothetical_voter_access.csv): 303 model outputs.
- [hypothetical_benefit_schedule.csv](results/hypothetical_benefit_schedule.csv): 601 model outputs.

The source summary includes **17 national, 34 racial, 51 age, and 30 CMS** year–column entries. There are 6 missing Black years, 12 missing adult/elderly age entries, and 7 pre-1973 disability category entries: 25 blanks in total.

## Interpretation checklist

1. Changes after legislation are **temporal associations**, not causal effects by default; several poverty rates fell before major 1964–65 bills.
2. Black–White differences are reported in **percentage points**, not the ratio of rates.
3. An annual poverty rate is not a measure of the probability that an individual person remains poor, nor a measure of wealth.
4. Historical poverty source tables may be revised, and their footnotes and category definitions matter when comparing vintages.
5. Administrative enrollment counts should not be converted to a coverage percentage without a matched eligible-population denominator.
6. Medicare's largely noncash coverage does not mechanically enter the official poverty resource calculation; Social Security cash payments can.
7. Medicare and Medicaid have materially different institutional designs and should not be merged into a single number.
8. A dated law and the enforcement capacity that makes it usable may arrive in different years.
9. Hypothetical illustrations are labeled in **figure, code, output table, caption, and manuscript**; they are not reconstructed historical effects.
10. The cited [historical SPM](https://www.nber.org/papers/w19789), [full-income](https://www.nber.org/papers/w26532), and [long-run dependency](https://www.nber.org/papers/w34759) studies answer different measurement questions and should not be treated as directly interchangeable estimates.

**Reproducible does not mean causally identified.** It means the source values, definitions, missingness, transformations, and illustrative assumptions are visible and testable.
