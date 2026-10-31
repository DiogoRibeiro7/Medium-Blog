# Historical midterm data: transcription and provenance

## Frozen forecast date and outcome definition

This study freezes the 2026 forecast at **31 October 2026**, before the 3
November 2026 midterm elections. **No 2026 outcomes** enter the training data.

The dependent variable is the American Presidency Project's *House seat change
for the president's party*, not the Democratic seat total or a national vote
share. The second predictor, *seats defended*, is the number of House seats held
by that same party before the election. Both columns come from the [American
Presidency Project midterm table](https://www.presidency.ucsb.edu/statistics/data/seats-congress-gainedlost-the-presidents-party-mid-term-elections),
which in turn cites the House Historian's [Party Divisions of the House of
Representatives](https://history.house.gov/Institution/Party-Divisions/Party-Divisions/).

The generic-ballot column is always **Democratic minus Republican points**.
The model multiplies it by +1 for a Democratic president and -1 for a
Republican president, so a positive fitted predictor consistently represents
an advantage for the president's party. This is a percentage-point difference,
not a two-party vote percentage, presidential approval rating, or seat share.

## Source selection, exactly as transcribed

The historical table contains 18 elections from 1950 through 2022. Its first
12 generic ballot observations (1950–1998) come from Gallup's [25 September
2002 retrospective table](https://news.gallup.com/poll/6877/democrats-enjoy-slight-lead-race-control-congress.aspx)
headed *Accuracy of Gallup Generic Ballot in Predicting the National Two-Party
Vote*, using the **% Democratic** column and the conversion `2*percent - 100`.
The 2002 election is from Gallup's [4 November 2004 retrospective
table](https://news.gallup.com/poll/13879/control-us-house-looks-too-close-call.aspx)
headed *Gallup Accuracy in Forecasting Republican Share of National Two-Party
Vote*, using the final Gallup **53.0% Republican** and the conversion
`100 - 2*percent = -6.0` Democratic-minus-Republican points.

The five modern elections use the last published **RCP Average**, with the
sign converted to Democratic-minus-Republican percentage points:

| Election | Transcribed final RCP spread | Original polling page |
| --- | ---: | --- |
| 2006 | D +11.5 | [RealClearPolitics](https://www.realclearpolitics.com/epolls/other/2006_generic_congressional_vote-2174.html) |
| 2010 | R +9.4 | [RealClearPolitics](https://www.realclearpolitics.com/epolls/other/2010_generic_congressional_vote-2171.html) |
| 2014 | R +2.4 | [RealClearPolitics](https://www.realclearpolitics.com/epolls/other/generic_congressional_vote-2170.html) |
| 2018 | D +7.3 | [RealClearPolitics](https://www.realclearpolitics.com/epolls/other/2018_generic_congressional_vote-6185.html) |
| 2022 | R +2.5 | [RealClearPolitics](https://www1.realclearpolitics.com/epolls/other/2022-generic-congressional-vote-7361.html) |

These are polling averages, **not** the `Final Results` rows printed on those
pages. Substituting election results for pre-election polling would create
information leakage. In the older Gallup table the measurement universes and
likely-voter screens are not completely identical across years. Likewise, RCP
averages are aggregates of changing pollsters and samples. The combined series
is not an invariant survey instrument.

## A missing year must remain missing

The 1986 row is **not present** because Gallup explicitly states that it did
not ask the generic-ballot question that year, in its [2002 historical
discussion](https://news.gallup.com/poll/6877/democrats-enjoy-slight-lead-race-control-congress.aspx).
The American Presidency Project does provide 1986 seats held (181) and seats
lost (-5), but not a compatible pre-election generic-ballot value. The
published Rattner chart appears to plot 1986; its source for that year's
polling input is not specified in the accompanying [original presentation](https://stevenrattner.com/2026/10/the-trump-economy-and-midterm-projections-steve-rattner-on-morning-joe/).
We **do not impute** it. Accordingly this is a close, transparent
reconstruction—not a claim to have reproduced his unpublished design matrix.

## A genuine inconsistency between Gallup retrospectives

Gallup's 2002 retrospective gives 55.5% Democratic for 1962 and 52.1% for
1998. Its 2004 retrospective gives 45.5% Republican for 1962 (equivalent to
54.5% Democratic), and 50.0% Republican for 1998 (equivalent to 50.0%
Democratic). These are **not numerically interchangeable**. They may reflect
changes in poll selection or voter-screen definitions; the published table
alone does not establish the cause. The file
[`gallup_alternative.csv`](gallup_alternative.csv) records the two alternatives.
Our analysis recomputes the entire model using them rather than silently
choosing whichever values best reproduce the headline 89%.

## The 2026 scenario is an input, not an observed result

Steve Rattner's [30 October 2026 presentation](https://stevenrattner.com/2026/10/the-trump-economy-and-midterm-projections-steve-rattner-on-morning-joe/)
states a D+8 generic ballot and 220 House seats defended by Republicans. The
model codes those as `(president_party='R', generic_dem_minus_rep=8.0,
seats_defended=220)`, producing a **president-party** generic margin of -8.
The reported -33 is a forecast, not a historical outcome or calibration target.

## Usage and audit

The CSVs are manually transcribed, small and human-verifiable. Each row's
`generic_source` keys the reference above. The analysis script validates
years, signs, ranges and completeness. Reproduction never refreshes data from
the web, so the numerical snapshot cannot change after election day.
