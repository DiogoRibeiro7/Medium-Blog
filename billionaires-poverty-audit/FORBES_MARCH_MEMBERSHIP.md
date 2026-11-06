# Forbes March 2026 country-membership cross-check

Source: Asia Alexander, [“The Richest Billionaire In Each Country”](https://www.forbes.com/sites/asia-alexander/2026/03/11/the-richest-billionaire-in-each-country/), Forbes, published 11 March 2026 (updated 12 March). The article explicitly describes the 80 areas as countries and territories with at least one **billionaire citizen**, with net worths as of 1 March 2026.

The file [forbes_march_2026_country_presence.csv](data/forbes_march_2026_country_presence.csv) transcribes the **country/territory headings only**, not individual names, fortunes or profiles. The published page visually splits “Czech Republic” across a heading and the next line; the transcription treats it as one name.

Running:

```bash
cd billionaires-poverty-audit
python forbes_march_membership.py
python -m unittest discover -s tests -p 'test_forbes_march_membership.py' -v
```

The March article's 80 country/territory headings match the 80 exact labels in the 5 November primary API citizenship aggregate. This independently corroborates **country membership** and explicit citizenship framing, including Afghanistan, Pakistan, Peru, Eswatini (Swaziland), St. Kitts and Nevis, Guernsey, Hong Kong and Taiwan.

**It does not validate country counts.** Forbes' separately published [March top-ten ranking](FORBES_MARCH_VINTAGE.md) already disagrees with the November API for the United States (+2), Russia (+2), and Italy (+1). Nor does membership establish a historical country-level residence series, the composition of the original 64-country chart, its poverty-year matching, or a causal effect.

Research gate: retain March counts as missing for countries without independently dated March numerical evidence. Do not fill those counts from the November API merely because the set of labels matches. The exploratory citizenship study remains distinct from a replication of the original chart.
