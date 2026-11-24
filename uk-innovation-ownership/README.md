# Who owns UK innovation: foreign ownership of inventions and ICT jobs, the UK against the EU, 1980–2021

Companion code for a Medium article that tests a conjecture about British capital: a "preserve, don't build" attitude keeps domestic money out of domestic risk, and foreign investors and buyers fill the gap. If that is right, it should not show up as slower growth of UK firms, which have no shortage of venture money, but as UK innovation ending up owned abroad more than the UK's size and language predict. The mentality itself is not measured here; only the footprint it predicts.

## What it finds

All numbers below are from `output.txt`.

**Growth is not the gap.**
- **High-growth ICT firms**, 2012–2018 mean: 17.5% of UK ICT firms with 10+ employees grew 10%+ a year, against an EU median of 14.7%; the UK ranks 7th of 28, behind Ireland, Sweden, the Netherlands, Denmark, Germany and Finland.
- **Venture capital** invested in UK companies, 2020–2025: 0.149% of GDP, 2.9 times the EU median (0.051%), with a later-stage share of 47.6% against 36.9%.

**The domestic pools are large.** Asset-backed pension investment averaged 94% of UK GDP in 2015–2024 (73.8% in 2025), against an EU median of 12.7%; only Denmark, the Netherlands and Sweden are comparable. Private-equity funds were 1.1% of UK pension-fund assets in 2019–2025 (about 1.1% of GDP), against 5.4% in the Netherlands (about 9.5% of GDP).

**Ownership ends abroad.**
- **Inventions.** Of PCT applications with an inventor in the UK, priority years 2008–2021, 38.3% were owned by applicants abroad and 14.8% by US applicants. Germany: 20.2% and 5.7%; France 24.2% and 5.9%; EU median 33.0% and 6.1%. Among countries with 3,000+ inventions a year only Canada (40.9%) is higher.
- **Trend.** The UK share owned abroad rose from 13.7% in 1980–84 to 41.2% in 2010–14, then eased to 34.7% in 2020–21; Germany went from 11.9% to 23.9%.
- **Owners of UK inventions**, 2008–2021: US 14.8%, Netherlands 3.8%, Switzerland 3.4%, Germany 3.2%, France 2.5%, Japan 2.3%.
- **ICT jobs** (Eurostat inward FATS, NACE J, 2014–2019; UK 2014–2017): 26.6% of UK ICT employment in foreign-controlled firms and 11.1% in US-controlled ones, against EU medians of 30.9% and 4.5% and 14.9% and 3.6% in Germany.

**Size and language explain part of it, not all.** A fractional logit across 29 countries (log inventions a year, English-language and transition-economy indicators), fitted without the UK:
- **Owned in the US:** UK 14.8%, predicted 13.6% (+1.1 points); 10th largest gap of 29. Being English-speaking accounts for the UK's US share; on size alone the gap would be +7.5 points.
- **Owned abroad, outside the US:** UK 23.6%, predicted 11.4% (+12.2 points, ratio 2.06); the largest gap of the 29 countries in leave-one-out prediction.
- **All foreign owners:** UK 38.3%, predicted 24.7% (+13.6 points); 3rd of 29, after Belgium and Romania.
- **Uncertainty.** The jackknife+ intervals for one country are wide (outside the US: 2.0%–49.4%), so the gap cannot be told apart from country-specific noise by the interval; the rank among countries is the sharper test.

**Pension pools do not keep inventions at home anywhere.** Adding log(1 + pension assets % of GDP): coefficient +0.009 (se 0.059) for the share owned abroad; the clustered country-year panel gives −0.002 (se 0.052). Moving from 10% to 90% of GDP changes the predicted share by almost nothing.

**Robustness.** Across PCT and EPO, 1995–2007 and 2008–2021, with and without small countries, without Canada and Ireland, and without the transition economies, the UK's non-US gap stays between +9.9 and +15.2 points and its leave-one-out rank between 1st and 5th. The US gap is +4.8 to +9.4 points in 1995–2007 and −0.1 to +1.8 in 2008–2021 (+7.4 without Canada and Ireland).

## Data

Downloaded by `fetch_data.py` into `.cache/` (not committed) and checked against SHA-256 digests taken on 24 November 2026; the sources are recorded as external references in [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data).
- `patents_foreign.csv`: OECD `DSD_PATENTS@DF_PATENTS_FOREIGN`, foreign ownership of domestic inventions; PCT and EPO applications, priority date, counts of inventions (W), owned abroad (WXD) and owned in the US.
- `patents_owners.csv`: the same table with every owner country, for the UK, Germany, France, Italy, the Netherlands and Sweden (PCT).
- `venture_capital.csv`: OECD `DSD_VC@DF_VC_INV`, venture capital investments (market statistics), by stage, % of GDP.
- `high_growth.csv`: OECD `DSD_SDBSBD_ISIC4@DF_BD_EMP_HG`, medium- and high-growth employer enterprises (10% a year in employment), % of enterprises with 10+ employees; ISIC J, J62, J63, M72.
- `pensions.csv`: OECD `DSD_FP@DF_FPS`, asset-backed pension investment (% of GDP) and private-equity funds (% of pension-fund assets).
- `fats_ict.csv`: Eurostat `fats_g1a_08`, persons employed in all, foreign-controlled and US-controlled enterprises; NACE J, J62, J63, M72; 2008–2020.

## Method notes

- **Shares** pool counts over the window (total owned abroad over total inventions), so each year weighs by its number of inventions.
- **Model**: quasi-binomial GLM with a logit link for a share (Papke and Wooldridge 1996), HC1 standard errors; covariates log mean inventions a year, English as an official and main business language (UK, Ireland, Malta, Canada, Australia, New Zealand), post-1989 transition economy. The US is excluded: it is the main foreign owner of everyone else's inventions. Main sample: EU-27 and nine OECD comparators with 1,000+ PCT inventions in 2008–2021 (29 countries).
- **Hold-out**: the UK's prediction comes from a fit without the UK. Every other country gets the same treatment (leave one out), and countries are ranked by the gap on the logit scale. Since the thesis names the UK before the data are seen, rank/n is a permutation-style measure of how unusual its gap is: 1/29 = 0.034 for owners outside the US.
- **Interval**: jackknife+ (Barber, Candès, Ramdas and Tibshirani 2021) on the logit scale, α = 0.1; coverage is at least 80% for exchangeable countries, whatever the model.
- **Panel check**: country-year fractional logit with standard errors clustered by country, for the pension coefficient.
- **Determinism**: no random number is drawn in the analysis (the tests use fixed seeds).

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the estimators on cases with a known answer
python fetch_data.py                       # download from the OECD and Eurostat and check the digests
python ownership_numbers.py > output.txt   # every number in the article
python figures.py                          # figures/
```

The workflow `.github/workflows/uk-innovation-ownership.yml` runs all of this, and fails if `output.txt` changes.

## Limits

- **Ownership is not acquisition.** A foreign-owned invention can come from a lab a foreign company opened in the UK, not from a UK start-up it bought. Country tables cannot separate the two; firm-level records (Companies House ownership filings) could.
- **Patents are not start-ups.** Inventions are the innovation output that can be counted across countries on one definition; most are filed by established firms.
- **Dual-listed companies.** Anglo-Dutch groups with a Dutch legal parent may appear as Dutch owners of UK inventions; this is not checked, and it would lower the non-US gap.
- **Few countries.** 29 countries make wide intervals; in the main sample the language indicator rests on four English-speaking comparators (Ireland, Canada, Australia, New Zealand).
- **The mentality is not measured.** The analysis tests a footprint the conjecture predicts; a takeover-friendly legal regime, the UK's science base or its role as an R&D location for multinationals would leave a similar footprint.
- **One vintage.** OECD and Eurostat revise these tables; the digests pin the 24 November 2026 vintage.

## Licence

Code: Apache-2.0. Data: OECD (OECD Terms and Conditions) and Eurostat (reuse authorised with acknowledgement); no data are redistributed here.
