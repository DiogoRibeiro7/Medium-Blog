# Wage setting in the EU as auction and bargaining, with Portugal as a special case

Computational companion to a Medium article. Question: labour laws are usually described as shifting "power" between employees and employers, but in a wage-setting model power is several parameters, not one. Which part of the wage distribution does each law move, what do EU earnings data show, and what changes when the minimum wage sits close to the median, as in Portugal?

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # Eurostat and OECD snapshot (run by the data workflow; see below)
python experiments.py        # model: results/model_*.csv, model_summary.txt (about 1 minute on two cores)
python analysis.py           # data: results/ses_*.csv, oecd_*.csv, floor_elasticities.csv, analysis_summary.txt
python figures.py            # figures/fig1_*.png ... fig6_*.png
python -m pytest -q tests
```

`results/model_summary.txt` and `results/analysis_summary.txt` hold every number used in the article. The workflow `.github/workflows/eu-wage-bargaining-power.yml` reruns everything and checks that both files are unchanged.

## Model (`model.py`)

A simulated labour market with 30,000 workers (log ability ~ N(0, 0.30²)) and 300 firms (log productivity ~ N(0, 0.35²)); a match produces y = ability × productivity a month. Wages follow the sequential-auction-with-bargaining structure of Postel-Vinay and Robin (2002) and Cahuc, Postel-Vinay and Robin (2006), without option values:

- hire from unemployment: w = max(m, b + β(y − b)), b = 0.45 × ability;
- outside offer from a more productive firm: the worker moves, w = max(m, y + β(y' − y));
- outside offer from a less productive firm that beats the current threat point q: the worker stays, w = max(m, y' + β(y − y')); wages are never cut;
- separations at 1.5% a month; offers at 20% a month when unemployed and 5% when employed.

Five laws map onto five parameters: minimum wage → reserve price m; collective bargaining strength → β; non-compete and no-poach rules → rate of rival offers λ1; unemployment insurance → outside option b; pay transparency → an equal-pay rule within firm × ability band, under which a raise to one worker is a raise to every peer, so firms counter outside offers only when that is worth it (Cullen and Pakzad-Hurson 2023).

`experiments.py` moves each lever alone from a common baseline (10 seeds), and then crosses minimum wages from 0.30 to 0.95 with a higher β and a higher λ1 (6 seeds).

## Data (`fetch.py`, `data/`)

- Eurostat: `earn_mw_cur` (minimum wages), `earn_mw_avgr2` (minimum wage relative to mean and median earnings), Structure of Earnings Survey deciles `earn_ses_mdeci`, `earn_ses10_mdeci`, `earn_ses14_mdeci`, `earn_ses18_mdeci`, `earn_ses22_mdeci`, employees and earnings by type of collective pay agreement (`earn_ses10_01` … `earn_ses22_01`, `earn_ses18_19`, `earn_ses22_19`), `earn_ses_pub1s`, `nama_10_gdp`, `nama_10_a10_e`, `une_rt_a`.
- OECD Data Explorer: `DSD_EARNINGS@MIN2AVE`, `DSD_EARNINGS@DEC_I`, OECD/AIAS ICTWSS `DSD_TUD_CBC@DF_TUD` and `@DF_CBC`, `DSD_EPL@DF_EPL`; EU members plus the UK and the US.

The workspace where the analysis was written cannot reach these APIs, so `.github/workflows/eu-wage-bargaining-power-data.yml` runs `fetch.py` on GitHub's runners and commits the snapshot with `data/fetch_log.txt` (URL, status and SHA-256 of every response). Eurostat data are reused under Commission Decision 2011/833/EU and OECD data under the OECD terms; they are not relicensed by the repository's licence.

## Definitions

- **Minimum wage as paid.** Eurostat reports Greece, Spain and Portugal on a 12-month basis (× 14 / 12); for comparison with October monthly earnings in the Structure of Earnings Survey the value is converted back. The value is the one in force from 1 July of the survey year.
- **Kaitz index.** Minimum wage / median. It is computed here from SES (October monthly earnings of full-time employees) and also read from Eurostat (`earn_mw_avgr2`, 12-month basis) and the OECD (median earnings of full-time workers). The Banco de Portugal figure (base wage, Social Security microdata) is quoted from its June 2026 Economic Bulletin.
- **Floor elasticity.** Slope of log(P10 / median) on log(minimum wage / median) with country and wave fixed effects, 2006–2022 SES waves, 21 countries with a statutory minimum wage; 90% intervals from 2,000 bootstrap draws over countries. Placebo: the same for P90 / median.
- **Agreement types (SES).** Multi-employer = national or interconfederal, industry, and industry-by-region agreements; single-employer = enterprise and local-unit agreements; enterprises with 10+ employees, NACE B–S excluding O.

## Caveats

- The model is a device for comparative statics. It has no participation margin, no firm entry and no option values, so its employment losses from a high floor and its wage levels are not estimates.
- The floor elasticity is a within-country association across five survey waves. Minimum wages are partly set in response to wage growth, so it is a fingerprint consistent with a reserve price, not a causal estimate.
- SES 2022 is not restricted to enterprises with 10+ employees as earlier waves are, and SES 2006 uses NACE Rev. 1.1; the estimate is repeated on 2006–2018 only.
- SES agreement types are reported by employers and their classification differs across countries (Portugal reports a large "other" category).

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
