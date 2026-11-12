# Inherited billionaire wealth, sector composition and productivity growth

Computational companion to a Medium article, and the third project in a line that starts with `wealth-concentration-misallocation` and continues with `wealth-origin-growth`. The second project found that, within countries, inherited billionaire wealth predicted slower TFP growth than self-made wealth over 2011–2023, and left open whether that was a property of inherited wealth or of the sectors in which inherited fortunes sit (technology fortunes are mostly self-made; retail, food and diversified holdings mostly inherited). This project adds the industry of every fortune and tests that alternative.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # run by the data workflow; see below
python analysis.py           # results/*.csv and results/summary.txt (about 40 seconds)
python figures.py            # figures/fig1_*.png ... fig4_*.png
python -m pytest -q tests
```

`results/summary.txt` holds every number used in the article. `.github/workflows/wealth-origin-sectors.yml` reruns the analysis and fails if that file changes.

## Data (`fetch.py`, `data/`)

- **Forbes annual billionaire lists, 2011–2026**, from the Forbes person-list API. Every list is aggregated to citizenship × year × industry × self-made flag *before* anything is written (number of people and summed net worth). No person-level row is saved. Forbes gives one industry per person (the share of people with more than one is zero in every year) and fills it for 99.9% or more of people. `data/forbes_coverage.csv` records, per year, the HTTP status, list size and field coverage. The 2026 request timed out; 2024–2026 are not used because growth data end in 2023.
- **Penn World Table 11.0** (growth, TFP, income levels) and **World Bank WDI** (GDP in current dollars, country codes), as in the previous project.

The workspace where the analysis was written cannot reach these hosts, so `.github/workflows/wealth-origin-sectors-data.yml` runs `fetch.py` on GitHub's runners and commits the tidy tables; `data/fetch_log.txt` holds URL, status, size and SHA-256 of every response.

**Industry labels.** Forbes used 34 distinct labels across the 2011–2023 lists, drifting between spellings ("Finance & Investments", "Finance and Investments", "Investments", "Finance"; "Fashion & Retail", "Retail"; "Healthcare", "Health care", "Medicine"; and so on). `analysis.py` maps them to the 18 categories Forbes uses today. Four labels that are not industries ("Billionaire", "Business", "Celebrity", "Politics") and empty entries are left unclassified; they hold 0.26% of listed wealth.

## Analysis (`analysis.py`)

**Panel.** As in `wealth-origin-growth`: countries with at least one million people in PWT 11.0 and a WDI GDP figure, list years 2011–2023, 1,926 country-years in 149 countries, 802 with at least one billionaire. Added: the share of each country-year's billionaire wealth in each sector, and six sector groups (technology and telecom; finance; fashion, retail, food and beverage; real estate and construction; energy and mining; manufacturing, automotive and logistics; the rest is the base).

**Sector controls.** The previous regression (growth from t to t + h on log(1 + inherited % of GDP) and on the sum of the inherited and self-made logs, so that the coefficient on the first is the inherited-minus-self-made difference; income level at t; year effects; country effects or not) re-estimated with the six group shares as controls, and, as a check, with all 18 sector shares. 24 fits: GDP per capita and TFP × h ∈ {1, 3, 5} × {billionaire countries (≥7 of 13 years), all country-years with billionaires} × {country fixed effects, pooled}.

**Shift-share decomposition.** A country's inherited share of billionaire wealth, Σₛ w(c,t,s) · inh(c,t,s), is split into a sector-predicted part, Σₛ w(c,t,s) · ρₛ(−c), where ρₛ(−c) is the inherited share of sector s among all *other* countries' billionaires (pooled 2011–2023), and the within-sector remainder. Growth is regressed on both parts with the size of billionaire wealth, log(1 + total % of GDP), held fixed. A sector effect would load on the first part, an origin effect on the second.

**Inference.** Standard errors clustered by country (CR1). A null-imposed joint test across the 24 fits (Simonsohn, Simmons and Nelson 2020) for the sector-controlled difference and for the within-sector coefficient. Placebo on past growth with a country-bootstrap test of forward minus backward. The forward coefficient with past growth as a control (mean reversion). Leave one country out; leave one list year out; windows ending by 2019 against windows including 2020.

## Caveats

- Sector is Forbes' label for a person's main source of wealth, one per person, assigned to the whole fortune. A family conglomerate labelled "Diversified" may hold technology businesses.
- The sector mix accounts for about a tenth of the variance of inherited shares across country-years and within countries; the sector-predicted part therefore has little variation of its own, and its coefficients are imprecise.
- Only about 16% of the variance of the inherited share is within countries; the fixed-effects estimates rest on that.
- The sector-controlled TFP difference is robust to dropping any country, to all 18 sector shares and to controlling for past growth, but leaving out the 2019 or 2020 list moves it toward zero, and the windows that end by 2019 do not show it on their own (−0.04 at three years, +0.17 at five).
- Every result is an association.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository. Forbes, PWT and WDI data are reused under their providers' terms.
