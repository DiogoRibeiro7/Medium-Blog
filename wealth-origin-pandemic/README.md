# Inherited billionaire wealth, pandemic exposure and productivity growth

Computational companion to a Medium article, and the fourth project in the line that runs through `wealth-concentration-misallocation`, `wealth-origin-growth` and `wealth-origin-sectors`. The third project showed that, within countries, inherited billionaire wealth predicts slower TFP growth than self-made wealth even with the sector mix of billionaire wealth held fixed, but that the estimate leans on the 2019 and 2020 Forbes lists, whose growth windows contain the 2020 collapse and the recovery. This project asks whether that reflects each economy's exposure to the pandemic shock, and whether it reflects how the Forbes lists measure wealth.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # run by the data workflow; see below
python analysis.py           # results/*.csv and results/summary.txt (about 2 minutes)
python figures.py            # figures/fig1_*.png ... fig3_*.png
python -m pytest -q tests
```

`results/summary.txt` holds every number used in the article. `.github/workflows/wealth-origin-pandemic.yml` reruns the analysis and fails if that file changes.

## Data (`fetch.py`, `data/`)

- **Billionaire, growth and GDP data**, copied unchanged from the `wealth-origin-sectors` snapshot so that any change in the estimates comes from the new controls. `data/PROVENANCE.txt` gives the source commit and SHA-256 of each file; a test checks them. The Forbes table holds aggregates only (citizenship × year × industry × self-made flag).
- **Oxford COVID-19 Government Response Tracker** (OxCGRT), national compact file: the stringency index averaged over 2020–2021, and cumulative confirmed deaths at the end of 2021 (divided by Penn World Table population).
- **World Bank WDI**: travel credits in the balance of payments (`BX.GSR.TRVL.ZS` × `BX.GSR.NFSV.CD`) as a share of GDP, averaged over 2015–2019; UNWTO tourism receipts (`ST.INT.RCPT.CD`) where the balance-of-payments series is missing (two billionaire countries). The two agree closely where both exist (Spearman 0.98).
- **ILOSTAT**: ILO modelled estimates of employment by ISIC Rev. 4 section (`EMP_2EMP_SEX_ECO_NB_A`), 2019; contact-intensive employment is trade (G), transport and communication (H–J), accommodation and food (I) and other services (R–U) as a share of the total. The survey-based series (`EMP_TEMP_SEX_ECO_NB_A`) is fetched as a check (Spearman 0.80 with the modelled measure where both exist).

The workspace where the analysis was written cannot reach most of these hosts, so `.github/workflows/wealth-origin-pandemic-data.yml` runs `fetch.py` on GitHub's runners and commits the tables; `data/fetch_log.txt` holds URL, status, size and SHA-256 of every response. All 63 countries with billionaires in at least seven of the thirteen lists have all four measures.

## Analysis (`analysis.py`)

**Exposure controls.** The sector-controlled regression of `wealth-origin-sectors` (growth from t to t + h on log(1 + inherited billionaire wealth as % of GDP) and the sum of the inherited and self-made logs, income level, six sector-group shares, year effects, country effects or not) is re-estimated with each standardised exposure measure interacted with list-year dummies. With country fixed effects, that absorbs any differential growth path that is linear in exposure, whatever its timing. 24 specifications (two outcomes × three horizons × two samples × two estimators), each exposure on its own and all four together, always compared with the estimate on the same country-years. Joint test across specifications with the null imposed (Simonsohn, Simmons and Nelson 2020).

**Around 2020.** Cross-sections of TFP growth on composition for the 2020 dip (2019 list, 2019–2020), the rebound (2020 list, 2020–2023), the net change and a pre-pandemic window; exposure-by-year coefficients; period split; leave one list year out; forward minus backward with a country bootstrap.

**Measurement.** The Forbes 2020 list values fortunes at stock prices of 18 March 2020, near the trough of the crash. Composition is therefore also measured by headcount (billionaires per 10 million people) and as wealth averaged over lists t − 1 and t; each is re-estimated with leave-one-list-year-out and the placebo, and the two-list average across all 24 specifications with its own joint test.

## Caveats

- Confirmed deaths undercount pandemic mortality where testing was limited; excess-mortality estimates do not cover enough of the panel to replace them.
- The modelled employment shares are ILO estimates, not survey counts, and section J (information and communication) cannot be separated from transport in them.
- Exposure is measured at the country level and enters through its interaction with list years; it cannot capture differences in exposure within an economy's billionaire sector mix, which the sector controls address separately.
- The forward-minus-backward placebo does not clearly separate forward from backward coefficients under any measure (shares of bootstrap draws at or above zero between 0.05 and 0.30), so pre-existing trends are not ruled out.
- Every result is an association.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository. Forbes, OxCGRT, ILOSTAT, PWT and WDI data are reused under their providers' terms.
