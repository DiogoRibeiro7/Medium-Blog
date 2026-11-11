# Wealth concentration, the coupling of wealth to productivity, and growth

Computational companion to a Medium article. Question: is concentrated wealth economically inefficient? In a frictionless economy the answer is no by construction (the second welfare theorem separates distribution from efficiency), so any inefficiency has to come through a friction. With a collateral constraint, what decides efficiency is not how concentrated wealth is but whether it sits with the people who can use it productively. The project measures that claim in a simulated economy and then asks what cross-country data can and cannot say about it.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # WID, PWT 11.0, WDI and the billionaire database (run by the data workflow; see below)
python experiments.py        # simulations: results/sim_*.csv, model_summary.txt (about 6 minutes on two cores)
python analysis.py           # cross-country: results/panel_*.csv, spec_curve_test.csv, billionaire_*.csv, analysis_summary.txt (about 3 minutes)
python figures.py            # figures/fig1_*.png ... fig6_*.png
python -m pytest -q tests
```

`results/model_summary.txt` and `results/analysis_summary.txt` hold every number used in the article. The workflow `.github/workflows/wealth-concentration-misallocation.yml` reruns everything and checks that both files are unchanged.

## Model (`model.py`, `experiments.py`)

**Static allocation.** N owners hold wealth a and run a firm with productivity z and technology y = z k^γ, γ < 1. Capital is rented at rate R; an owner can rent at most λa (collateral constraint, as in Evans and Jovanovic 1989, Buera and Shin 2013, Moll 2014). Total capital equals total wealth. Efficiency is output relative to the frictionless allocation with the same capital and productivities. Wealth is lognormal with a target top-1% share; wealth and productivity are joined by a Gaussian copula with a target Spearman correlation ("coupling"). E1 maps efficiency over top-1% share (4% to 50%) × coupling (−0.2 to 1) with γ = 0.5, log-sd of productivity 1.0 and λ = 1.25 (one period of the dynamic economy at a given wage), plus slices for λ = 2, 4, 10.

**Dynamic economy.** 10,000 dynasties; firm technology y = z (k^α n^(1−α))^η with α = 1/3, η = 0.75; each household supplies one unit of labour; λ = 1.25; depreciation 6%; saving rates 5% of wage income and 50% of capital income (the rich save more: Dynan, Skinner and Zeldes 2004). Log productivity is AR(1) with persistence ψ = 0.98 and stationary sd 0.5; each year a dynasty passes to an heir with probability 1/40, wealth passes in full, and the heir's log productivity has correlation ρ_g with the parent's (baseline 0.3). Both factor markets clear exactly each year. Baseline steady state: top-1% wealth share about 17%, top-10% about 50%, wealth/output about 4.6.

- E2: ρ_g from 0 to 1.
- E3: 25% tax on capital income (profits plus rent) against a flat wealth tax that raises the same revenue in steady state (3.52%). Both are deducted from disposable capital income before saving, so the regimes differ only in their base. This is the comparison in Guvenen, Kambourov, Kuruscu, Ocampo and Chen (2023), in a much simpler model.
- E4: at t = 0 every dynasty's productivity is redrawn independently of its wealth (a technology shock that re-ranks firms); paths of coupling and efficiency for 120 years under each tax regime.
- E5: ψ ∈ {0.80, 0.90, 0.95, 0.98} × ρ_g ∈ {0, 1}.
- E6: sensitivity of the E4 transition (no tax) to the saving rate out of capital income (s_k ∈ {0.3, 0.4, 0.5}) and the persistence of productivity (ψ ∈ {0.90, 0.95, 0.98}): years until coupling recovers half and 90% of its pre-shock level, and cumulative output lost over 120 years in years of pre-shock output.

Burn-in 350 years, averages over the next 100, three seeds.

## Data (`fetch.py`, `data/`)

- **World Inequality Database** full bulk download: top 1%, top 10% and bottom 50% shares of net personal wealth (`shweal992j`) and of pre-tax national income (`sptinc992j`), adults, equal-split, with the per-observation `data_quality` score and the variable metadata.
- **Penn World Table 11.0** (Feenstra, Inklaar and Timmer), through the DataverseNL API (doi:10.34894/FABVLR): `rgdpna`, `rgdpe`, `pop`, `rtfpna`, `hc`, `csh_i`. CC BY 4.0.
- **World Bank WDI**: GDP in current US dollars (`NY.GDP.MKTP.CD`) and the country list (ISO codes, income groups). CC BY 4.0.
- **Billionaire Characteristics Database** (Freund and Oliver 2016, PIIE Working Paper 16-1), Forbes lists of 1996, 2001 and 2014 with the source of each fortune, as redistributed by the CORGIS project. The `was political` and `was founder` fields are constant in that copy and are not used.

The workspace where the analysis was written cannot reach WID, DataverseNL or the World Bank API, so `.github/workflows/wealth-concentration-misallocation-data.yml` runs `fetch.py` on GitHub's runners and commits the tidy tables with `data/fetch_log.txt` (URL, status, size and SHA-256 of every response, including the 900 MB WID archive, which is not committed). The data are reused under their providers' terms and are not relicensed by the repository's licence.

## Cross-country analysis (`analysis.py`)

**Multiverse.** Annualised growth of real GDP per capita or TFP over the following window, regressed on a top share at the start of the window, for every combination of measure (wealth top 1%, top 10%, bottom 50% with sign flipped; income top 1%, top 10%), outcome (GDP per capita, TFP), window (5 or 10 years, 1980–2019), controls (none; initial income; initial income, human capital and investment share), estimator (pooled with period effects; country and period fixed effects; between) and sample (all WID areas matched to PWT; only window starts with a source-based WID value within two years, `data_quality > 0`; World Bank high-income countries): 540 specifications. Coefficients are percentage points of annual growth per one standard deviation more concentration in the estimation sample; 95% intervals are country-clustered (CR1) for pooled and fixed-effects fits and HC1 for between fits.

**Joint test.** Following Simonsohn, Simmons and Nelson (2020), the estimated effect is removed from the outcome in every specification (y − b̂x), countries are resampled with replacement (one draw shared by all 540 specifications, so their dependence is kept), and the whole curve is re-estimated, 1,000 times. The observed median estimate, the numbers of significantly negative and positive specifications, and Stouffer's Z are compared with their null distributions, for the full curve and for subsets (wealth or income measures; wealth by sample; each estimator). Results in `results/spec_curve_test.csv`.

**Composition.** Billionaire wealth in 1996, 2001 and 2014 split into inherited, self-made (founders, finance, executives) and privatisation/resources, as a share of GDP in current dollars; growth of GDP per capita or TFP over the next 5 or 9 years; controls for initial income, human capital, investment share and wave; countries with at least one million people. 24 variants (outcome × horizon × log(1 + share) or raw share × all countries / only countries with billionaires / without financial centres). 95% intervals from 2,000 country-cluster bootstrap draws.

## Caveats

- The models are devices for comparative statics. Saving rates are fixed rather than chosen, there is no occupational choice, no entry, no aggregate demand and no political economy (rent-seeking, entrenchment), which is where much of the argument about concentrated wealth lives.
- Efficiency is relative to a first best with the same capital stock; it does not count the effect of distribution on saving or on the size of the capital stock.
- Cross-country regressions of growth on inequality are associations. They are reported to show how much the sign depends on choices, not to estimate an effect.
- About 94% of WID's published wealth-share country-years are imputations (`data_quality = 0`); the "observed" sample has 142 five-year windows in 36 countries.
- The billionaire composition regressions have 61 countries with at least one billionaire and three waves; they are underpowered for differences of the size the literature reports.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
