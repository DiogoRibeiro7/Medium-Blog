# Founder-to-heir successions and productivity growth: a country-level event study

Computational companion to a Medium article, and the fifth project in the line that runs through `wealth-concentration-misallocation`, `wealth-origin-growth`, `wealth-origin-sectors` and `wealth-origin-pandemic`. Those projects found that, within countries, inherited billionaire wealth predicts slower TFP growth than self-made wealth, and that the association survives controls for sector, past growth and pandemic exposure. It remains an association. A succession, when a self-made founder leaves the Forbes list and heirs appear, changes the origin of a fortune without changing its firm or its sector, and its timing is closer to an accident than any change in composition used so far. This project uses successions as events.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # run by the data workflow; see below
python analysis.py           # results/*.csv and results/summary.txt (about one minute)
python figures.py            # figures/fig1_*.png ... fig5_*.png
python -m pytest -q tests
```

`results/summary.txt` holds every number used in the article. `.github/workflows/wealth-origin-successions.yml` reruns the analysis and fails if that file changes.

## Data (`fetch.py`, `data/`)

`fetch.py` downloads every Forbes list from 2011 to 2026 from the Forbes person-list API and works on them in memory only. Between consecutive lists it looks for self-made billionaires who left the list and, in the same country, new inherited billionaires who share the founder's last name or a source of wealth specific to one fortune. It writes only country × year aggregates (`data/succession_country_year.csv`) and per-year matching diagnostics (`data/succession_diagnostics.csv`); no name, identifier or person-level row is stored. The rules:

- People are identified across lists by whichever of Forbes' `uri`, `naturalId` or name keeps the most people from one list to the next; `naturalId` changes every year (0% persistence), `uri` keeps 91% of people and is used.
- An exit is a self-made person absent from the next list and not back on the one after.
- A new heir is an inherited person on the next list who was not on the previous one.
- A last name identifies a family unless more than four people of the same citizenship carry it on the lists compared; a source identifies a fortune unless people of three or more citizenships, or more than thirty people, share it.
- `strict` events have heirs on the next list; `window` events also accept heirs who appear a list later.

Of 1,650 exits of self-made billionaires between 2011 and 2026, 63 are matched to new heirs on the next list; 49 of them involve founders aged 70 or more. Most exits are fortunes falling below one billion dollars.

Growth (Penn World Table 11.0), GDP (WDI), the composition table and the pandemic-exposure measures are copied unchanged from earlier snapshots; `data/PROVENANCE.txt` gives source commits and SHA-256 checksums, and a test checks them.

## Analysis (`analysis.py`)

**Event study.** Annual TFP (or GDP per capita) growth on leads and lags of a succession shock, log(1 + founder's wealth passing to heirs as % of the previous year's GDP), with country and year effects and lagged income. Successions are dated by the first list that omits the founder; the departure happened in the twelve months before, mostly in the previous calendar year (k = −1). Leads k = −3 and −2 are placebos; the effect is the sum over k = 0, 1, 2. Standard errors clustered by country.

**Contrasts.** The same design with every exit of a founder aged 70 or more (heirs found or not), with exits that have no matched heirs, and with new heirs that have no matched exit.

**Inference and checks.** A permutation test that reshuffles each country's shock years (1,000 draws); previous year's growth as a control; changes in billionaire wealth with the same leads and lags as controls (for selection on valuations); growth years 2020 and 2021 left out; the United States left out; pandemic exposure × year (from `wealth-origin-pandemic`); an indicator for successions above 0.1% of GDP; leave one country out. The within-country composition coefficient from the earlier projects is used to predict the effect of each observed succession, for comparison.

## Caveats

- Successions are found only when heirs appear on the list, which requires each heir's share to be worth more than a billion dollars. Detection therefore depends on valuations at the time of the next list.
- Founders whose heirs were already billionaires, whose fortunes were split below the threshold or went to foundations are not detected as successions.
- 39 country-years in the billionaire countries carry a succession, seven of them above 1% of GDP; inference rests on few events.
- The effect depends on growth years 2020 and 2021 and is imprecise once pandemic exposure is controlled for.
- Every result is an association conditional on the design; a death is closer to an exogenous event than a change in composition, but its detection is not.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository. Forbes, PWT, WDI, OxCGRT and ILOSTAT data are reused under their providers' terms.
