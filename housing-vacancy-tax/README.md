# Empty homes and the state's bargaining power: Portugal vacancy tax

Reproducible companion for the standalone Medium article **Empty Homes Are a Policy Choice: The Economics of Taxing Housing Left Unused**, written 21 November 2026.

## Research question

How does Portugal's *existing* vacancy tax alter the private incentives to hold an otherwise usable dwelling idle? What can the 2021 Census establish about the size of the vacant stock, and what does it **not** establish about legal eligibility, response and affordability?

This project **does not claim to estimate a causal tax effect for Portugal**. It separates three layers:

1. **Observed descriptive data:** INE Census 2021, transcribed from IHRU/OHARU (2024) Table 11, in `data/census_2021_ihru_table11.csv`.
2. **Statutory tax arithmetic:** ordinary IMI; 3× for buildings legally classified as vacant outside pressure zones; 10× in the first year and an ordinary-rate 20× standard cap in designated pressure zones, with qualifications and exceptions in law. The assumed VPT and municipal rate are **not observed**.
3. **Hypothetical owner decisions and mobilisation:** a one-period opportunity-cost identity and two-way sensitivity grid, **not a forecast**.

## Reproduce

Python 3.10 or later. From the `housing-vacancy-tax/` directory:

```bash
python -m pip install -r requirements.txt
python run.py
python figures.py
python -m unittest discover -s tests -v
```

`run.py` needs only the standard library. `figures.py` uses matplotlib and produces three PNG figures. Everything operates offline from the pinned source table.

Outputs:

- `results/summary.json`: observed counts, assumptions and outputs.
- `results/tax_scenarios.csv`: annual hypothetical IMI and option-value thresholds.
- `results/sensitivity.csv`: hypothetical two-stage filtering of the 248,534 dwellings.
- `figures/fig01_vacant_stock_2021.png`: observed vacant stock by market status and repairs.
- `figures/fig02_imi_scenarios.png`: illustrated tax outcomes for VPT €150,000 and IMI 0.35%.
- `figures/fig03_sensitivity.png`: scenario grid, **not a causal prediction**.

## Data provenance

IHRU/OHARU, *Territórios com falta ou desadequação da oferta habitacional em Portugal*, September 2024, **Table 11, printed page 32** ([official report](https://www.portaldahabitacao.pt/documents/20126/58203/1_Estudo%2BPNH-_VFinal.pdf/9ed08595-a9d2-0708-d845-ec3a93d56c3e)), using INE Census 2021.

The 5,970,677 classic dwellings comprised 4,142,581 habitual residences, 1,104,881 secondary residences and 723,215 vacant dwellings. Of the vacant stock, 348,097 were offered for sale or rent and 375,118 were vacant for other reasons. The repair-status cross-tabulation is in `data/`. Buildings requiring no or only minor repairs account for 485,461 vacant dwellings, of which 248,534 were vacant for other reasons.

**Do not interpret 248,534 as the number of legally taxable, immediately lettable, or deliberately withheld dwellings.** The Census categories do not identify a legal vacancy classification, the property's condition inside, inheritance disputes, municipal housing pressure, or property location relative to demand.

Sources for tax law: [Article 112, CIMI](https://info.portaldasfinancas.gov.pt/pt/informacao_fiscal/codigos_tributarios/cimi/Pages/cimi112.aspx), [Article 112-B, CIMI](https://info.portaldasfinancas.gov.pt/pt/informacao_fiscal/codigos_tributarios/cimi/Pages/cimi112b.aspx), and [Decree-Law 159/2006](https://diariodarepublica.pt/dr/detalhe/decreto-lei/159-2006-538621). These rules include safeguards, exemptions and possible municipal rate variations; this is not legal advice.

Further evidence: [OECD Economic Survey Portugal 2026](https://www.oecd.org/en/publications/oecd-economic-surveys-portugal-2026_025b3445-en.html) discusses outdated property assessments and limited enforcement of vacancy surcharges; [Caracciolo and Miglino (2024)](https://cdhowe.org/publication/ripple-effects-impact-empty-homes-tax-housing-market/) find reduced vacancy in Vancouver, but no statistically significant decline in average rent.

## One-period model

`Expected net cashflow from letting before IMI` is calculated as 12 × monthly rent × occupancy × fraction of rent retained after variable expenses and income tax − additional annual rental cost. Ordinary IMI is paid under both choices; a legally applicable vacant-property multiplier creates an extra annual charge `(multiplier − 1) × VPT × municipal_rate` on the empty choice. Renting becomes more attractive if the owner's option value of continued vacancy is lower than cashflow plus the additional vacant-property tax.

In the example, monthly rent (€900), occupancy (90%), retained fraction (70%), extra fixed rental costs (€1,200), VPT (€150,000) and tax rate (0.35%) are *hypothetical*, not estimates. Capital appreciation common to both choices cancels out. These are comparative-static assumptions, not estimates of landlord behaviour.

## Identification and further work

The census snapshot does not tell us what municipalities actually assess and collect, which dwellings qualify for Article 112-B, how rental contracts change, or how much of any tax is shifted onto tenants. A causal study for Portugal needs property-linked data on legal vacancy classifications, VPT, municipal rates, occupancy, lease registrations, property quality and location; ideally variation in actual enforcement with a credible comparison design. Housing availability, housing affordability and tax receipts are distinct outcomes.

## License

Apache-2.0, matching the surrounding repository.
