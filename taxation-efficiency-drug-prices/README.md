# Taxes, efficiency and the price of medicines

Public reproducibility companion for a series on Medium by Diogo Ribeiro. The editorial manuscripts are kept in the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium/tree/main/series/taxation-efficiency-drug-prices) repository; this folder holds the code, the printed numbers and the figures. The series starts from the slogan "taxation is theft", separates the premises of that argument that are normative from the one that can be measured, and then measures the efficiency claim where it is most concrete: what a single public buyer such as Portugal's SNS pays for medicines, and what that costs in other currencies (time to access, leakage through parallel export). Every number quoted in the articles is printed by the script in the corresponding folder, and the committed `output.txt` is what the script prints.

- **Part 1 — the premises of "taxation is theft", and the one that data can test** ([`part1/`](part1)). A linear public-goods economy. Voluntary contributions with imperfect conditional cooperation (Fischbacher and Gächter 2010), calibrated to the first-round and tenth-round contribution levels summarised by Ledyard (1995) and by Andreoni (1995) from Isaac and Walker (1988), with a sensitivity analysis over the parts the calibration does not identify; costly peer punishment calibrated to two results of Fehr and Gächter (2000) and checked out of sample against the 50-round result of Gächter, Renner and Sefton (2008); antisocial punishment (Herrmann, Thöni and Gächter 2008); a tax with a marginal cost of public funds from Browning (1976) and Ballard, Shoven and Whalley (1985) and a share of revenue that never reaches the good; the break-even frontier between voluntary provision and the tax; and a map of the group sizes and returns that laboratory studies have covered against a national public good.
- **Part 2 — how much more the US pays for medicines, and how the answer depends on the method** ([`part2/`](part2)). Medicare Part D against the French Assurance Maladie in 2024, for tablets and capsules dispensed by retail pharmacies, molecule by molecule: 432 molecules covering $121.9 bn of Part D spending. The molecules are sorted into single-source (no generic competition in either country), multi-source (generics in both) and mixed baskets, and compared with Laspeyres, Paasche, Fisher and Jevons indices, with a bootstrap over molecules. The script shows why a brand-against-brand comparison misleads, how an all-drug index moves with the generic share of units alone, and how the single-source gap responds to the French dispensing fee, the exchange rate, the basket threshold, the oral-solid classification, outliers, the largest molecules, pharmacy acquisition prices (NADAC) and confidential rebates on both sides. It then sets the Medicare negotiated prices of the oral drugs selected for 2026 and 2027 against 2024 acquisition prices and current French prices.
- **Part 3 — how a small single buyer sets the price of a medicine** ([`part3/`](part3)). Portugal does not publish its prices presentation by presentation, so the script rebuilds the rule that sets them: Infarmed's external reference pricing formula (average ex-factory price in Spain, France, Italy and Belgium, adjusted for strength and pack size, then the regulated margins), as implemented in Infarmed's own 2026 calculation form. It feeds the rule the public price lists of the four countries (France's manufacturer price is backed out of the retail price; 99 percent of original brands invert to a whole-cent price), applies it to the centrally authorised brands with no generic competition in France (French 2024 volumes as weights), measures how far the Portuguese ceiling sits from each reference country and how much the choice of rule and countries moves it, and checks the rebuilt rule against the only Portuguese prices published by presentation, the original brands in Infarmed's list of homogeneous groups. It also reads Infarmed's tables of exceptional price revisions and the SNS ambulatory aggregates.

## Running

Python 3.13 with the pinned versions in `requirements.txt`.

```
pip install -r requirements.txt
cd part1
python premises_numbers.py > output.txt   # about two minutes, single-threaded; writes figure_data.json
python figures.py
cd ../part2
python prices_numbers.py > output.txt     # about ten seconds; reads data/, writes figure_data.json
python figures.py
cd ../part3
python reference_numbers.py > output.txt  # about twenty seconds; reads data/ and ../part2/data
python figures.py
cd ..
python -m unittest discover -s tests      # tests of the shared tools
```

`publicgoods.py` holds the tools of Part 1: the conditional-cooperation model, peer punishment with prosocial and antisocial sanctions, and the welfare algebra of voluntary provision against a tax (the break-even share `s_star` and the break-even marginal group return `g_star`).

`drugprices.py` holds the tools of the drug-price parts: name matching across languages (international non-proprietary names, US adopted names, Part D's truncated names), the oral-solid and segment classification of Part D rows through the FDA NDC directory, French units per box, molecule baskets, the index formulas and the bootstrap, NADAC by molecule, and the negotiated-price table.

`erp.py` holds Portugal's reference pricing rule and the conversions it needs: the pack-size factor and the strength and pack adjustment of Infarmed's form, the average (ambulatory) and lowest (hospital) rules, the six retail-price tiers and their inverse, the factors that turn Spanish and Italian retail prices into ex-factory prices, the French margins and their inverse, a molecule-name key that works across five languages, a strength parser, and the choice of the most similar reference presentation. `part3/countries.py` turns each country's file into the same table.

Parts 2 and 3 read public files that `partN/fetch.py` downloads and reduces. A second workflow (`.github/workflows/taxation-efficiency-drug-prices-data.yml`) runs it on a part branch, or on demand, and commits the reduced files to `partN/data` with a log of what each download returned, so the analysis runs from a fixed snapshot.

Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` to reproduce the committed output exactly.

A GitHub Actions workflow (`.github/workflows/taxation-efficiency-drug-prices.yml`) reruns each part on every change and checks that the output matches the committed `output.txt`.

## Data

- **Part 1** downloads nothing. The behavioural model is calibrated to published summary figures, each quoted with its source in `part1/premises_numbers.py`: first-round contributions of 40–60 percent of the endowment (Ledyard 1995); 15–25 percent by the tenth round (Andreoni 1995, citing Isaac and Walker 1988); 82.5 percent of subjects giving their whole endowment in the final period of the partner treatment with punishment, and a total cost of punishment about as large as the benefit from the extra cooperation in that experiment (Fehr and Williams 2017, footnote 3). The laboratory designs in Section 6 are the cells reported by Isaac, Walker and Williams (1994, Table 1), Fehr and Gächter (2000) and Weimann et al. (2019). The model is not a reanalysis of any experiment's microdata.

- **Part 2** uses the snapshot in `part2/data`, taken on 12 November 2026:
  - CMS, *Medicare Part D Spending by Drug*, data years 2020–2024 (released June 2026). Gross spending (ingredient cost, dispensing fees, sales tax; before manufacturer rebates), dosage units and claims by brand name and generic name. Only the rows that add up all manufacturers of a brand name are used.
  - CMS / Medicaid, *National Average Drug Acquisition Cost (NADAC)*, weekly file for 2024, reduced to the 2024 mean per NDC.
  - FDA, *National Drug Code Directory* (product and package files), for dosage form, route, marketing category and non-proprietary name.
  - CMS, *Medicare Drug Price Negotiation Program: Selected Drug List and Maximum Fair Price data file* (version of 21 September 2026).
  - Assurance Maladie, *Open Medic 2024*, boxes and reimbursement base by CIP13 presentation dispensed by retail pharmacies.
  - ANSM / HAS / Assurance Maladie, *Base de données publique des médicaments* (BDPM): specialities, presentations with prices and dispensing fee, compositions and generic groups. This file is the current one, so its prices are those in force when the snapshot was taken.
  - ECB reference rates, US dollars per euro, annual averages.

  Published figures used for comparison or as scenarios are quoted with their sources in `part2/prices_numbers.py`: the brand and generic shares of US volume and spending and the rebate adjustment in Mulcahy, Schwam and Lovejoy (2024); Part D rebates on brand-name drugs in CBO (2018, 2022); French product rebates in DREES (2026); CMS's estimates of the savings from the negotiated prices (CMS 2024; Milliman 2026).

- **Part 3** uses the snapshot in `part3/data`, taken on 12 November 2026, and the French files of `part2/data`:
  - Ministerio de Sanidad, *Nomenclátor de facturación*: every presentation billed to the Spanish health system, with its retail price including VAT.
  - AIFA, *Lista dei farmaci di classe A per principio attivo*, 30 June 2026: retail prices of reimbursed presentations in Italy.
  - FPS Public Health / RIZIV-INAMI, *SAM v2* full export (version 12680): every authorised pack in Belgium with its ex-factory price; reduced to the current version of each record.
  - Infarmed, Deliberação 071/CD/2026: homogeneous groups and reference prices for the third quarter of 2026, with the maximum and current retail price of every presentation in a group (published as a PDF, kept as layout text).
  - Infarmed, tables of exceptional price revisions decided by the Ministry of Health, 2025 and 2026 (ambulatory market).
  - Infarmed, calculation form for the 2026 annual price revision of non-generic medicines and the note on margins and factors used to compute retail prices.
  - SNS transparency portal: monthly ambulatory retail value, packs and SNS spending by region.

  The rule itself comes from Decreto-Lei 97/2015, Portaria 195-C/2015 and, for 2026, Portaria 394/2025/1 (reference countries; retail prices up to 30 euros exempt from the annual revision; cuts above that capped at 20 percent).

## References

Andreoni, J. (1995). Cooperation in public-goods experiments: kindness or confusion? *American Economic Review*, 85(4), 891–904.

Ballard, C. L., Shoven, J. B., & Whalley, J. (1985). General equilibrium computations of the marginal welfare costs of taxes in the United States. *American Economic Review*, 75(1), 128–138.

Browning, E. K. (1976). The marginal cost of public funds. *Journal of Political Economy*, 84(2), 283–298.

Centers for Medicare & Medicaid Services (2024). Negotiating lower drug prices works, saves billions. Press release, 15 August 2024.

Centers for Medicare & Medicaid Services (2026). Medicare Part D Spending by Drug: methodology, data years 2020–2024.

Congressional Budget Office (2018). Options for Reducing the Deficit: 2019 to 2028. Option 54735: require manufacturers to pay a minimum rebate on drugs covered under Part D.

Congressional Budget Office (2022). *Prescription Drugs: Spending, Use, and Prices*. Publication 57050.

Danzon, P. M., & Chao, L.-W. (2000). Cross-national price differences for pharmaceuticals: how large, and why? *Journal of Health Economics*, 19(2), 159–195.

Diewert, W. E. (1976). Exact and superlative index numbers. *Journal of Econometrics*, 4(2), 115–145.

DREES (2026). Innovation thérapeutique et régulation des prix : les déterminants de la hausse des dépenses de médicaments en France (2016-2024). M. Cadillac & É. Maugendre, *Les Dossiers de la DREES*, 139.

Decreto-Lei n.º 97/2015, de 1 de junho. *Diário da República*, 1.ª série, n.º 105.

Fehr, E., & Gächter, S. (2000). Cooperation and punishment in public goods experiments. *American Economic Review*, 90(4), 980–994.

Fehr, E., & Williams, T. (2017). Creating an efficient culture of cooperation. IZA Discussion Paper No. 11131.

Fischbacher, U., & Gächter, S. (2010). Social preferences, beliefs, and the dynamics of free riding in public goods experiments. *American Economic Review*, 100(1), 541–556.

Gächter, S., Renner, E., & Sefton, M. (2008). The long-run benefits of punishment. *Science*, 322(5907), 1510.

Herrmann, B., Thöni, C., & Gächter, S. (2008). Antisocial punishment across societies. *Science*, 319(5868), 1362–1367.

Isaac, R. M., & Walker, J. M. (1988). Group size effects in public goods provision: the voluntary contributions mechanism. *Quarterly Journal of Economics*, 103(1), 179–199.

Isaac, R. M., Walker, J. M., & Williams, A. W. (1994). Group size and the voluntary provision of public goods: experimental evidence utilizing large groups. *Journal of Public Economics*, 54(1), 1–36.

Ledyard, J. O. (1995). Public goods: a survey of experimental research. In J. H. Kagel & A. E. Roth (Eds.), *The Handbook of Experimental Economics* (pp. 111–194). Princeton University Press.

Milliman (2026). Medicare drug price negotiation: navigating the next wave of maximum fair prices. 12 January 2026.

Infarmed (2025). Circular Informativa n.º 132/CD/100.20.200, de 17 de novembro de 2025: revisão anual de preços 2026. Formulário de cálculo e informações para cálculo de PVP de medicamentos (margens e fatores).

Mulcahy, A. W., Schwam, D., & Lovejoy, S. L. (2024). *International Prescription Drug Price Comparisons: Estimates Using 2022 Data*. RAND Corporation, RR-A788-3.

Portaria n.º 195-C/2015, de 30 de junho. *Diário da República*, 1.ª série, n.º 125.

Portaria n.º 394/2025/1, de 14 de novembro. *Diário da República*, 1.ª série.

Weimann, J., Brosig-Koch, J., Heinrich, T., Hennig-Schmidt, H., & Keser, C. (2019). Public good provision by large groups – the logic of collective action revisited. *European Economic Review*, 118, 348–363.
