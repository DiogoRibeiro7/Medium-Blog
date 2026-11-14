# Taxes, efficiency and the price of medicines

Public reproducibility companion for a series on Medium by Diogo Ribeiro. The editorial manuscripts are kept in the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium/tree/main/series/taxation-efficiency-drug-prices) repository; this folder holds the code, the printed numbers and the figures. The series starts from the slogan "taxation is theft", separates the premises of that argument that are normative from the one that can be measured, and then measures the efficiency claim where it is most concrete: what a single public buyer such as Portugal's SNS pays for medicines, and what that costs in other currencies (time to access, leakage through parallel export). Every number quoted in the articles is printed by the script in the corresponding folder, and the committed `output.txt` is what the script prints.

- **Part 1 — the premises of "taxation is theft", and the one that data can test** ([`part1/`](part1)). A linear public-goods economy. Voluntary contributions with imperfect conditional cooperation (Fischbacher and Gächter 2010), calibrated to the first-round and tenth-round contribution levels summarised by Ledyard (1995) and by Andreoni (1995) from Isaac and Walker (1988), with a sensitivity analysis over the parts the calibration does not identify; costly peer punishment calibrated to two results of Fehr and Gächter (2000) and checked out of sample against the 50-round result of Gächter, Renner and Sefton (2008); antisocial punishment (Herrmann, Thöni and Gächter 2008); a tax with a marginal cost of public funds from Browning (1976) and Ballard, Shoven and Whalley (1985) and a share of revenue that never reaches the good; the break-even frontier between voluntary provision and the tax; and a map of the group sizes and returns that laboratory studies have covered against a national public good.
- **Part 2 — how much more the US pays for medicines, and how the answer depends on the method** ([`part2/`](part2)). Medicare Part D against the French Assurance Maladie in 2024, for tablets and capsules dispensed by retail pharmacies, molecule by molecule: 432 molecules covering $121.9 bn of Part D spending. The molecules are sorted into single-source (no generic competition in either country), multi-source (generics in both) and mixed baskets, and compared with Laspeyres, Paasche, Fisher and Jevons indices, with a bootstrap over molecules. The script shows why a brand-against-brand comparison misleads, how an all-drug index moves with the generic share of units alone, and how the single-source gap responds to the French dispensing fee, the exchange rate, the basket threshold, the oral-solid classification, outliers, the largest molecules, pharmacy acquisition prices (NADAC) and confidential rebates on both sides. It then sets the Medicare negotiated prices of the oral drugs selected for 2026 and 2027 against 2024 acquisition prices and current French prices.
- **Part 3 — how a small single buyer sets the price of a medicine** ([`part3/`](part3)). Portugal does not publish its prices presentation by presentation, so the script rebuilds the rule that sets them: Infarmed's external reference pricing formula (average ex-factory price in Spain, France, Italy and Belgium, adjusted for strength and pack size, then the regulated margins), as implemented in Infarmed's own 2026 calculation form. It feeds the rule the public price lists of the four countries (France's manufacturer price is backed out of the retail price; 99 percent of original brands invert to a whole-cent price), applies it to the centrally authorised brands with no generic competition in France (French 2024 volumes as weights), measures how far the Portuguese ceiling sits from each reference country and how much the choice of rule and countries moves it, and checks the rebuilt rule against the only Portuguese prices published by presentation, the original brands in Infarmed's list of homogeneous groups. It also reads Infarmed's tables of exceptional price revisions and the SNS ambulatory aggregates.
- **Part 4 — what Portugal's wait for new medicines is made of** ([`part4/`](part4)). Tests whether the time from EU authorisation to availability in Portugal is explained by market size, distance from the centre of Europe or price, using European comparisons only. Across 29 European countries (W.A.I.T. 2025), the time to availability is set against population, the distance from each capital to Brussels, actual individual consumption per head and the EURIPID price level of medicines, with bootstrap intervals, leave-one-out fits, Portugal's out-of-sample prediction and groups of countries that resemble Portugal on one dimension at a time. Portugal's own pipeline is then split into the time before the company files and the time after, from Infarmed's 2020–2025 access report against the European averages of the CRA Access Hurdles Portal, and the time after filing is decomposed process by process from Infarmed's public process tracker (SIATS) into days attributed to Infarmed, to the company and to price negotiation, set against the legal deadlines and split by company size, orphan status and oncology.
- **Part 5 — medicines that leave Portugal, and the lists that try to keep them** ([`part5/`](part5)). Measures parallel export from Portugal and Infarmed's export controls from public records only. How much leaves and to where: Infarmed's counts of export notifications and prohibitions (2021, 2024, 2025) and Eurostat's trade in medicaments (HS 3004). Where parallel distribution goes and where it may come from: every notice in the EMA register of parallel distribution of centrally authorised medicines, with the destinations named, how often each country is left out as a possible source, and how that lines up with the EURIPID price level of medicines, with bootstrap intervals. What Portugal restricts: every prior-notification list and suspended-export list Infarmed published from May 2021 to November 2026, recovered from the live page and from Wayback Machine snapshots, with their size, composition and persistence; and the five monthly lists of export prohibitions of 2026 with Infarmed's own indicators (export intentions and exports against the quantity sold, satisfaction ratios, pharmacy shortage reports). Why medicines run short, by the reasons companies report. Last, the Portuguese ceilings that Part 3's rebuilt pricing rule gives the brands Infarmed lists for export reasons.
- **Part 6 — who pays for pharmaceutical innovation** ([`part6/`](part6)). Tests the argument that countries paying low prices for medicines free-ride on the United States. Reads the annual reports (10-K and 20-F) of seventeen research-based companies from their XBRL data at the SEC, 2015–2025: total revenue, revenue in the United States and in Europe as each company tags it, R&D, cost of sales, selling and administrative costs. Compares the US share of revenue with the US share of world and high-income GDP, turns the revenue share into a share of the revenue left after cost of sales and selling costs under the US–foreign price gaps estimated by RAND (2024), and asks whether revenue earned outside the United States would cover the companies' R&D. Then sets each European country's pharmaceutical market against the industry R&D carried out in it (EFPIA country tables) and against public and academic medical research (OECD), and applies the literature's elasticities of innovation to market size to what Portugal, or the largest European markets, would add by paying more.
- **Part 7 — a ledger of what Portugal's public buyer saves, and where the saving comes from** ([`part7/`](part7)). Puts the results of Parts 2 to 6 into one account for Portugal, in euros a year. SNS spending on medicines in hospitals and pharmacies, 2015–2025 (SNS transparency portal), and the industry's returns to the SNS (Infarmed's estimate for 2025). A log decomposition of the gap between US prices and Portugal's net price into the part carried by the reference countries' list prices, the reference rule's own premium (Part 3) and the SNS's own returns, under four US benchmarks (Part 2's index before and after US rebates, RAND 2024 gross and net). A regression of the EURIPID price level of medicines on population and consumption per head across 25 European countries, with a bootstrap, to ask what buyer size is worth. The time account from Part 4 (days beyond the EU average and beyond the legal deadlines, and the spending a delay defers), leakage and the contribution to innovation from Parts 5 and 6, and administration and the public share of pharmaceutical spending from the OECD health accounts. A simulation over the ranges of the uncertain inputs gives each account a median and an interval.

Follow-ups test conjectures left open by the parts.

- **Follow-up 1 — confidential returns in Portugal and the countries it references** ([`followup1/`](followup1)). Tests the conjecture of Part 7: that reference pricing turns list prices into something other payers use, so that a small country that is both referenced and referencing should bargain through confidential returns, more so over time and in hospitals, and should have a wider gap between list and net prices than the countries it references. Puts together every published piece on Portugal's returns (the State–industry agreements 2013–2025, the extraordinary contribution on the industry, Infarmed's figures for 2023–2025 and the industry's own studies, with the rounding of each figure carried through), France's rebates 2016–2024 by setting (DREES, IGAS-IGF), Belgium's compensations under article 81/111 conventions and levies 2015–2024 by setting (INAMI), and what Italy and Spain publish. Compares the share returned in 2024, turns it into net prices with Part 3's list-price ratios (with a simulation over the uncertain inputs), and computes what the referenced countries' own rebates do to Portugal's price ceiling.
- **Follow-up 2 — where the rise in the US share of drugmakers' revenue comes from** ([`followup2/`](followup2)). Tests the second conjecture of Part 6: that the US share rises because new medicines are launched first, and priced highest, in the United States, so that the rise is concentrated in medicines launched in the period and the share is higher for companies with more revenue from medicines under five years old. Rebuilds, from the XBRL instances of the seventeen companies' annual reports, the revenue of each medicine in the United States and in total (2016–2025 for the companies that tag revenue by product and country), dates each medicine by its first US approval (Drugs@FDA and the Purple Book), and splits the change in the pooled US share exactly, medicine by medicine (Griliches and Regev 1995), by launch cohort and by age. Adds the life cycle of a medicine's US share from launch and the relation, across and within companies, between revenue from new medicines and the US share.
- **Follow-up 3 — whether old medicines lose US share in the United States or abroad** ([`followup3/`](followup3)). Tests the second conjecture of follow-up 2: that the fall in the US share of established medicines comes from the United States (falling net prices, earlier generic and biosimilar entry) rather than from growth abroad. Uses follow-up 2's medicine-by-medicine revenue from the SEC filings and splits each medicine's change in US share exactly into the part due to its US revenue and the part due to its revenue abroad (a two-factor Shapley split), by years since first US approval. Tests the two predictions (the fall concentrated in medicines whose US revenue fell; medicines whose US revenue grew keeping their share), follows the medicines on the market for ten years or more in 2016 to 2025, compares which market fell first with the FDA and EMA approval dates of generics and biosimilars, and sets the companies' US revenue against Medicare Part D and Part B volumes, 2020–2024.

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
cd ../part4
python access_numbers.py > output.txt     # a few seconds; reads data/, writes figure_data.json
python figures.py
cd ../part5
python export_numbers.py > output.txt     # a few seconds; reads data/, ../part4/data and ../part3/output.txt
python figures.py
cd ../part6
python innovation_numbers.py > output.txt # a few seconds; reads data/ and ../part2/output.txt
python figures.py
cd ../part7
python ledger_numbers.py > output.txt     # a few seconds; reads data/ and the results of Parts 2 to 6
python figures.py
cd ../followup1
python rebates_numbers.py > output.txt    # a few seconds; reads data/, ../part7/data and Parts 3 and 7
python figures.py
cd ../followup2
python portfolio_numbers.py > output.txt  # about ten seconds; reads data/ and ../part6/data, ../part6/output.txt
python figures.py
cd ../followup3
python us_side_numbers.py > output.txt    # about thirty seconds; reads data/, ../followup2/data, ../part6/data and ../part4/data
python figures.py
cd ..
python -m unittest discover -s tests      # tests of the shared tools
```

`publicgoods.py` holds the tools of Part 1: the conditional-cooperation model, peer punishment with prosocial and antisocial sanctions, and the welfare algebra of voluntary provision against a tax (the break-even share `s_star` and the break-even marginal group return `g_star`).

`drugprices.py` holds the tools of the drug-price parts: name matching across languages (international non-proprietary names, US adopted names, Part D's truncated names), the oral-solid and segment classification of Part D rows through the FDA NDC directory, French units per box, molecule baskets, the index formulas and the bootstrap, NADAC by molecule, and the negotiated-price table.

`erp.py` holds Portugal's reference pricing rule and the conversions it needs: the pack-size factor and the strength and pack adjustment of Infarmed's form, the average (ambulatory) and lowest (hospital) rules, the six retail-price tiers and their inverse, the factors that turn Spanish and Italian retail prices into ex-factory prices, the French margins and their inverse, a molecule-name key that works across five languages, a strength parser, and the choice of the most similar reference presentation. `part3/countries.py` turns each country's file into the same table.

Parts 2 to 7 and the follow-ups read public files that `partN/fetch.py` (or `followupN/fetch.py`) downloads and reduces. A second workflow (`.github/workflows/taxation-efficiency-drug-prices-data.yml`) runs it on a part or follow-up branch, or on demand, and commits the reduced files to its `data` folder with a log of what each download returned, so the analysis runs from a fixed snapshot.

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

- **Part 4** uses the snapshot in `part4/data`, taken on 12 November 2026:
  - Infarmed, *Tracker de Processos* (SIATS public process search): every reimbursement and hospital prior-evaluation process for a new active substance and for a new indication that was open on 1 January 2026 or was submitted or concluded after it, with the dates of authorisation, submission, start of assessment, conclusion and access, and the days the tracker attributes to Infarmed, to the company and to negotiation. Tracker updated 12 November 2026.
  - Infarmed, *Acesso a novos medicamentos 2020-2025* (May 2026), kept as layout text: filing rates and mean times for the 282 new active substances authorised centrally in 2020–2025.
  - Charles River Associates for EFPIA, *European Access Hurdles Portal: results from the fourth year of data collection* (May 2026), kept as layout text: European averages of the time to file and the time to a reimbursement decision.
  - EFPIA / IQVIA, *Patients W.A.I.T. Indicator 2025 survey* (medicines authorised 2021–2024, availability on 5 January 2026) and the 2024 survey: number of medicines available, mean and median time to availability by country for all medicines, oncology and orphans, read off the charts of the PDF by the position of each country label. The EU27 means of the values read this way equal the averages printed in the report (597, 655 and 614 days).
  - EMA, list of centrally authorised medicines (authorisation date, orphan status, ATC code), joined to the tracker by brand name.
  - EURIPID / Gesundheit Österreich, *Purchasing power parities of pharmaceutical products* (2018): price level indices of medicines by country, 2015–2017, all medicines and by ATC main group.
  - Eurostat: population on 1 January (`tps00001`), GDP per head in PPS (`tec00114`), and actual individual consumption per head in PPS with its price level index (`prc_ppp_ind`).

  The legal deadlines are those of Portaria 195-A/2015 (decision within 75 days, the clock stopping while the company answers; contract negotiation within 30 working days), which still govern processes opened before Decreto-Lei 118/2026 came into force on 1 July 2026.

- **Part 5** uses the snapshot in `part5/data`, taken on 12 November 2026, with Part 4's population and EURIPID files and Part 3's `output.txt`:
  - Infarmed, lists published under the regulation on transactions of medicines to other countries (Deliberação 391/2021, replaced by Deliberação 234/2025 from 1 March 2025): the prior-notification lists (LNP), the lists of medicines whose export is temporarily suspended, and, from 2026, the monthly lists of export prohibitions with Infarmed's indicators. The lists were found on the live page *Gestão da disponibilidade do medicamento* and on its Wayback Machine snapshots since 2021, then downloaded from Infarmed and parsed (XLSX, or PDF kept as layout text): 54 lists, of which 51 were read. Three could not be downloaded because Infarmed's server refused the connection (LNP of April and August 2024; suspended list of January 2026). The deliberations that approve each list are kept as dates only. The two annexes of Deliberação 391/2021 are read as the first lists.
  - Infarmed, *Relatório anual: gestão da disponibilidade de medicamentos* for 2021, 2024 and 2025, kept as layout text: export notifications and prohibitions, the most exported listed medicines, export destinations and the reasons companies give for shortages. The destination and shortage-reason charts of 2024 and 2025 are read off the charts; the script checks that the percentages printed on each chart are exactly the ones it uses.
  - EMA, public register of parallel distribution notices (IRIS): every notice, with product, distributor, status, date, destination and source countries.
  - Eurostat, *EU trade since 2002 by HS2-4-6 and CN8* (`ds-045409`): Portugal's imports and exports of medicaments in measured doses (HS 3004) by partner, 2015–2025, in euros and in 100 kg.

  The legal basis is article 100 of Decreto-Lei 176/2006 (Estatuto do Medicamento), as amended in 2019.

- **Part 6** uses the snapshot in `part6/data`, taken on 12 November 2026, with Part 2's `output.txt`:
  - SEC EDGAR: the 10-K or 20-F of each fiscal year 2015–2025 for Pfizer, Merck & Co, Johnson & Johnson, AbbVie, Bristol-Myers Squibb, Eli Lilly, Amgen, Gilead, Regeneron, Vertex, Biogen, Novartis, AstraZeneca, GSK, Sanofi, Novo Nordisk and Takeda, read from the XBRL instance of each filing (167 filings; the foreign filers file XBRL from 2017–2021 onwards). Revenue in the United States is the value a company tags with the United States member of a geographic axis; it is found for 14 companies in fiscal year 2025 and for 7 in every year.
  - ECB reference exchange rates, annual averages.
  - EFPIA, *The Pharmaceutical Industry in Figures, Key Data 2026*, kept as layout text: market value at ex-factory prices, industry R&D and production by European country (2024), the world market by region (IQVIA MIDAS, 2025), the sales of new medicines by market and the origin of new molecules.
  - World Bank: GDP in current dollars and population. Eurostat: GDP in euros (`nama_10_gdp`), GBARD for health (`gba_nabsfin07`) and business R&D in pharmaceuticals (`rd_e_berdindr2`).
  - OECD: business R&D performed in the pharmaceutical industry and GDP in PPP dollars (MSTI), government budget allocations for R&D by objective (health and total), and R&D in medical and health sciences by sector of performance (GERD by field; the United States does not report it).

  Parameters taken from the literature are quoted with their sources in `part6/innovation_numbers.py`: the US–foreign price ratios for brand-name originator medicines (Mulcahy, Schwam and Lovejoy 2024) and the elasticities of the number of new medicines to market size (Acemoglu and Linn 2004; Dubois, de Mouzon, Scott Morton and Seabright 2015).

- **Part 7** uses the snapshot in `part7/data`, taken on 13 November 2026, with the committed `figure_data.json`, `output.txt` and data of Parts 2 to 6:
  - SNS transparency portal: *Despesa com Medicamentos nos Hospitais do SNS* and *Despesa com Medicamentos no Ambulatório*, monthly SNS charges by region from January 2011 (pharmacy: SNS charges, patients' charges, retail value).
  - Infarmed, *Nota Informativa: Reforço do acesso dos cidadãos e impacto da inovação*, 15 November 2025, kept as text: the estimated annual total of industry returns to the SNS (EUR 500–600 million).
  - APIFARMA, *Boletim de Conjuntura* n.º 113, December 2025, kept as text: the pharmacy market at ex-factory prices and the share of it in homogeneous groups (with generic competition). The 2025 agreement between the Portuguese State and the pharmaceutical industry, kept as text.
  - OECD Health Statistics, System of Health Accounts (`DSD_SHA@DF_SHA`): current health expenditure, pharmaceuticals and other medical non-durables (HC.5.1), prescribed medicines (HC.5.1.1) and governance and administration (HC.7), by financing scheme (total, government and compulsory schemes, household out-of-pocket), 2015 onwards.

  The share of hospital spending on medicines without generic or biosimilar competition has no public figure; the script uses 60 to 90 percent and shows how the results move with it. The legal deadlines are those of Portaria 195-A/2015.

- **Follow-up 1** uses the snapshot in `followup1/data`, taken on 13 November 2026, kept as text (each document is checked for the sentences the analysis reads), with the SNS spending series and Infarmed's note in `part7/data` and the list-price ratios of Part 3:
  - Portugal: the agreements between the State and the pharmaceutical industry for 2013, 2014, 2015, 2016 and 2017 (English versions published by APIFARMA), 2024 and 2025; APIFARMA's *e-Pharma* newsletters of September 2024 and September 2025; the Conselho das Finanças Públicas reports *Evolução do desempenho do Serviço Nacional de Saúde* for 2020, 2022, 2023, 2024 and 2025 (revenue of the extraordinary contribution on the pharmaceutical industry); ECO's report of 29 May 2025 on Infarmed's monitoring of SNS spending in 2024, gross and net of returns.
  - France: DREES, *Les Dossiers de la Drees* n.º 139 (July 2026), and *Les établissements de santé*, 2026 edition, sheet 23; the IGAS-IGF mission report on the regulation of medicine prices (July 2025).
  - Belgium: INAMI-RIZIV, MORSE reports with data for 2022–2023 and for 2024 (gross and net spending, compensations under article 81/111 conventions, levies, and the cumulative table of convention turnover and compensations).
  - Italy: AIFA, monitoring of national and regional pharmaceutical spending, January–December 2024, final report and summary.

  The Spanish deductions (Real Decreto-ley 8/2010, as amended by Real Decreto-ley 9/2011) and the Italian rule that companies repay half of the overrun of the ceiling on direct purchases (Law 145/2018, art. 1, c. 580) are quoted from the law in `followup1/rebates_numbers.py`. No source splits Portugal's returns between hospitals and pharmacies, and the APIFARMA newsletter does not state the base of its 2018 and 2022 shares; the script shows both readings.

- **Follow-up 2** uses the snapshot in `followup2/data`, taken on 13 November 2026, with the exchange rates and output of Part 6:
  - SEC EDGAR: the annual reports (10-K and 20-F) of the seventeen companies of Part 6, fiscal years 2015–2025; every annual revenue fact of each XBRL instance is kept with its dimensions (product, geography, segment), for the year reported and the comparative years. Thirteen companies tag revenue by product and by country in some years (most from fiscal 2016, through the comparatives of their 2018 reports); Bristol-Myers Squibb, GSK, Pfizer and Takeda tag products without countries.
  - openFDA, Drugs@FDA bulk file: brand names, applications and the date of the original approval of each NDA and BLA.
  - FDA Purple Book, January 2026 extract: licensed biological products of CDER and CBER, with the licence date of each product.

  A product member of a filing is matched to a US brand name by its words (the leftmost word that begins a brand name; four names used outside the US or misspelt are aliased); members that name a group, a business or a kind of revenue stay out. The decomposition covers the medicines a company reports one by one with US and total revenue, not the company's whole revenue; the script prints how much of each company's revenue that is.

- **Follow-up 3** uses the snapshot in `followup3/data`, taken on 14 November 2026, with the revenue and FDA files of follow-up 2 (`followup2/data`), the exchange rates of Part 6 and the EMA medicines list of Part 4:
  - CMS, Medicare Part D Spending by Drug and Medicare Part B Spending by Drug, the release covering 2020–2024 (the only one in the CMS data catalogue): spending, dosage or billing units, claims and beneficiaries by brand (Part D, the "Overall" row of each brand and generic name) and by HCPCS code (Part B). Spending is gross of manufacturer rebates in Part D; Part B pays on average sales prices.

  Medicare rows are assigned to a medicine when their brand name begins with its name, or is contained in the name of the XBRL member that reports it (members such as `PREZISTAPREZCOBIXREZOLSTASYMTUZA` report several brands as one line). First US generic approvals are the earliest ANDA with the same active moieties (salt and hydrate words removed); first US biosimilars, the earliest 351(k) licence naming the medicine as reference product; first EU biosimilars, the earliest centrally authorised biosimilar of the same substance. Approval is not launch, and the script compares both with the year each market's revenue peaked.

## References

Acemoglu, D., & Linn, J. (2004). Market size in innovation: theory and evidence from the pharmaceutical industry. *Quarterly Journal of Economics*, 119(3), 1049–1090.

AIFA (2025). *Monitoraggio della Spesa Farmaceutica Nazionale e Regionale, Gennaio-Dicembre 2024, Consuntivo*. 29 luglio 2025; and its *Sintesi*.

Andreoni, J. (1995). Cooperation in public-goods experiments: kindness or confusion? *American Economic Review*, 85(4), 891–904.

APIFARMA (2024). *e-Pharma*, newsletter, setembro de 2024.

APIFARMA (2025). *Boletim de Conjuntura* n.º 113, dezembro de 2025. Núcleo de Estudos e Análise.

APIFARMA (2025). *e-Pharma*, newsletter, setembro de 2025.

Ballard, C. L., Shoven, J. B., & Whalley, J. (1985). General equilibrium computations of the marginal welfare costs of taxes in the United States. *American Economic Review*, 75(1), 128–138.

Blume-Kohout, M. E., & Sood, N. (2013). Market size and innovation: effects of Medicare Part D on pharmaceutical research and development. *Journal of Public Economics*, 97, 327–336.

Browning, E. K. (1976). The marginal cost of public funds. *Journal of Political Economy*, 84(2), 283–298.

Centers for Medicare & Medicaid Services (2024). Negotiating lower drug prices works, saves billions. Press release, 15 August 2024.

Centers for Medicare & Medicaid Services (2026). Medicare Part D Spending by Drug: methodology, data years 2020–2024.

Centers for Medicare & Medicaid Services (2026). Medicare Part B Spending by Drug: data years 2020–2024.

Charles River Associates (2026). *European Access Hurdles Portal: results from the fourth year of data collection*. Final report for EFPIA, May 2026.

Cleary, E. G., Beierlein, J. M., Khanuja, N. S., McNamee, L. M., & Ledley, F. D. (2018). Contribution of NIH funding to new drug approvals 2010–2016. *Proceedings of the National Academy of Sciences*, 115(10), 2329–2334.

Cockburn, I. M., Lanjouw, J. O., & Schankerman, M. (2016). Patents and the global diffusion of new drugs. *American Economic Review*, 106(1), 136–164.

Congressional Budget Office (2018). Options for Reducing the Deficit: 2019 to 2028. Option 54735: require manufacturers to pay a minimum rebate on drugs covered under Part D.

Congressional Budget Office (2022). *Prescription Drugs: Spending, Use, and Prices*. Publication 57050.

Conselho das Finanças Públicas (2021–2026). *Evolução do desempenho do Serviço Nacional de Saúde* em 2020, 2022, 2023, 2024 e 2025. Relatórios n.º 06/2021, 07/2023, 07/2024, 05/2025 e 05/2026.

Costa-Font, J., & Kanavos, P. (2007). Medicines parallel trade in the European Union: a gravity specification. LSE Health Working Paper 6/2007.

Council of Economic Advisers (2018). *Reforming Biopharmaceutical Pricing at Home and Abroad*. White paper, February 2018.

Court of Justice of the European Union (2008). Judgment of 16 September 2008, Sot. Lélos kai Sia EE and Others v GlaxoSmithKline AEVE, Joined Cases C-468/06 to C-478/06, ECLI:EU:C:2008:504.

Danzon, P. M., & Chao, L.-W. (2000). Cross-national price differences for pharmaceuticals: how large, and why? *Journal of Health Economics*, 19(2), 159–195.

Danzon, P. M., & Towse, A. (2003). Differential pricing for pharmaceuticals: reconciling access, R&D and patents. *International Journal of Health Care Finance and Economics*, 3(3), 183–205.

Danzon, P. M., Wang, Y. R., & Wang, L. (2005). The impact of price regulation on the launch delay of new drugs: evidence from twenty-five major markets in the 1990s. *Health Economics*, 14(3), 269–292.

Decreto-Lei n.º 176/2006, de 30 de agosto (Estatuto do Medicamento). *Diário da República*, 1.ª série, n.º 167.

Decreto-Lei n.º 97/2015, de 1 de junho. *Diário da República*, 1.ª série, n.º 105.

Decreto-Lei n.º 118/2026, de 17 de junho. *Diário da República*, 1.ª série.

Diewert, W. E. (1976). Exact and superlative index numbers. *Journal of Econometrics*, 4(2), 115–145.

DREES (2026). Innovation thérapeutique et régulation des prix : les déterminants de la hausse des dépenses de médicaments en France (2016-2024). M. Cadillac & É. Maugendre, *Les Dossiers de la DREES*, 139.

DREES (2026). Les médicaments et dispositifs médicaux onéreux. In *Les établissements de santé*, édition 2026, fiche 23.

Dubois, P., de Mouzon, O., Scott-Morton, F., & Seabright, P. (2015). Market size and pharmaceutical innovation. *RAND Journal of Economics*, 46(4), 844–871.

Duso, T., Herr, A., & Suppliet, M. (2014). The welfare impact of parallel imports: a structural approach applied to the German market for oral anti-diabetics. *Health Economics*, 23(9), 1036–1057.

ECO (2025). Despesa do SNS com medicamentos subiu para 3.960 milhões de euros em 2024. Lusa, 29 de maio de 2025.

EFPIA & IQVIA (2026). *EFPIA Patients W.A.I.T. Indicator 2025 survey*. May 2026.

EFPIA (2026). *The Pharmaceutical Industry in Figures: Key Data 2026*. Brussels.

EURIPID & Gesundheit Österreich (2018). *Purchasing power parities of pharmaceutical products: final report*. Vienna.

European Commission, Directorate-General for Health and Food Safety (2018). Paper on the obligation of continuous supply to tackle the problem of shortages of medicines. Agreed by the ad-hoc technical meeting under the Pharmaceutical Committee on shortages of medicines, 25 May 2018.

FDA (2026). *Drugs@FDA*, via openFDA bulk download (drug/drugsfda), November 2026.

FDA (2026). *Purple Book: Database of Licensed Biological Products*, data download, January 2026.

Fehr, E., & Gächter, S. (2000). Cooperation and punishment in public goods experiments. *American Economic Review*, 90(4), 980–994.

Fehr, E., & Williams, T. (2017). Creating an efficient culture of cooperation. IZA Discussion Paper No. 11131.

Fischbacher, U., & Gächter, S. (2010). Social preferences, beliefs, and the dynamics of free riding in public goods experiments. *American Economic Review*, 100(1), 541–556.

Frech, H. E., Pauly, M. V., Comanor, W. S., & Martinez, J. R. (2023). Pharmaceutical pricing and R&D as a global public good. NBER Working Paper 31272.

Gächter, S., Renner, E., & Sefton, M. (2008). The long-run benefits of punishment. *Science*, 322(5907), 1510.

Ganslandt, M., & Maskus, K. E. (2004). Parallel imports and the pricing of pharmaceutical products: evidence from the European Union. *Journal of Health Economics*, 23(5), 1035–1057.

Goldman, D., & Lakdawalla, D. (2018). *The Global Burden of Medical Innovation*. USC Schaeffer Center White Paper, January 2018.

Griliches, Z., & Regev, H. (1995). Firm productivity in Israeli industry 1979–1988. *Journal of Econometrics*, 65(1), 175–203.

Herrmann, B., Thöni, C., & Gächter, S. (2008). Antisocial punishment across societies. *Science*, 319(5868), 1362–1367.

IGAS & IGF (2025). *Bilan et évolution du dispositif de régulation du prix des médicaments*. Rapport IGF n.º 2025-M-015-03, IGAS n.º 2025-019, juillet 2025.

INAMI-RIZIV (2024). *Monitoring Of Reimbursement Significant Expenses (MORSE)*, rapport 2024, données 2022-2023.

INAMI-RIZIV (2026). *Monitoring Of Reimbursement Significant Expenses (MORSE)*, data 2024. January 2026. doi:10.5281/zenodo.18154911.

Infarmed (2021). Deliberação n.º 391/2021: regulamento sobre a notificação prévia de transações de medicamentos para o exterior do país. *Diário da República*, 2.ª série, n.º 78, 22 de abril de 2021.

Infarmed (2022). *Relatório anual 2021: gestão da disponibilidade de medicamentos*. Fevereiro de 2022.

Infarmed (2025). *Relatório anual 2024: gestão da disponibilidade de medicamentos*.

Infarmed (2025). Circular Informativa n.º 132/CD/100.20.200, de 17 de novembro de 2025: revisão anual de preços 2026. Formulário de cálculo e informações para cálculo de PVP de medicamentos (margens e fatores).

Infarmed (2025). Deliberação n.º 234/2025: regulamento sobre o controlo de transações de medicamentos para o exterior do país. *Diário da República*, 2.ª série, n.º 34, 18 de fevereiro de 2025.

Infarmed (2026). *Acesso a novos medicamentos 2020-2025*. Relatório, maio de 2026.

Infarmed (2026). *Relatório anual 2025: gestão da disponibilidade de medicamentos*.

Infarmed (2026). Circular Informativa n.º 044/CD/100.20.200, de 30 de abril de 2026: monitorização de tempos de avaliação de financiamento de medicamentos (Tracker de Processos).

Infarmed (2025). Nota Informativa: Reforço do acesso dos cidadãos e impacto da inovação: evolução da utilização e despesa com medicamentos em 2025. 15 de novembro de 2025.

Isaac, R. M., & Walker, J. M. (1988). Group size effects in public goods provision: the voluntary contributions mechanism. *Quarterly Journal of Economics*, 103(1), 179–199.

Isaac, R. M., Walker, J. M., & Williams, A. W. (1994). Group size and the voluntary provision of public goods: experimental evidence utilizing large groups. *Journal of Public Economics*, 54(1), 1–36.

Kanavos, P., & Costa-Font, J. (2005). Pharmaceutical parallel trade in Europe: stakeholder and competition effects. *Economic Policy*, 20(44), 751–798.

Kanavos, P., Costa-i-Font, J., Merkur, S., & Gemmill, M. (2004). *The economic impact of pharmaceutical parallel trade in European Union member states: a stakeholder analysis*. LSE Health and Social Care Special Research Paper.

Kyle, M. K. (2007). Pharmaceutical price controls and entry strategies. *Review of Economics and Statistics*, 89(1), 88–99.

Ledyard, J. O. (1995). Public goods: a survey of experimental research. In J. H. Kagel & A. E. Roth (Eds.), *The Handbook of Experimental Economics* (pp. 111–194). Princeton University Press.

Legge 30 dicembre 2018, n. 145. Bilancio di previsione dello Stato per l'anno finanziario 2019 (art. 1, commi 574-584). *Gazzetta Ufficiale* n. 302.

Maini, L., & Pammolli, F. (2023). Reference pricing as a deterrent to entry: evidence from the European pharmaceutical market. *American Economic Journal: Microeconomics*, 15(2), 345–383.

Milliman (2026). Medicare drug price negotiation: navigating the next wave of maximum fair prices. 12 January 2026.

Mulcahy, A. W., Schwam, D., & Lovejoy, S. L. (2024). *International Prescription Drug Price Comparisons: Estimates Using 2022 Data*. RAND Corporation, RR-A788-3.

Nolen, N., & Balling, S. (2021). *Parallel trade of pharmaceuticals and its problems in the EU*. cepInput 11/2021, Centrum für Europäische Politik.

OECD (2026). *OECD Health Statistics 2026*: System of Health Accounts, health expenditure and financing (DSD_SHA@DF_SHA).

Portaria n.º 195-A/2015, de 30 de junho. *Diário da República*, 1.ª série, n.º 125.

Portaria n.º 195-C/2015, de 30 de junho. *Diário da República*, 1.ª série, n.º 125.

Portaria n.º 394/2025/1, de 14 de novembro. *Diário da República*, 1.ª série.

Portuguese State & Indústria Farmacêutica (2013–2024). Agreements for 2013 (amendment to the agreement of 2012), 2014, 2015 and 2016–2018, the amendment for 2017, and the renewal for 2024 (texts published by APIFARMA).

Portuguese State & Indústria Farmacêutica (2025). Acordo entre o Estado Português e a Indústria Farmacêutica, 20 de março de 2025.

Real Decreto-ley 8/2010, de 20 de mayo, por el que se adoptan medidas extraordinarias para la reducción del déficit público. *Boletín Oficial del Estado*; amended by Real Decreto-ley 9/2011, de 19 de agosto.

Shorrocks, A. F. (2013). Decomposition procedures for distributional analysis: a unified framework based on the Shapley value. *Journal of Economic Inequality*, 11(1), 99–126.

Weimann, J., Brosig-Koch, J., Heinrich, T., Hennig-Schmidt, H., & Keser, C. (2019). Public good provision by large groups – the logic of collective action revisited. *European Economic Review*, 118, 348–363.
