# Saving, investment and cross-border capital flows of the EU27: what "€300 billion a year" measures

Computational companion to Part 4 of a Medium series on Europe's low-wage path. Question: the claim that about €300 billion of European savings leave the EU every year, mainly for the United States, is used to argue that Europe lacks capital at home. Which number in the national accounts and the balance of payments does it correspond to, who saves and who lends abroad, in which instruments, and what has accumulated in the EU's external balance sheet?

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # Eurostat and OECD snapshot (already committed; rerun to refresh)
python analysis.py           # results/*.csv and results/analysis_summary.txt
python figures.py            # figures/fig1_*.png ... fig3_*.png
python -m pytest -q tests
```

`results/analysis_summary.txt` holds every number used in the article. The workflow `.github/workflows/eu-capital-exports.yml` reruns the analysis and the tests and checks that the summary is unchanged.

## Data (`fetch.py`, `data/`)

Eight requests made on 12 November 2026; `data/fetch_log.txt` records the URL, HTTP status, size and SHA-256 of each response, and the raw responses are kept gzipped in `data/raw/`.

Eurostat (JSON-stat dissemination API), the EU27 treated as one economy:

- `bop_eu6_q`: EU balance of payments with non-EU residents and with selected partners (United States, United Kingdom, Switzerland, offshore centres, China, Japan), quarterly, summed to calendar years; and portfolio investment by resident sector.
- `bop_iip6_q`: international investment position against non-EU residents at the end of each year, by functional category.
- `nasa_10_nf_tr`: gross saving, gross capital formation, consumption of fixed capital and net lending by institutional sector.
- `nama_10_gdp`: GDP at current prices.
- `nasa_10_f_bs`: financial assets of households (and non-profit institutions serving households) by instrument.

OECD Data Explorer:

- `DSD_NAMAIN10@DF_TABLE2`: net saving, consumption of fixed capital, gross capital formation and net lending for the United States, the euro area, the EU27, Japan and the United Kingdom.
- `DSD_NASEC20@DF_T720R_A`: financial assets of households by instrument, United States and Germany (Germany as a cross-check of the Eurostat balance sheet; the two agree exactly).

Eurostat and OECD data are reused under their terms; they are not relicensed by the repository's licence.

## Definitions

- **Gross saving** is disposable income not consumed; **gross capital formation** is investment in fixed assets and inventories; their difference, plus net capital transfers, is **net lending** to the rest of the world. For the EU27 as a whole, net lending in the sector accounts equals the balance of the current and capital accounts with non-EU residents in the balance of payments (430 billion euro in 2024 in both).
- **Net lending by sector** splits the same surplus among households, non-financial corporations, financial corporations and general government; the four add up to the total.
- **Balance of payments, financial account.** Net = net acquisition of foreign assets minus net incurrence of liabilities to non-residents, by functional category (direct, portfolio, derivatives, other investment, reserves). Reserve assets are not published separately for the EU27 in `bop_eu6_q`; they are computed as the residual of the published total. Gross portfolio "purchases" are net acquisitions of assets (purchases minus sales and redemptions).
- **BoP identity.** Current account + capital account + net errors and omissions = net financial account; the repository checks it for every year.
- **Stock-flow reconciliation.** The change in a net international position between two year-ends equals the cumulative net flows plus "other changes", which are mainly revaluations from prices and exchange rates.
- **Household portfolio shares** are instruments as a share of total financial assets, at market value, non-consolidated.

## Caveats

- Bilateral portfolio liabilities to US residents are not compiled (securities are held through custodians), so there is no net portfolio flow "to the United States"; only EU purchases of US securities are observable.
- Portfolio flows are recorded by the residence of the holder. Investment funds domiciled in Ireland and Luxembourg are EU residents: their purchases of US securities count as EU outflows even when their shareholders are not EU residents, and EU households' purchases of those funds are intra-EU.
- Net errors and omissions are large: over 2015–2025 they equal −21% of the cumulative current account surplus.
- The international investment position by functional category excludes reserve assets, which are not published for the EU27 in `bop_iip6_q`.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
