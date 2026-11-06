# From concentration to exposure: a supply-based screen for EU import dependence

Code and data for the article *From Concentration to Exposure: Redesigning the European Union's Import-Dependence Screen with Production and Trade Data*.

The European Commission's 2021 review of strategic dependencies flags a product when (1) the Herfindahl index of its outside suppliers is above 0.4, (2) extra-EU imports are more than half of all EU imports, and (3) extra-EU imports exceed total EU exports. This folder diagnoses each criterion and tests a replacement: the share of EU supply (production plus extra-EU imports) provided by the largest outside supplier.

## Scripts

- `panel.py`: the HS 6-digit panel 2002–2025 from the companion analysis (`../europe-deindustrialisation/data`): the bounds that tie the Herfindahl index to the largest supplier's share, the products a dominant supplier with a spread fringe keeps out of the screen, threshold sensitivity, and year-to-year stability of the Commission screen and of a single-supplier screen.
- `energy.py`: natural gas, 1990–2024, as a known case: the Herfindahl index of outside suppliers against the largest supplier's share of EU supply and consumption.
- `supply.py`: the 2025 screen at the Prodcom level. EU sold production (Prodcom DS-059367) and 2025 trade at the CN 8-digit level summed to Prodcom codes with the CN–Prodcom correspondence give EU supply, the outside share of supply and single-supplier exposure; the Commission's three criteria are computed on the same products for comparison.
- `figures.py`: the seven figures.
- `check_results.py`: compares regenerated summaries with the committed ones within a small tolerance.

## Data

Fetched by GitHub Actions (`.github/workflows/dependency-screen-design-data.yml`) and committed to `data/`:

- `fetch_prodcom.py`: Eurostat Prodcom DS-059367 (CPA 2.2), reference year 2025, EU-27 sold production, exports and imports in euros; and the CN 2025 – Prodcom 2025 correspondence published by Statistics Netherlands (CBS).
- `fetch_cn8.py`: Eurostat Comext DS-045409, 2025, EU-27 imports and exports with partners outside and inside the EU for every CN 8-digit code, and imports by partner for every code in the correspondence.

## Reproducing

```
python -m pip install -r requirements.txt
python -m pytest -q tests
python panel.py && python energy.py && python supply.py && python figures.py
```

## Points to keep in mind

- Sold production is valued at producer prices and imports at the border (CIF); the two are close enough for shares but not identical.
- Prodcom codes whose CN codes are shared with another Prodcom code (276 of them) are left out, because their trade cannot be allocated. Eight products have no published production.
- Apparent consumption (production + imports − exports) is computed for reference but not used: it is zero or negative for 111 products, where exports include goods imported and shipped on.
- Imports are attributed to the country of origin; a component made in one country and assembled in another counts for the second.
