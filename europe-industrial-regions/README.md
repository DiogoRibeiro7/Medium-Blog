# Regions, not countries? Industrialisation on the map of Europe, 1900-2021

Code for an article testing Sidney Pollard's argument that Europe industrialised by regions rather than by countries, with the Rosés-Wolf database on regional GDP (version 7, 2025): 173 regions in 16 countries, with the share of industry and mining in employment, GDP and population at benchmark years from 1900 to 2015 and 2021.

## Data (`check_data.py`, `data/raw/`, not tracked)

CEPR, which hosts the database, refuses automated downloads, so the two files cannot be fetched by a script or by CI. Download them in a browser and save them in `data/raw/`:

- `RosesWolf_RegionalGDP_v7.xlsx` from https://cepr.org/system/files/2025-10/RosesWolf_RegionalGDP_v7.xlsx
- `RosesWolf_RegionalGDP_shapefile.zip` from https://cepr.org/system/files/2022-11/RosesWolf_RegionalGDP_shapefile.zip

Both are registered in the data registry as [`external/roses-wolf-regional-gdp-v7`](https://github.com/DiogoRibeiro7/data/tree/476fcef9fde3131168e7af1e2843cde2018bcdbe/external/roses-wolf-regional-gdp-v7) with size and SHA-256; [`data_sources/roses-wolf-regional-gdp-v7.json`](../data_sources/roses-wolf-regional-gdp-v7.json) copies that contract. `check_data.py` verifies the files before anything is read. No licence is stated, so the files are not stored here; cite *Rosés-Wolf database on regional GDP, version 7 (2025)*.

## Run

```
pip install -r requirements.txt
python check_data.py      # both files present and matching the registry
python -m pytest -q tests
python regions.py         # results/summary.txt and the CSVs figures.py reads; about a minute
python figures.py         # figures/fig1-fig5
```

`python regions.py 50` uses 50 random partitions instead of 1,000. CI runs the tests on synthetic maps only (the data tests skip without the files); `results/summary.txt` is the output of a full run with the files above.

## Method

- **Countries against random lines.** For each benchmark year, the population-weighted share of regional variation that lies between the 16 countries, against 1,000 random partitions of the map into 16 contiguous groups of about the countries' sizes (each the best of 50 draws by size match), and against random groupings that ignore geography.
- **Border premium.** For the 386 pairs of regions that share a land boundary, the absolute difference in the industry share (or log GDP per head) regressed on a cross-border dummy and log centroid distance, with Fafchamps-Gubert dyadic-robust standard errors; the same regression with the lines of each random partition gives the null distribution.
- **The 1900 core.** The 35 regions with the highest industry share in 1900, the land-connected clusters among them, and what happened to them by 2021.
- **Price base.** The workbook gives regional GDP in 2011 PPP dollars (to 2022) and in 1990 international dollars (to 2015). Inside a country and year the two differ by one factor, so regional ratios within a country are the same and only national levels move. The income tests are repeated in 1990 dollars, and section 4 of `results/summary.txt` lists national levels and every land border in both bases.
- **Borders of 1900.** The data use today's borders. Section 5 repeats the border test for 1900 and 1910 with Alsace in Germany, Ireland in the United Kingdom and Trentino-South Tyrol in Austria; regions split by a border of the time (Lorraine, Southern Denmark, Friuli-Venezia Giulia, Liège) keep their present country.

Islands and the three separate land masses are joined to the nearest region by straight links only to draw random partitions; the border test uses land neighbours alone.

Contact: dfr@esmad.ipp.pt
