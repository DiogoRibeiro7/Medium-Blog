# China's manufacturing labour costs against Mexico, Vietnam and the EU, 1995–2022

Computational companion to Part 2 of a Medium series on Europe's low-wage path. Question: does China still compete in manufacturing on low pay? The repository compares compensation per worker, value added per worker and the labour share of value added across 25 economies, splits China's dollar wage growth into yuan wage growth and exchange-rate movement, decomposes the growth of value added per worker into within-industry and between-industry terms, and puts the published robot statistics next to the employment counts.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # OECD snapshot (already committed; rerun to refresh)
python analysis.py           # results/*.csv and results/analysis_summary.txt
python figures.py            # figures/fig1_*.png ... fig5_*.png
python -m pytest -q tests
```

`results/analysis_summary.txt` holds every number used in the article. The workflow `.github/workflows/china-manufacturing-labour-costs.yml` reruns the analysis and the tests and checks that the summary is unchanged.

## Data (`fetch.py`, `data/`)

Three requests to the OECD Data Explorer SDMX API, made on 12 November 2026; `data/fetch_log.txt` records the URL, HTTP status, size and SHA-256 of each response, and the raw responses are kept gzipped in `data/raw/`. The tidy files keep the code columns only and about 47 economies plus the world and EU27 totals.

- `OECD.STI.PIE DSD_TIM_2025@DF_TIM_2025` (Trade in Employment, 2025 edition): persons engaged (`EMPN`, thousands), labour compensation (`LABR`, million US dollars and percent of value added) and value added per person engaged (`VALU_EMPN`, US dollars) for total manufacturing and 17 manufacturing industries, 1995–2022.
- `OECD.STI.PIE DSD_TIVA_MAINLV@DF_MAINLV` version 1.1 (Trade in Value Added, 2025 edition, served from the `sti-public` endpoint): value added in US dollars for manufacturing, textiles and apparel, computers and electronics, motor vehicles and the total economy, counterpart world.
- `OECD.SDD.NAD DSD_NAMAIN10@DF_TABLE4`: annual average exchange rates (`EXC_A`) and GDP purchasing power parities (`PPP_B1GQ`) for 19 economies from 1995.

`data/ifr_published.csv` holds robot densities, installations and stocks as published by the International Federation of Robotics in press releases and in the World Robotics 2025 foreword; `data/ifr_sources.csv` gives the title and URL of each source. The IFR's own databases are licensed and are not reproduced here; only the published headline figures are used, with the source of each value recorded.

`data/nbs_published.csv` holds average annual wages of urban units (non-private and private, all sectors and manufacturing, 2022–2025) as published by China's National Bureau of Statistics, with the release URL for each value. They extend the picture past 2022, where the OECD tables end, and are converted to dollars at the OECD's average exchange rate. They are wages, not compensation: they exclude employer social contributions and cover urban units only, so they are not on the same basis as the OECD series.

OECD data are reused under the OECD terms and conditions; they are not relicensed by the repository's licence.

## Definitions

- **Compensation per person engaged** = labour compensation (compensation of employees) ÷ persons engaged (employees and the self-employed), in US dollars at market exchange rates. Market rates, not PPPs, because the question is what a producer pays in the currency in which goods are sold.
- **Labour share** = compensation of employees ÷ value added. It is a unit labour cost that needs no exchange rate: the wage bill per unit of output value. The computed share matches the OECD's published `LABR` in percent of value added to within 0.0004 for every country and year.
- **China's relative pay** is China's compensation per person divided by the other economy's. The crossing year is the first year from which the ratio stays above 1 to 2022.
- **Yuan decomposition.** ln(USD pay) = ln(yuan pay) − ln(yuan per USD), so the change in log dollar pay over a period is the change in log yuan pay plus the change in the exchange rate (positive when the yuan appreciates). Pay at GDP PPP is reported for comparison.
- **Shift-share.** With industry employment shares s_i and value added per person v_i, the change in manufacturing value added per person between years a and b is Σ s_i,a Δv_i (within) + Σ Δs_i v_i,a (between) + Σ Δs_i Δv_i (interaction). The 17 industries sum exactly to total manufacturing employment in China.
- **World shares** are value added in US dollars divided by the TiVA world total. "Low-wage Asia" is Vietnam, Bangladesh, Cambodia, India, Indonesia, Pakistan and Myanmar.
- **Wage cut to reach China's labour share** = 1 − (China's 2022 labour share ÷ the country's 2022 labour share): the cut in compensation, with value added unchanged, that would bring the country's wage bill per unit of value added down to China's. Negative values mean the country's share is already lower than China's.
- **Years until equal** = ln(country pay ÷ China pay) ÷ (China's annual growth − the country's annual growth), with growth rates of 2015–2022 in US dollars. This is arithmetic on past rates, not a forecast.
- **Implied employees in the robot statistics** = operational stock ÷ robot density × 10,000.

## Caveats

- Compensation of employees does not pay the self-employed, while persons engaged include them. Where self-employment in manufacturing is large (India, Indonesia, Mexico, Vietnam) compensation per person engaged understates what an employee earns; the labour share understates labour's income for the same reason.
- Dollar values move with exchange rates. 2022 had a strong dollar and a weak euro, which lowers European pay relative to China's in that year.
- Trade in Employment ends in 2022. The NBS wages for 2024–2025 are reported next to the OECD series, never combined with it.
- The IFR's China robot density for 2024 was published as 567 and revised to 166 after the employment denominator was updated; the operational stock did not change. Densities before 2024 use the old denominator and are not comparable with the revised value; the IFR's stated 17% increase in 2024 implies about 142 for 2023 on the new basis.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
