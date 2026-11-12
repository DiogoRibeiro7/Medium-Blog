# Where North-West Europe's lower operating surplus per hour sits: an industry decomposition against the US, 2019 and 2021

Computational companion to a follow-up of the Medium series *Europe's Low-Wage Path, Measured*. Part 1 of the series found that an hour of work in eight North-West European countries produces slightly less value added than an hour in the United States, pays labour slightly more and leaves about half the net operating surplus. It stated a conjecture about where that missing surplus sits: in information and communication, finance and professional services, with manufacturing contributing little or the opposite. This repository tests it.

The method splits the gap in net operating surplus per hour of the whole economy into the contributions of the 19 ISIC Rev. 4 sections, splits each contribution into an industry-structure part and a within-industry margin part, repeats the exercise under four accountings (net or gross of depreciation, with or without an imputed wage for the self-employed), computes currency-free profit shares by industry, opens up information and communication, finance, professional services, manufacturing and real estate one level down, and repeats the industry split for each of the eight countries.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # OECD snapshot (already committed; rerun to refresh)
python analysis.py           # results/*.csv and results/analysis_summary.txt
python figures.py            # figures/fig1_*.png ... fig4_*.png
python -m pytest -q tests
```

`results/analysis_summary.txt` holds every number used in the article. The workflow `.github/workflows/eu-us-surplus-by-industry.yml` reruns the analysis and the tests and checks that the summary is unchanged.

## Data (`fetch.py`, `data/`)

Two requests to the OECD Data Explorer SDMX API, made on 12 November 2026; `data/fetch_log.txt` records the URL, HTTP status, size and SHA-256 of each response, and the raw responses are kept gzipped in `data/raw/`.

- `OECD.STI.PIE DSD_STAN@DF_STAN_2025` (STAN Database for Structural Analysis, 2025 edition), current prices, national currency, from 2015, for the United States, Austria, Belgium, Denmark, Finland, France, Germany, the Netherlands and Sweden, by ISIC Rev. 4 industry:
  - value added (`B1G`), compensation of employees (`D1`), gross and net operating surplus and mixed income (`B2A3G`, `B2A3N`), consumption of fixed capital (`P51C`), other taxes less subsidies on production (`D29X39`);
  - total employment, employees and the self-employed (`EMP`, `SAL`, `SELF`) in persons and in hours worked.
- `OECD.SDD.NAD DSD_NAMAIN10@DF_TABLE4`: GDP purchasing power parities (`PPP_B1GQ`) and average exchange rates (`EXC_A`).

STAN 2025 has US income components by industry up to 2021; for 2022 onwards it has them for the total economy only. 2021 is therefore the main year and 2019, the last year before the pandemic, is the check.

OECD data are reused under the OECD terms and conditions; they are not relicensed by the repository's licence.

## Definitions

- **North-West Europe (NW)** is the sum of the eight countries of Part 1, each converted to US dollars at its GDP PPP, divided by the sum of their hours worked.
- **Value added at factor cost (FCVA)** = compensation of employees + gross operating surplus and mixed income. It leaves out other taxes less subsidies on production, which differ in size and treatment between countries.
- **Imputed labour income of the self-employed** = compensation per employee hour in the industry × hours of the self-employed. Hours of the self-employed are total hours minus employee hours for the European countries; for the US, where STAN has persons but not hours for the self-employed, they are self-employed persons × employee hours per employee.
- **Labour income (LC)** = compensation of employees + the imputation. **Net remainder (NR)** = net operating surplus and mixed income − the imputation: what is left for capital after depreciation and after paying the self-employed as if they were employees of their own industry. Where net surplus is not published it is gross surplus − depreciation; where both are published, the two differ by at most 0.04% of gross surplus. Households as employers (section T) have no operating surplus in the data and are given zero.
- **Contribution of industry i to the gap** = NR_i(NW) ÷ total hours(NW) − NR_i(US) ÷ total hours(US), in US dollars per hour worked in the whole economy. The 19 contributions add up to the gap.
- **Structure and margin.** With a_i the industry's hours as a share of total hours and r_i its net remainder per hour worked in the industry, the contribution a_i(NW) r_i(NW) − a_i(US) r_i(US) = (a_NW − a_US)(r_NW + r_US)/2 [structure] + (r_NW − r_US)(a_NW + a_US)/2 [margin]. The split is exact.
- **Profit share** = NR ÷ FCVA, within an industry or for the economy. It needs no exchange rate. The gap in the aggregate share is split into within-industry and between-industry parts with average value-added weights and average shares.
- **Four accountings.** The baseline is net of depreciation and with the imputation, as in Part 1. The other three leave depreciation in (gross operating surplus and mixed income − the imputation), leave mixed income whole (net operating surplus and mixed income, no imputation), or both. Moving from one to another changes each industry's contribution by exactly its gap in depreciation per hour or in imputed income per hour; the tests check this.
- **Sub-industries** (publishing and broadcasting, telecoms, IT services; banks, insurance, auxiliary finance; legal-accounting-consulting, R&D, advertising; pharmaceuticals, electronics, motor vehicles) are shown as net operating surplus and mixed income per hour of the whole economy, *without* the self-employment imputation, because US hours for the finance sub-industries are not usable in STAN 2025.
- **Owner-occupied housing** is the imputed rent that owner-occupiers pay themselves (`L68A`). It is compared on a gross basis, because depreciation is not published for it separately. Germany publishes value added but not gross surplus for `L68A`; its value added is used, which overstates the surplus by the small compensation and taxes in the industry.

## Results files

- `contributions_2019.csv`, `contributions_2021.csv`: per section, the contribution, its share of the gap, the structure and margin parts, hours shares, net remainder per hour within the industry, profit shares and value-added shares, NW and US.
- `variants_2019.csv`, `variants_2021.csv`: the contribution of each section under the four accountings.
- `owner_occupied_housing_2021.csv`, `subindustries_2021.csv`, `countries_2021.csv`: the inputs to figures 3 and 4.

## Caveats

- GDP PPPs convert the whole economy at one rate. An industry whose prices are relatively higher in one economy (finance, telecoms, housing) gets a larger share of value added there at the same volume of output. The profit shares in `contributions_*.csv` avoid the conversion and are the check on that.
- The imputation assumes a self-employed hour earns what an employee hour earns in the same industry. It is large: the self-employed work 14.7% of hours in the North-West against 6.4% in the US (2021), and in several European countries they record far more hours per person than employees (France 2,380 against 1,359). Where the self-employed earn less per hour than employees, it moves too much income from the remainder into labour and can make the remainder negative, as in professional services. Without it the North-West's gap falls from 8.89 to 4.68 dollars per hour, which is why every result is also shown unimputed.
- In the US national accounts, owners of incorporated businesses are employees of those businesses: their salaries are compensation and the rest is corporate profit. Only unincorporated owners (sole proprietors and partners) earn mixed income. The imputation therefore covers a narrower group in the US than in countries where professionals work unincorporated.
- The US self-employed hours assume the self-employed work as many hours per person as employees. At the total-economy level, where STAN has both, they worked 1,741 hours per person in 2021 against 1,676 for employees, so the US imputation is about 4% too small.
- Section T has a small negative remainder in the North-West sum (−0.03 dollars per hour in 2021), from the imputation for self-employed domestic workers in countries that record them; there is no T in the US data.
- Imputed rent on owner-occupied housing is a national-accounts construction, not a cash profit. It is part of operating surplus in both economies and is shown separately so that it can be left out.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
