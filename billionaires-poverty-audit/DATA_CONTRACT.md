# Input contract

The research program accepts **an existing local CSV** at `data/countries.csv`. It does not include any invented country observations.

| Field | Type | Meaning |
| --- | --- | --- |
| `country` | non-empty text | Distinct country label |
| `iso3` | text, three uppercase letters | Country identifier |
| `billionaires` | integer ≥ 0 | Forbes count under one documented attribution method |
| `population` | integer > 0 | Population denominator and reference year from a consistent source |
| `poverty_pct` | real in [0, 100] | World Bank share below $30/day, 2021 PPP |
| `gdp_ppp_per_capita` | real > 0 | WDI constant PPP GDP per capita |
| `poverty_year` | integer | Survey reference year |
| `gdp_year` | integer | GDP reference year |
| `billionaire_method` | `residence` or `citizenship` | Do not mix attribution methods in one model |

`analysis.py` explicitly refuses mixed citizenship/residence records and duplicate ISO codes. It models only rows with positive billionaire density, which mirrors the domain of the original log plot; zero-count records remain separately reported for audit. A second specification allowing zeros must be pre-specified, not substituted into this one.

For genuine cross-country estimates, archive source URLs, retrieval dates, country mappings, SHA-256 file hashes, original downloadable files where redistribution is permitted, and resulting joined CSV. The input sample MUST be reconstructed and checked before quoting regression outputs.
