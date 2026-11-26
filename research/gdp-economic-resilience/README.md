# GDP economic resilience: World Bank baseline

This is the first empirical implementation of the [research protocol](RESEARCH_PROTOCOL.md).
The data repository has already promoted the two reviewed snapshots in [data PR #291](https://github.com/DiogoRibeiro7/data/pull/291).
No raw data is duplicated here.

## Reproduce

Python 3.12+; standard library only. From this directory:

```bash
python -m unittest discover -s tests -v
python pipeline.py --fetch
python pipeline.py --output results
```

The first run downloads only exact GitHub raw URLs pinned to registry commit
`ce3cc0995411c124534408e2086772275ee2c7c8`. Every subsequent run validates size
and SHA-256 again. A mismatched cache is rejected, not silently refreshed.
Remove the offending local copy explicitly before fetching it again.

Data contracts: [WDI](../../data_sources/world-bank-wdi-gdp-resilience-2026-11-26.json)
and [CWON](../../data_sources/world-bank-cwon-2024-gdp-resilience-2026-11-26.json).
World Bank CC BY 4.0 dataset terms and the original provider attribution apply.
Consult the registry metadata for retrieval dates, units and methodological caveats.

## Outputs

- `gdp_rankings_2025.csv`: nominal and PPP GDP, population and per capita values for
  33 countries and separately the EU27 plus six non-EU comparators. Ranks cover only
  the defined sample, not the world. EU aggregates require all 27 constituent observations.
- `sector_composition.csv`: all 1,023 WDI country-year rows, even when observations are missing.
  Manufacturing is a subset of industry. Three-sector HHI uses agriculture, industry and
  services; four-sector HHI splits industry into manufacturing and nonmanufacturing.
  Normalize only complete, nonnegative sector vectors by their observed sum; record the
  unallocated GDP residual rather than interpreting it as another sector.
  These coarse HHI measures are proxies for composition, not input-output or export concentration.
- `gdp_wealth_endpoint_growth.csv`: country-level real GDP per capita versus real comprehensive
  wealth per capita percentage changes. Primary 1996–2020; alternatives 1995–2020,
  2000–2020 and 1996–2019. No summation of chained wealth components or cross-country
  chained levels. Cyprus remains excluded where wealth endpoints are absent. Flag Bulgaria's
  anomalous 1996 wealth baseline explicitly; report sensitivity excluding it.
- `coverage.csv`: indicator-year counts of observed, explicit-null and absent cells.
- `summary.json`: compact counts, primary comparison and reproducibility metadata.

The CI workflow reruns the analysis offline after the first verified fetch, compares
the output bytes, checks the reviewed observation counts, and exposes the result
tables as an Actions artifact. Outputs are not claimed as verified publication findings
until CI succeeds.

## Boundaries

Nominal USD and current international-dollar PPP ranks measure different things.
PPP is not foreign-exchange purchasing power. GDP is a flow; comprehensive wealth
is a model-based stock estimate. Compare growth rates within countries, not dollar
levels across different price bases. A GDP/wealth divergence is descriptive and not
proof of a particular causal mechanism.

The [detailed Eurostat sectoral decomposition](EUROSTAT.md) is available as a
separate reproducible analysis, using the existing registry snapshots. The
[historical shock analysis](SHOCKS.md) adds transparent trend benchmarks and
recovery measures. Later changes add sector-specific simulations, figures and
the manuscript evidence audit. IMF and OECD source access and licensing remain
separate; no unverified source is assumed in this baseline.
