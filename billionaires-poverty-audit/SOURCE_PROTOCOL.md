# Data-source reconstruction protocol

## Scope

This source pipeline is a **data-ingestion and validation step** for
[A Billionaire Is Not a Poverty Policy](article.md). It does not reproduce
the empirical coefficients by itself. The 2026 Forbes aggregates, survey
vintages, and full input files must still be audited before the
article's provisional numbers can be accepted.

The pipeline intentionally separates external retrieval from offline
analysis and distinguishes **country of residence** from **citizenship**.

## Public sources and vintage

| File | Meaning | Source |
| --- | --- | --- |
| owid_30usd_2021_ppp.csv | Share below $30/day at 2021 PPP | [OWID live Grapher CSV API](https://ourworldindata.org/grapher/poverty-share-on-less-than-30-per-day.csv?v=1&csvType=full&useColumnShortNames=false) |
| wdi_gdp_ppp_constant.json | GDP per capita, constant international dollars | [World Bank API](https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.PP.KD?format=json&date=2020:2025&per_page=20000) |
| wdi_population.json | Population (total) | [World Bank API](https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&date=2020:2025&per_page=20000) |
| forbes_2026_country_counts.csv | Reviewed Forbes country totals | [Forbes 2026 ranking](https://www.forbes.com/billionaires/) |

The live OWID API is not an immutable snapshot; each successful run preserves exact downloaded bytes, retrieval date and SHA-256. The OWID source page also provides a dated archive of the indicator page, but the previous archived CSV URL returned HTTP 404 in GitHub Actions. The source snapshot date is not a guarantee of an underlying
household survey in the same year. The source file's Year column is
called poverty_year only for compatibility with the existing analysis.py;
it remains an **observation year** and requires PIP survey-date metadata
before matched-year interpretations can be published.

Likewise, latest-available GDP, population, and poverty years are selected
independently. Country-year alignment must be checked prior to substantive
inference. Neither WDI source is a permanently frozen vintage merely because
the API query includes a historical date range.

## Forbes input (manually reviewed)

The Forbes country data must be independently obtained and reviewed.
Only aggregate counts—not Forbes person records—are persisted.

Required CSV columns:

    iso3,country,billionaires,billionaire_method,forbes_year

- iso3: uppercase ISO-3166-1 alpha-3 country code, reviewed against a
  documented country/territory crosswalk.
- country: human-readable country name.
- billionaires: nonnegative integer count from the 2026 annual source.
- billionaire_method: either residence or citizenship.
- forbes_year: 2026.

Do **not** mix attribution methods in one input file. A country absent
from Forbes data is *missing*, not automatically a verified zero.
Countries with an explicitly verified zero may be recorded as zero, and
are retained in the panel, although the existing log-density regression
excludes zeros by construction.

Document each aggregate's upstream source and retrieval date and verify
that the sum and country definitions match the Forbes edition being
used. Record a separate citizenship and residence panel if both are to be
compared.

## Fetch (optional) and build (offline)

From the project directory:

    python source_pipeline.py fetch --output-dir data/source_snapshots

The fetch command downloads only the three public OWID/WDI files,
creates SOURCE_MANIFEST.json, records a SHA-256 digest of each source,
and refuses to overwrite an existing snapshot.

Prepare the reviewed Forbes aggregate CSV independently, then run:

    python source_pipeline.py build \
      --poverty-csv data/source_snapshots/owid_30usd_2021_ppp.csv \
      --gdp-json data/source_snapshots/wdi_gdp_ppp_constant.json \
      --population-json data/source_snapshots/wdi_population.json \
      --forbes-csv data/source_snapshots/forbes_2026_country_counts.csv \
      --output data/countries.csv \
      --manifest results/source_panel_manifest.json

This produces an ISO3-joined panel and a report including input hashes,
unmatched Forbes countries, and year ranges. It does not invent data
for unmatched countries. It does not download on the analysis path.

After reviewing the panel and survey-year metadata:

    python analysis.py --input data/countries.csv --output results/models.json

Finally, inspect the sample sizes, poverty line, country mapping,
original Forbes geography rule, and robustness specifications before
quoting model coefficients.

## Deterministic tests

Run:

    python -m unittest discover -s tests -p 'test_*.py' -v

The accompanying tests use only clearly synthetic country codes and
fixtures. They verify schema checks, latest-observation selection,
duplicate rejection, API pagination, zero handling, unmatched reporting,
deterministic panel hashes, and a valid offline end-to-end join.

No numeric result from synthetic tests is an empirical claim.

## Remaining release gates

1. Capture the source files and retrieval checksums in the repository
   or an immutable, publicly documented research archive.
2. Recover actual household survey reference years from World Bank PIP,
   along with income-versus-consumption survey classification.
3. Validate Forbes country/territory attribution and country totals.
4. Compare the reconstructed sample with the original 64 plotted
   countries; record unmatched and excluded observations.
5. Reproduce cross-country regressions and all robustness tables.
6. Build eight permanent PNG publication figures and SVG sources,
   with captions, alt text, and generation scripts.
7. Update the manuscript's numerical claims *only* from verified output,
   keeping all counterexamples on comparable poverty definitions.

This is a deliberately narrower deliverable than a complete published
empirical study: it makes the source and reproducibility process
inspectable without pretending the absent inputs have already been
recovered.
