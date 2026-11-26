# Detailed Eurostat sectoral composition

This phase reuses four existing canonical Eurostat files from the two
[services-energy-time](../../data_sources/eurostat-services-energy-time.json)
and [services-relative-prices](../../data_sources/eurostat-services-relative-prices.json)
data contracts. These were acquired on 22 November 2026 and are pinned to the
original registry commits, not the latest API vintage. No new data download
from Eurostat or raw data duplication is necessary.

## Reproduce

From this directory, Python 3.12+ (standard library only):

```bash
python -m unittest discover -s tests -v
python eurostat_sectors.py --fetch
python eurostat_sectors.py
```

The separate Actions workflow verifies input bytes, runs tests, reproduces the
tables twice, and uploads the artifact `gdp-resilience-eurostat-sectors`.

## Tables

| Output | Content |
| --- | --- |
| eurostat_sections.csv | Nominal and real GVA, hours, shares of total GVA, real GVA per hour, flags and presence indicators by A10/A64 code |
| eurostat_concentration.csv | Coverage, reconciliation residual and concentration for both classifications |
| eurostat_groups.csv | Nominal value added and shares for twelve disjoint A64 analytical groups |
| eurostat_endpoint_changes.csv | Real GVA, hours and productivity growth; nominal share changes and relative volume/deflator decomposition |
| summary.json | Source commits, scope, complete partitions and table sizes |

Scope: EU27_2020 as published by Eurostat plus the 27 member countries,
1995–2025. Missing or flagged observations remain explicit. Endpoint comparisons
use 1995–2024, 2000–2024, 2010–2024, 2019–2020 and 2019–2024. Values are
reported only where their required inputs exist. 2024 is the main comparison
endpoint; the 2025 coverage table is retained rather than assumed complete.

## Accounting rules

- A10 is the primary complete partition: A, B-E, F, G-I, J, K, L, M_N, O-Q,
  R-U. C (manufacturing) is nested within B-E and is shown separately without
  entering the ten-sector HHI.
- The A64 snapshot contains sections A–T; U is not included. J61, Q86 and
  Q87_Q88 are nested diagnostic series, not independent sections. The twelve
  analytical groups partition the observed A–T sections, not the missing U.
- Shares use current-price total GVA, not GDP. HHI is emitted only when every
  partition member is present and nonnegative, the total is positive, and
  the absolute reconciliation residual is at most 0.5 percentage points.
  HHI is normalized over the reconciled sector sum; the residual is kept.
  A64 HHI is not directly comparable to A10 HHI because classification
  granularity differs. Incomplete groups are not summed.
- Current-price GVA can be added. Chain-linked 2020 volumes cannot be added
  across industries. Real growth and productivity use the original series
  for each section or published aggregate. Million euros divided by thousand
  hours is multiplied by 1,000 for euros per hour.
- The change in the logarithm of a sector's nominal GVA share is exactly the
  relative real-volume change plus the relative implicit-deflator change.
  Both contributions are in log points; nominal share changes are separately
  reported in percentage points. Nonpositive or missing endpoints invalidate
  logarithmic comparisons.
- This classification is descriptive. Information and communication, finance,
  professional/scientific/technical, and administrative/support activities are
  separate groups. The O–Q group is by activity, not public ownership. Real
  estate includes imputed rents. No category is a validated measure of
  technological substitutability, knowledge intensity, economic essentiality,
  or resilience. Those concepts require additional evidence.

Eurostat (European Commission), NACE Rev. 2 national accounts, reused under
the Commission reuse policy with source acknowledgement. Results are not
publication-ready until the workflow and evidence review pass.
