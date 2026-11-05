# Exploratory 2026 citizenship comparison — not the original 64-country plot

This research stage tests a narrower claim with **observed international data**:

> Does Forbes billionaire density, attributed by **citizenship**, add descriptive
> explanatory power to $30/day poverty variation after accounting for GDP per
> capita in an independently matched sample?

It is **not a replication of the original residence-based scatterplot** and
not a causal test of whether billionaire wealth reduces poverty.

## Observed sources

- Forbes 2026 citizenship totals: **76-country secondary transcription** in
  [data/forbes_2026_citizenship_secondary.csv](data/forbes_2026_citizenship_secondary.csv).
  The count totals 3,359, below the reported 3,428 people on the Forbes list.
  Source table: [List of countries by number of billionaires](https://en.wikipedia.org/wiki/List_of_countries_by_number_of_billionaires).
  This transcription is **not verified against the complete original Forbes
  person list**; it is a deliberately incomplete observational proxy.
- World Bank PIP / OWID $30/day poverty (2021 PPP), source snapshot from
  GitHub Actions run [41820926349](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/41820926349).
- WDI GDP per capita PPP (constant 2021 international dollars), same run.
- WDI population, same run.

The successful source capture archived exact bytes with SHA-256 checksums.
The workflow artifact is **temporary** (expires December 4, 2026), so
publication requires an immutable long-term archive.

## First independently calculated comparison

The following results were calculated against the captured OWID/WDI inputs
and the separately labelled secondary citizenship transcription. The
executable workflow in this PR independently regenerates the results.

| Descriptive OLS model, HC3 covariance | Matched countries | R² |
| --- | ---: | ---: |
| log10 billionaire density | 54 | 0.5471 |
| log10 GDP per capita PPP | 54 | 0.8049 |
| Both predictors | 54 | 0.8113 |

Billionaire coefficient in the combined model: **−7.91 percentage points per
10× increase** (HC3 p≈0.119; 95% normal CI approximately [−17.87, 2.04]).
Incremental R² relative to GDP alone: approximately **0.00634**.

These are **in-sample associations**, conditional on the 54 selected and
matched countries. Do not interpret the p-value as the probability that the
economic narrative is correct. In particular, an insignificant coefficient
does not prove the true causal effect is zero.

Of the 76 country rows in the secondary transcription, **22 do not have
all required observations** in the selected 2020–2025 OWID/WDI window.
They are not imputed as zero. Survey reference years are often older than
the 2026 Forbes snapshot.

## Repeat from actual captured inputs

From the study directory, place the downloaded source artifact in
`data/source_snapshots/`, preserving the original manifest. Then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
python secondary_citizenship.py \
  --snapshots data/source_snapshots \
  --forbes data/forbes_2026_citizenship_secondary.csv \
  --output results/citizenship_exploratory
```

The run refuses existing nonempty output folders, validates the source
manifest, and saves a matched-panel CSV, unmatched-country ledger and
machine-readable regression results. It never calls the internet while
fitting the models.

A GitHub Actions workflow attached to this PR can capture and verify
new official-source responses, then run this model end to end. A **new
capture is a new data vintage**: its estimates may differ from those
listed above. Compare captured checksums and values before publishing.

## What would be required for the final study?

The central challenge remains: independently reconstruct the *residence*
counts used in the image and establish the exact 64-country selection;
retrieve PIP survey metadata; report income versus consumption series,
regional controls, exclusion rules and historical development paths;
and generate publication figures from verified data.

The current evidence says that GDP explains much more **cross-country
variation** in this exploratory model. It cannot establish that corporate
scaling or wealth concentration is beneficial or harmful.
