# Refreshing from the private Google Sheet

The private Google Sheet is the authoritative raw source for this analysis. The public repository stores only privacy-safe aggregate outputs.

## Required GitHub Actions secrets

Configure these repository secrets outside the source tree:

- `BLOOD_PRESSURE_SHEET_URL`: the private Google Sheets URL.
- `GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON`: the complete Google service-account credential JSON.
- `BLOOD_PRESSURE_WORKSHEET`: optional worksheet name. If omitted, the first worksheet is used.

Share the private Google Sheet with the service account's `client_email` as a viewer. Do not make the Sheet public.

The workflow never prints the Sheet URL, spreadsheet ID, credentials, raw measurement rows, row-level pulse-oximeter values, or row-level date-temperature matches.

## Source schema

The blood-pressure source includes the established blood-pressure and context fields plus two optional measurements from the pulse oximeter:

- `spo2`: oxygen saturation from the pulse oximeter;
- `bpm_spo2`: pulse rate from that same device.

The existing `bpm` field remains the pulse rate from the blood-pressure monitor. `bpm` and `bpm_spo2` are kept as distinct device-specific measurements.

Historical rows may leave the pulse-oximeter fields blank. Missing pulse-oximeter values are not imputed.

## Refresh pipeline

Run the GitHub Actions workflow **Refresh blood-pressure analysis** manually.

It performs:

1. secret-backed authentication to Google Sheets with `gspread`;
2. source-schema validation, including the optional `spo2` and `bpm_spo2` fields;
3. the explicit legacy date repairs documented by the analysis;
4. removal of blank placeholder and spreadsheet-summary rows;
5. 15-minute measurement-session construction;
6. generation of `data/analysis_snapshot.csv` and `data/source_audit.json`;
7. generation of aggregate pulse-oximeter coverage and cross-device BPM diagnostics in `data/pulse_oximeter_diagnostics.json`;
8. generation of the date-free relative-day pulse-oximeter snapshot in `data/pulse_oximeter_daily_snapshot.csv`;
9. generation of coverage-aware longitudinal pulse-oximeter diagnostics in `data/pulse_oximeter_longitudinal_diagnostics.json`;
10. retrieval of hourly 2 m air temperature for Fiães from Open-Meteo and in-memory matching to the actual measurement times;
11. fitting of privacy-safe temperature-adjusted exploratory models without writing the date-temperature series;
12. regeneration of the remaining statistical outputs and figures;
13. a guard that fails if an `.xlsx` or `.xls` source file appears in the project tree;
14. creation of a pull request containing only privacy-safe generated outputs when they changed.

The pulse-oximeter longitudinal layer reports coverage before any time association. It uses the same relative `day_index` origin as the established blood-pressure snapshot and fits descriptive HC3 time associations only when at least four usable pulse-oximeter days are available. Otherwise the output records `estimable: false` rather than producing an unstable slope.

No clinical thresholds or causal interpretations are applied to the pulse-oximeter outputs.

## Local reproduction

For local use, expose the same environment variables and run the canonical package modules:

```bash
cd Blood_Pressure_Missingness
python -m pip install -e .

python -m blood_pressure_missingness.data_sources.google_sheets \
  --data-dir data

python -m blood_pressure_missingness.analyses.pulse_oximeter \
  --output data/pulse_oximeter_diagnostics.json \
  --daily-output data/pulse_oximeter_daily_snapshot.csv

python -m blood_pressure_missingness.analyses.pulse_oximeter_longitudinal \
  --pulse-data data/pulse_oximeter_daily_snapshot.csv \
  --bp-data data/analysis_snapshot.csv \
  --output data/pulse_oximeter_longitudinal_diagnostics.json

python -m blood_pressure_missingness.analyses.temperature \
  --output figures/temperature_covariate.json

python -m blood_pressure_missingness.public_analysis \
  --data data/analysis_snapshot.csv \
  --audit data/source_audit.json \
  --output-dir figures
```

The source-refresh module writes only the privacy-safe blood-pressure snapshot and aggregate source audit. The pulse-oximeter module reuses the private Sheet in memory and writes only aggregate diagnostics plus the relative-day daily snapshot. The longitudinal pulse-oximeter module consumes only those public aggregate inputs.

The temperature analysis also reuses the private Sheet in memory, joins it to Open-Meteo hourly temperature, and writes only aggregate model diagnostics. It does not persist calendar dates, row-level times, row-level matched temperatures, or a day-indexed temperature series.

The established blood-pressure analysis and the pulse-oximeter extension remain separate analysis surfaces. Adding `spo2` and `bpm_spo2` does not automatically change the published blood-pressure estimands or article conclusions.