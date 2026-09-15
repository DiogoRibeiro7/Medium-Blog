# Refreshing from the private Google Sheet

The private Google Sheet is the authoritative raw source for this analysis. The public repository stores only privacy-safe aggregate outputs.

## Required GitHub Actions secrets

Configure these repository secrets outside the source tree:

- `BLOOD_PRESSURE_SHEET_URL`: the private Google Sheets URL.
- `GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON`: the complete Google service-account credential JSON.
- `BLOOD_PRESSURE_WORKSHEET`: optional worksheet name. If omitted, the workflow resolves the unique worksheet containing the complete expected source schema.

Share the private Google Sheet with the service account's `client_email` as a viewer. Do not make the Sheet public.

The workflow never prints the Sheet URL, spreadsheet ID, credentials, raw measurement rows, row-level pulse-oximeter values, or row-level date-temperature matches.

## Source schema

The blood-pressure source includes the established blood-pressure and context fields plus two optional measurements from the pulse oximeter:

- `SpO2`: oxygen saturation from the pulse oximeter in the private Sheet; it is mapped internally to the lowercase `spo2` field used by the analysis code and public aggregate names;
- `bpm_spo2`: pulse rate from that same device.

The existing `bpm` field remains the pulse rate from the blood-pressure monitor. `bpm` and `bpm_spo2` are kept as distinct device-specific measurements.

Historical rows may leave the pulse-oximeter fields blank. Missing pulse-oximeter values are not imputed.

## Refresh pipeline

Run the GitHub Actions workflow **Refresh blood-pressure analysis** manually, or change the reviewed `.github/refresh-blood-pressure.request` token to request a refresh on merge.

It performs:

1. secret-backed authentication to Google Sheets with `gspread`;
2. worksheet resolution and source-schema validation, including the optional `SpO2` and `bpm_spo2` fields;
3. the explicit legacy date repairs documented by the analysis;
4. removal of blank placeholder and spreadsheet-summary rows;
5. 15-minute measurement-session construction;
6. generation of `data/analysis_snapshot.csv` and `data/source_audit.json`;
7. generation of aggregate pulse-oximeter coverage and cross-device BPM diagnostics in `data/pulse_oximeter_diagnostics.json`;
8. generation of the date-free relative-day pulse-oximeter snapshot in `data/pulse_oximeter_daily_snapshot.csv`;
9. generation of coverage-aware longitudinal pulse-oximeter diagnostics in `data/pulse_oximeter_longitudinal_diagnostics.json`;
10. generation of observed-window pulse coverage in `data/pulse_oximeter_observation_window.json`, using the first observed pulse day through the end of the public BP calendar;
11. retrieval of hourly 2 m air temperature for Fiães from Open-Meteo and in-memory matching to the actual measurement times;
12. fitting of privacy-safe temperature-adjusted exploratory models without writing the date-temperature series;
13. regeneration of the remaining statistical outputs and figures;
14. a guard that fails if an `.xlsx` or `.xls` source file appears in the project tree;
15. creation of a pull request containing only privacy-safe generated outputs when they changed.

The pulse-oximeter longitudinal layer reports coverage before any time association. It uses the same relative `day_index` origin as the established blood-pressure snapshot and fits descriptive HC3 time associations only when at least four usable pulse-oximeter days are available. Otherwise the output records `estimable: false` rather than producing an unstable slope.

The observation-window layer answers a different question. Full-history pulse coverage uses all BP rows or calendar days as denominators, including historical periods before any pulse value is observed. The observation-window artifact instead begins at the first observed pulse day and reports day-level and reading-level capture from that point forward. This boundary is empirical: it does **not** identify the date when the device was introduced, and rows before it are not reclassified as missing pulse measurements.

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

python -m blood_pressure_missingness.analyses.pulse_oximeter_observation_window \
  --pulse-data data/pulse_oximeter_daily_snapshot.csv \
  --bp-data data/analysis_snapshot.csv \
  --output data/pulse_oximeter_observation_window.json

python -m blood_pressure_missingness.analyses.temperature \
  --output figures/temperature_covariate.json

python -m blood_pressure_missingness.public_analysis \
  --data data/analysis_snapshot.csv \
  --audit data/source_audit.json \
  --output-dir figures
```

The source-refresh module writes only the privacy-safe blood-pressure snapshot and aggregate source audit. The pulse-oximeter module reuses the private Sheet in memory and writes only aggregate diagnostics plus the relative-day daily snapshot. The longitudinal and observation-window pulse modules consume only public aggregate inputs.

The temperature analysis also reuses the private Sheet in memory, joins it to Open-Meteo hourly temperature, and writes only aggregate model diagnostics. It does not persist calendar dates, row-level times, row-level matched temperatures, or a day-indexed temperature series.

The established blood-pressure analysis and the pulse-oximeter extension remain separate analysis surfaces. Adding `SpO2` and `bpm_spo2` to the private source does not automatically change the published blood-pressure estimands or article conclusions.