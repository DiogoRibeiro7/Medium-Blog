# Refreshing from the private Google Sheet

The private Google Sheet is the authoritative raw source for this analysis. The public repository stores only privacy-safe aggregate outputs.

## Required GitHub Actions secrets

Configure these repository secrets outside the source tree:

- `BLOOD_PRESSURE_SHEET_URL`: the private Google Sheets URL.
- `GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON`: the complete Google service-account credential JSON.
- `BLOOD_PRESSURE_WORKSHEET`: optional worksheet name. If omitted, the first worksheet is used.

Share the private Google Sheet with the service account's `client_email` as a viewer. Do not make the Sheet public.

The workflow never prints the Sheet URL, spreadsheet ID, credentials, or raw measurement rows.

## Refresh pipeline

Run the GitHub Actions workflow **Refresh blood-pressure analysis** manually.

It performs:

1. secret-backed authentication to Google Sheets with `gspread`;
2. source-schema validation;
3. the explicit legacy date repairs documented by the analysis;
4. removal of blank placeholder and spreadsheet-summary rows;
5. 15-minute measurement-session construction;
6. generation of the date-free daily aggregate snapshot and aggregate audit metadata;
7. regeneration of the statistical outputs and figures;
8. a guard that fails if an `.xlsx` or `.xls` source file appears in the project tree;
9. creation of a pull request containing only privacy-safe generated outputs when they changed.

For local use, expose the same environment variables and run:

```bash
cd Blood_Pressure_Missingness
python -m pip install -r requirements.txt
python refresh_from_google_sheets.py --data-dir data
python analysis.py \
  --data data/analysis_snapshot.csv \
  --audit data/source_audit.json \
  --output-dir figures
```

Raw rows are processed in memory by `refresh_from_google_sheets.py`; the script writes only `data/analysis_snapshot.csv` and `data/source_audit.json`.
