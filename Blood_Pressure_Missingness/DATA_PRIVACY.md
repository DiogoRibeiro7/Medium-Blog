# Data privacy

The original workbook and Google Sheet are personal health records and are intentionally not stored in this public repository.

The private Google Sheet is the authoritative raw source. GitHub Actions accesses it only through secret-backed environment variables and processes the raw rows in memory. The sheet URL, spreadsheet ID, service-account credentials, row-level timestamps, free-text symptoms, meal labels, and other contextual fields are never committed.

The committed public snapshot removes calendar dates and row-level health context. It retains only day-indexed aggregate counts and daily means needed to reproduce the statistical arguments in the article.

Pulse-oximeter measurements follow the same boundary. Row-level `spo2` and `bpm_spo2` values remain private. The public repository may contain aggregate coverage and descriptive summaries, joint-field completeness counts, aggregate cross-device BPM-difference diagnostics, and a separate relative-day pulse-oximeter snapshot containing only daily counts and means. That snapshot uses the same date-free `day_index` origin as the blood-pressure snapshot and contains no calendar dates, timestamps, contextual fields, row-level measurements, or clinical classifications.

The ambient-temperature covariate follows the same boundary. Measurement timestamps are matched to hourly temperature for Fiães only in memory. The repository does not persist row-level matched temperatures or a day-indexed temperature sequence, because a detailed local weather sequence could act as a quasi-identifier for the hidden calendar dates. Only non-temporal aggregate diagnostics and fitted covariate coefficients are committed.

Do not add the source workbook, Google Sheet URL, service-account JSON, row-level pulse-oximeter measurements, row-level weather joins, or other row-level health exports to this directory.
