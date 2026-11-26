# What happens to UK technology companies that raise equity, by founder nationality: Companies House, 2016–2026

Work in progress. Companion code for a Medium article that follows UK technology companies incorporated from July 2016, when persons with significant control began to be recorded at incorporation, and compares how often companies founded by British and by non-British founders are sold to foreign owners, sold to UK owners, dissolved or still independent.

Companies House is not reachable from every machine, so the data steps run on a GitHub runner. The first step, `ch_probe.py`, profiles the bulk files (aggregate counts only, no company-level rows or personal data) and, if a `COMPANIES_HOUSE_API_KEY` secret is set, checks the API endpoints the cohort will need. The workflow `.github/workflows/uk-founder-exits-probe.yml` runs it and uploads `ch_probe.json` as an artifact.

Data: Companies House, Free Company Data Product and People with Significant Control snapshot, Open Government Licence v3.0.
