# What happens to UK technology companies after they first issue new shares, by founder nationality: Companies House, 2016–2026

Companion code for a Medium article that tests a conjecture about British founders: that a domestic attitude to risk and ownership makes British-founded UK technology companies more likely to be sold, above all to foreign owners, or to stop raising capital, than comparable UK companies founded by people of other nationalities. The attitude is not measured; the comparison only looks for the footprint it predicts, in companies that share the same legal system, tax rules, investors and talent pool.

Results are not in yet. The analysis runs on a GitHub runner and commits `output.txt`, `figdata.json`, `build_report.json` and `figures/` back to this folder when it finishes.

## Design

**Cohort.** Private limited companies incorporated in England, Wales, Scotland or Northern Ireland from 1 July 2016 (when persons with significant control began to be recorded at incorporation) to 31 December 2020, with a technology SIC code: software publishing and programming (58210, 58290, 62011, 62012), data processing and web portals (63110, 63120), research and development (72110, 72190), pharmaceuticals (21100, 21200), medical instruments (32500) and electronics (26xxx). IT consultancy (62020, 62030, 62090) is left out because it is dominated by one-person contracting companies. Live and dissolved companies are both included, so the cohort is not conditioned on survival.

**Origin.** The first allotment of new shares (form SH01) within three years of incorporation, while the company is still controlled by individuals. An allotment is a public trace of new equity; it is not proof of outside investment, so a sensitivity analysis starts the clock instead at the first allotment with a shareholder resolution within 60 days, the usual paperwork of a priced round.

**Founders.** Individuals with significant control (more than 25% of shares or votes) notified within 30 days of incorporation. A company is British-founded if all such founders give a British, English, Scottish, Welsh or Northern Irish nationality (dual nationals with British included), non-British-founded if none does, and mixed otherwise. Companies controlled by another company at incorporation are left out.

**Outcomes**, whichever comes first after the origin:
- *sold to a foreign owner*: a company registered abroad takes majority control (more than half the shares or votes, or the right to appoint the board) under a different name, or a UK company does so and is itself controlled from abroad, followed up to three levels;
- *moved under a foreign parent of its own*: a company registered abroad with a name like the company's own takes control, usually the founders re-domiciling under a Delaware parent;
- *sold to a UK owner*: a UK company with a different name takes majority control and has no foreign controller above it;
- *moved under an offshore parent*: Jersey, Guernsey, the Isle of Man, the British Virgin Islands, Cayman, Bermuda and similar;
- *closed*: the first insolvency filing or dissolution.

A UK holding company with the company's own name is treated as a reorganisation: its own controllers are followed instead. Follow-up ends at the date of the PSC snapshot.

**Estimators** (`exits_core.py`): Aalen–Johansen cumulative incidence with bootstrap intervals, cause-specific Cox models (lifelines), a Fine–Gray subdistribution hazard model written out with inverse-probability-of-censoring weights and cumulative sums (checked against lifelines and against simulated data with known coefficients), inverse-probability weights that balance incorporation year, activity, region, number of founders and early allotments between British and non-British founders, and the share of companies in each state five years after the origin, including *independent, no further allotment*, the "stalled" outcome.

## Data

Companies House, Open Government Licence v3.0:
- Advanced company search API, one query per SIC code and calendar month, for every company incorporated in the window, live or dissolved.
- Filing history API for every company found: allotments (SH01), resolutions and insolvency filings.
- People with Significant Control snapshot (bulk JSON), for the founders' nationality and country of residence and for corporate controllers. Names of individuals and dates of birth are dropped as the file is read.

The API needs a free key from the Companies House developer hub, stored as the repository secret `COMPANIES_HOUSE_API_KEY`. The company-level table stays on the runner (kept 30 days as a workflow artifact); only aggregates are committed.

## Run

The workflow `.github/workflows/uk-founder-exits.yml` runs the tests, the search crawl, up to four rounds of filing-history crawling (about 1.9 requests a second, under the limit of 600 per five minutes), the PSC extraction and the analysis. The crawl database is carried between jobs in the Actions cache, so a run that stops early can be resumed by running the workflow again.

Locally, with a key in the environment:

    pip install -r requirements.txt
    python ch_api.py --db .cache/crawl.sqlite search
    python ch_api.py --db .cache/crawl.sqlite filings
    python ch_bulk.py --db .cache/crawl.sqlite --work-dir .cache/ch --out .cache/psc.json.gz
    python ch_build.py --db .cache/crawl.sqlite --psc .cache/psc.json.gz --out .cache/cohort.csv --report build_report.json
    python exits_numbers.py --cohort .cache/cohort.csv --report build_report.json --out output.txt --figdata figdata.json
    python figures.py --figdata figdata.json --out figures
    python -m unittest discover -s tests

`ch_probe.py` (manual workflow `uk-founder-exits-probe.yml`) profiles the bulk files with aggregate counts only.

## Limits

- Nationality is self-declared on the register and describes the founders, not the company's culture, its investors or where its customers are.
- Only founders holding more than 25% are seen; a founder below that threshold at incorporation is invisible.
- The date of a change of control is the date Companies House was notified, which can lag the transaction.
- Name similarity separates a founder's own foreign parent from a buyer, and it can misclassify generic names.
- An SH01 records new shares, not who paid for them or how much.

## Licence

Code: Apache-2.0. Data: Companies House, Open Government Licence v3.0.
