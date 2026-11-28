# Insider selling around buyback announcements, from EDGAR

Companion to `stock-buybacks-ceo-wealth`. That article could not settle whether insiders sell after
buyback announcements: an SEC staff sample says yes, and a 2025 study says the corporate calendar
explains it. This folder tests that disagreement on open data.

## Design

- **Events** come from each firm's 8-K filings. An 8-K with item 8.01, 7.01 or 2.02 whose text pairs
  buyback wording with authorization and board wording is an *announcement*. If it is filed with
  item 2.02 (earnings) it is kept apart as *announcement_with_earnings*.
- **Placebos** are 2.02 earnings 8-Ks with no authorization wording. Insiders trade in open windows
  after earnings, so the placebo carries the calendar lift.
- **Outcomes** (Form 4, open-market sales, code S): number of selling insiders, whether the CEO sold,
  and log shares sold, in the 10 trading days before versus from the event day to 10 days after.
- **Estimate**: firm-level mean change at announcements minus the firm-level mean change at
  placebo dates; 95% interval from a firm bootstrap (2,000 draws, seed 7). Firms need both kinds.

## Run

Needs a User-Agent that identifies you, as SEC fair access requires.

```
pip install -r requirements.txt
python -m pytest -q tests
EDGAR_USER_AGENT="Name contact@email" python3 -I src/run.py --tickers config/tickers_example.txt --out out
```

On GitHub: set the repository variable `EDGAR_USER_AGENT`, then run the workflow
`stock-buybacks-insider-sales` (Actions, workflow_dispatch). Results are uploaded as an artifact;
nothing is committed. The EDGAR cache is kept by `actions/cache`.

## First live run (10 tickers)

The first run read only the 8-K cover page and found 11 announcements in 650 events: the press-release
exhibits were missing. Even so, the placebo earnings dates show the calendar effect clearly: 0.12 insiders
sold in the 10 trading days before an earnings filing and 1.94 in the 10 days from it. A naive
before-and-after test around any earnings-linked announcement would read that as an announcement effect.

## Status and limits

- `out/announcement_audit_sample.csv` lists up to 50 detected announcements with the matched text. Check it by hand.
- The offline tests pass (Form 4 parsing, event classification, windows, the difference-in-differences
  on synthetic data, and an end-to-end run against a fake EDGAR). **No live EDGAR run has been done**:
  the authoring environment could not reach sec.gov. Check the first live output before trusting it.
- Windows use weekdays and ignore exchange holidays.
- Plan sales: the `aff10b5One` flag exists only in filings from 2023, so earlier 10b5-1 sales are not
  separated. Do the main analysis from 2023 on as well as the full sample.
- Announcements are detected by text matching, so some are missed or wrongly flagged. Hand-check a
  sample of 50 before reporting.
- `config/tickers_example.txt` has ten tickers. Replace it with S&P 500 constituents at a dated snapshot.
- Insiders are the reporting owners of the issuer's own Form 4s: officers and directors, plus 10% holders.
