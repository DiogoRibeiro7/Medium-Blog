"""Build the event table from EDGAR and run the analysis.

python3 -I src/run.py --tickers config/tickers_example.txt --out out --since 2010-01-01
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import analysis
import events
import form4
import windows
from edgar import Edgar


def build_events(ed, ticker, cik, since):
    subs = ed.submissions(cik)
    f4 = [s for s in subs if s["form"] in ("4",) and s["filing_date"] >= since]
    sales = []
    for s in f4:
        raw = ed.get(Edgar.doc_url(cik, s["accession"], form4.raw_xml_name(s["primary_doc"])))
        if not raw:
            continue
        try:
            sales += form4.open_market_sales(form4.parse_form4(raw))
        except Exception:
            continue
    n_f4 = len(f4)
    sales = pd.DataFrame(sales)
    print(f"  {ticker}: {n_f4} Form 4 filings, {len(sales)} open-market sales rows", file=sys.stderr, flush=True)
    if sales.empty:  # no insider sales at all: events still count, with zeros
        sales = pd.DataFrame({"tx_date": pd.to_datetime([]), "shares": [], "owner_cik": [], "ceo": [], "plan_10b5": []})
    sales["tx_date"] = pd.to_datetime(sales.tx_date)
    # keep only issuer's own insiders, exclude 10b5-1 plan sales from the main outcome
    sales = sales[~sales.plan_10b5]
    out = []
    for s in subs:
        if s["form"] != "8-K" or s["filing_date"] < since:
            continue
        its = {i.strip() for i in s["items"].split(",")}
        if not its & {"2.02", "8.01", "7.01"}:
            continue
        raw = ed.full_submission(cik, s["accession"])
        if not raw:
            continue
        kind, snippet = events.classify(s["items"], raw.decode("utf-8", "ignore"))
        if kind is None:
            continue
        stats = windows.window_stats(sales, s["filing_date"])
        out.append({"ticker": ticker, "cik": cik, "date": s["filing_date"], "kind": kind,
                    "accession": s["accession"], "snippet": snippet[:400], **stats})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--since", default="2010-01-01")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    ed = Edgar(a.cache)
    tmap = ed.ticker_map()
    tickers = [t.strip().upper() for t in open(a.tickers) if t.strip() and not t.startswith("#")]
    rows = []
    for t in tickers:
        if t not in tmap:
            print("no CIK for", t, file=sys.stderr)
            continue
        try:
            rows += build_events(ed, t, tmap[t], a.since)
        except Exception as e:  # one bad firm must not kill the run
            print(f"FAILED {t}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        print(t, len(rows), "events so far", file=sys.stderr, flush=True)
    os.makedirs(a.out, exist_ok=True)
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(a.out, "events.csv"), index=False)
    if len(ev):
        aud = ev[ev.kind.str.startswith("announcement")]
        aud = aud.sample(min(50, len(aud)), random_state=7) if len(aud) else aud
        aud[["ticker", "date", "kind", "accession", "snippet"]].to_csv(
            os.path.join(a.out, "announcement_audit_sample.csv"), index=False)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(analysis.summary(ev) if len(ev) else {}, f, indent=2)
    print("done", len(ev), "events")


if __name__ == "__main__":
    main()
