import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import analysis
import events
import form4
import windows


def test_form4_parse_keeps_only_open_market_sales():
    rows = form4.parse_form4(open(os.path.join(HERE, "fixtures", "form4_sale.xml"), "rb").read())
    assert len(rows) == 2
    s = form4.open_market_sales(rows)
    assert len(s) == 1 and s[0]["shares"] == 1000 and s[0]["ceo"] and not s[0]["plan_10b5"]


def test_raw_xml_name_strips_xsl_prefix():
    assert form4.raw_xml_name("xslF345X05/wk-form4_1.xml") == "wk-form4_1.xml"


def test_classify():
    bb = "<p>The Board of Directors authorized a new share repurchase program of up to $10 billion.</p>"
    assert events.classify("8.01,9.01", bb) == "announcement"
    assert events.classify("2.02,8.01", bb) == "announcement_with_earnings"
    assert events.classify("2.02,9.01", "Quarterly results were strong.") == "earnings_placebo"
    assert events.classify("8.01", "The company named a new director.") is None
    # earnings release that merely mentions repurchases already done is not an authorization
    assert events.classify("2.02", "We repurchased shares during the quarter.") == "earnings_placebo"


def test_window_stats_counts_pre_and_post():
    sales = pd.DataFrame({
        "tx_date": pd.to_datetime(["2024-03-04", "2024-03-12", "2024-03-13"]),
        "shares": [100.0, 200.0, 300.0], "owner_cik": ["a", "b", "c"], "ceo": [False, True, False]})
    st = windows.window_stats(sales, "2024-03-11")
    assert st["sellers_pre"] == 1 and st["sellers_post"] == 2
    assert st["shares_post"] == 500.0 and st["ceo_sells_post"] == 1 and st["ceo_sells_pre"] == 0


def _synth(effect, n_firms=80, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_firms):
        base = rng.poisson(2)
        # both kinds share the same calendar lift (+1 after the event), announcements add `effect`
        for kind, add in (("announcement", effect), ("earnings_placebo", 0)):
            for _ in range(4):
                rows.append({"cik": c, "kind": kind, "sellers_pre": rng.poisson(base),
                             "sellers_post": rng.poisson(base + 1 + add), "shares_pre": 0.0,
                             "shares_post": 0.0, "ceo_sells_pre": 0, "ceo_sells_post": 0})
    return pd.DataFrame(rows)


def test_did_removes_calendar_lift_and_detects_effect():
    null = analysis.did(_synth(0.0))
    assert null["ci"][0] < 0 < null["ci"][1]            # calendar lift alone is not an effect
    assert null["announcement_mean_change"] > 0.5       # raw change is positive, DiD is not
    eff = analysis.did(_synth(1.5))
    assert eff["ci"][0] > 0 and 1.0 < eff["estimate"] < 2.0


def test_did_needs_enough_firms():
    ev = _synth(1.0, n_firms=3)
    assert analysis.did(ev)["estimate"] is None


class FakeEdgar:
    """Serves one firm: a Form 4 sale five days after a buyback 8-K, plus an earnings 8-K."""
    def __init__(self):
        self.f4 = open(os.path.join(HERE, "fixtures", "form4_sale.xml"), "rb").read()

    def submissions(self, cik):
        return [
            {"accession": "0000000001-24-000001", "filing_date": "2024-02-01", "form": "8-K",
             "primary_doc": "bb.htm", "items": "8.01,9.01"},
            {"accession": "0000000001-24-000002", "filing_date": "2024-05-02", "form": "8-K",
             "primary_doc": "er.htm", "items": "2.02,9.01"},
            {"accession": "0000000001-24-000003", "filing_date": "2024-02-06", "form": "4",
             "primary_doc": "xslF345X05/f4.xml", "items": ""},
        ]

    def get(self, url):
        if url.endswith("bb.htm"):
            return b"<p>The Board of Directors authorized a share repurchase program.</p>"
        if url.endswith("er.htm"):
            return b"<p>Results for the quarter.</p>"
        if url.endswith("f4.xml"):
            return self.f4
        return None


def test_build_events_end_to_end_offline():
    import run
    rows = run.build_events(FakeEdgar(), "TEST", 320193, "2024-01-01")
    kinds = {r["kind"]: r for r in rows}
    assert set(kinds) == {"announcement", "earnings_placebo"}
    assert kinds["announcement"]["sellers_post"] == 1 and kinds["announcement"]["sellers_pre"] == 0
    assert kinds["earnings_placebo"]["sellers_post"] == 0
