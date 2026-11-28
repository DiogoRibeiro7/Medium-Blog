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
    assert events.classify("8.01,9.01", bb)[0] == "announcement"
    assert events.classify("2.02,8.01", bb)[0] == "announcement_with_earnings"
    assert "$10 billion" in events.classify("8.01", bb)[1]
    assert events.classify("2.02,9.01", "Quarterly results were strong.")[0] == "earnings_placebo"
    assert events.classify("8.01", "The company named a new director.")[0] is None
    # reporting past purchases or remaining capacity is not a new authorization
    past = ("The company repurchased $5 billion of shares in the quarter and has $20 billion of "
            "remaining authorization from its Board of Directors under the share repurchase program.")
    assert events.classify("2.02", past)[0] == "earnings_placebo"
    # exhibit text inside a full-submission file is stripped of markup before matching
    sub = "<DOCUMENT><TYPE>EX-99.1<TEXT><html><b>The Board of Directors</b> has authorized an additional $110 billion for share repurchases.</html>"
    assert events.classify("2.02,9.01", sub)[0] == "announcement_with_earnings"


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

    def full_submission(self, cik, accession):
        return self.get(accession + "/bb.htm" if accession.endswith("1") else accession + "/er.htm")

    def get(self, url):
        if url.endswith("bb.htm"):
            return b"<p>The Board of Directors authorized a new share repurchase program of up to $10 billion.</p>"
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


def test_edgar_retries_transient_5xx(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request
    import edgar
    calls = {"n": 0}

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"ok"

    def fake(req, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 502, "Bad Gateway", {}, None)
        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    ed = edgar.Edgar(str(tmp_path), user_agent="Test test@example.org")
    assert ed.get("https://example.org/x") == b"ok" and calls["n"] == 3


def test_combine_concatenates_shards_and_dedupes(tmp_path):
    import combine
    for i in range(2):
        d = tmp_path / f"s{i}"
        d.mkdir()
        _synth(0.5, n_firms=10, seed=i).assign(ticker="T", date="2024-01-01", accession=lambda x: x.index.astype(str) + f"-{i}",
                                              snippet="").to_csv(d / "events.csv", index=False)
    combine.main(str(tmp_path), str(tmp_path / "out"))
    out = pd.read_csv(tmp_path / "out" / "summary.json".replace("summary.json", "events.csv"))
    assert len(out) == 2 * 10 * 8


def test_combine_survives_empty_shard(tmp_path):
    import combine
    (tmp_path / "s0").mkdir()
    (tmp_path / "s0" / "events.csv").write_text("")  # a shard that found nothing writes an empty file
    (tmp_path / "s1").mkdir()
    _synth(0.5, n_firms=10, seed=1).assign(ticker="T", date="2024-01-01",
                                          accession=lambda x: x.index.astype(str), snippet="").to_csv(
        tmp_path / "s1" / "events.csv", index=False)
    combine.main(str(tmp_path), str(tmp_path / "out"))
    assert len(pd.read_csv(tmp_path / "out" / "events.csv")) == 10 * 8
