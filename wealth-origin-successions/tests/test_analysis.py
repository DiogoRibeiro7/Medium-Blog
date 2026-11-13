import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import analysis as A  # noqa: E402
import fetch as F  # noqa: E402


def test_ols_cluster_equals_hc1_with_singleton_clusters():
    rng = np.random.default_rng(0)
    n = 300
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ np.array([0.5, -1.0]) + rng.standard_normal(n) * (1 + np.abs(X[:, 1]))
    b, se_hc1 = A.ols(y, X)
    _, se_cr1 = A.ols(y, X, np.arange(n))
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])
    assert np.allclose(se_cr1, se_hc1)


def test_copied_snapshot_matches_provenance():
    text = (HERE / "data" / "PROVENANCE.txt").read_text()
    pairs = re.findall(r"^\s+(\S+\.csv)\s+sha256 ([0-9a-f]{64})$", text, flags=re.M)
    assert len(pairs) == 5
    for name, digest in pairs:
        assert hashlib.sha256((HERE / "data" / name).read_bytes()).hexdigest() == digest, name


def test_matching_rules():
    # (country, worth, self_made, source, age, last name)
    L = {2015: {"f": ("US", 10e3, True, "casinos", 88.0, "adel"), "g": ("US", 3e3, True, "widgets", 81.0, "li"),
                "z": ("US", 1e3, True, "zinc", 40.0, "smith"), "q": ("US", 2e3, True, "quux", 79.0, "roe")},
         2016: {"m": ("US", 9e3, False, "casinos", 75.0, "adel"),            # same last name: match
                "h": ("US", 2e3, False, "other", 50.0, "li"),                # common name, other source: no match
                "z": ("US", 1e3, True, "zinc", 41.0, "smith"),
                "c1": ("MO", 1e3, True, "casinos", 50.0, "ho"), "c2": ("AU", 1e3, True, "casinos", 50.0, "pa")},
         2017: {"m": ("US", 9e3, False, "casinos", 76.0, "adel"), "z": ("US", 1e3, True, "zinc", 42.0, "smith"),
                "w": ("US", 2e3, False, "quux", 45.0, "doe")}}               # same source, a list later: window
    for k in range(5):  # make "li" a common US name
        L[2015][f"li{k}"] = ("US", 1e3, True, f"s{k}", 50.0, "li")
        L[2016][f"li{k}"] = L[2015][f"li{k}"]
    rows, d = F.detect(L)
    us16 = next(r for r in rows if r["citizenship"] == "US" and r["year"] == 2016)
    assert us16["n_succ_strict"] == 1 and us16["worth_succ_strict"] == 10e3
    assert us16["n_succ_window"] == 2 and us16["n_exit_unmatched"] == 1  # g has no heir; q matched a list later
    assert us16["n_heir_unmatched"] == 1                                  # h
    us17 = [r for r in rows if r["citizenship"] == "US" and r["year"] == 2017]
    assert not us17 or "n_heir_unmatched" not in us17[0]                 # w was already matched through the window


def test_identifier_choice_prefers_the_persistent_key():
    def rec(uri, nid, c="US"):
        return ({"uri": uri, "naturalId": nid, "name": None}, c, 1e3, True, "x", 50.0, "a")
    raw = {2015: [rec("a", "15a"), rec("b", "15b")], 2016: [rec("a", "16a"), rec("b", "16b")]}
    p = F.persistence(raw)
    assert p["uri"] == 1.0 and p["naturalId"] == 0.0 and max(p, key=p.get) == "uri"


def test_leads_and_lags_are_rebuilt_exactly():
    rng = np.random.default_rng(4)
    iso, years = ["A", "B", "C"], list(range(2012, 2027))
    M = rng.exponential(0.2, (3, len(years))) * (rng.random((3, len(years))) < 0.3)
    a = pd.DataFrame([{"iso3": i, "year": s} for i in iso for s in range(2012, 2024)])
    rows, cols = {i: r for r, i in enumerate(iso)}, {t: c for c, t in enumerate(years)}
    A.fill_leads_lags(a, M, rows, cols, "x")
    for _, r in a.iterrows():
        for k in A.LEADS_LAGS:
            t = r["year"] - k
            want = M[rows[r["iso3"]], cols[t]] if t in cols else np.nan
            got = r[f"x_k{k}"]
            assert (np.isnan(want) and np.isnan(got)) or want == got


def test_event_study_recovers_a_simulated_effect():
    rng = np.random.default_rng(5)
    rows = []
    shocks = {(c, t): (rng.exponential(1.0) if rng.random() < 0.15 else 0.0) for c in range(80) for t in range(2009, 2027)}
    for c in range(80):
        for s in range(2012, 2024):
            r = {"iso3": str(c), "year": s, "billionaire_country": True, "ever_billionaire": True,
                 "ln_y_prev": rng.standard_normal()}
            for k in A.LEADS_LAGS:
                r[f"strict_k{k}"] = shocks[(c, s - k)]
            effect = -0.5 * (r["strict_k0"] + r["strict_k1"] + r["strict_k2"])
            r["g_tfp"] = effect + rng.standard_normal()
            rows.append(r)
    a = pd.DataFrame(rows)
    out = A.event_study(a, "strict", "tfp", "billionaire countries")
    assert out["post"] == pytest.approx(-1.5, abs=0.3)
    assert abs(out["pre"]) < 0.4


def test_panel_coefficient_matches_previous_articles():
    panel, _ = A.build_panel(A.load_countries())
    d = A.prep(A.sample(panel, "billionaire_countries"))
    r = A.fit(d, "g_tfp_3", ["x_inh", "x_sum"], ["ln_y"] + A.SECTOR_CTRL, "fe")
    assert r["b_x_inh"] == pytest.approx(-1.048, abs=5e-4)
