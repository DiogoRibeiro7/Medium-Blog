"""How much of Portugal's spread came from Greece and Ireland.

    python contagion.py

Monthly changes in 10-year spreads to Germany (percentage points). The rest of the euro area is
summarised by the plain average of the changes in France, Italy, Spain, Belgium, Austria, the
Netherlands and Finland (the "common" move; for a country in that group, the average of the
others). Three regressions through the origin, so that a steady widening is not hidden in an
intercept:

    Greece   = c * common + greek
    Ireland  = c * common + d * greek + irish
    country  = a * common + b_gr * greek + b_ie * irish + own

``greek`` is the part of Greece's move that the rest of the euro area did not share; ``irish``
is the part of Ireland's move shared neither with the rest nor with Greece. b_gr is the
response, in percentage points, to a one-point Greece-specific widening.

The greek and irish components contain anything that moved Greece (or Ireland) without moving the
seven: Greek news, but also news about the periphery as a whole (a summit decision on private
creditors, say). So b_gr * greek is an upper bound on pure Greek contagion, and the same holds
for Ireland. Reverse spillovers from Portugal to Greece and Ireland push the same way.

Cumulative decomposition: summing the last regression over months splits Portugal's total change
into common, Greek, Irish and own parts.

Windows:
- crisis: changes from November 2009 (the month after the Greek revision) to July 2012 (the
  month before the OMT announcement);
- before Portugal's request: November 2009 to March 2011;
- crisis without the Greek bond exchange (February-April 2012 changes removed: the Greek series
  switches to the new bonds in March 2012);
- after OMT: 2013-2019, and after Portugal's programme: June 2014 to 2019 (Greek data for July
  2015 are missing, so that month is dropped).

Intervals: moving-block bootstrap (blocks of 3 months, 2,000 draws), 5th and 95th percentiles.
A calibration on simulated data with known responses checks the estimator.

Writes ``results/contagion_*.csv`` and ``results/contagion_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import NAMES, RESULTS, monthly_spreads, write

SEED = 20261109
N_BOOT = 2000
BLOCK = 3
REST = ("FR", "IT", "ES", "BE", "AT", "NL", "FI")
RESPONDERS = ("PT", "ES", "IT", "BE", "FR")
WINDOWS = {
    "crisis": ("2009-11", "2012-07", ()),
    "before_request": ("2009-11", "2011-03", ()),
    "crisis_no_exchange": ("2009-11", "2012-07", ("2012-02", "2012-03", "2012-04")),
    "after_omt": ("2013-01", "2019-12", ()),
    "after_programme": ("2014-06", "2019-12", ()),
}
LABELS = {
    "crisis": "Crisis, Nov 2009 - Jul 2012",
    "before_request": "Before Portugal's request, Nov 2009 - Mar 2011",
    "crisis_no_exchange": "Crisis without Feb-Apr 2012",
    "after_omt": "After OMT, 2013-2019",
    "after_programme": "After Portugal's programme, Jun 2014 - 2019",
}


def ols0(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def components(ch: pd.DataFrame, j: str, coefs: tuple[float, np.ndarray] | None = None) -> pd.DataFrame:
    """Common move (excluding j), Greek- and Irish-specific moves, for responder j.

    ``coefs`` (Greece on common; Ireland on common and greek) are estimated on ``ch`` unless given.
    """
    rest = [c for c in REST if c != j]
    common = ch[rest].mean(axis=1).to_numpy()
    el, ie = ch["EL"].to_numpy(), ch["IE"].to_numpy()
    c_el = ols0(el, common[:, None])[0] if coefs is None else coefs[0]
    greek = el - common * c_el
    b = ols0(ie, np.column_stack([common, greek])) if coefs is None else coefs[1]
    irish = ie - np.column_stack([common, greek]) @ b
    out = pd.DataFrame({"common": common, "greek": greek, "irish": irish}, index=ch.index)
    out.attrs["coefs"] = (c_el, b)
    return out


def responses(ch: pd.DataFrame, j: str) -> dict[str, float]:
    comp = components(ch, j)
    X = comp.to_numpy()
    y = ch[j].to_numpy()
    beta = ols0(y, X)
    own = y - X @ beta
    out = dict(zip(("b_common", "b_greek", "b_irish"), map(float, beta)))
    parts = X * beta
    out.update(total=float(y.sum()), part_common=float(parts[:, 0].sum()), part_greek=float(parts[:, 1].sum()),
               part_irish=float(parts[:, 2].sum()), part_own=float(own.sum()),
               r2=float(1 - own.var() / y.var()))
    return out


def window(ch: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b, drop = WINDOWS[name]
    c = ch.loc[a:b]
    if drop:
        c = c.drop(index=[m for m in c.index if str(m)[:7] in drop])
    return c.dropna()


def bootstrap(c: pd.DataFrame, j: str, rng: np.random.Generator) -> pd.DataFrame:
    n = len(c)
    draws = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, size=int(np.ceil(n / BLOCK)))
        idx = np.concatenate([np.arange(k, k + BLOCK) for k in starts])[:n]
        draws.append(responses(c.iloc[idx], j))
    return pd.DataFrame(draws)


def calibration(n: int = 33, reps: int = 400) -> dict[str, float]:
    """Simulated months with known responses of Portugal: 0.30 to Greece, 0.50 to Ireland, 1.0 to the rest.

    The seven others load on a common shock (loading 1 + N(0, 0.3)); Greece and Ireland load on it
    too and carry their own shocks (standard deviation 3 and 1.5 against 0.3 for the common one,
    Ireland also 0.2 times Greece's); Portugal adds noise of standard deviation 0.3. All shocks have
    a positive mean (a crisis drifts upward), which is what an intercept would have absorbed.
    """
    est = []
    for seed in range(reps):
        rng = np.random.default_rng(SEED + seed)
        common = 0.05 + 0.3 * rng.normal(size=n)
        g = 0.4 + 3.0 * rng.normal(size=n)
        i = 0.1 + 1.5 * rng.normal(size=n)
        cols = {c: (1 + 0.3 * rng.normal()) * common + 0.1 * rng.normal(size=n) for c in REST}
        cols["EL"] = common + g
        cols["IE"] = common + 0.2 * g + i
        cols["PT"] = common + 0.3 * g + 0.5 * i + 0.3 * rng.normal(size=n)
        est.append(responses(pd.DataFrame(cols), "PT"))
    m = pd.DataFrame(est)
    return dict(greek=float(m.b_greek.mean()), greek_sd=float(m.b_greek.std()), irish=float(m.b_irish.mean()),
                irish_sd=float(m.b_irish.std()), common=float(m.b_common.mean()))


def run() -> str:
    s = monthly_spreads()
    ch = s[["EL", "IE", *REST, "PT"]].diff()
    ch.index = ch.index.to_period("M") if not isinstance(ch.index, pd.PeriodIndex) else ch.index
    rng = np.random.default_rng(SEED)
    rows, lines = [], []
    lines.append("Responses to a one-point country-specific widening (pp), point estimate [5-95% block bootstrap]; "
                 "cumulative decomposition of each country's change over the window (pp)")
    for w in WINDOWS:
        c = window(ch, w)
        lines.append(f"  {LABELS[w]} ({len(c)} months):")
        for j in RESPONDERS:
            est = responses(c, j)
            D = bootstrap(c, j, rng)
            row = dict(window=w, country=j, months=len(c), **est)
            for k in ("b_greek", "b_irish", "b_common", "part_greek", "part_irish", "part_common", "part_own"):
                row[f"{k}_lo"], row[f"{k}_hi"] = float(D[k].quantile(.05)), float(D[k].quantile(.95))
            rows.append(row)
            lines.append(f"    {NAMES[j]}: Greece {est['b_greek']:+.3f} [{row['b_greek_lo']:+.3f}, {row['b_greek_hi']:+.3f}]; "
                         f"Ireland {est['b_irish']:+.3f} [{row['b_irish_lo']:+.3f}, {row['b_irish_hi']:+.3f}]; "
                         f"common {est['b_common']:+.2f}; R2 {est['r2']:.2f}")
            if not w.startswith("after"):
                lines.append(f"      change {est['total']:+.2f} = common {est['part_common']:+.2f} + Greek {est['part_greek']:+.2f} "
                             f"[{row['part_greek_lo']:+.2f}, {row['part_greek_hi']:+.2f}] + Irish {est['part_irish']:+.2f} "
                             f"[{row['part_irish_lo']:+.2f}, {row['part_irish_hi']:+.2f}] + own {est['part_own']:+.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "contagion_responses.csv", index=False, float_format="%.4f")

    # monthly decomposition for Portugal, slopes estimated on the crisis window without the Greek
    # bond exchange and applied to every month from November 2009 to December 2012
    c = window(ch, "crisis_no_exchange")
    full = ch.loc["2009-11":"2012-12"].dropna()
    est = responses(c, "PT")
    comp = components(full, "PT", components(c, "PT").attrs["coefs"])
    m = pd.DataFrame({"month": full.index.astype(str), "pt": full["PT"].to_numpy(),
                      "common": est["b_common"] * comp["common"].to_numpy(),
                      "greek": est["b_greek"] * comp["greek"].to_numpy(),
                      "irish": est["b_irish"] * comp["irish"].to_numpy()})
    m["own"] = m["pt"] - m[["common", "greek", "irish"]].sum(axis=1)
    m["in_window"] = m["month"] <= "2012-07"
    m.to_csv(RESULTS / "contagion_monthly_pt.csv", index=False, float_format="%.4f")

    pt = df[(df.window == "crisis_no_exchange") & (df.country == "PT")].iloc[0]
    lines += ["", "Portugal over the crisis window without February-April 2012, shares of the total change:"]
    for k, lab in (("part_common", "rest of the euro area"), ("part_greek", "Greece-specific"), ("part_irish", "Ireland-specific"), ("part_own", "Portugal's own")):
        lines.append(f"  {lab}: {pt[k]:+.2f} pp ({pt[k] / pt['total']:.0%})")
    peak = m[m["month"] <= "2012-01"]
    lines.append("  Monthly decomposition with these slopes, every month from November 2009 (the exchange months included, so sums differ):")
    lines.append(f"  To the January 2012 peak: change {peak.pt.sum():+.2f} = common {peak.common.sum():+.2f} + Greek {peak.greek.sum():+.2f} "
                 f"+ Irish {peak.irish.sum():+.2f} + own {peak.own.sum():+.2f}")
    pre = m[m["month"] <= "2011-03"]
    lines.append(f"  To March 2011 (before the request): change {pre.pt.sum():+.2f} = common {pre.common.sum():+.2f} + Greek {pre.greek.sum():+.2f} "
                 f"+ Irish {pre.irish.sum():+.2f} + own {pre.own.sum():+.2f}")
    post = m[(m["month"] > "2011-03") & (m["month"] <= "2012-01")]
    lines.append(f"  April 2011 to January 2012: change {post.pt.sum():+.2f} = common {post.common.sum():+.2f} + Greek {post.greek.sum():+.2f} "
                 f"+ Irish {post.irish.sum():+.2f} + own {post.own.sum():+.2f}")

    lines += ["", "The same Greek shock with and without a backstop: Greece-specific response of Portugal, crisis (without Feb-Apr 2012) vs after OMT"]
    a_ = df[(df.window == "crisis_no_exchange") & (df.country == "PT")].iloc[0]
    b_ = df[(df.window == "after_omt") & (df.country == "PT")].iloc[0]
    c_ = df[(df.window == "after_programme") & (df.country == "PT")].iloc[0]
    lines.append(f"  crisis {a_.b_greek:+.3f} [{a_.b_greek_lo:+.3f}, {a_.b_greek_hi:+.3f}]; after OMT {b_.b_greek:+.3f} [{b_.b_greek_lo:+.3f}, {b_.b_greek_hi:+.3f}]; "
                 f"after the programme {c_.b_greek:+.3f} [{c_.b_greek_lo:+.3f}, {c_.b_greek_hi:+.3f}]")
    s15 = s.loc["2014-12":"2015-08", ["EL", "PT", "IE", "ES", "IT"]]
    lines.append("  2015 Greek crisis, December 2014 to June 2015 (pp): " + ", ".join(f"{NAMES[g]} {s15.loc['2015-06', g] - s15.loc['2014-12', g]:+.2f}" for g in s15.columns))
    s10 = s.loc["2009-12":"2010-05", ["EL", "PT", "IE", "ES", "IT"]]
    lines.append("  2010, first six months, December 2009 to May 2010 (pp): " + ", ".join(f"{NAMES[g]} {s10.loc['2010-05', g] - s10.loc['2009-12', g]:+.2f}" for g in s10.columns))

    cal = calibration()
    lines += ["", f"Calibration (simulated windows of 33 months; true responses Greece 0.30, Ireland 0.50, common 1.0): mean estimates Greece {cal['greek']:.3f} (sd {cal['greek_sd']:.3f}), "
              f"Ireland {cal['irish']:.3f} (sd {cal['irish_sd']:.3f}), common {cal['common']:.2f}"]
    return write("contagion_summary.txt", lines)


if __name__ == "__main__":
    print(run())
