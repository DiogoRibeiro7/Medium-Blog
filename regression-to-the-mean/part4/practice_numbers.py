"""Part 4 of the regression-to-the-mean series: data-science practice.

Every number quoted in the article is printed by this script. Run:

    python practice_numbers.py > output.txt

Data downloaded on first run, checked against SHA-256 digests and cached in data/:
  - causaldata::nsw_mixtape: the Dehejia-Wahba sample of the National Supported Work demonstration
    (185 trainees, 260 randomised controls) with real earnings in 1974, 1975 and 1978;
  - DAAG::psid1 and DAAG::cps1: the non-experimental comparison groups (PSID and CPS) used by LaLonde (1986)
    and Dehejia and Wahba (1999), with the same earnings variables;
  - nycflights13::flights: every flight from the three New York airports in 2013.

Sections
1. Picking winners: a simulated archive of click-through tests (assumptions stated in SCENARIOS), split into
   halves, with normal and nonparametric empirical-Bayes predictions; and a real check on choosing the
   carrier with the best on-time record on each New York route, odd days against even days.
2. Interventions on the worst performers: the NSW experiment, Ashenfelter's dip and the comparison groups.
3. Alerts: daily late-arrival shares by carrier, what follows an alert, system-wide days, segments of
   different size, and persistence rules.
"""

from __future__ import annotations

import hashlib
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 1886
HERE = Path(__file__).parent
DATA = HERE / "data"
RD = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/csv/"
FILES = {
    "nsw_mixtape.csv": RD + "causaldata/nsw_mixtape.csv",
    "psid1.csv": RD + "DAAG/psid1.csv",
    "cps1.csv": RD + "DAAG/cps1.csv",
    "flights.csv": RD + "nycflights13/flights.csv",
}
DIGESTS = {
    "nsw_mixtape.csv": "72efd333a4b94a2bee55149d3542021d0adde03201d010ebe971e3e863360cd6",
    "psid1.csv": "e66261aba97aec4bb67950efb4af49f9cd31444beeaab9a5c5150f69c19770f3",
    "cps1.csv": "06bb86f71868bf92a85db7e172426cfc3f1f8d5bedcbaf784674a56e6810a73c",
    "flights.csv": "b791a2590c9167aa0bd3cb45af3f64101180878cf088c183eae4011e3a654f42",
}
YEARS = ("re74", "re75", "re78")
# Simulated archives: (label, tail degrees of freedom or None for normal, scale of the relative effects).
SCENARIOS = (("heavy-tailed effects, small", 3.0, 0.05), ("normal effects, small", None, 0.08),
             ("heavy-tailed effects, larger", 3.0, 0.10))
N_TESTS = 6000


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fetch(name: str) -> Path:
    path = DATA / name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FILES[name], timeout=600) as r:
            path.write_bytes(r.read())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != DIGESTS[name]:
        raise RuntimeError(f"{name}: SHA-256 {got} does not match the pinned digest {DIGESTS[name]}")
    return path


def mean_se(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def diff_se(a, b):
    ma, sa = mean_se(a)
    mb, sb = mean_se(b)
    return ma - mb, math.hypot(sa, sb)


# ---------------------------------------------------------------------------
# Picking winners
# ---------------------------------------------------------------------------

def simulate_portfolio(rng, n_tests: int = 10_000, tail_df: float = 3.0, rel_scale: float = 0.05) -> pd.DataFrame:
    """A portfolio of click-through tests: k arms each, a base rate per test, heavy-tailed relative effects.

    Every parameter is an assumption stated in the article, not an estimate from any company's data.
    """
    k = rng.choice([2, 3, 4, 5, 6], size=n_tests, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    base = rng.beta(6, 394, size=n_tests)                          # mean 1.5%
    rows = []
    for j in range(n_tests):
        rel = rel_scale * rng.standard_t(tail_df, size=k[j]) if tail_df else rel_scale * rng.standard_normal(k[j])
        p = np.clip(base[j] * (1 + rel), 1e-4, 0.5)
        n = np.maximum(200, rng.lognormal(math.log(5000), 0.6, size=k[j]).astype(int))
        rows.append(pd.DataFrame({"test": j, "arm": np.arange(k[j]), "p_true": p, "impressions": n,
                                  "clicks": rng.binomial(n, p)}))
    return pd.concat(rows, ignore_index=True)


def split_halves(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Split each arm's impressions at random into two halves (hypergeometric thinning of the counts)."""
    out = df.copy()
    out["nA"] = out.impressions // 2
    out["nB"] = out.impressions - out.nA
    out["cA"] = rng.hypergeometric(out.clicks, out.impressions - out.clicks, out.nA)
    out["cB"] = out.clicks - out.cA
    out["pA"], out["pB"] = out.cA / out.nA, out.cB / out.nB
    return out


def npmle_prior(d: np.ndarray, v: np.ndarray, grid: np.ndarray, iters: int = 400) -> np.ndarray:
    """Nonparametric maximum-likelihood prior on a fixed grid for d_i ~ N(theta_i, v_i): EM for the grid weights."""
    loglike = -0.5 * (d[:, None] - grid[None, :]) ** 2 / v[:, None]
    loglike -= loglike.max(1, keepdims=True)
    like = np.exp(loglike)
    w = np.full(len(grid), 1 / len(grid))
    for _ in range(iters):
        post = like * w
        post /= post.sum(1, keepdims=True)
        w = post.mean(0)
    return w


def posterior_mean(d: np.ndarray, v: np.ndarray, grid: np.ndarray, w: np.ndarray) -> np.ndarray:
    loglike = -0.5 * (d[:, None] - grid[None, :]) ** 2 / v[:, None]
    loglike -= loglike.max(1, keepdims=True)
    post = np.exp(loglike) * w
    return (post * grid).sum(1) / post.sum(1)


def within_group(df: pd.DataFrame, group: str, y: str, v: str) -> pd.DataFrame:
    """Deviation of each arm from its group's mean and the sampling variance of that deviation."""
    out = df[df.groupby(group)[y].transform("size") >= 2].copy()
    k = out.groupby(group)[y].transform("size")
    out["k"] = k
    out["d"] = out[y] - out.groupby(group)[y].transform("mean")
    out["vd"] = out[v] * (1 - 2 / k) + out.groupby(group)[v].transform("sum") / k ** 2
    return out


def shrink(df: pd.DataFrame, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Normal-prior (linear) and nonparametric empirical-Bayes posterior means of the deviations."""
    d, vd = df.d.to_numpy(), df.vd.to_numpy()
    tau2 = max(d.var() - vd.mean(), 1e-12)
    linear = d * tau2 / (tau2 + vd)
    w = npmle_prior(d, vd, grid)
    return linear, posterior_mean(d, vd, grid, w), tau2, w


def winner_leads(df: pd.DataFrame, group: str, cols: list[str], select: str) -> pd.DataFrame:
    """For each group, the arm with the highest value of `select`, and its lead over the other arms in each column."""
    rows = []
    for j, g in df.groupby(group, sort=False):
        if len(g) < 2:
            continue
        i = g[select].to_numpy().argmax()
        other = np.arange(len(g)) != i
        rows.append([j, len(g)] + [g[c].to_numpy()[i] - g[c].to_numpy()[other].mean() for c in cols])
    return pd.DataFrame(rows, columns=[group, "k"] + cols)


# ---------------------------------------------------------------------------
# Worst performers: NSW
# ---------------------------------------------------------------------------

def load_nsw():
    dw = pd.read_csv(fetch("nsw_mixtape.csv"))
    psid = pd.read_csv(fetch("psid1.csv"))
    cps = pd.read_csv(fetch("cps1.csv"))
    return dw, psid, cps


def holdout_se(sd_treated: float, sd_control: float, n: int, share: float) -> float:
    """Standard error of treated-minus-holdout when a share of n flagged units is held out at random."""
    return math.sqrt(sd_treated ** 2 / (n * (1 - share)) + sd_control ** 2 / (n * share))


# ---------------------------------------------------------------------------
# Alerts: flights
# ---------------------------------------------------------------------------

def load_flights() -> pd.DataFrame:
    f = pd.read_csv(fetch("flights.csv"), usecols=["year", "month", "day", "arr_delay", "carrier", "origin", "dest"])
    f["date"] = pd.to_datetime(dict(year=f.year, month=f.month, day=f.day))
    # Late: arrived 15 or more minutes after schedule, or did not arrive (cancelled or diverted).
    f["late"] = ~((f.arr_delay < 15) & f.arr_delay.notna())
    return f


def daily_series(f: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    s = f.groupby(keys + ["date"]).agg(n=("late", "size"), k=("late", "sum")).reset_index()
    s["x"] = s.k / s.n
    return s


def complete_days(s: pd.DataFrame, key: str) -> pd.DataFrame:
    """One row per series and calendar day of 2013, with NaN where the series had no flights."""
    days = pd.date_range("2013-01-01", "2013-12-31", freq="D")
    idx = pd.MultiIndex.from_product([sorted(s[key].unique()), days], names=[key, "date"])
    return s.set_index([key, "date"]).reindex(idx).reset_index()


def lag_corr(x: np.ndarray, lag: int) -> float:
    a, b = x[:-lag], x[lag:]
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def alert_study(cc: pd.DataFrame, col: str, q: float, consecutive: bool = False, window: int = 28) -> dict:
    """Alerts on the excess of a daily value over its trailing mean, with horizon-specific predictions."""
    frames = []
    for name, g in cc.groupby("carrier"):
        g = g.sort_values("date").copy()
        v = g[col]
        g["base"] = v.shift(1).rolling(window, min_periods=20).mean()
        g["ex"] = v - g.base
        for k in range(1, 8):
            g[f"lead{k}"] = v.shift(-k) - g.base
        for k in (1, 2):
            g[f"lag{k}"] = v.shift(k)
        g["ex_prev"] = g.ex.shift(1)
        g["thr"] = g.ex[g.h1].quantile(q)
        frames.append(g)
    d = pd.concat(frames)
    train = d[d.h1 & d.ex.notna()]
    slopes, intercepts = [], []
    for k in range(1, 8):
        t = train.dropna(subset=[f"lead{k}"])
        b = np.cov(t.ex, t[f"lead{k}"])[0, 1] / t.ex.var()
        slopes.append(b)
        intercepts.append(t[f"lead{k}"].mean() - b * t.ex.mean())
    test = d[~d.h1 & d.ex.notna() & d.lead1.notna()]
    sel = test.ex >= test.thr
    if consecutive:
        sel &= test.ex_prev >= test.thr
    a = test[sel].copy()
    a["x"] = a[col]
    a["nxt"] = a.base + a.lead1
    a["nxt2"] = a.base + a.lead2
    a["pred"] = a.base + intercepts[0] + slopes[0] * a.ex
    a["all_n"] = a.date.map(cc.groupby("date")["all"].first().shift(-1))
    a["raw"] = a["x"] if col == "x" else a.date.map(lambda _: np.nan)
    if col != "x":
        a["raw"] = cc.set_index(["carrier", "date"]).loc[list(zip(a.carrier, a.date)), "x"].to_numpy()
    full = a.dropna(subset=["lag2", "lag1"] + [f"lead{k}" for k in range(1, 8)])
    P = np.column_stack([full.lag2, full.lag1, full[col]] + [full.base + full[f"lead{k}"] for k in range(1, 8)])
    M = np.column_stack([np.full(len(full), np.nan)] * 2 + [full[col]]
                        + [full.base + intercepts[k - 1] + slopes[k - 1] * full.ex for k in range(1, 8)])
    allv = cc.pivot(index="date", columns="carrier", values="all").iloc[:, 0]
    A = np.array([[allv.get(t + pd.Timedelta(days=k), np.nan) for k in range(-2, 8)] for t in full.date])
    return {"alerts": a, "slopes": slopes, "intercepts": intercepts, "paths": (P, M, A)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    out = {}

    # -----------------------------------------------------------------------
    header("1. Picking winners")
    grid = np.linspace(-1.2, 1.2, 241)
    print(f"simulated archives of {N_TESTS:,} tests each: 2-6 arms (shares 0.30/0.25/0.20/0.15/0.10), base click rate")
    print("Beta(6, 394) (mean 1.5%), impressions per arm lognormal with median 5,000, relative effects scale * t or normal;")
    print("each arm's impressions split at random into halves A and B; the winner is the arm with the highest rate in A")
    sims = []
    for si, (label, tail, scale) in enumerate(SCENARIOS):
        df = simulate_portfolio(np.random.default_rng(SEED + 10 + si), n_tests=N_TESTS, tail_df=tail, rel_scale=scale)
        sp = split_halves(df, np.random.default_rng(SEED + 20 + si))
        dropped = int(((sp.cA == 0) | (sp.cB == 0)).sum())
        sp = sp[(sp.cA > 0) & (sp.cB > 0)].copy()
        sp["yA"], sp["yB"], sp["yt"] = np.log(sp.pA), np.log(sp.pB), np.log(sp.p_true)
        sp["vA"] = (1 - sp.pA) / (sp.nA * sp.pA)
        w = within_group(sp, "test", "yA", "vA")
        lin, npm, tau2, _ = shrink(w, grid)
        w["lin"], w["npm"] = lin, npm
        wl = winner_leads(w, "test", ["yA", "yB", "lin", "npm", "yt"], "yA")
        best = w.groupby("test").apply(lambda g: g.yA.to_numpy().argmax() == g.yt.to_numpy().argmax())
        oracle = w.groupby("test").apply(
            lambda g: g.yt.to_numpy().max() - np.delete(g.yt.to_numpy(), g.yt.to_numpy().argmax()).mean())
        R = tau2 / (tau2 + w.vd.mean())
        true_dev = w.yt - w.groupby("test").yt.transform("mean")
        tau2_true = true_dev.var()
        noise_actual = ((w.d - true_dev) ** 2).mean()
        print(f"{label}: tests {wl.test.nunique():,}, arms {len(w):,} ({dropped} arms dropped for a half with no clicks); "
              f"SD of true deviations (log scale) estimated {math.sqrt(tau2):.3f}, actual {math.sqrt(tau2_true):.3f}; "
              f"noise SD assumed {math.sqrt(w.vd.mean()):.3f}, actual {math.sqrt(noise_actual):.3f}")
        print(f"   reliability of one arm (half the traffic): estimated {R:.3f}, actual "
              f"{tau2_true / (tau2_true + noise_actual):.3f}; with the full traffic, estimated "
              f"{tau2 / (tau2 + w.vd.mean() / 2):.3f}")
        lin_true = w.d * tau2_true / (tau2_true + w.vd)
        vt = (1 - np.exp(w.yt)) / (w.nA * np.exp(w.yt))
        vdt = vt * (1 - 2 / w.k) + vt.groupby(w.test).transform("sum") / w.k ** 2
        npm_tv = posterior_mean(w.d.to_numpy(), vdt.to_numpy(), grid, npmle_prior(w.d.to_numpy(), vdt.to_numpy(), grid))
        extra = winner_leads(w.assign(lin_t=lin_true.to_numpy(), npm_t=npm_tv), "test", ["yA", "lin_t", "npm_t"], "yA")
        print(f"   normal prior with the actual prior variance: {100 * math.expm1(extra.lin_t.mean()):.1f}%; nonparametric "
              f"prior with variances from the true rates: {100 * math.expm1(extra.npm_t.mean()):.1f}%; true deviations "
              f"outside the grid +-1.2: {int((true_dev.abs() > 1.2).sum())}")
        print(f"   winner's lead in half A {wl.yA.mean():.4f}, in half B {wl.yB.mean():.4f}, true {wl.yt.mean():.4f}; "
              f"share kept in B {wl.yB.mean() / wl.yA.mean():.3f}")
        print(f"   predicted from half A alone: normal prior {wl.lin.mean():.4f}, nonparametric prior {wl.npm.mean():.4f}; "
              f"half-A winner is the truly best arm in {best.mean():.3f} of tests")
        pct = lambda v: 100 * math.expm1(v)
        print(f"   as a percentage lead in click rate: A {pct(wl.yA.mean()):.1f}%, B {pct(wl.yB.mean()):.1f}%, true "
              f"{pct(wl.yt.mean()):.1f}%, normal prior {pct(wl.lin.mean()):.1f}%, nonparametric {pct(wl.npm.mean()):.1f}%")
        print(f"   true lead of the truly best arm (what an oracle would pick) {100 * math.expm1(oracle.mean()):.1f}%; "
              f"the half-A winner's true lead is {wl.yt.mean() / oracle.mean():.3f} of it")
        for k, g in wl.groupby("k"):
            print(f"   {k} arms ({len(g):,} tests): A {g.yA.mean():.4f}, B {g.yB.mean():.4f}, true {g.yt.mean():.4f}, "
                  f"normal {g.lin.mean():.4f}, nonparametric {g.npm.mean():.4f}; in percent A "
                  f"{100 * math.expm1(g.yA.mean()):.1f}%, B {100 * math.expm1(g.yB.mean()):.1f}%, true "
                  f"{100 * math.expm1(g.yt.mean()):.1f}%")
        sims.append((label, R, wl))
        if si == 0:
            rows = []
            for _, g in w.groupby("test"):
                i = g.yA.to_numpy().argmax()
                o = np.arange(len(g)) != i
                se = math.sqrt(g.vA.to_numpy()[i] + g.vA.to_numpy()[o].sum() / o.sum() ** 2)
                lead = g.yA.to_numpy()[i] - g.yA.to_numpy()[o].mean()
                rows.append((lead, g.yB.to_numpy()[i] - g.yB.to_numpy()[o].mean(),
                             g.yt.to_numpy()[i] - g.yt.to_numpy()[o].mean(), g.nA.mean(), lead / se))
            z = pd.DataFrame(rows, columns=["a", "b", "t", "n", "z"])
            for lab, sel in (("winner's lead significant at 5% (z >= 1.96)", z.z >= 1.96), ("not significant", z.z < 1.96)):
                g = z[sel]
                print(f"   {lab}: {len(g):,} tests ({len(g) / len(z):.3f}); lead in A {100 * math.expm1(g.a.mean()):.1f}%, "
                      f"in B {100 * math.expm1(g.b.mean()):.1f}%, true {100 * math.expm1(g.t.mean()):.1f}%; share of the "
                      f"measured lead that is real {g.t.mean() / g.a.mean():.3f}")
            z["size"] = pd.qcut(z.n, 3, labels=["smallest third", "middle third", "largest third"])
            for lab, g in z.groupby("size", observed=True):
                print(f"   tests in the {lab} by impressions per arm (median {g.n.median():,.0f} per half): lead in A "
                      f"{100 * math.expm1(g.a.mean()):.1f}%, true {100 * math.expm1(g.t.mean()):.1f}%, share real "
                      f"{g.t.mean() / g.a.mean():.3f}")
    out["sims"] = sims

    f = load_flights()
    f["ontime"] = ~f.late
    f["half"] = np.where(f.day % 2 == 1, "A", "B")
    f["route"] = f.origin + "-" + f.dest
    rt = f.groupby(["route", "carrier", "half"]).agg(n=("ontime", "size"), k=("ontime", "sum")).unstack("half")
    rt.columns = [a + b for a, b in rt.columns]
    rt = rt.dropna()
    rt = rt[(rt.nA >= 20) & (rt.nB >= 20)].reset_index()
    rt["pA"], rt["pB"] = rt.kA / rt.nA, rt.kB / rt.nB
    rt["vA"] = rt.pA * (1 - rt.pA) / rt.nA
    w = within_group(rt, "route", "pA", "vA")
    lin, npm, tau2, _ = shrink(w, np.linspace(-0.5, 0.5, 201))
    w["lin"], w["npm"] = lin, npm
    wl = winner_leads(w, "route", ["pA", "pB", "lin", "npm"], "pA")
    same = w.groupby("route").apply(lambda g: g.pA.to_numpy().argmax() == g.pB.to_numpy().argmax())
    R = tau2 / (tau2 + w.vd.mean())
    print(f"routes from New York served by 2+ carriers with 20+ flights on both odd and even days: {wl.route.nunique()}, "
          f"carrier-routes {len(w)}; median flights per carrier-route on odd days {w.nA.median():.0f}")
    print(f"   SD of true differences in on-time share {math.sqrt(tau2):.4f}, of noise {math.sqrt(w.vd.mean()):.4f}; "
          f"reliability {R:.3f}")
    print(f"   best carrier on odd days: lead {wl.pA.mean():.4f} on odd days, {wl.pB.mean():.4f} on even days "
          f"(kept {wl.pB.mean() / wl.pA.mean():.3f}); predicted from odd days: normal prior {wl.lin.mean():.4f}, "
          f"nonparametric {wl.npm.mean():.4f}; same carrier best on even days in {same.mean():.3f} of routes")
    for k, g in wl.groupby("k"):
        print(f"   {k} carriers ({len(g)} routes): odd {g.pA.mean():.4f}, even {g.pB.mean():.4f}, normal {g.lin.mean():.4f}, "
              f"nonparametric {g.npm.mean():.4f}")
    out["routes"] = (R, wl)

    header("2. Interventions on the worst performers: the NSW experiment")
    dw, psid, cps = load_nsw()
    tr, ct = dw[dw.treat == 1], dw[dw.treat == 0]
    print(f"Dehejia-Wahba sample: {len(dw)} men, {len(tr)} trainees and {len(ct)} randomised controls")
    for lab, g in (("trainees", tr), ("controls", ct), ("PSID", psid), ("CPS", cps)):
        means = " / ".join(f"{g[y].mean():,.0f}" for y in YEARS)
        zeros = " / ".join(f"{(g[y] == 0).mean():.3f}" for y in YEARS)
        print(f"   {lab:9s} n {len(g):6d}  mean earnings 1974 / 1975 / 1978: {means};  share with zero: {zeros}")
    d_ct, se_ct = diff_se(ct.re78, ct.re75)
    d_ct74, _ = diff_se(ct.re75, ct.re74)
    print(f"controls: 1974 -> 1975 {d_ct74:+,.0f}; 1975 -> 1978 {(ct.re78 - ct.re75).mean():+,.0f} "
          f"(SE of the paired change {mean_se(ct.re78 - ct.re75)[1]:,.0f})")
    pp, pp_se = mean_se(tr.re78 - tr.re75)
    exp, exp_se = diff_se(tr.re78, ct.re78)
    did_exp = pp - (ct.re78 - ct.re75).mean()
    did_exp_se = math.hypot(pp_se, mean_se(ct.re78 - ct.re75)[1])
    print(f"trainees' change 1975 -> 1978 (the before-and-after estimate): {pp:+,.0f} (SE {pp_se:,.0f})")
    print(f"experimental estimate, trainees minus controls in 1978: {exp:+,.0f} (SE {exp_se:,.0f}); "
          f"difference in changes {did_exp:+,.0f} (SE {did_exp_se:,.0f})")
    print(f"before-and-after over experimental: {pp / exp:.2f}; the control group's change is "
          f"{(ct.re78 - ct.re75).mean() / pp:.2f} of the trainees' change")
    print(f"1975 earnings, trainees minus controls: {tr.re75.mean() - ct.re75.mean():+,.0f} "
          f"(SE {diff_se(tr.re75, ct.re75)[1]:,.0f})")
    comps = {}
    for lab, c in (("PSID", psid), ("CPS", cps)):
        rules = (("everyone", np.ones(len(c), bool)),
                 ("no 1975 earnings", (c.re75 == 0).to_numpy()),
                 ("1975 earnings under 2,000", (c.re75 < 2000).to_numpy()),
                 ("under 2,000 in 1974 and 1975", ((c.re74 < 2000) & (c.re75 < 2000)).to_numpy()))
        for rule, sel in rules:
            g = c[sel]
            change = (g.re78 - g.re75).mean()
            did = pp - change
            comps[(lab, rule)] = (len(g), g.re74.mean(), g.re75.mean(), g.re78.mean(), change, did)
            print(f"   {lab} {rule:28s} n {len(g):6d}: 1974 / 1975 / 1978 {g.re74.mean():7,.0f} / {g.re75.mean():7,.0f} / "
                  f"{g.re78.mean():7,.0f}; change {change:+7,.0f}; difference-in-differences {did:+7,.0f}")
    for lab, g in (("NSW controls", ct), ("PSID", psid), ("CPS", cps)):
        print(f"   correlation of 1975 and 1978 earnings, {lab}: {np.corrcoef(g.re75, g.re78)[0, 1]:.3f}; "
              f"of 1974 and 1975: {np.corrcoef(g.re74, g.re75)[0, 1]:.3f}")
    rule_fns = (("no 1975 earnings", lambda d: d.re75 == 0),
                ("1975 earnings under 2,000", lambda d: d.re75 < 2000),
                ("under 2,000 in 1974 and 1975", lambda d: (d.re74 < 2000) & (d.re75 < 2000)))
    print("the same rule applied to trainees, controls and comparison group alike:")
    like = {}
    for rule, fn in rule_fns:
        t, c0 = tr[fn(tr)], ct[fn(ct)]
        ch_t, ch_c = (t.re78 - t.re75).mean(), (c0.re78 - c0.re75).mean()
        exp_r = ch_t - ch_c
        exp_se = math.hypot(mean_se(t.re78 - t.re75)[1], mean_se(c0.re78 - c0.re75)[1])
        print(f"   {rule}: trainees {len(t)} (change {ch_t:+,.0f}), controls {len(c0)} (1974 / 1975 / 1978 "
              f"{c0.re74.mean():,.0f} / {c0.re75.mean():,.0f} / {c0.re78.mean():,.0f}, change {ch_c:+,.0f}); experimental "
              f"difference in changes {exp_r:+,.0f} (SE {exp_se:,.0f})")
        for lab, c in (("PSID", psid), ("CPS", cps)):
            g = c[fn(c)]
            ch_g = (g.re78 - g.re75).mean()
            se_g = math.hypot(mean_se(t.re78 - t.re75)[1], mean_se(g.re78 - g.re75)[1])
            like[(lab, rule)] = (ch_t - ch_g, exp_r)
            print(f"      {lab}: change {ch_g:+,.0f}; difference-in-differences {ch_t - ch_g:+,.0f} (SE {se_g:,.0f}); "
                  f"miss against the experiment {ch_t - ch_g - exp_r:+,.0f}")
    for lab, c in (("NSW controls", ct), ("PSID", psid), ("CPS", cps)):
        g = c[c.re75 < 2000]
        z = (2000 - c.re75.mean()) / c.re75.std(ddof=1)
        X = np.column_stack([np.ones(len(c)), c.re75])
        beta = np.linalg.lstsq(X, c.re78.to_numpy(float), rcond=None)[0]
        print(f"   {lab}, 1975 earnings under 2,000: share who earned 5,000 or more in 1974 {(g.re74 >= 5000).mean():.3f}; "
              f"threshold {z:+.2f} SD from the population's 1975 mean; within-population line predicts 1978 mean "
              f"{(beta[0] + beta[1] * g.re75).mean():,.0f}, actual {g.re78.mean():,.0f}")
    out["like"] = like
    sd_t, sd_c = tr.re78.std(ddof=1), ct.re78.std(ddof=1)
    print(f"SD of 1978 earnings: trainees {sd_t:,.0f}, controls {sd_c:,.0f}")
    for n in (445, 1000, 5000):
        print(f"   {n:5d} flagged units, standard error of the effect with a random holdout of "
              + ", ".join(f"{int(100 * s)}%: {holdout_se(sd_t, sd_c, n, s):,.0f}" for s in (0.05, 0.1, 0.2, 0.5)))
    out["nsw"] = {"dw": dw, "psid": psid, "cps": cps, "comps": comps}

    # A scorecard: the worst carrier-routes of January-June, measured in July-December, with no intervention.
    fl = load_flights()
    fl["ontime"] = ~fl.late
    fl["H"] = np.where(fl.month <= 6, "A", "B")
    fl["route"] = fl.origin + "-" + fl.dest
    sc = fl.groupby(["route", "carrier", "H"]).agg(n=("ontime", "size"), k=("ontime", "sum")).unstack("H")
    sc.columns = [a + b for a, b in sc.columns]
    sc = sc.dropna()
    sc = sc[(sc.nA >= 20) & (sc.nB >= 20)].reset_index()
    sc["pA"], sc["pB"] = sc.kA / sc.nA, sc.kB / sc.nB
    sc["vA"] = sc.pA * (1 - sc.pA) / sc.nA
    tau2 = sc.pA.var() - sc.vA.mean()
    print(f"scorecard: {len(sc)} carrier-routes with 20+ flights in each half of 2013; on-time share January-June "
          f"{sc.pA.mean():.3f}, July-December {sc.pB.mean():.3f}")
    print(f"   reliability of a half-year on-time share as a measure of that half-year (sampling noise only) "
          f"{tau2 / (tau2 + sc.vA.mean()):.3f}; correlation between the two halves {np.corrcoef(sc.pA, sc.pB)[0, 1]:.3f}")
    q = int(round(0.1 * len(sc)))
    for lab, g in (("worst tenth", sc.nsmallest(q, "pA")), ("best tenth", sc.nlargest(q, "pA"))):
        kept = (g.pB.mean() - sc.pB.mean()) / (g.pA.mean() - sc.pA.mean())
        print(f"   {lab} in January-June ({q}): {g.pA.mean():.3f} -> {g.pB.mean():.3f}; share of the gap to the mean "
              f"kept {kept:.3f}")
    out["scorecard"] = sc

    # -----------------------------------------------------------------------
    header("3. Alerts: daily late-arrival shares at the New York airports, 2013")
    print(f"flights {len(f):,}; late (15+ minutes or did not arrive) {f.late.mean():.3f}")
    dayall = f.groupby("date").late.mean()
    print(f"all flights, daily late share: mean {dayall.mean():.3f}, SD {dayall.std():.3f}, lag-1 correlation "
          f"{lag_corr(dayall.to_numpy(), 1):.3f}, lag-2 {lag_corr(dayall.to_numpy(), 2):.3f}, lag-7 "
          f"{lag_corr(dayall.to_numpy(), 7):.3f}")
    car = daily_series(f, ["carrier"])
    vol = car.groupby("carrier").n.mean()
    big = sorted(vol[vol >= 20].index)
    print(f"carriers {car.carrier.nunique()}; with 20+ flights a day on average: {len(big)} "
          f"({', '.join(big)}), carrying {f.carrier.isin(big).mean():.3f} of flights")
    cc = complete_days(car[car.carrier.isin(big)], "carrier")
    cc["all"] = cc.date.map(dayall)
    cc["h1"] = cc.date.dt.month <= 6
    print("alert rule: a carrier's daily late share exceeds its trailing 28-day mean by at least the 95th percentile")
    print("of that excess in January-June; predictions x(t+k) = baseline + a_k + b_k * excess, with a_k and b_k fitted")
    print("on January-June days of all nine carriers pooled; alerts counted from July to December")
    res = alert_study(cc, "x", q=0.95)
    r = res["alerts"]
    print(f"   horizon slopes b_1..b_7 (share of the excess expected to remain): "
          + ", ".join(f"{b:.3f}" for b in res["slopes"]))
    print(f"   intercepts a_1..a_7: " + ", ".join(f"{a:+.3f}" for a in res["intercepts"]))
    print(f"alerts {len(r)} on {r.carrier.nunique()} carriers: baseline {r.base.mean():.3f}, late share on the alert day "
          f"{r.x.mean():.3f}, next day {r.nxt.mean():.3f}, predicted {r.pred.mean():.3f}")
    print(f"   fall from the alert day to the next {(r.x - r.nxt).mean():.3f}, predicted {(r.x - r.pred).mean():.3f}; "
          f"share of the excess still present the next day {(r.nxt - r.base).mean() / (r.x - r.base).mean():.3f}")
    print(f"   all-flights late share on alert days {r['all'].mean():.3f}, next day {r.all_n.mean():.3f}")
    hi_all = dayall[dayall.index.month > 6].quantile(0.9)
    print(f"   alerts on days when the all-flights share was in its own July-December top 10% (>= {hi_all:.3f}): "
          f"{(r['all'] >= hi_all).mean():.3f}")
    print("   alerts by month: " + ", ".join(f"{pd.Timestamp(2013, mth, 1):%b} {c}" for mth, c in
                                         r.date.dt.month.value_counts().sort_index().items())
          + f"; on {r.date.nunique()} distinct days, at most {r.groupby('date').size().max()} carriers on one day")
    dv = dayall.to_numpy()
    first = dayall.index.month <= 6
    for lag in (1, 2, 3):
        print(f"   all-flights late share, lag-{lag} correlation: January-June {lag_corr(np.where(first, dv, np.nan), lag):.3f}, "
              f"July-December {lag_corr(np.where(~first, dv, np.nan), lag):.3f}")
    valid = cc.dropna(subset=["x"])
    tot = f.groupby("date").late.agg(["sum", "size"])
    other = (valid.date.map(tot["sum"]) - valid.k) / (valid.date.map(tot["size"]) - valid.n)
    print(f"correlation of a large carrier's daily late share with the all-flights share: "
          f"{np.corrcoef(valid.x, valid['all'])[0, 1]:.3f}; with the share of all other carriers' flights "
          f"{np.corrcoef(valid.x, other)[0, 1]:.3f}")
    P, M, A = res["paths"]
    print(f"alerts with a complete window from 2 days before to 7 after: {len(P)}")
    print("   day       " + "  ".join(f"{d:+d}".rjust(6) for d in range(-2, 8)))
    print("   observed  " + "  ".join(f"{v:6.3f}" for v in P.mean(0)))
    print("   predicted " + "  ".join("      " if np.isnan(v) else f"{v:6.3f}" for v in M.mean(0)))
    print("   baseline  " + f"{r.base.mean():6.3f} (trailing 28 days)")
    print("   all       " + "  ".join(f"{v:6.3f}" for v in A.mean(0)))
    out["alert_path"] = (P.mean(0), M.mean(0), A.mean(0), r.base.mean())
    cc["u"] = cc.x - cc["all"]
    rel = alert_study(cc, "u", q=0.95)
    rl = rel["alerts"]
    print(f"relative alerts (carrier share minus all-flights share, same rule): {len(rl)}; excess over its baseline on "
          f"the alert day {(rl.x - rl.base).mean():+.3f}, next day {(rl.nxt - rl.base).mean():+.3f}, predicted "
          f"{(rl.pred - rl.base).mean():+.3f}; next-day slope b_1 {rel['slopes'][0]:.3f}")
    print(f"   carrier late share on those days {rl.raw.mean():.3f} against all flights {rl['all'].mean():.3f}; "
          f"share of relative alerts on all-flights top-10% days {(rl['all'] >= hi_all).mean():.3f}")
    anyb = alert_study(cc, "x", q=0.90)["alerts"]
    dbl = anyb[anyb.ex_prev >= anyb.thr]
    sgl = anyb[~(anyb.ex_prev >= anyb.thr)]
    kept_fn = lambda q: (q.nxt - q.base).mean() / (q.x - q.base).mean()
    for lab, q in (("any breach of the 90th percentile", anyb), ("second consecutive breach", dbl),
                   ("first breach only (not preceded by one)", sgl)):
        print(f"   {lab}: {len(q)} alerts, alert day {q.x.mean():.3f}, next day {q.nxt.mean():.3f}, two days later "
              f"{q.nxt2.mean():.3f}; share of the excess left the next day {kept_fn(q):.3f}")
    brng = np.random.default_rng(SEED + 40)
    dk, d2 = [], []
    for _ in range(2000):
        a = dbl.iloc[brng.integers(0, len(dbl), len(dbl))]
        b = sgl.iloc[brng.integers(0, len(sgl), len(sgl))]
        dk.append(kept_fn(a) - kept_fn(b))
        d2.append(a.nxt2.mean() - b.nxt2.mean())
    print(f"   second breach minus first only, bootstrap 95% intervals (2,000 resamples of alerts): share left the next "
          f"day {kept_fn(dbl) - kept_fn(sgl):+.3f} ({np.quantile(dk, 0.025):+.3f} to {np.quantile(dk, 0.975):+.3f}); late "
          f"share two days later {dbl.nxt2.mean() - sgl.nxt2.mean():+.3f} ({np.quantile(d2, 0.025):+.3f} to "
          f"{np.quantile(d2, 0.975):+.3f})")

    resid = (r.nxt - r.pred).to_numpy()
    sd_res = resid.std(ddof=1)
    per_year = 2 * len(r)
    print(f"SD of the next-day late share around its prediction, over the alerts: {sd_res:.3f}; about {per_year} "
          f"alerts a year at this rate")
    day_res = pd.Series(resid, index=r.date.to_numpy()).groupby(level=0).mean()
    sd_day = day_res.std(ddof=1)
    days_year = 2 * len(day_res)
    print(f"   the alerts fall on {len(day_res)} days; SD of the day-average residual {sd_day:.3f}; about {days_year} "
          f"alert days a year")
    for share in (0.1, 0.2, 0.5):
        se = sd_res * math.sqrt(1 / (per_year * (1 - share)) + 1 / (per_year * share))
        se_d = sd_day * math.sqrt(1 / (days_year * (1 - share)) + 1 / (days_year * share))
        print(f"   withholding the response from {int(100 * share)}%: standard error {se:.3f} if alerts were independent; "
              f"{se_d:.3f} withholding whole alert days; smallest effect detectable at 5% with 80% power "
              f"{2.8 * se_d:.3f}")

    # Segments of different size: carrier by airport.
    s = daily_series(f, ["carrier", "origin"])
    s["series"] = s.carrier + "-" + s.origin
    s["all"] = s.date.map(dayall)
    vol = s.groupby("series").n.mean()
    s["vol"] = s.series.map(vol)
    s["tier"] = pd.cut(s.vol, [0, 10, 40, 1000], labels=["under 10", "10 to 40", "40 or more"])
    s = s.sort_values(["series", "date"])
    s["nxt"] = s.groupby("series").x.shift(-1)
    s["gap"] = (s.groupby("series").date.shift(-1) - s.date).dt.days
    s = s[s.gap == 1].copy()
    s["z"] = (s.x - s["all"]) / np.sqrt(s["all"] * (1 - s["all"]) / s.n)
    tiers = s.groupby("tier", observed=True).agg(series=("series", "nunique"), days=("n", "size"), flights=("n", "sum"))
    tiers["flight_share"] = tiers.flights / tiers.flights.sum()
    print(f"carrier-airport series: {s.series.nunique()}, series-days with a next day: {len(s):,}")
    for t, row in tiers.iterrows():
        print(f"   average flights a day {t:10s}: {int(row.series):2d} series, {int(row.days):5d} series-days, "
              f"{row.flight_share:.3f} of flights")
    print(f"   series averaging over 100 flights a day: {', '.join(sorted(vol[vol > 100].index))}")
    for t, g in s.groupby("tier", observed=True):
        print(f"   binomial z against the day, series with {t} flights a day: SD {g.z.std():.2f}, share of series-days "
              f"with z >= 3 {(g.z >= 3).mean():.3f} (normal theory 0.001)")
    s["h1"] = s.date.dt.month <= 6
    zsd = s[s.h1].groupby("series").z.std()
    s["zc"] = s.z / s.series.map(zsd)
    print(f"   z rescaled by each series' own January-June SD of z: share of July-December series-days at 3 or more "
          f"{(s.zc[~s.h1] >= 3).mean():.3f}")
    seg = {}
    for lab, sel in (("late share of 75% or more", s.x >= 0.75), ("z of 3 or more against the day's all-flights share", s.z >= 3),
                     ("rescaled z of 3 or more, July-December", (s.zc >= 3) & ~s.h1),
                     ("z of 3 or more, July-December", (s.z >= 3) & ~s.h1),
                     ("late share of 75% or more, July-December", (s.x >= 0.75) & ~s.h1)):
        g = s[sel]
        shares = g.tier.value_counts(normalize=True)
        seg[lab] = (shares, g.groupby("tier", observed=True).apply(lambda h: (len(h), h.x.mean(), h.nxt.mean(), h["all"].mean())))
        print(f"alert rule '{lab}': {len(g)} alerts; share from series with under 10 / 10 to 40 / 40 or more flights a day: "
              + " / ".join(f"{shares.get(t, 0):.3f}" for t in ("under 10", "10 to 40", "40 or more")))
        for t, h in g.groupby("tier", observed=True):
            print(f"   {t:10s}: {len(h):4d} alerts, late share {h.x.mean():.3f} -> next day {h.nxt.mean():.3f}; "
                  f"all flights that day {h['all'].mean():.3f}")
    s["u"] = s.x - s["all"]
    s["u_n"] = s.groupby("series").u.shift(-1)
    for t, g in s.groupby("tier", observed=True):
        g = g.dropna(subset=["u_n"])
        print(f"   lag-1 correlation of the excess over all flights, series with {t} flights a day: "
              f"{np.corrcoef(g.u, g.u_n)[0, 1]:.3f}")
    out["segments"] = (tiers, seg)
    out["alerts"] = r
    return out


if __name__ == "__main__":
    main()
