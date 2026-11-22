"""Portugal's fiscal reaction function, its EU context, and what it implies for debt in 2035.

    python fetch_data.py && python build_data.py && python run_analysis.py

Writes results/results.json, results/summary.txt and figures/*.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import debt_model as M  # noqa: E402
import frf  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
SEED, N_SIM, N_DIAL, BLOCK = 20261122, 10_000, 4_000, 3
I_LR, G_LR = 3.6, (1.011 * 1.02 - 1) * 100  # long-run effective rate and nominal growth in the debt model
DEBT_2027 = 84.5

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update(
    {
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 11,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "legend.labelcolor": INK,
    }
)
NAMES = {
    "AUT": "Austria", "BEL": "Belgium", "BGR": "Bulgaria", "HRV": "Croatia", "CYP": "Cyprus", "CZE": "Czechia",
    "DNK": "Denmark", "EST": "Estonia", "FIN": "Finland", "FRA": "France", "DEU": "Germany", "GRC": "Greece",
    "HUN": "Hungary", "IRL": "Ireland", "ITA": "Italy", "LVA": "Latvia", "LTU": "Lithuania", "LUX": "Luxembourg",
    "MLT": "Malta", "NLD": "Netherlands", "POL": "Poland", "PRT": "Portugal", "ROM": "Romania", "SVK": "Slovakia",
    "SVN": "Slovenia", "ESP": "Spain", "SWE": "Sweden",
}  # fmt: skip


def note(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8.5, color=INK2, ha="left", va="bottom")


# ----------------------------------------------------------------------------- data
def portugal(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[panel.country == "PRT"].set_index("year").sort_index().copy()
    p["g"] = 100 * p["gdp_nom"].pct_change()
    p["real_growth"] = 100 * p["gdp_real"].pct_change()
    p["deflator_growth"] = 100 * p["deflator"].pct_change()
    p["sfa"] = p["debt"] - p["debt"].shift(1) / (1 + p["g"] / 100) - p["interest"] + p["pb"]
    p["d_debt"] = p["debt"].diff()
    return p


def historical_shocks(p: pd.DataFrame, resid: pd.Series, h: float):
    """Joint yearly shock vectors (gap innovation, deflator, yield, stock-flow, rule residual), 1998-2025."""
    hh = p.loc[1996:2025]
    gap = hh["gap"]
    x, y = gap.shift(1).dropna(), gap.iloc[1:]
    rho_gap = float(np.dot(x, y) / np.dot(x, x))
    u = (gap - rho_gap * gap.shift(1)).loc[1998:2025]
    X = np.column_stack([np.ones(len(hh)), hh["gap"].values])
    b = np.linalg.lstsq(X, hh["deflator_growth"].values, rcond=None)[0]
    v = pd.Series(hh["deflator_growth"].values - X @ b, index=hh.index).loc[1998:2025]
    frame = pd.DataFrame(
        {
            "u": u / (1 - h),
            "v": v,
            "dm": hh["ltr"].diff().loc[1998:2025],
            "sfa": hh["sfa"].loc[1998:2025],
            "e": resid.reindex(range(1998, 2026)),
        }
    ).dropna()
    return rho_gap, float(b[1]), frame - frame.mean()


# ----------------------------------------------------------------------- estimation
def estimate(panel: pd.DataFrame, p: pd.DataFrame) -> dict:
    out: dict = {}
    y, X = frf.design(p.reset_index())
    main = frf.ols(y, X)
    b = dict(zip(main["cols"], main["b"], strict=True))
    se = dict(zip(main["cols"], main["se"], strict=True))
    out["main"] = {
        "b": b,
        "se": se,
        "n": main["n"],
        "r2": main["r2"],
        "cov": main["cov"].tolist(),
        "cols": main["cols"],
    }
    out["resid"] = pd.Series(main["e"], index=X.index)

    # Long-run quantities with uncertainty (draws from the estimated sampling distribution)
    rng = np.random.default_rng(SEED)
    idx = [main["cols"].index(c) for c in ("const", "debt_l", "pb_l")]
    draws = rng.multivariate_normal(main["b"][idx], main["cov"][np.ix_(idx, idx)], size=20_000)
    draws = draws[np.abs(draws[:, 2]) < 0.95]
    a_d, r_d, f_d = draws.T

    def lr_pb(d, a=a_d, r=r_d, f=f_d):
        return (a + r * d) / (1 - f)

    slope_req = (I_LR - G_LR) / (100 + G_LR)
    anchor = (M.SFA_LATER - a_d / (1 - f_d)) / (r_d / (1 - f_d) - slope_req)
    out["long_run"] = {
        "slope": b["debt_l"] / (1 - b["pb_l"]),
        "slope_ci": list(np.percentile(r_d / (1 - f_d), [5, 95])),
        "pb_at": {
            str(d): {
                "point": (b["const"] + b["debt_l"] * d) / (1 - b["pb_l"]),
                "ci90": list(np.percentile(lr_pb(d), [5, 95])),
            }
            for d in (60, 85, 100, 130)
        },
        "anchor_point": (M.SFA_LATER - b["const"] / (1 - b["pb_l"])) / (b["debt_l"] / (1 - b["pb_l"]) - slope_req),
        "anchor_ci90": list(np.percentile(anchor[(anchor > 0) & (anchor < 400)], [5, 95])),
        "p_anchor_above_60": float(np.mean(anchor > 60)),
        "i_lr": I_LR,
        "g_lr": G_LR,
    }
    out["one_step_2026"] = b["const"] + b["debt_l"] * p.loc[2025, "debt"] + b["pb_l"] * p.loc[2025, "pb"]
    r = out["resid"]
    out["resid_recent"] = {int(k): float(v) for k, v in r.loc[2016:].items()}
    out["resid_mean_2016_2025_ex_pandemic"] = float(r.loc[2016:].drop([2020, 2021]).mean())
    out["fitted"] = {int(k): float(v) for k, v in (pd.Series(y, index=X.index) - r).items()}

    # Robustness
    rob = {}
    for lags in (1, 3):
        rob[f"hac_lags_{lags}"] = {"se_rho": float(frf.ols(y, X, hac_lags=lags)["se"][1])}
    hub = frf.huber(y, X)
    hb = dict(zip(hub["cols"], hub["b"], strict=True))
    rob["huber"] = {"rho": hb["debt_l"], "phi": hb["pb_l"], "gamma": hb["gap"], "lr": hb["debt_l"] / (1 - hb["pb_l"])}
    yx, Xx = frf.design(p.reset_index().query("year not in (2020, 2021)"))
    ex = frf.ols(yx, Xx)
    eb_ = dict(zip(ex["cols"], ex["b"], strict=True))
    rob["drop_2020_2021"] = {"rho": eb_["debt_l"], "se": float(ex["se"][1]), "phi": eb_["pb_l"]}
    yc, Xc = frf.design(p.reset_index(), y_col="capb")
    Xc = Xc.drop(columns=["gap"])
    cap = frf.ols(yc, Xc)
    cb = dict(zip(cap["cols"], cap["b"], strict=True))
    rob["cyclically_adjusted"] = {
        "rho": cb["debt_l"],
        "se": float(cap["se"][1]),
        "phi": cb["pb_l"],
        "lr": cb["debt_l"] / (1 - cb["pb_l"]),
        "lr_pb_at_85": (cb["const"] + cb["debt_l"] * 85) / (1 - cb["pb_l"]),
    }
    for start in (2012, 2015, 2016):
        ys, Xs = frf.design(p.reset_index(), extra={"post": start})
        s = frf.ols(ys, Xs)
        sb = dict(zip(s["cols"], s["b"], strict=True))
        ss = dict(zip(s["cols"], s["se"], strict=True))
        rob[f"level_shift_from_{start}"] = {
            "shift": sb["post"],
            "se": ss["post"],
            "rho": sb["debt_l"],
            "rho_se": ss["debt_l"],
            "lr_pb_at_85": (sb["const"] + sb["post"] + sb["debt_l"] * 85) / (1 - sb["pb_l"]),
        }
    yi, Xi = frf.design(p.reset_index(), extra={"post2008": 2008})
    Xi["debt_post2008"] = Xi["debt_l"] * Xi["post2008"]
    it = frf.ols(yi, Xi)
    ib = dict(zip(it["cols"], it["b"], strict=True))
    iss = dict(zip(it["cols"], it["se"], strict=True))
    rob["pre_post_2008"] = {
        "rho_pre": ib["debt_l"],
        "rho_post": ib["debt_l"] + ib["debt_post2008"],
        "diff": ib["debt_post2008"],
        "diff_se": iss["debt_post2008"],
    }
    out["robustness"] = rob

    # Rolling 15-year windows
    roll = {}
    for end in range(2010, 2026):
        yr, Xr = frf.design(p.reset_index(), years=(end - 14, end))
        rr = frf.ols(yr, Xr)
        roll[end] = {"rho": float(rr["b"][1]), "se": float(rr["se"][1])}
    out["rolling15"] = roll

    # EU countries
    rows = {}
    for c, g in panel.groupby("country"):
        yc_, Xc_ = frf.design(g)
        rc = frf.ols(yc_, Xc_)
        bc = dict(zip(rc["cols"], rc["b"], strict=True))
        rows[c] = {
            "rho": bc["debt_l"],
            "se": float(rc["se"][1]),
            "phi": bc["pb_l"],
            "gamma": bc["gap"],
            "n": rc["n"],
            "lr": bc["debt_l"] / (1 - bc["pb_l"]),
        }
    eu = pd.DataFrame(rows).T
    re = frf.random_effects(eu.rho.values, eu.se.values)
    eu["eb"], eu["eb_se"] = re["eb"], re["eb_se"]
    out["eu"] = {
        "countries": {c: {k: float(v) for k, v in r_.items()} for c, r_ in eu.iterrows()},
        "re_mean": re["mu"],
        "re_se": re["se_mu"],
        "tau": float(np.sqrt(re["tau2"])),
        "q": re["q"],
        "n_positive_significant": int(((eu.rho - 1.96 * eu.se) > 0).sum()),
        "n_negative": int((eu.rho < 0).sum()),
        "rank_portugal": int(eu.rho.rank(ascending=False)["PRT"]),
        "median_lr": float(eu.lr.median()),
    }
    fe = frf.pooled_fe(panel)
    out["pooled_fe"] = {
        "b": dict(zip(fe["cols"], fe["b"].tolist(), strict=True)),
        "se": dict(zip(fe["cols"], fe["se"].tolist(), strict=True)),
        "n": fe["n"],
    }
    return out


# ---------------------------------------------------------------------- simulation
def policies(est: dict) -> dict[str, M.Policy]:
    b = est["main"]["b"]
    hist = M.Policy("hist", "Portugal's average rule, 1996–2025", kind="rule", a=b["const"], rho=b["debt_l"],
                    gamma=b["gap"], phi=b["pb_l"])  # fmt: skip
    anchored = hist.copy(key="anch", label="Today's surplus, Portugal's response to debt",
                         a=M.anchored_intercept(b["debt_l"], b["pb_l"], DEBT_2027, M.PB_2027))  # fmt: skip
    fixed = M.Policy("fixed", "Primary surplus held at 2.5%", kind="fixed", pbs=2.5)
    return {"fixed": fixed, "anch": anchored, "hist": hist}


def simulate(pol: M.Policy, shocks: pd.DataFrame, est: dict | None, n: int, seed: int, anchored: bool = False):
    rng = np.random.default_rng(seed)
    arr = shocks[["u", "v", "dm", "sfa", "e"]].values
    T, H = len(arr), len(M.YEARS)
    debt, bal, pbs = np.empty((n, H)), np.empty((n, H)), np.empty((n, H))
    if est is not None:
        cols = est["main"]["cols"]
        idx = [cols.index(c) for c in ("const", "debt_l", "gap", "pb_l")]
        mean = np.array(est["main"]["cov"])[np.ix_(idx, idx)]
        mu = np.array([est["main"]["b"][c] for c in ("const", "debt_l", "gap", "pb_l")])
    for k in range(n):
        ix = []
        while len(ix) < H:
            st = int(rng.integers(0, T - BLOCK + 1))
            ix.extend(range(st, st + BLOCK))
        draws = {y: tuple(arr[j]) for y, j in zip(M.YEARS, ix[:H], strict=True)}
        q = pol
        if est is not None and pol.kind == "rule":
            while True:
                a, r, g, f = rng.multivariate_normal(mu, mean)
                if abs(f) < 0.95:
                    break
            if anchored:
                a = M.anchored_intercept(r, f, DEBT_2027, M.PB_2027)
            q = pol.copy(a=a, rho=r, gamma=g, phi=f)
        df = M.run(q, draws=draws)
        debt[k], bal[k], pbs[k] = df["debt"].values, df["balance"].values, df["primary_balance"].values
    return debt, bal, pbs


def summarise(debt, bal, pbs) -> dict:
    yrs = np.array(M.YEARS)
    j = {y: k for k, y in enumerate(M.YEARS)}
    first = np.array([yrs[np.argmax(r <= 60)] if (r <= 60).any() else 9999 for r in debt])
    pct = np.percentile(debt, [5, 10, 25, 50, 75, 90, 95], axis=0)
    return {
        "p_at_or_below_60_in_2035": float((debt[:, j[2035]] <= 60).mean()),
        "p_at_or_below_70_in_2035": float((debt[:, j[2035]] <= 70).mean()),
        "p_above_2027_in_2035": float((debt[:, j[2035]] > debt[:, j[2027]]).mean()),
        "p_deficit_above_3_by_2035": float((bal[:, : j[2035] + 1] < -3).any(axis=1).mean()),
        "median_2035": float(pct[3, j[2035]]),
        "p10_2035": float(pct[1, j[2035]]),
        "p90_2035": float(pct[5, j[2035]]),
        "median_2045": float(pct[3, j[2045]]),
        "median_pb_2030": float(np.median(pbs[:, j[2030]])),
        "median_pb_2035": float(np.median(pbs[:, j[2035]])),
        "median_first_year_at_or_below_60": float(np.median(first)),
        "p_not_at_60_by_2045": float((first == 9999).mean()),
        "p_by_year": {int(y): float((debt[:, k] <= 60).mean()) for k, y in enumerate(M.YEARS)},
        "fan": {
            q: {int(y): float(pct[i, k]) for k, y in enumerate(M.YEARS)}
            for i, q in enumerate(["p5", "p10", "p25", "p50", "p75", "p90", "p95"])
        },
    }


# ------------------------------------------------------------------------- figures
def fig_rule(p: pd.DataFrame, est: dict):
    b = est["main"]["b"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = p.loc[1996:2025]
    periods = [(1996, 2008, "1996–2008", C[0]), (2009, 2015, "2009–2015", C[1]), (2016, 2025, "2016–2025", C[2])]
    for a, z, lab, col in periods:
        s = sub.loc[a:z]
        xs = p["debt"].shift(1).loc[a:z].values
        ax.scatter(xs, s["pb"].values, s=46, color=col, label=lab, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for yr in (2000, 2009, 2010, 2014, 2017, 2019, 2020, 2023, 2025):
        ax.annotate(str(yr), (p["debt"].shift(1).loc[yr], p.loc[yr, "pb"]), xytext=(5, 4),
                    textcoords="offset points", fontsize=8.5, color=INK2)  # fmt: skip
    d = np.linspace(45, 140, 200)
    lr = (b["const"] + b["debt_l"] * d) / (1 - b["pb_l"])
    req = M.SFA_LATER + d * (I_LR - G_LR) / (100 + G_LR)
    ax.plot(d, lr, color=INK, linewidth=2, label="Long-run primary balance of Portugal's rule")
    ax.plot(d, req, color=INK2, linewidth=1.6, linestyle=(0, (4, 3)), label="Primary balance that holds debt steady")
    anchor = est["long_run"]["anchor_point"]
    ax.axvline(anchor, color=C[3], linewidth=1.4)
    ax.text(anchor + 1, 4.2, f"where the rule settles: {anchor:.0f}%", color=INK, fontsize=9.5)
    ax.axvline(60, color=GRID, linewidth=1.6, zorder=0)
    ax.text(61, -9.6, "60%", color=INK2, fontsize=9.5)
    ax.set_xlim(45, 140)
    ax.set_ylim(-10, 5)
    ax.set_xlabel("Debt at the end of the previous year, % of GDP")
    ax.set_ylabel("Primary balance, % of GDP")
    ax.set_title("Portugal's primary balance against its debt, 1996–2025")
    ax.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.0, 0.0))
    note(
        fig, "Source: European Commission, AMECO (autumn 2026); author's estimates. Gap closed for the long-run lines."
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig1_rule_and_anchor.png", dpi=200)
    plt.close(fig)


def fig_eu(est: dict):
    eu = pd.DataFrame(est["eu"]["countries"]).T.sort_values("rho")
    fig, ax = plt.subplots(figsize=(9, 9))
    yy = np.arange(len(eu))
    for k, (c, r) in enumerate(eu.iterrows()):
        col = C[1] if c == "PRT" else C[0]
        ax.plot([r.rho - 1.96 * r.se, r.rho + 1.96 * r.se], [k, k], color=col, linewidth=2, alpha=0.6)
        ax.scatter(r.rho, k, color=col, s=40, zorder=3)
        ax.scatter(r.eb, k, color=SURFACE, edgecolor=col, s=40, zorder=4, linewidth=1.5)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.axvline(est["eu"]["re_mean"], color=C[2], linewidth=1.6, linestyle=(0, (4, 3)))
    ax.text(est["eu"]["re_mean"] + 0.003, len(eu) - 0.3, "EU average", color=INK, fontsize=9.5)
    ax.set_yticks(yy, [NAMES[c] for c in eu.index])
    for t in ax.get_yticklabels():
        if t.get_text() == "Portugal":
            t.set_color(INK)
            t.set_fontweight("bold")
    ax.set_xlabel("Change in primary balance per point of debt, pp of GDP (95% interval)")
    ax.set_title("How strongly each EU government's budget responds to its debt")
    ax.scatter([], [], color=C[0], s=40, label="Country estimate")
    ax.scatter([], [], color=SURFACE, edgecolor=C[0], s=40, label="Pulled towards the EU average")
    ax.legend(loc="lower right", fontsize=9)
    note(fig, "Source: European Commission, AMECO (autumn 2026); author's estimates, 1996–2025.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig2_eu_responses.png", dpi=200)
    plt.close(fig)


def fig_rolling(est: dict):
    r = pd.DataFrame(est["rolling15"]).T
    fig, ax = plt.subplots(figsize=(10, 5))
    x = r.index.astype(int)
    ax.fill_between(x, r.rho - 1.96 * r.se, r.rho + 1.96 * r.se, color="#cde2fb", linewidth=0, label="95% interval")
    ax.plot(x, r.rho, color=C[0], linewidth=2.2, marker="o", markersize=5, label="Estimate")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.axhline(est["main"]["b"]["debt_l"], color=C[1], linewidth=1.4, linestyle=(0, (4, 3)),
               label="Full sample, 1996–2025")  # fmt: skip
    ax.set_xticks(list(range(2010, 2026, 2)))
    ax.set_xlabel("Last year of the 15-year window")
    ax.set_ylabel("Response to debt, pp per point of debt")
    ax.set_title("Portugal's response to debt, estimated on 15-year windows")
    ax.legend(loc="lower left", fontsize=9)
    note(fig, "Source: European Commission, AMECO (autumn 2026); author's estimates.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig3_rolling_response.png", dpi=200)
    plt.close(fig)


def fig_paths(paths: dict, labels: dict, p: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    hist_years = list(range(2015, 2026))
    ax.plot(hist_years, p.loc[2015:2025, "debt"].values, color=INK, linewidth=2.2, label="Actual, 2015–2025")
    for k, key in enumerate(["fixed", "anch", "hist"]):
        yrs = [2025, 2026] + M.YEARS
        vals = [p.loc[2025, "debt"], M.D0] + [paths[key][y] for y in M.YEARS]
        ax.plot(yrs, vals, color=C[k], linewidth=2.2, label=labels[key])
    ax.axhline(60, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(2015.2, 61.2, "60% reference value", color=INK2, fontsize=9.5)
    ax.axvline(2035, color=GRID, linewidth=1.5, zorder=0)
    ax.set_ylim(40, 140)
    ax.set_xlim(2015, 2045)
    ax.set_ylabel("Gross public debt, % of GDP")
    ax.set_title("Debt to 2045 under three ways of setting the primary balance, no shocks")
    ax.legend(loc="upper right", fontsize=9.3)
    note(fig, "Source: European Commission, AMECO; Orçamento do Estado 2027; author's model and estimates.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig4_debt_paths.png", dpi=200)
    plt.close(fig)


def fig_prob(summ: dict, labels: dict):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for k, key in enumerate(["fixed", "anch", "hist"]):
        pr = summ[key]["p_by_year"]
        yrs = sorted(pr)
        ax.plot(yrs, [100 * pr[y] for y in yrs], color=C[k], linewidth=2.2, marker="o", markersize=4.5,
                label=labels[key])  # fmt: skip
    ax.axvline(2035, color=GRID, linewidth=1.5, zorder=0)
    ax.set_ylim(0, 100)
    ax.set_xlim(2027, 2045)
    ax.set_ylabel("Share of simulated futures at or below 60%")
    ax.set_title("The chance of being at or below 60% of GDP, year by year")
    ax.legend(loc="upper left", fontsize=9.3)
    note(
        fig,
        "Source: author's model, 10,000 paths per policy; rule coefficients drawn from their estimated uncertainty.",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig5_probability_by_year.png", dpi=200)
    plt.close(fig)


def fig_dial(dial: dict, marks: dict):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    lv = sorted(dial)
    ax.plot(lv, [100 * dial[x] for x in lv], color=C[0], linewidth=2.4, marker="o", markersize=6)
    for n_, (x, lab) in enumerate(marks.items()):
        ax.axvline(x, color=INK2, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.text(x + 0.05, 96 - 9 * (n_ % 2), lab, color=INK2, fontsize=8.8, va="top")
    ax.axhline(50, color=GRID, linewidth=1.5, zorder=0)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Primary balance the rule still asks for once debt is at 60% of GDP, % of GDP")
    ax.set_ylabel("Probability debt is at or below 60% in 2035")
    ax.set_title("What the rule must still demand at low debt")
    note(
        fig,
        f"Source: author's model, {N_DIAL:,} paths per level; Portugal's estimated response to debt and to the cycle.",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig6_rule_level_dial.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    panel = pd.read_csv(HERE / "data" / "panel.csv")
    p = portugal(panel)
    est = estimate(panel, p)
    pols = policies(est)
    rho_gap, kappa, shocks = historical_shocks(p, est["resid"], M.Policy("x", "x").h)
    for k in pols:
        pols[k] = pols[k].copy(rho_gap=rho_gap)

    det = {}
    for key, pol in pols.items():
        r = M.run(pol)
        hit = r.index[r["debt"] <= 60]
        det[key] = {
            "label": pol.label,
            "debt": {int(y): float(v) for y, v in r["debt"].items()},
            "pb": {int(y): float(v) for y, v in r["primary_balance"].items()},
            "first_year_at_or_below_60": int(hit[0]) if len(hit) else None,
        }

    summ = {}
    for i_, (key, pol) in enumerate(pols.items()):
        est_ = est if pol.kind == "rule" else None
        d, b_, pb_ = simulate(pol, shocks, est_, N_SIM, SEED + i_, anchored=(key == "anch"))
        summ[key] = summarise(d, b_, pb_)

    b = est["main"]["b"]
    dial = {}
    for x in np.round(np.arange(-2.5, 4.01, 0.5), 2):
        a = M.anchored_intercept(b["debt_l"], b["pb_l"], 60.0, float(x))
        pol = pols["hist"].copy(a=a)
        d, _, _ = simulate(pol, shocks, None, N_DIAL, SEED + 10)
        dial[float(x)] = float((d[:, M.YEARS.index(2035)] <= 60).mean())
    lr_slope = b["debt_l"] / (1 - b["pb_l"])
    lv = sorted(dial)
    hit = [x for x in lv if dial[x] >= 0.5]
    x50 = None
    if hit and hit[0] > lv[0]:
        x1 = lv[lv.index(hit[0]) - 1]
        x50 = x1 + (hit[0] - x1) * (0.5 - dial[x1]) / (dial[hit[0]] - dial[x1])
    marks = {
        est["long_run"]["pb_at"]["60"]["point"]: "Portugal's\naverage rule",
        M.PB_2027 - lr_slope * (DEBT_2027 - 60): "Today's surplus,\nsame response",
    }
    if x50 is not None:
        marks[x50] = "Even chance\nby 2035"

    res = {
        "meta": {"seed": SEED, "n_sim": N_SIM, "n_dial": N_DIAL, "block": BLOCK, "rho_gap": rho_gap, "kappa": kappa},
        "portugal": {
            "debt": {int(k): float(v) for k, v in p["debt"].loc[1995:2025].items()},
            "pb": {int(k): float(v) for k, v in p["pb"].loc[1995:2025].items()},
        },
        "estimates": {k: v for k, v in est.items() if k != "resid"},
        "deterministic": det,
        "monte_carlo": summ,
        "dial": dial,
        "dial_marks": {f"{k:.3f}": v for k, v in marks.items()},
        "dial_first_level_reaching_50pct": hit[0] if hit else None,
        "dial_level_for_even_chance": x50,
        "dial_even_chance_pb_at_2027_debt": (x50 + lr_slope * (DEBT_2027 - 60)) if x50 is not None else None,
        "long_run_slope": lr_slope,
    }
    (RES / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    write_summary(res)

    fig_rule(p, est)
    fig_eu(est)
    fig_rolling(est)
    fig_paths({k: v["debt"] for k, v in det.items()}, {k: v.label for k, v in pols.items()}, p)
    fig_prob(summ, {k: v.label for k, v in pols.items()})
    fig_dial(dial, marks)
    print((RES / "summary.txt").read_text())


def write_summary(res: dict) -> None:
    e = res["estimates"]
    m, se = e["main"]["b"], e["main"]["se"]
    L = [
        "portugal rule 1996-2025: "
        + ", ".join(f"{k}={m[k]:.3f} (se {se[k]:.3f})" for k in m)
        + f"; n={e['main']['n']}, r2={e['main']['r2']:.3f}",
        f"long-run slope {e['long_run']['slope']:.3f} (90%: {e['long_run']['slope_ci'][0]:.3f} to "
        f"{e['long_run']['slope_ci'][1]:.3f})",
    ]
    for d, v in e["long_run"]["pb_at"].items():
        L.append(f"long-run pb at debt {d}: {v['point']:.2f} (90%: {v['ci90'][0]:.2f} to {v['ci90'][1]:.2f})")
    lr = e["long_run"]
    L.append(
        f"anchor: {lr['anchor_point']:.1f} (90%: {lr['anchor_ci90'][0]:.1f} to {lr['anchor_ci90'][1]:.1f}); "
        f"p(anchor > 60) {lr['p_anchor_above_60']:.3f}; i_lr {lr['i_lr']:.2f}, g_lr {lr['g_lr']:.2f}"
    )
    L.append(f"one-step 2026 from 2025 actuals: {e['one_step_2026']:.2f}")
    L.append("residuals 2016-2025: " + ", ".join(f"{k}={v:.2f}" for k, v in e["resid_recent"].items()))
    L.append(f"residual mean 2016-2025 ex 2020-21: {e['resid_mean_2016_2025_ex_pandemic']:.2f}")
    for k, v in e["robustness"].items():
        L.append(f"robustness {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in v.items()))
    L.append("rolling15 rho: " + ", ".join(f"{k}={v['rho']:.3f}" for k, v in e["rolling15"].items()))
    eu = e["eu"]
    L.append(
        f"eu: re mean {eu['re_mean']:.4f} (se {eu['re_se']:.4f}), tau {eu['tau']:.4f}, positive&sig "
        f"{eu['n_positive_significant']}, negative {eu['n_negative']}, portugal rank {eu['rank_portugal']}/27, "
        f"median lr {eu['median_lr']:.3f}"
    )
    pt = eu["countries"]["PRT"]
    L.append(f"eu portugal: rho {pt['rho']:.4f} (se {pt['se']:.4f}), eb {pt['eb']:.4f} (se {pt['eb_se']:.4f})")
    fe = e["pooled_fe"]
    L.append("pooled fe: " + ", ".join(f"{k}={fe['b'][k]:.4f} (se {fe['se'][k]:.4f})" for k in fe["b"]))
    for k, v in res["deterministic"].items():
        d, pb = v["debt"], v["pb"]
        L.append(
            f"deterministic {k}: debt 2030 {d[2030]:.1f} 2035 {d[2035]:.1f} 2045 {d[2045]:.1f}; pb 2028 "
            f"{pb[2028]:.2f} 2030 {pb[2030]:.2f} 2035 {pb[2035]:.2f}; first<=60 {v['first_year_at_or_below_60']}"
        )
    keys = (
        "p_at_or_below_60_in_2035", "p_at_or_below_70_in_2035", "p_above_2027_in_2035",
        "p_deficit_above_3_by_2035", "median_2035", "p10_2035", "p90_2035", "median_2045",
        "median_pb_2030", "median_pb_2035", "median_first_year_at_or_below_60", "p_not_at_60_by_2045",
    )  # fmt: skip
    for k, v in res["monte_carlo"].items():
        L.append(f"monte carlo {k}: " + ", ".join(f"{q}={v[q]:.3f}" for q in keys))
        L.append(f"monte carlo {k} p by year: " + ", ".join(f"{y}={p:.3f}" for y, p in v["p_by_year"].items()))
    L.append("dial (pb at 60% debt -> p<=60 in 2035): " + ", ".join(f"{x}={p:.3f}" for x, p in res["dial"].items()))
    L.append("dial marks: " + ", ".join(f"{k}={v.replace(chr(10), ' ')}" for k, v in res["dial_marks"].items()))
    L.append(f"dial first level reaching 50%: {res['dial_first_level_reaching_50pct']}")
    if res["dial_level_for_even_chance"] is not None:
        L.append(
            f"dial even chance at pb {res['dial_level_for_even_chance']:.2f} at 60% debt, "
            f"{res['dial_even_chance_pb_at_2027_debt']:.2f} at {DEBT_2027}% debt"
        )
    (RES / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
