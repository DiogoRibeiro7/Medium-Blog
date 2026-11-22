"""Real-time versus current estimates of the output gap and the structural balance.

Vocabulary used throughout:

- a release is one AMECO vintage (spring or autumn of a year); the current release (autumn 2026) is the
  benchmark, called "latest";
- the age of an estimate is the release year minus the target year: -1 is the autumn forecast for the
  following year (the one a budget is built on), 0 is the autumn estimate for the current year, 1 the
  estimate a year after, and so on;
- the error of an estimate is the estimate minus the latest estimate for the same country and year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LATEST = (2026, "autumn")
MATURE = 2022  # target years whose latest estimate is at least four years old


def latest(v: pd.DataFrame, var: str) -> pd.Series:
    s = v[(v.vintage_year == LATEST[0]) & (v.season == LATEST[1]) & (v["var"] == var)]
    return s.set_index(["country", "year"])["value"]


def estimates(v: pd.DataFrame, var: str, age: int, season: str = "autumn") -> pd.DataFrame:
    """Every archived estimate of `var` at the given age, with the latest estimate and the error."""
    s = v[(v.season == season) & (v["var"] == var) & (v.vintage_year != LATEST[0])]
    s = s[(s.vintage_year - s.year) == age]
    s = s[s.year <= MATURE]
    lt = latest(v, var)
    out = s[["country", "year", "vintage_year", "value"]].rename(columns={"value": "estimate"}).copy()
    out["latest"] = [lt.get((c, y), np.nan) for c, y in zip(out.country, out.year, strict=True)]
    out = out.dropna(subset=["latest"])
    out["error"] = out["estimate"] - out["latest"]
    return out.reset_index(drop=True)


def summary(err: pd.Series) -> dict:
    return {
        "n": int(err.size),
        "bias": float(err.mean()),
        "sd": float(err.std(ddof=1)) if err.size > 1 else float("nan"),
        "mae": float(err.abs().mean()),
        "p10": float(err.quantile(0.1)),
        "p25": float(err.quantile(0.25)),
        "median": float(err.median()),
        "p75": float(err.quantile(0.75)),
        "p90": float(err.quantile(0.9)),
    }


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = e @ e / max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se, e


def with_innovation(df: pd.DataFrame, lt_gap: pd.Series, rho: float) -> pd.DataFrame:
    """Add the year's innovation in the latest output gap, latest(t) - rho * latest(t-1)."""
    out = df.copy()
    prev = np.array([lt_gap.get((c, y - 1), np.nan) for c, y in zip(out.country, out.year, strict=True)])
    out["innovation"] = out["latest"].to_numpy() - rho * prev
    return out.dropna(subset=["innovation"]).reset_index(drop=True)


def miss_share(df: pd.DataFrame) -> dict:
    """Regress latest - estimate on the year's innovation: the slope is the share of the innovation missed.

    The constant is minus the bias; the residual is the part of the error unrelated to the year's surprise."""
    y = (df["latest"] - df["estimate"]).to_numpy()
    X = np.column_stack([np.ones(len(df)), df["innovation"].to_numpy()])
    b, se, e = ols(y, X)
    return {"c": float(b[1]), "c_se": float(se[1]), "const": float(b[0]), "resid_sd": float(e.std(ddof=2)), "n": len(y)}


def cluster_bootstrap_c(df: pd.DataFrame, reps: int, seed: int) -> list[float]:
    """Resample countries with replacement and re-estimate the share missed: a 90% interval robust to
    errors that are correlated within a country."""
    rng = np.random.default_rng(seed)
    groups = {c: g for c, g in df.groupby("country")}
    names = list(groups)
    out = []
    for _ in range(reps):
        pick = rng.choice(names, size=len(names), replace=True)
        d = pd.concat([groups[c] for c in pick], ignore_index=True)
        out.append(miss_share(d)["c"])
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))]


def attenuation(df: pd.DataFrame) -> dict:
    """estimate = a + lam * latest: lam below one means real-time gaps are compressed towards zero."""
    X = np.column_stack([np.ones(len(df)), df["latest"].to_numpy()])
    b, se, _ = ols(df["estimate"].to_numpy(), X)
    return {"a": float(b[0]), "lam": float(b[1]), "lam_se": float(se[1]), "n": len(df)}


def residual_persistence(df: pd.DataFrame, c: float, bias: float) -> dict:
    """AR(1) coefficient of the error left after the bias and the missed share, across consecutive years."""
    d = df.copy()
    d["nu"] = d["estimate"] - d["latest"] + c * d["innovation"] - bias
    d = d.sort_values(["country", "year"])
    d["nu_prev"] = d.groupby("country")["nu"].shift(1)
    d["gap_years"] = d.groupby("country")["year"].diff()
    d = d[(d.gap_years == 1)].dropna(subset=["nu_prev"])
    if len(d) < 3:
        return {"rho": 0.0, "n": len(d)}
    r = float(np.corrcoef(d["nu"], d["nu_prev"])[0, 1])
    return {"rho": r, "n": len(d)}


def decomposition(v: pd.DataFrame) -> pd.DataFrame:
    """Revisions to the current-year estimate split into real growth and potential growth.

    For each country and year with an autumn estimate in the same year (age 0): the revision to real
    growth, to potential growth, to the change in the gap, and to the level of the gap."""
    a = v[(v.season == "autumn") & (v.vintage_year != LATEST[0])]
    a = a.pivot_table(index=["vintage_year", "country", "year"], columns="var", values="value")
    lt = v[(v.vintage_year == LATEST[0]) & (v.season == LATEST[1])]
    lt = lt.pivot_table(index=["country", "year"], columns="var", values="value")
    rows = []
    for (vy, c, y), r in a.iterrows():
        if (
            y != vy
            or y > MATURE
            or (vy, c, y - 1) not in a.index
            or (c, y) not in lt.index
            or (c, y - 1) not in lt.index
        ):
            continue
        p, l1, l0 = a.loc[(vy, c, y - 1)], lt.loc[(c, y)], lt.loc[(c, y - 1)]
        rows.append(
            {
                "country": c,
                "year": y,
                "growth_rt": 100 * (r["Y"] / p["Y"] - 1),
                "growth_latest": 100 * (l1["Y"] / l0["Y"] - 1),
                "potential_rt": 100 * (r["Ypot"] / p["Ypot"] - 1),
                "potential_latest": 100 * (l1["Ypot"] / l0["Ypot"] - 1),
                "dgap_rt": r["gap"] - p["gap"],
                "dgap_latest": l1["gap"] - l0["gap"],
                "gap_rt": r["gap"],
                "gap_latest": l1["gap"],
            }
        )
    d = pd.DataFrame(rows).dropna()
    d["growth_rev"] = d.growth_latest - d.growth_rt
    d["potential_rev"] = d.potential_latest - d.potential_rt
    d["dgap_rev"] = d.dgap_latest - d.dgap_rt
    d["level_rev"] = d.gap_latest - d.gap_rt
    return d.reset_index(drop=True)


def structural_effort(v: pd.DataFrame, var: str, benchmark: float = 0.5) -> pd.DataFrame:
    """The year's change in a structural measure as seen in the autumn release of that year, and now.

    `flip` marks the cases where the verdict on "an improvement of at least `benchmark` points" differs."""
    a = v[(v.season == "autumn") & (v.vintage_year != LATEST[0]) & (v["var"] == var)]
    a = a.set_index(["vintage_year", "country", "year"])["value"]
    lt = latest(v, var)
    rows = []
    for (vy, c, y), val in a.items():
        if y != vy or y > MATURE or (vy, c, y - 1) not in a.index:
            continue
        if (c, y) not in lt.index or (c, y - 1) not in lt.index:
            continue
        rt = val - a[(vy, c, y - 1)]
        now = lt[(c, y)] - lt[(c, y - 1)]
        rows.append({"country": c, "year": y, "realtime": rt, "now": now})
    d = pd.DataFrame(rows).dropna()
    d["error"] = d.realtime - d.now
    d["flip"] = (d.realtime >= benchmark) != (d.now >= benchmark)
    return d.reset_index(drop=True)
