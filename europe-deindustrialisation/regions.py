"""What happened later to the regions that lost the most manufacturing jobs? (NUTS 2, 2000-2023)

    python regions.py

Exposure: the change in manufacturing employment (NACE C) between 2000 and 2008, as a share of
the region's total employment in 2000 (percentage points; negative = loss). Outcomes, measured
over 2008-2023 so that they come after the exposure window:

- GDP per head in PPS relative to the national average (log change),
- population (log change),
- employment rate of people aged 20-64 (change in points),
- unemployment rate (change in points),
- manufacturing employment itself (log change), to see whether the lost jobs came back.

Each outcome is regressed on the exposure with country fixed effects (so regions are compared
with other regions of the same country), controlling for the 2000 manufacturing share and the
2000 GDP per head relative to the national average. Standard errors are clustered by country.
The estimates describe which regions diverged; they are not causal effects of losing industry.
Writes ``results/regions_*.csv`` and ``results/regions_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"

EU27 = {"AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"}
T0, T1, T2 = 2000, 2008, 2023  # 2024 regional manufacturing employment is still sparse


def nuts2(name: str, **filters) -> pd.DataFrame:
    d = pd.read_csv(DATA / f"{name}.csv", low_memory=False)
    for k, v in filters.items():
        d = d[d[k] == v]
    d = d[(d.geo.str.len() == 4) & d.geo.str[:2].isin(EU27) & ~d.geo.str.endswith("Z")]
    return d.pivot_table(index="geo", columns="time", values="value")


def build() -> pd.DataFrame:
    emp_t = nuts2("regional_employment", nace_r2="TOTAL")
    emp_c = nuts2("regional_employment", nace_r2="C")
    gdp = nuts2("regional_gdp_per_head")
    pop = nuts2("regional_population")
    er = nuts2("regional_employment_rate")
    ur = nuts2("regional_unemployment")
    r = pd.DataFrame(index=emp_t.index)
    r["country"] = r.index.str[:2]
    r["manuf_share_2000"] = 100 * emp_c[T0] / emp_t[T0]
    r["manuf_share_2023"] = 100 * emp_c[T2] / emp_t[T2]
    r["exposure"] = 100 * (emp_c[T1] - emp_c[T0]) / emp_t[T0]
    r["manuf_jobs_0008"] = np.log(emp_c[T1] / emp_c[T0])
    r["manuf_jobs_0823"] = np.log(emp_c[T2] / emp_c[T1])
    r["total_jobs_0823"] = np.log(emp_t[T2] / emp_t[T1])
    # GDP per head relative to the national (employment-weighted) mean of NUTS 2 regions
    for y in (T0, T1, T2):
        g = gdp[y]
        w = emp_t[y].reindex(g.index)
        nat = (g * w).groupby(g.index.str[:2]).sum() / w.groupby(g.index.str[:2]).sum()
        r[f"rel_gdp_{y}"] = np.log(g / nat.reindex(g.index.str[:2]).to_numpy()).reindex(r.index)
    r["d_rel_gdp_0823"] = r[f"rel_gdp_{T2}"] - r[f"rel_gdp_{T1}"]
    r["d_log_pop_0823"] = np.log(pop[T2] / pop[T1]).reindex(r.index)
    r["d_emp_rate_0823"] = (er[T2] - er[T1]).reindex(r.index)
    r["d_unemp_0823"] = (ur[T2] - ur[T1]).reindex(r.index)
    return r


def fe_ols(d: pd.DataFrame, y: str, xs: list[str], group: str = "country") -> dict:
    """OLS with group fixed effects (within transformation) and group-clustered standard errors."""
    d = d.dropna(subset=[y] + xs + [group])
    d = d[d.groupby(group)[y].transform("size") > 1]
    Y = d[y] - d.groupby(group)[y].transform("mean")
    X = pd.concat([d[x] - d.groupby(group)[x].transform("mean") for x in xs], axis=1).to_numpy()
    Yv = Y.to_numpy()
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ Yv
    e = Yv - X @ b
    meat = np.zeros((len(xs), len(xs)))
    groups = d[group].to_numpy()
    G = len(np.unique(groups))
    for gname in np.unique(groups):
        m = groups == gname
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    n, k = X.shape
    adj = G / (G - 1) * (n - 1) / (n - k - G)
    V = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    return {"n": n, "groups": G, **{f"b_{x}": bi for x, bi in zip(xs, b)}, **{f"se_{x}": si for x, si in zip(xs, se)}}


def main() -> None:
    RES.mkdir(exist_ok=True)
    r = build()
    r.to_csv(RES / "regions_panel.csv", float_format="%.6g")
    controls = ["manuf_share_2000", f"rel_gdp_{T0}"]
    outcomes = {
        "manuf_jobs_0823": "manufacturing jobs 2008-2023 (log change)",
        "total_jobs_0823": "total jobs 2008-2023 (log change)",
        "d_rel_gdp_0823": "GDP per head relative to the national average 2008-2023 (log change)",
        "d_log_pop_0823": "population 2008-2023 (log change)",
        "d_emp_rate_0823": "employment rate 20-64, 2008-2023 (points)",
        "d_unemp_0823": "unemployment rate 2008-2023 (points)",
    }
    rows = []
    for y, label in outcomes.items():
        res = fe_ols(r, y, ["exposure"] + controls)
        res.update({"outcome": y, "label": label})
        rows.append(res)
    est = pd.DataFrame(rows)
    est.to_csv(RES / "regions_estimates.csv", index=False, float_format="%.6g")

    # descriptive: terciles of exposure within country, outcomes relative to the country mean
    r2 = r.dropna(subset=["exposure"]).copy()
    r2["tercile"] = r2.groupby("country").exposure.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["largest loss", "middle", "smallest loss"]) if len(s) >= 3 else np.nan)
    for c in outcomes:
        r2[f"{c}_vs_country"] = r2[c] - r2.groupby("country")[c].transform("mean")
    terc = r2.dropna(subset=["tercile"]).groupby("tercile", observed=True)[[f"{c}_vs_country" for c in outcomes] + ["exposure", "manuf_share_2000"]].mean()
    terc.to_csv(RES / "regions_terciles.csv", float_format="%.6g")

    L = []
    say = L.append
    v = r.dropna(subset=["exposure"])
    say(f"NUTS 2 regions with manufacturing employment in 2000 and 2008: {len(v)} in {v.country.nunique()} member states")
    say(f"Exposure (manufacturing jobs 2000-2008, % of 2000 employment): median {v.exposure.median():+.2f}, 10th percentile {v.exposure.quantile(0.1):+.2f}, 90th percentile {v.exposure.quantile(0.9):+.2f}; regions that lost manufacturing jobs: {(v.exposure < 0).mean():.2f}")
    say(f"Manufacturing share of employment, median region: 2000 {v.manuf_share_2000.median():.1f}%, 2023 {v.manuf_share_2023.median():.1f}%")
    say("\nWithin-country regressions on the 2000-2008 exposure (per percentage point of 2000 employment lost; negative exposure = loss)")
    say("controls: manufacturing share 2000, GDP per head relative to the national average 2000; country fixed effects; SE clustered by country")
    for _, e in est.iterrows():
        b, se = e.b_exposure, e.se_exposure
        say(f"  {e.label}: coefficient {b:+.4f} (SE {se:.4f}, t {b / se:+.1f}); n={int(e.n)}, countries {int(e.groups)}")
    say("\nWithin-country terciles of exposure: mean outcome relative to the country mean")
    for t, row in terc.iterrows():
        say(
            f"  {t}: exposure {row.exposure:+.2f} pp, manufacturing share 2000 {row.manuf_share_2000:.1f}%; manufacturing jobs 2008-23 {row['manuf_jobs_0823_vs_country']:+.3f}, total jobs {row['total_jobs_0823_vs_country']:+.3f}, "
            f"relative GDP per head {row['d_rel_gdp_0823_vs_country']:+.3f}, population {row['d_log_pop_0823_vs_country']:+.3f}, employment rate {row['d_emp_rate_0823_vs_country']:+.2f}, unemployment {row['d_unemp_0823_vs_country']:+.2f}"
        )
    text = "\n".join(L) + "\n"
    (RES / "regions_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
