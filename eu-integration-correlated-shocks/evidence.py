"""EU-27 evidence: how correlated are national supply shocks, and how synchronised are prices?

    python evidence.py

Writes ``results/evidence_summary.txt`` and ``results/evidence_*.csv``.

Shocks are deviations of log national cereal production from a Hodrick-Prescott trend
(lambda = 100, the usual value for annual data). Malta, which grows no cereals, is
dropped. Correlation uncertainty comes from a moving-block bootstrap over years.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import multivariate_normal, norm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"

EU27 = (
    "BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV",
    "LT", "LU", "HU", "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE",
)
NAMES = {
    "BE": "Belgium", "BG": "Bulgaria", "CZ": "Czechia", "DK": "Denmark", "DE": "Germany", "EE": "Estonia",
    "IE": "Ireland", "EL": "Greece", "ES": "Spain", "FR": "France", "HR": "Croatia", "IT": "Italy",
    "CY": "Cyprus", "LV": "Latvia", "LT": "Lithuania", "LU": "Luxembourg", "HU": "Hungary", "MT": "Malta",
    "NL": "Netherlands", "AT": "Austria", "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SI": "Slovenia",
    "SK": "Slovakia", "FI": "Finland", "SE": "Sweden", "EU27_2020": "EU-27",
}
SHARE_YEARS = (2015, 2024)


def hp_trend(y: np.ndarray, lam: float = 100.0) -> np.ndarray:
    n = len(y)
    eye = sparse.eye(n, format="csc")
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    return spsolve((eye + lam * d2.T @ d2).tocsc(), np.asarray(y, dtype=float))


def cereal_production() -> pd.DataFrame:
    d = pd.read_csv(DATA / "crops.csv", low_memory=False)
    p = d[(d.crops == "C0000") & (d.strucpro == "HPRD_HUMD_EU_THS_T")].pivot_table(index="time", columns="geo", values="value")
    return p.astype(float)


def shocks(prod: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Log deviations from an HP trend, by country (Malta dropped) and for the EU aggregate."""
    nat = prod.drop(columns=["EU27_2020", "MT"])
    logp = np.log(nat)
    s = logp - logp.apply(lambda c: pd.Series(hp_trend(c.values), index=c.index))
    eu = np.log(prod["EU27_2020"])
    s_eu = eu - hp_trend(eu.values)
    return s, s_eu


def production_weights(prod: pd.DataFrame) -> pd.Series:
    nat = prod.drop(columns=["EU27_2020", "MT"]).loc[SHARE_YEARS[0] : SHARE_YEARS[1]].mean()
    return nat / nat.sum()


def pairwise(c: np.ndarray) -> np.ndarray:
    return c[np.triu_indices(c.shape[0], 1)]


def diversification(s: pd.DataFrame, w: pd.Series) -> dict:
    sig = s.std()
    cov = s.cov().values
    wv = w[s.columns].values
    v = float(wv @ cov @ wv)
    avg_sd = float((w[s.columns] * sig).sum())
    return {
        "eu_sd_from_cov": np.sqrt(v),
        "weighted_national_sd": avg_sd,
        "diversification_ratio": np.sqrt(v) / avg_sd,
        "n_eff": avg_sd**2 / v,
        "n_eff_if_independent": avg_sd**2 / float(np.sum(wv**2 * sig.values**2)),
    }


def block_bootstrap(s: pd.DataFrame, w: pd.Series, n: int = 2000, block: int = 3, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = len(s)
    out = []
    for _ in range(n):
        idx = []
        while len(idx) < T:
            start = rng.integers(0, T - block + 1)
            idx.extend(range(start, start + block))
        b = s.iloc[idx[:T]]
        c = b.corr().values
        dv = diversification(b, w)
        out.append({"mean_corr": pairwise(c).mean(), "n_eff": dv["n_eff"], "diversification_ratio": dv["diversification_ratio"]})
    return pd.DataFrame(out)


def tail_coexceedance(s: pd.DataFrame, q: float = 0.2) -> dict:
    """Average over pairs of P(both in their bottom q) / q, empirical and implied by a Gaussian copula."""
    u = s.rank(pct=True)
    c = s.corr().values
    iu = np.triu_indices(s.shape[1], 1)
    z = norm.ppf(q)
    emp_lo, emp_hi, gau = [], [], []
    for i, j in zip(*iu):
        a, b = u.iloc[:, i], u.iloc[:, j]
        emp_lo.append(((a <= q) & (b <= q)).mean() / q)
        emp_hi.append(((a > 1 - q) & (b > 1 - q)).mean() / q)
        r = c[i, j]
        gau.append(multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([z, z]) / q)
    return {"empirical_lower": float(np.mean(emp_lo)), "empirical_upper": float(np.mean(emp_hi)), "gaussian": float(np.mean(gau)), "independent": q}


def crop_shocks(crop: str, detrend: str = "hp") -> pd.DataFrame:
    """Shocks for one crop code, countries with a complete 2000-2024 series and non-zero output."""
    d = pd.read_csv(DATA / "crops.csv", low_memory=False)
    p = d[(d.crops == crop) & (d.strucpro == "HPRD_HUMD_EU_THS_T")].pivot_table(index="time", columns="geo", values="value")
    p = p.loc[2000:2024].drop(columns=[c for c in ("EU27_2020", "MT") if c in p.columns]).astype(float)
    p = p.loc[:, (p.notna().all()) & (p.min() > 0)]
    return detrended(np.log(p), detrend)


def detrended(logp: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "hp":
        return logp - logp.apply(lambda c: pd.Series(hp_trend(c.values), index=c.index))
    if how == "quadratic":
        t = np.arange(len(logp))
        X = np.column_stack([np.ones_like(t), t, t**2]).astype(float)
        beta, *_ = np.linalg.lstsq(X, logp.values, rcond=None)
        return logp - X @ beta
    if how == "diff":
        return logp.diff().dropna()
    raise ValueError(how)


def bloc_correlations(s: pd.DataFrame, k: int = 7) -> dict:
    """Mean correlation within the two ends of the second principal component and across them."""
    c = s.corr()
    ev, evec = np.linalg.eigh(c.values)
    pc2 = pd.Series(evec[:, -2], index=c.index).drop("CY", errors="ignore").sort_values()
    a, b = list(pc2.index[:k]), list(pc2.index[-k:])
    within = lambda g: pairwise(c.loc[g, g].values).mean()
    return {"bloc_a": a, "bloc_b": b, "within_a": within(a), "within_b": within(b), "across": float(c.loc[a, b].values.mean())}


def food_inflation() -> pd.DataFrame:
    f = pd.read_csv(DATA / "food_prices.csv")
    f = f[(f.unit == "INX_A_AVG") & (f.coicop == "CP011")].pivot_table(index="time", columns="geo", values="value")
    cols = [c for c in f.columns if c in EU27]
    return np.log(f[cols]).diff().loc[2001:2025]


def house_price_growth() -> pd.DataFrame:
    h = pd.read_csv(DATA / "house_prices.csv").pivot_table(index="time", columns="geo", values="value")
    cols = [c for c in h.columns if c in EU27]
    return np.log(h[cols]).diff()


def common_factor_share(x: pd.DataFrame) -> tuple[float, float, int]:
    x = x.dropna(axis=1)
    c = np.corrcoef(((x - x.mean()) / x.std()).T.values)
    ev = np.linalg.eigvalsh(c)
    return float(ev[-1] / ev.sum()), float(pairwise(c).mean()), x.shape[1]


def construction() -> pd.DataFrame:
    e = pd.read_csv(DATA / "employment_industry.csv")
    F = e[(e.nace_r2 == "F")].pivot_table(index="time", columns="geo", values="value")
    return F[[c for c in F.columns if c in EU27]]


def construction_asymmetry(F: pd.DataFrame) -> pd.DataFrame:
    """Peak in 2004-2010, trough after it, average log rates down and up to the last year."""
    rows = []
    for c in F.columns:
        s = F[c].dropna().loc[2000:]
        if len(s) < 20:
            continue
        pk = int(s.loc[2004:2010].idxmax())
        tr = int(s.loc[pk:].idxmin())
        if tr <= pk:
            continue
        last = int(s.index[-1])
        down = np.log(s[tr] / s[pk]) / (tr - pk)
        up = np.log(s[last] / s[tr]) / (last - tr) if last > tr else np.nan
        rows.append({"geo": c, "peak": pk, "trough": tr, "drop": s[tr] / s[pk] - 1, "down_rate": down, "up_rate": up, "last_vs_peak": s[last] / s[pk], "last": last})
    return pd.DataFrame(rows).set_index("geo")


def farm_ages() -> pd.DataFrame:
    f = pd.read_csv(DATA / "farm_managers_age.csv", low_memory=False)
    p = f.pivot_table(index=["geo", "time"], columns="age", values="value")
    p["ge65_share"] = 100 * p["Y_GE65"] / p["TOTAL"]
    p["lt35_share"] = 100 * p[["Y_LT25", "Y25-34"]].sum(axis=1) / p["TOTAL"]
    latest = p.reset_index().sort_values("time").groupby("geo").tail(1).set_index("geo")
    return latest[["time", "TOTAL", "ge65_share", "lt35_share"]]


def farm_output_ages() -> pd.DataFrame:
    """Share of standard output on holdings whose manager is 65+, latest wave per country."""
    f = pd.read_csv(DATA / "farm_output_by_age.csv", low_memory=False)
    p = f.pivot_table(index=["geo", "time"], columns="age", values="value")
    p["ge65_output_share"] = 100 * p["Y_GE65"] / p["TOTAL"]
    p["lt35_output_share"] = 100 * p[["Y_LT25", "Y25-34"]].sum(axis=1) / p["TOTAL"]
    latest = p.reset_index().sort_values("time").groupby("geo").tail(1).set_index("geo")
    return latest[["time", "ge65_output_share", "lt35_output_share"]]


def agri_labour() -> pd.DataFrame:
    a = pd.read_csv(DATA / "agri_labour_input.csv").pivot_table(index="time", columns=["geo", "am_item"], values="value")
    out = {}
    for g in [c for c in a.columns.get_level_values(0).unique() if c in EU27 or c == "EU27_2020"]:
        try:
            tot = a[(g, "AM400000")]
            nons = a[(g, "AM410000")]
        except KeyError:
            continue
        out[g] = {
            "total_2005": tot.get(2005, np.nan),
            "total_2025": tot.get(2025, np.nan),
            "nonsal_2005": nons.get(2005, np.nan),
            "nonsal_2025": nons.get(2025, np.nan),
        }
    df = pd.DataFrame(out).T
    df["total_change"] = df.total_2025 / df.total_2005 - 1
    df["nonsal_change"] = df.nonsal_2025 / df.nonsal_2005 - 1
    return df


def latest_by_geo(name: str, **filters) -> pd.Series:
    d = pd.read_csv(DATA / f"{name}.csv", low_memory=False)
    for k, v in filters.items():
        d = d[d[k] == v]
    d = d.sort_values("time").groupby("geo").tail(1)
    return d.set_index("geo")["value"]


def country_table(prod: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Everything the multi-country model needs, one row per member state."""
    pop = pd.read_csv(DATA / "population.csv")
    pop = pop[pop.time == 2024].set_index("geo")["value"]
    pop = pop[[c for c in EU27]]
    nat = prod.drop(columns=["EU27_2020"]).loc[SHARE_YEARS[0] : SHARE_YEARS[1]].mean()
    nat = nat.reindex(list(EU27)).fillna(0.0)
    df = pd.DataFrame({"population": pop, "production": nat})
    df["pop_share"] = df.population / df.population.sum()
    df["prod_share"] = df.production / df.production.sum()
    df["self_sufficiency"] = df.prod_share / df.pop_share
    df["shock_sd"] = s.std().reindex(df.index)
    md = latest_by_geo("meal_deprivation", hhcomp="TOTAL", rskpovth="TOTAL")
    df["meal_deprivation"] = md.reindex(df.index) / 100.0
    df["gini"] = latest_by_geo("gini", age="TOTAL").reindex(df.index) / 100.0
    return df


def main() -> None:
    RES.mkdir(exist_ok=True)
    L: list[str] = []
    say = L.append
    prod = cereal_production()
    s, s_eu = shocks(prod)
    w = production_weights(prod)
    s.to_csv(RES / "evidence_cereal_shocks.csv")
    s_eu.rename("EU27").to_csv(RES / "evidence_eu_shock.csv")

    say("== Cereal production shocks, 26 member states, 2000-2024 (log deviation from HP trend, lambda 100)")
    say(f"years {int(s.index.min())}-{int(s.index.max())}; countries {s.shape[1]}")
    sig = s.std().sort_values()
    say("national shock sd: " + ", ".join(f"{k} {v:.3f}" for k, v in sig.items()))
    say(f"production-weighted mean national sd: {(w * s.std()).sum():.4f}; EU aggregate shock sd (from EU series): {s_eu.std():.4f}")
    c = s.corr().values
    pw = pairwise(c)
    W = np.outer(w[s.columns], w[s.columns])
    np.fill_diagonal(W, 0)
    say(f"mean pairwise correlation {pw.mean():.3f} (median {np.median(pw):.3f}); production-weighted {float((c * W).sum() / W.sum()):.3f}")
    dv = diversification(s, w)
    say("diversification: " + ", ".join(f"{k} {v:.4f}" for k, v in dv.items()))
    bs = block_bootstrap(s, w)
    for k in ("mean_corr", "n_eff", "diversification_ratio"):
        say(f"  bootstrap 90% interval, {k}: {bs[k].quantile(0.05):.3f} to {bs[k].quantile(0.95):.3f}")
    for a, b in ((2000, 2008), (2008, 2016), (2016, 2024), (2000, 2012), (2012, 2024)):
        sub = s.loc[a:b]
        say(f"  mean pairwise correlation {a}-{b}: {pairwise(sub.corr().values).mean():.3f}; n_eff {diversification(sub, w)['n_eff']:.2f}")
    tc = tail_coexceedance(s)
    say("tail co-exceedance at q = 0.2: " + ", ".join(f"{k} {v:.3f}" for k, v in tc.items()))
    ev, evec = np.linalg.eigh(c)
    say(f"principal components of the correlation matrix: variance shares {', '.join(f'{x:.3f}' for x in (ev[::-1][:3] / ev.sum()))}")
    pc2 = pd.Series(evec[:, -2], index=s.columns).sort_values()
    say("second component (sign arbitrary): " + ", ".join(f"{k} {v:+.2f}" for k, v in pc2.items()))
    say("EU aggregate shocks by year: " + ", ".join(f"{int(y)} {v:+.3f}" for y, v in s_eu.items()))
    shares = prod.drop(columns=["EU27_2020", "MT"])
    shares = shares.div(shares.sum(axis=1), axis=0)
    hhi = (shares**2).sum(axis=1)
    say(f"concentration of production (HHI): 2000 {hhi.loc[2000]:.4f}, 2010 {hhi.loc[2010]:.4f}, 2024 {hhi.loc[2024]:.4f}; equivalent number of equal producers 2000 {1 / hhi.loc[2000]:.1f}, 2024 {1 / hhi.loc[2024]:.1f}")
    pd.DataFrame({"hhi": hhi, "equivalent_producers": 1 / hhi}).to_csv(RES / "evidence_concentration.csv")
    pd.DataFrame(c, index=s.columns, columns=s.columns).to_csv(RES / "evidence_correlation.csv")
    bs.to_csv(RES / "evidence_bootstrap.csv", index=False)

    bl = bloc_correlations(s)
    say(f"blocs from the second component: {', '.join(bl['bloc_a'])} | {', '.join(bl['bloc_b'])}")
    say(f"  mean correlation within first bloc {bl['within_a']:.3f}, within second {bl['within_b']:.3f}, across {bl['across']:.3f}")
    say("robustness to detrending (cereals): ")
    nat = prod.drop(columns=["EU27_2020", "MT"])
    for how in ("hp", "quadratic", "diff"):
        sh = detrended(np.log(nat), how)
        dvh = diversification(sh, w)
        say(f"  {how}: mean pairwise correlation {pairwise(sh.corr().values).mean():.3f}; n_eff {dvh['n_eff']:.2f}; diversification ratio {dvh['diversification_ratio']:.3f}")
    say("by crop (HP detrending; countries with complete, non-zero 2000-2024 series):")
    crop_rows = []
    for code, name in (("C0000", "cereals"), ("C1100", "wheat"), ("C1300", "barley"), ("C1500", "grain maize")):
        sh = crop_shocks(code)
        d = pd.read_csv(DATA / "crops.csv", low_memory=False)
        pp = d[(d.crops == code) & (d.strucpro == "HPRD_HUMD_EU_THS_T")].pivot_table(index="time", columns="geo", values="value")
        wc = pp.loc[SHARE_YEARS[0] : SHARE_YEARS[1], sh.columns].mean()
        wc = wc / wc.sum()
        dvc = diversification(sh, wc)
        mc_ = pairwise(sh.corr().values).mean()
        crop_rows.append({"crop": name, "countries": sh.shape[1], "mean_corr": mc_, "n_eff": dvc["n_eff"], "weighted_sd": dvc["weighted_national_sd"], "union_sd": dvc["eu_sd_from_cov"]})
        say(f"  {name}: {sh.shape[1]} countries; mean pairwise correlation {mc_:.3f}; weighted national sd {dvc['weighted_national_sd']:.3f}; union sd {dvc['eu_sd_from_cov']:.3f}; n_eff {dvc['n_eff']:.2f}")
    pd.DataFrame(crop_rows).to_csv(RES / "evidence_by_crop.csv", index=False)

    say("\n== Food prices (HICP CP011, annual log change)")
    fi = food_inflation()
    fi.to_csv(RES / "evidence_food_inflation.csv")
    for y in (2019, 2021, 2022, 2023, 2024):
        r = fi.loc[y].dropna()
        say(f"{y}: mean {r.mean():.3f}, min {r.min():.3f} ({r.idxmin()}), max {r.max():.3f} ({r.idxmax()}), countries {len(r)}")
    share, mc, n = common_factor_share(fi)
    say(f"food inflation 2001-2025: first component {share:.3f} of variance; mean pairwise correlation {mc:.3f}; {n} countries")
    sh_c, mc_c, n_c = common_factor_share(s)
    say(f"cereal production shocks: first component {sh_c:.3f}; mean pairwise correlation {mc_c:.3f}; {n_c} countries")
    say(f"EU cereal shock 2022: {s_eu.loc[2022]:+.3f}; 2003: {s_eu.loc[2003]:+.3f}; 2007: {s_eu.loc[2007]:+.3f}")

    say("\n== Housing")
    hg = house_price_growth()
    hg.to_csv(RES / "evidence_house_price_growth.csv")
    share_h, mc_h, n_h = common_factor_share(hg.loc[2011:2025])
    say(f"house price growth 2011-2025: first component {share_h:.3f}; mean pairwise correlation {mc_h:.3f}; {n_h} countries")
    F = construction()
    F.to_csv(RES / "evidence_construction_employment.csv")
    falling = (np.log(F).diff() < 0).mean(axis=1)
    say("share of member states with falling construction employment: " + ", ".join(f"{int(y)} {v:.2f}" for y, v in falling.loc[2006:2025].items()))
    falling.rename("share_falling").to_csv(RES / "evidence_construction_falling.csv")
    asym = construction_asymmetry(F)
    asym.to_csv(RES / "evidence_construction_asymmetry.csv")
    big = asym[asym["drop"] < -0.30].sort_values("drop")
    for g, r in big.iterrows():
        say(f"  {g}: peak {r.peak} trough {r.trough}, fall {r['drop']:.3f}, down {r.down_rate:.4f}/yr, up {r.up_rate:.4f}/yr, {int(r['last'])} vs peak {r.last_vs_peak:.3f}")
    say(f"  member states losing more than 30% of construction employment after 2004-2010 peak: {len(big)}")
    say(f"  median down rate among them {big.down_rate.median():.4f}, median up rate {big.up_rate.median():.4f}, median ratio {(-big.down_rate / big.up_rate).median():.2f}")

    say("\n== Farm managers and labour")
    fa = farm_ages()
    fa.to_csv(RES / "evidence_farm_ages.csv")
    for g in fa.sort_values("ge65_share", ascending=False).index:
        r = fa.loc[g]
        say(f"  {g} {int(r.time)}: 65+ {r.ge65_share:.1f}%, under 35 {r.lt35_share:.1f}%")
    fo = farm_output_ages()
    fo.to_csv(RES / "evidence_farm_output_ages.csv")
    both = fa.join(fo[["ge65_output_share", "lt35_output_share"]], how="left")
    for g in both.sort_values("ge65_output_share", ascending=False).index:
        r = both.loc[g]
        say(f"  {g}: share of standard output on holdings run by managers 65+ {r.ge65_output_share:.1f}% (share of holdings {r.ge65_share:.1f}%)")
    al = agri_labour()
    al.to_csv(RES / "evidence_agri_labour.csv")
    say(f"  EU-27 labour input 2005 -> 2025: total {al.loc['EU27_2020', 'total_change']:+.3f}, non-salaried {al.loc['EU27_2020', 'nonsal_change']:+.3f}")

    ct = country_table(prod, s)
    ct.to_csv(RES / "country_table.csv")
    say("\n== Country table for the model")
    for g, r in ct.sort_values("self_sufficiency", ascending=False).iterrows():
        say(
            f"  {g}: pop share {r.pop_share:.4f}, production share {r.prod_share:.4f}, self-sufficiency proxy {r.self_sufficiency:.2f}, "
            f"shock sd {r.shock_sd:.3f}, meal deprivation {r.meal_deprivation:.3f}, Gini {r.gini:.3f}"
        )
    text = "\n".join(L) + "\n"
    (RES / "evidence_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
