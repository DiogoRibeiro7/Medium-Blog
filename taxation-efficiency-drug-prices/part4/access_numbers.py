"""Part 4 of the taxation-efficiency-drug-prices series: what the wait for new medicines in Portugal is
made of, and whether price, market size or distance explains it.

Every number quoted in the article is printed by this script. Run, after part4/fetch.py has put the
public files in part4/data (the GitHub Actions workflow does this and commits them):

    python access_numbers.py > output.txt

The question is whether Portugal waits longer than other European countries for new medicines because
of the prices it pays, or because it is a small market far from the centre of Europe, or because of
its own administrative process. Only European comparisons are used.

Sections
1. Data and coverage.
2. Portugal in the European ranking: EFPIA / IQVIA W.A.I.T. 2025 (medicines authorised 2021-2024).
3. Size, distance, income and price across European countries: correlations with the time to
   availability and with the number of medicines available, regressions with Portugal's residual,
   leave-one-out and other checks, and groups of countries similar to Portugal on one dimension.
4. Portugal's pipeline in Infarmed's own report (medicines authorised 2020-2025), against the
   European averages of the CRA Access Hurdles Portal: time before the company files and time after.
5. Inside the time after filing: Infarmed's public process tracker (SIATS), process by process, with
   the days it attributes to Infarmed, to the company and to negotiation; the legal deadlines.
6. Size inside Portugal: large groups against other companies, orphan and oncology medicines.
7. The arithmetic the article uses.
"""

from __future__ import annotations

import gzip
import json
import re
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 20261112
REPS = 5000

pd.set_option("display.width", 200)

# Capital cities (latitude, longitude), for the distance to Brussels.
CAPITALS = {
    "AT": ("Austria", 48.21, 16.37), "BE": ("Belgium", 50.85, 4.35), "BG": ("Bulgaria", 42.70, 23.32),
    "HR": ("Croatia", 45.81, 15.98), "CY": ("Cyprus", 35.17, 33.36), "CZ": ("Czechia", 50.08, 14.44),
    "DK": ("Denmark", 55.68, 12.57), "EE": ("Estonia", 59.44, 24.75), "FI": ("Finland", 60.17, 24.94),
    "FR": ("France", 48.86, 2.35), "DE": ("Germany", 52.52, 13.40), "EL": ("Greece", 37.98, 23.73),
    "HU": ("Hungary", 47.50, 19.04), "IE": ("Ireland", 53.35, -6.26), "IT": ("Italy", 41.90, 12.50),
    "LV": ("Latvia", 56.95, 24.11), "LT": ("Lithuania", 54.69, 25.28), "LU": ("Luxembourg", 49.61, 6.13),
    "MT": ("Malta", 35.90, 14.51), "NL": ("Netherlands", 52.37, 4.90), "PL": ("Poland", 52.23, 21.01),
    "PT": ("Portugal", 38.72, -9.14), "RO": ("Romania", 44.43, 26.10), "SK": ("Slovakia", 48.15, 17.11),
    "SI": ("Slovenia", 46.06, 14.51), "ES": ("Spain", 40.42, -3.70), "SE": ("Sweden", 59.33, 18.07),
    "IS": ("Iceland", 64.15, -21.94), "NO": ("Norway", 59.91, 10.75), "CH": ("Switzerland", 46.95, 7.45),
}
EU27 = [g for g in CAPITALS if g not in ("IS", "NO", "CH")]
BRUSSELS = (50.85, 4.35)

# Twenty large research-based groups, fixed before the tracker was split by company. Matched on the
# name of the marketing authorisation holder in the tracker.
LARGE_GROUPS = {
    "Johnson & Johnson": r"janssen|johnson", "Roche": r"roche|genentech", "Merck & Co (MSD)": r"merck sharp|msd",
    "Pfizer": r"pfizer", "AbbVie": r"abbvie|allergan", "AstraZeneca": r"astrazeneca|alexion",
    "Novartis": r"novartis", "Bristol Myers Squibb": r"bristol|celgene", "Eli Lilly": r"lilly",
    "Sanofi": r"sanofi|genzyme", "Novo Nordisk": r"novo nordisk", "GSK": r"glaxo|gsk|viiv",
    "Amgen": r"amgen", "Takeda": r"takeda|shire|baxalta", "Boehringer Ingelheim": r"boehringer",
    "Gilead": r"gilead|kite", "Bayer": r"bayer", "Merck KGaA": r"merck europe|merck serono|merck kgaa",
    "Daiichi Sankyo": r"daiichi", "Astellas": r"astellas",
}

# Portaria 195-A/2015 (procedure for reimbursement and hospital prior evaluation): decision within 75
# days for non-generic medicines (article 11), the clock stopping while the company answers; contract
# negotiation within 30 working days (article 9), about 42 calendar days.
LEGAL_DECISION_DAYS = 75
LEGAL_NEGOTIATION_DAYS = 42


def section(title: str) -> None:
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)


def text(name: str) -> str:
    with gzip.open(DATA / f"{name}.txt.gz", "rt", encoding="utf-8") as f:
        return re.sub(r"\s+", " ", f.read())


def grab(t: str, pattern: str, n: int = 1) -> tuple[int, ...] | int:
    """Integers captured by `pattern` in the whitespace-collapsed text; fails if the sentence is gone."""
    m = re.search(pattern, t)
    if not m:
        raise SystemExit(f"pattern not found: {pattern}")
    vals = tuple(int(v) for v in m.groups())
    return vals[0] if n == 1 else vals


def km(a: tuple, b: tuple) -> float:
    la1, lo1, la2, lo2 = map(radians, (*a, *b))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def ols(y: np.ndarray, xs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    X = np.column_stack([np.ones(len(y))] + xs)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ b
    r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, fit, float(r2)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def boot_corr(x: np.ndarray, y: np.ndarray, rng) -> tuple[float, float]:
    n = len(x)
    out = []
    for _ in range(REPS):
        i = rng.integers(0, n, n)
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            out.append(pearson(x[i], y[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def partial_corr(x: np.ndarray, y: np.ndarray, z: list[np.ndarray]) -> float:
    _, fx, _ = ols(x, z)
    _, fy, _ = ols(y, z)
    return pearson(x - fx, y - fy)


# ---------------------------------------------------------------------------------------- data
def countries() -> pd.DataFrame:
    c = pd.DataFrame([(g, v[0], km((v[1], v[2]), BRUSSELS)) for g, v in CAPITALS.items()],
                     columns=["geo", "country", "dist_km"])
    pop = pd.read_csv(DATA / "eurostat_population.csv.gz")
    pop = pop[pop["time"] == 2025][["geo", "value"]].rename(columns={"value": "pop"})
    aic = pd.read_csv(DATA / "eurostat_aic_pps.csv.gz")
    aic = aic[aic["time"] == 2024].pivot(index="geo", columns="na_item", values="value").reset_index()
    aic = aic.rename(columns={"VI_PPS_EU27_2020_HAB": "aic", "PLI_EU27_2020": "pli_aic"})
    gdp = pd.read_csv(DATA / "eurostat_gdp_pps.csv.gz")
    gdp = gdp[gdp["time"] == 2024][["geo", "value"]].rename(columns={"value": "gdp"})
    pli = pd.read_csv(DATA / "euripid_pli_2017.csv.gz")[["geo", "pli_last", "pli_L"]]
    pli = pli.rename(columns={"pli_last": "pli"})
    w25 = pd.read_csv(DATA / "wait_2025.csv.gz")
    w24 = pd.read_csv(DATA / "wait_2024.csv.gz")[["country", "mean_days"]].rename(columns={"mean_days": "mean_2024"})
    grp = pd.read_csv(DATA / "wait_2025_groups.csv.gz")
    onc = grp[grp["group"] == "oncology"][["country", "mean_days"]].rename(columns={"mean_days": "onc_days"})
    orph = grp[grp["group"] == "orphan"][["country", "mean_days"]].rename(columns={"mean_days": "orphan_days"})
    x = (c.merge(w25, on="country", how="left").merge(w24, on="country", how="left")
         .merge(onc, on="country", how="left").merge(orph, on="country", how="left")
         .merge(pop, on="geo", how="left").merge(aic[["geo", "aic", "pli_aic"]], on="geo", how="left")
         .merge(gdp, on="geo", how="left").merge(pli, on="geo", how="left"))
    x["lpop"] = np.log10(x["pop"])
    x["eu"] = x["geo"].isin(EU27)
    return x


def tracker() -> pd.DataFrame:
    d = pd.read_csv(DATA / "pt_siats_processes.csv.gz", dtype=str)
    d.columns = ["name", "dci", "mah", "ptype", "mtype", "areas", "ind", "status", "ma", "submitted", "start",
                 "access", "concl", "d_inf", "d_co", "d_neg", "page", "search"]
    for c in ("ma", "submitted", "start", "access", "concl"):
        d[c] = pd.to_datetime(d[c], format="%d-%m-%Y", errors="coerce")
    for c in ("d_inf", "d_co", "d_neg"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["setting"] = np.where(d["ptype"].str.contains("Hospitalar"), "hospital", "ambulatory")
    d["outcome"] = d["status"].map(lambda s: s.split(" - ")[1] if s.startswith("Concluído") else "open")
    d["lag"] = (d["submitted"] - d["ma"]).dt.days
    d["span"] = (d["concl"] - d["submitted"]).dt.days
    d["total"] = (d["concl"] - d["ma"]).dt.days
    d["actor_sum"] = d[["d_inf", "d_co", "d_neg"]].sum(axis=1)
    mah = d["mah"].str.lower()
    d["group"] = "other"
    for g, pat in LARGE_GROUPS.items():
        d.loc[mah.str.contains(pat, regex=True) & d["group"].eq("other"), "group"] = g
    d["large"] = d["group"] != "other"

    # Join the EMA list of centrally authorised medicines by brand name.
    e = pd.read_csv(DATA / "ema_medicines.csv.gz", dtype=str)
    e["key"] = e["Name of medicine"].str.replace(r"\s*\(previously.*\)$", "", regex=True).str.upper().str.strip()
    e["ema_ma"] = pd.to_datetime(e["Marketing authorisation date"], dayfirst=True, errors="coerce")
    e["rank"] = np.where(e["Medicine status"].eq("Authorised"), 0, 1)
    e = e.sort_values(["key", "rank", "ema_ma"]).drop_duplicates("key")
    e = e.rename(columns={"Orphan medicine": "orphan", "ATC code (human)": "atc", "Conditional approval": "conditional",
                          "PRIME: priority medicine": "prime"})
    d["key"] = d["name"].str.upper().str.strip()
    d = d.merge(e[["key", "ema_ma", "orphan", "atc", "conditional", "prime"]], on="key", how="left")
    d["central"] = d["ema_ma"].notna()
    d["oncology"] = d["atc"].fillna("").str.startswith("L01") | d["areas"].fillna("").str.startswith("Oncologia")
    d["orphan"] = d["orphan"].eq("Yes")
    return d


def perm_test(a: np.ndarray, b: np.ndarray, rng, stat=np.median) -> float:
    """Two-sided permutation p-value for the difference of `stat` between two groups."""
    obs = abs(stat(a) - stat(b))
    pooled = np.concatenate([a, b])
    hits = 0
    for _ in range(REPS):
        rng.shuffle(pooled)
        hits += abs(stat(pooled[:len(a)]) - stat(pooled[len(a):])) >= obs
    return (hits + 1) / (REPS + 1)


def fmt_days(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) == 0:
        return "n=0"
    return f"n={len(s):3d}  mean {s.mean():6.0f}  median {s.median():6.0f}  p25 {s.quantile(.25):5.0f}  p75 {s.quantile(.75):5.0f}"


# ---------------------------------------------------------------------------------------- main
def main() -> None:
    rng = np.random.default_rng(SEED)
    figure: dict = {}

    # ------------------------------------------------------------------------------------- 1
    section("1. Data and coverage")
    x = countries()
    d = tracker()
    wait = text("wait_2025_report")
    pt = text("pt_access_report_2020_2025")
    cra = text("cra_portal_2026")
    log = (DATA / "fetch_log.txt").read_text(encoding="utf-8")
    updated = re.search(r"tracker last updated: (\S+)", log).group(1)
    n_cohort = grab(wait, r"availability of (\d+) innovative medicines")
    print(f"W.A.I.T. 2025: {n_cohort} medicines authorised centrally in 2021-2024, availability on 5 January 2026.")
    print(f"  countries with a mean time to availability: {x['mean_days'].notna().sum()} of the {len(x)} EU/EEA/Swiss countries kept")
    print(f"EURIPID price level indices 2017 (all medicines): {x['pli'].notna().sum()} of them; ATC group L: {x['pli_L'].notna().sum()}")
    print(f"Eurostat: population 1 January 2025, AIC and GDP per head in PPS 2024 for {x['aic'].notna().sum()} of them")
    print(f"Infarmed process tracker (SIATS), last updated {updated}: {len(d)} processes, "
          f"{(d['search'] == 'new active substance').sum()} for a new active substance and "
          f"{(d['search'] == 'new indication').sum()} for a new indication")
    nas = d[d["search"] == "new active substance"].copy()
    print(f"  new active substances: {nas['name'].nunique()} medicines; centrally authorised (found in the EMA list): "
          f"{nas['central'].sum()} processes, {nas.loc[nas['central'], 'name'].nunique()} medicines")
    print("  conclusion year of concluded processes (new active substances):")
    print("   ", nas["concl"].dt.year.value_counts().sort_index().to_dict())

    # ------------------------------------------------------------------------------------- 2
    section("2. Portugal in the European ranking (W.A.I.T. 2025)")
    eu = x[x["eu"]]
    avg_all, avg_onc, avg_orph = grab(wait, r"Average time (\d+) (\d+) (\d+) \d+ \d+ to availability", 3)
    rates = re.search(r"Average rate (\d+)% (\d+)% (\d+)%", wait).groups()
    print(f"EU27 averages printed by the report: time to availability {avg_all} days (oncology {avg_onc}, orphans "
          f"{avg_orph}); rate of availability {rates[0]}% (oncology {rates[1]}%, orphans {rates[2]}%)")
    calc = eu.loc[eu["dates_submitted"] >= 3, "mean_days"].mean()
    calc_onc = eu["onc_days"].mean()
    calc_orph = eu["orphan_days"].mean()
    print(f"  the same averages from the values read off the charts: {calc:.0f}, {calc_onc:.0f}, {calc_orph:.0f}")
    p = x[x["geo"] == "PT"].iloc[0]
    ranked = eu.dropna(subset=["mean_days"]).sort_values("mean_days").reset_index(drop=True)
    rank = int(ranked.index[ranked["geo"] == "PT"][0]) + 1
    print(f"Portugal: mean {p['mean_days']:.0f} days, median {p['median_days']:.0f}; available {p['available']:.0f} of "
          f"{n_cohort} ({p['available'] / n_cohort:.0%}); dates submitted {p['dates_submitted']:.0f}")
    print(f"  rank by mean time among the {len(ranked)} EU countries with a mean: {rank} (1 = fastest); slower: "
          f"{', '.join(ranked.loc[ranked['mean_days'] > p['mean_days'], 'country'])}")
    print(f"  gap to the EU27 average: {p['mean_days'] - avg_all:.0f} days")
    eu_avail = eu["available"].mean()
    rank_av = int((eu["available"] > p["available"]).sum()) + 1
    print(f"  medicines available: Portugal {p['available']:.0f}, EU27 mean {eu_avail:.1f}; rank {rank_av} of {eu['available'].notna().sum()}")
    print(f"  oncology: Portugal {p['onc_days']:.0f} days against EU27 {avg_onc}; orphans: {p['orphan_days']:.0f} against {avg_orph}")
    print(f"  W.A.I.T. 2024 (medicines of 2020-2023): Portugal {p['mean_2024']:.0f} days")
    print("Fastest and slowest EU countries (mean days):")
    for _, r in ranked.iterrows():
        print(f"    {r['country']:12s} {r['mean_days']:6.0f}   available {r['available']:4.0f}")

    # ------------------------------------------------------------------------------------- 3
    section("3. Size, distance, income and price across countries")
    s = x.dropna(subset=["mean_days"]).copy()
    print(f"Countries with a mean time: {len(s)} (EU27 without Malta, plus Iceland, Norway and Switzerland)")
    print("Variables: population 2025 (log10), distance from the capital to Brussels (km), actual individual "
          "consumption per head in PPS 2024 (EU27 = 100), GDP per head in PPS 2024, EURIPID price level of "
          "medicines 2017 (EURIPID 20 = 100), all medicines and ATC group L (cancer and immune medicines).")
    print(f"Portugal: population {p['pop'] / 1e6:.1f} million, {p['dist_km']:.0f} km from Brussels, AIC per head "
          f"{p['aic']:.0f}, GDP per head {p['gdp']:.0f}, price level {p['pli']:.1f} (ATC L {p['pli_L']:.1f})")
    print(f"  sample means: price level {s['pli'].mean():.1f} (n={s['pli'].notna().sum()}), ATC L "
          f"{s['pli_L'].mean():.1f} (n={s['pli_L'].notna().sum()}), AIC {s['aic'].mean():.0f}, "
          f"distance {s['dist_km'].mean():.0f} km, mean time {s['mean_days'].mean():.0f} days")
    lo_L = s.dropna(subset=["pli_L"]).sort_values("pli_L")
    print(f"  ATC L price level, lowest first: " + ", ".join(f"{r['geo']} {r['pli_L']:.0f}" for _, r in lo_L.head(6).iterrows()))
    print()
    print("Correlation with the mean time to availability and with the number of medicines available")
    print("  (Pearson r with a 95% bootstrap interval over countries; Spearman rank correlation)")
    corr_rows = []
    labels = {"lpop": "size (log population)", "dist_km": "distance to Brussels", "aic": "AIC per head",
              "gdp": "GDP per head", "pli": "price level, all medicines", "pli_L": "price level, ATC L"}
    for v, lab in labels.items():
        z = s.dropna(subset=[v])
        a, b = z[v].to_numpy(float), z["mean_days"].to_numpy(float)
        r, (lo, hi), rs = pearson(a, b), boot_corr(a, b, rng), spearman(a, b)
        zz = z.dropna(subset=["available"])
        ra = pearson(zz[v].to_numpy(float), zz["available"].to_numpy(float))
        print(f"  {lab:28s} n={len(z):2d}  time r={r:+.2f} [{lo:+.2f}, {hi:+.2f}]  rho={rs:+.2f}   available r={ra:+.2f}")
        corr_rows.append({"var": v, "label": lab, "n": len(z), "r": round(r, 3), "lo": round(lo, 3), "hi": round(hi, 3)})
    figure["correlations"] = corr_rows

    z = s.dropna(subset=["pli", "aic"])
    pc = partial_corr(z["pli"].to_numpy(float), z["mean_days"].to_numpy(float), [z["aic"].to_numpy(float)])
    pa = partial_corr(z["aic"].to_numpy(float), z["mean_days"].to_numpy(float), [z["pli"].to_numpy(float)])
    print(f"  partial correlation, price level with time holding AIC per head fixed: {pc:+.2f} (n={len(z)}); "
          f"AIC with time holding price fixed: {pa:+.2f}")
    print(f"  correlation of the price level with AIC per head: {pearson(z['pli'].to_numpy(float), z['aic'].to_numpy(float)):+.2f}")

    print()
    print("Regressions of the mean time (days) on each set of variables; Portugal's fitted value, residual,")
    print("the rank of its residual (1 = most delayed relative to the model) and its prediction when the model")
    print("is fitted without Portugal.")
    models = {
        "size": ["lpop"], "distance": ["dist_km"], "size + distance": ["lpop", "dist_km"],
        "size + distance + AIC": ["lpop", "dist_km", "aic"], "price + AIC": ["pli", "aic"],
        "size + distance + AIC + price": ["lpop", "dist_km", "aic", "pli"],
        "size + distance + AIC + price L": ["lpop", "dist_km", "aic", "pli_L"],
    }
    model_rows = []
    for name, vs in models.items():
        z = s.dropna(subset=vs).reset_index(drop=True)
        y = z["mean_days"].to_numpy(float)
        b, fit, r2 = ols(y, [z[v].to_numpy(float) for v in vs])
        res = y - fit
        i = int(z.index[z["geo"] == "PT"][0])
        rk = int((res > res[i]).sum()) + 1
        zo = z[z["geo"] != "PT"]
        bo, _, _ = ols(zo["mean_days"].to_numpy(float), [zo[v].to_numpy(float) for v in vs])
        pred_out = float(bo[0] + sum(bo[k + 1] * z.loc[i, v] for k, v in enumerate(vs)))
        coefs = ", ".join(f"{v} {b[k + 1]:+.2f}" for k, v in enumerate(vs))
        print(f"  {name:34s} n={len(z):2d}  R2 {r2:.2f}  [{coefs}]")
        print(f"  {'':34s} Portugal fitted {fit[i]:5.0f}, residual {res[i]:+5.0f}, rank {rk:2d} of {len(z)}; "
              f"predicted without Portugal {pred_out:5.0f} (actual {y[i]:.0f}); sample mean {y.mean():.0f}")
        model_rows.append({"model": name, "n": len(z), "r2": round(r2, 3), "pt_fit": round(float(fit[i]), 1),
                           "pt_resid": round(float(res[i]), 1), "pt_pred_out": round(pred_out, 1),
                           "mean": round(float(y.mean()), 1)})
    figure["models"] = model_rows

    print()
    print("Price slope: days per point of the price level, in 'price + AIC' and in the full model; 95% bootstrap")
    print("interval over countries; range when each country is left out in turn.")
    for vs in (["pli", "aic"], ["lpop", "dist_km", "aic", "pli"]):
        z = s.dropna(subset=vs).reset_index(drop=True)
        y = z["mean_days"].to_numpy(float)
        X = [z[v].to_numpy(float) for v in vs]
        k = vs.index("pli") + 1
        b, _, _ = ols(y, X)
        bs = []
        for _ in range(REPS):
            i = rng.integers(0, len(z), len(z))
            bb, _, _ = ols(y[i], [a[i] for a in X])
            bs.append(bb[k])
        loo = []
        for j in range(len(z)):
            m = np.arange(len(z)) != j
            bb, _, _ = ols(y[m], [a[m] for a in X])
            loo.append(bb[k])
        print(f"  {' + '.join(vs):28s} slope {b[k]:+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]  "
              f"leave-one-out {min(loo):+.2f} to {max(loo):+.2f}")

    print()
    print("Other checks of the price correlation (Pearson r of the price level with the time):")
    checks = {
        "median time instead of mean": (s, "pli", "median_days"),
        "W.A.I.T. 2024 (2020-2023 medicines)": (x, "pli", "mean_2024"),
        "oncology time, price level ATC L": (x, "pli_L", "onc_days"),
        "orphan time, price level all": (x, "pli", "orphan_days"),
        "EU members only": (s[s["eu"]], "pli", "mean_days"),
        "without Romania": (s[s["geo"] != "RO"], "pli", "mean_days"),
        "without Switzerland, Denmark and Iceland": (s[~s["geo"].isin(["CH", "DK", "IS"])], "pli", "mean_days"),
        "without Portugal": (s[s["geo"] != "PT"], "pli", "mean_days"),
    }
    for lab, (df, a, b) in checks.items():
        z = df.dropna(subset=[a, b])
        print(f"  {lab:42s} n={len(z):2d}  r={pearson(z[a].to_numpy(float), z[b].to_numpy(float)):+.2f}  "
              f"rho={spearman(z[a].to_numpy(float), z[b].to_numpy(float)):+.2f}")
    z = s.dropna(subset=["lpop"])
    print(f"  size against time, EU members only: r={pearson(z.loc[z['eu'], 'lpop'].to_numpy(float), z.loc[z['eu'], 'mean_days'].to_numpy(float)):+.2f}")

    print()
    print("Countries like Portugal on one dimension (mean days; price level all / ATC L):")
    peers = {
        "size: population 7.5 to 15 million": s[(s["pop"] >= 7.5e6) & (s["pop"] <= 15e6)],
        "distance: capital 1,400 km or more from Brussels": s[s["dist_km"] >= 1400],
        "income: AIC per head within 10 points of Portugal": s[(s["aic"] - p["aic"]).abs() <= 10],
    }
    for lab, df in peers.items():
        df = df.sort_values("mean_days")
        print(f"  {lab}")
        for _, r in df.iterrows():
            pli = "  n/a" if pd.isna(r["pli"]) else f"{r['pli']:5.0f}"
            pl = "  n/a" if pd.isna(r["pli_L"]) else f"{r['pli_L']:5.0f}"
            print(f"    {r['country']:12s} {r['mean_days']:5.0f} days  pop {r['pop'] / 1e6:5.1f} m  {r['dist_km']:5.0f} km  "
                  f"AIC {r['aic']:4.0f}  price {pli} / {pl}")
        zz = df.dropna(subset=["pli"])
        if len(zz) >= 4:
            print(f"    price level against time within the group: r={pearson(zz['pli'].to_numpy(float), zz['mean_days'].to_numpy(float)):+.2f} (n={len(zz)})")
    figure["countries"] = [
        {"geo": r["geo"], "country": r["country"], "days": float(r["mean_days"]), "pop_m": round(float(r["pop"]) / 1e6, 2),
         "dist_km": round(float(r["dist_km"]), 0), "pli": None if pd.isna(r["pli"]) else float(r["pli"]),
         "available": None if pd.isna(r["available"]) else int(r["available"])}
        for _, r in s.sort_values("geo").iterrows()]

    # ------------------------------------------------------------------------------------- 4
    section("4. Portugal's pipeline in Infarmed's report, against the European averages")
    n_ma, n_not, pct_not = grab(pt, r"Dos (\d+) processos com AIM centralizada entre 2020 e 2025, (\d+) \((\d+)%\) processos não foram submetidos", 3)
    n_sub, filing = grab(pt, r"Para os (\d+) processos submetidos, o tempo médio de submissão ao INFARMED, I\.P\. foi de (\d+) dias", 2)
    n_conc, n_ok, n_arch, n_ref = grab(pt, r"(\d+) foram concluídos até 31 de dezembro de 2025, dos quais (\d+) foram aprovados para financiamento público, (\d+) foram arquivados no decurso da avaliação a pedido da empresa e (\d+) foram indeferidos", 4)
    span = grab(pt, r"Para os \d+ processos concluídos, o tempo médio entre a submissão e a conclusão foi de (\d+) dias")
    inf_ms, rest = grab(pt, r"INFARMED/MS foi de (\d+) dias\. O restante tempo engloba o tempo da empresa assim como o processo de negociação \(em média (\d+) dias\)", 2)
    early = grab(pt, r"Dos \d+ processos com AIM entre 2020 e 2025, (\d+) tiveram programas de acesso precoce")
    print(f"Medicines (new active substances) authorised centrally in 2020-2025: {n_ma}")
    print(f"  not submitted to Infarmed by 31 December 2025: {n_not} ({pct_not}%); submitted {n_sub} ({n_sub / n_ma:.0%})")
    print(f"  mean time from authorisation to submission: {filing} days")
    print(f"  concluded: {n_conc} (approved {n_ok}, withdrawn by the company {n_arch}, refused {n_ref})")
    print(f"  mean time from submission to conclusion: {span} days = Infarmed and Ministry {inf_ms} + company and negotiation {rest}")
    print(f"  mean authorisation to conclusion (filing + assessment): {filing + span} days")
    print(f"  early access programmes while the assessment ran: {early} of {n_ma}")
    prime = grab(pt, r"Dos (\d+) medicamentos que beneficiaram do estatuto PRIME, (\d+) não foram submetidos \((\d+)%\)", 3)
    prime_f = grab(pt, r"PRIME submetidos ao INFARMED, o tempo médio de submissão foi de (\d+) dias")
    onc = grab(pt, r"Dos (\d+) processos de medicamentos oncológicos, (\d+) processos \((\d+)%\) não foram submetidos", 3)
    orp = grab(pt, r"Dos (\d+) processos de medicamentos órfãos, (\d+) processos \((\d+)%\) não foram submetidos", 3)
    filing_of = lambda n: grab(pt, rf"Para os {n} processos submetidos, o tempo médio de submissão ao INFARMED, I\.P\. foi de (\d+) dias")
    onc_f, orp_f = filing_of(onc[0] - onc[1]), filing_of(orp[0] - orp[1])
    concl = re.findall(r"Para estes (\d+) processos, o tempo médio de conclusão foi de (\d+) dias", pt)
    inf = re.findall(r"avaliação da responsabilidade do INFARMED(?:/MS)? foi de (\d+) dias", pt)
    if len(concl) != 2 or len(inf) != 3:
        raise SystemExit(f"report layout changed: {concl} {inf}")
    onc_c, orp_c = tuple(map(int, concl[0])), tuple(map(int, concl[1]))
    onc_i, orp_i = int(inf[1]), int(inf[2])
    print(f"  oncology: {onc[0]} medicines, {onc[1]} ({onc[2]}%) not submitted; filing {onc_f} days; {onc_c[0]} concluded in "
          f"{onc_c[1]} days on average, of which Infarmed and Ministry {onc_i}")
    print(f"  orphans: {orp[0]} medicines, {orp[1]} ({orp[2]}%) not submitted; filing {orp_f} days; {orp_c[0]} concluded in "
          f"{orp_c[1]} days on average, of which Infarmed {orp_i}")
    prime_c = grab(pt, r"Dos \d+ processos submetidos, (\d+) foram concluídos com um tempo médio de conclusão de (\d+) dias", 2)
    print(f"  PRIME (EMA priority) medicines: {prime[0]}, {prime[1]} ({prime[2]}%) not submitted; filing {prime_f} days; "
          f"{prime_c[0]} concluded in {prime_c[1]} days")

    print()
    eu_total, eu_filing, eu_decision = grab(cra, r"average number of (\d+) days between EMA approval and reimbursement; within this, companies file after (\d+) days, and the remaining (\d+) days are spent", 3)
    share_f, share_d = grab(cra, r"(\d+)% of the total time between MA and availability can be attributed to the time between MA and P&R filing, while (\d+)%", 2)
    nordic_f, cee_f = grab(cra, r"lowest in Nordic markets \(\d+% or (\d+) days on average\), while in CEE countries, it is \d+% \(or (\d+) days", 2)
    cee_d, eu4_d, nord_d, west_d = grab(cra, r"longer on average in CEE countries \((\d+) days\) than in the EU4\+UK, Nordic and Western regions \((\d+) days, (\d+) days, and (\d+) days", 4)
    filed = grab(cra, r"taking an average across European countries, (\d+)% of products have already been filed")
    west = re.search(r"Western \(non-EU4\+UK\): ([A-Za-z ,]+?) •", cra).group(1)
    print("CRA Access Hurdles Portal 2026 (91 medicines authorised January 2022 to June 2025; reimbursed medicines):")
    print(f"  European average from authorisation to reimbursement {eu_total} days: filing {eu_filing} ({share_f}%), "
          f"decision after filing {eu_decision} ({share_d}%)")
    print(f"  filing: Nordic {nordic_f} days, Central and Eastern {cee_f}; decision: CEE {cee_d}, EU4+UK {eu4_d}, Nordic {nord_d}, "
          f"Western {west_d} (Western = {west})")
    print(f"  share of medicines filed, average across countries: {filed}% (Portugal, Infarmed report: {n_sub / n_ma:.0%})")
    for lab, n in (("Health system infrastructure", None), ("Economic viability", None), ("P&R process", None),
                   ("Value assessment process", None)):
        k = grab(cra, re.escape(lab) + r" \(indicated as a reason (\d+) times\)")
        print(f"  reasons for not filing, {lab}: {k}")
    ex_f = filing - eu_filing
    ex_d = span - eu_decision
    print(f"Portugal minus the European average: filing {ex_f:+d} days, after filing {ex_d:+d} days; "
          f"share of the excess after filing {ex_d / (ex_f + ex_d):.0%}")
    print("  (Portugal's figures are means over all concluded processes, approvals, refusals and withdrawals,")
    print("   for medicines of 2020-2025; the CRA figures cover reimbursed medicines of 2022-2025.)")
    figure["pipeline"] = {
        "europe": {"filing": eu_filing, "decision": eu_decision},
        "portugal_report": {"filing": filing, "infarmed": inf_ms, "company_negotiation": rest},
    }

    # ------------------------------------------------------------------------------------- 5
    section("5. Inside the time after filing: the process tracker")
    print("New active substances, all processes in the tracker:")
    print("  ", nas["status"].value_counts().to_dict())
    print("  setting:", nas["setting"].value_counts().to_dict())
    allok = nas[nas["outcome"] == "Deferido"]
    t0 = allok[["d_inf", "d_co", "d_neg"]].sum()
    print(f"All approved processes, including {(~allok['central']).sum()} medicines new to Portugal but not authorised "
          f"centrally (mostly fixed-dose combinations): n={len(allok)}; means Infarmed {allok['d_inf'].mean():.0f}, company "
          f"{allok['d_co'].mean():.0f}, negotiation {allok['d_neg'].mean():.0f}; medians {allok['d_inf'].median():.0f} / "
          f"{allok['d_co'].median():.0f} / {allok['d_neg'].median():.0f}; shares {t0['d_inf'] / t0.sum():.0%} / "
          f"{t0['d_co'] / t0.sum():.0%} / {t0['d_neg'] / t0.sum():.0%}")
    nas_c = nas[nas["central"]]
    print("From here on, centrally authorised medicines only (the population of W.A.I.T. and of Infarmed's report):")
    print("  ", nas_c["status"].value_counts().to_dict())
    ok = nas_c[nas_c["outcome"] == "Deferido"].copy()
    print(f"Approved processes: {len(ok)} ({ok['setting'].value_counts().to_dict()}), {ok['name'].nunique()} medicines")
    print(f"  conclusion year: {ok['concl'].dt.year.value_counts().sort_index().to_dict()}")
    print(f"  days attributed to Infarmed  {fmt_days(ok['d_inf'])}")
    print(f"  days attributed to company   {fmt_days(ok['d_co'])}")
    print(f"  days of negotiation          {fmt_days(ok['d_neg'])}")
    print(f"  sum of the three             {fmt_days(ok['actor_sum'])}")
    print(f"  submission to conclusion     {fmt_days(ok['span'])}")
    diff = (ok["span"] - ok["actor_sum"]).dropna()
    print(f"  submission-to-conclusion minus the sum of the three: median {diff.median():.0f}, "
          f"within 5 days for {(diff.abs() <= 5).mean():.0%}")
    tot = ok[["d_inf", "d_co", "d_neg"]].sum()
    shares = tot / tot.sum()
    print(f"  shares of all days: Infarmed {shares['d_inf']:.0%}, company {shares['d_co']:.0%}, negotiation {shares['d_neg']:.0%}")
    print(f"  processes with some negotiation: {(ok['d_neg'] > 0).mean():.0%}")
    print(f"  authorisation to submission    {fmt_days(ok['lag'])}")
    print(f"  authorisation to conclusion    {fmt_days(ok['total'])}")
    print(f"  access date before the conclusion date: {(ok['access'] < ok['concl']).sum()} processes")
    for st, g in ok.groupby("setting"):
        print(f"  {st}: n={len(g)}  median Infarmed {g['d_inf'].median():.0f}, company {g['d_co'].median():.0f}, "
              f"negotiation {g['d_neg'].median():.0f}, submission to conclusion {g['span'].median():.0f}; means "
              f"{g['d_inf'].mean():.0f} / {g['d_co'].mean():.0f} / {g['d_neg'].mean():.0f} / {g['span'].mean():.0f}")
    print("Against the legal deadlines (Portaria 195-A/2015): decision in 75 days of Infarmed time, negotiation")
    print("in 30 working days (about 42 calendar days):")
    print(f"  Infarmed days above 75: {(ok['d_inf'] > LEGAL_DECISION_DAYS).mean():.0%} of approved processes; "
          f"median {ok['d_inf'].median() / LEGAL_DECISION_DAYS:.1f} times the deadline")
    neg = ok.loc[ok["d_neg"] > 0, "d_neg"]
    print(f"  negotiation above 42 days: {(neg > LEGAL_NEGOTIATION_DAYS).mean():.0%} of processes that negotiated; "
          f"median {neg.median() / LEGAL_NEGOTIATION_DAYS:.1f} times the deadline")
    over_inf = (ok["d_inf"] - LEGAL_DECISION_DAYS).clip(lower=0)
    over_neg = (ok["d_neg"] - LEGAL_NEGOTIATION_DAYS).clip(lower=0)
    print(f"  mean days beyond the deadlines per approved process: Infarmed {over_inf.mean():.0f}, negotiation {over_neg.mean():.0f}")
    for outcome, lab in (("Arquivado", "withdrawn by the company"), ("Indeferido", "refused")):
        g = nas_c[nas_c["outcome"] == outcome]
        print(f"{outcome} ({lab}): n={len(g)}; median Infarmed {g['d_inf'].median():.0f}, "
              f"company {g['d_co'].median():.0f}, negotiation {g['d_neg'].median():.0f}; with negotiation {(g['d_neg'] > 0).sum()}")
    op = nas_c[nas_c["outcome"] == "open"].copy()
    asof = pd.Timestamp(pd.to_datetime(updated, format="%d/%m/%Y"))
    op["open_days"] = (asof - op["submitted"]).dt.days
    print(f"Open processes: {len(op)}; days since submission {fmt_days(op['open_days'])}")
    print(f"  open for more than one year: {(op['open_days'] > 365).sum()}, more than two years: {(op['open_days'] > 730).sum()}")
    for st, g in op.groupby("status"):
        print(f"  {st:46s} n={len(g):3d}  median so far: Infarmed {g['d_inf'].median():4.0f}, company {g['d_co'].median():4.0f}, "
              f"negotiation {g['d_neg'].median():4.0f}")
    ni = d[(d["search"] == "new indication") & (d["outcome"] == "Deferido")]
    t2 = ni[["d_inf", "d_co", "d_neg"]].sum()
    print(f"New indications, approved: n={len(ni)}; medians Infarmed {ni['d_inf'].median():.0f}, company {ni['d_co'].median():.0f}, "
          f"negotiation {ni['d_neg'].median():.0f}; shares {t2['d_inf'] / t2.sum():.0%} / {t2['d_co'] / t2.sum():.0%} / {t2['d_neg'] / t2.sum():.0%}")
    figure["tracker"] = [
        {"name": r["name"], "setting": r["setting"], "infarmed": int(r["d_inf"]), "company": int(r["d_co"]),
         "negotiation": int(r["d_neg"])}
        for _, r in ok.sort_values(["actor_sum", "name"]).iterrows()]
    figure["tracker_means"] = {"infarmed": round(float(ok["d_inf"].mean()), 1), "company": round(float(ok["d_co"].mean()), 1),
                               "negotiation": round(float(ok["d_neg"].mean()), 1),
                               "lag": round(float(ok["lag"].mean()), 1), "n": int(len(ok))}

    # ------------------------------------------------------------------------------------- 6
    section("6. Size inside Portugal: company size, orphan and oncology medicines (tracker)")
    c = nas[nas["central"]].copy()
    print(f"Centrally authorised new active substances: {len(c)} processes; large groups {c['large'].sum()}, others {(~c['large']).sum()}")
    print("  groups present:", ", ".join(sorted(c.loc[c["large"], "group"].unique())))
    recent = c["ma"].dt.year >= 2020
    print(f"  of which authorised in 2020 or later: {recent.sum()}")
    for lab, m in (("large groups", c["large"]), ("other companies", ~c["large"])):
        g = c[m]
        a = g[g["outcome"] == "Deferido"]
        print(f"  {lab:16s} filing lag {fmt_days(g['lag'])}")
        print(f"  {'':16s} authorised 2020 or later: filing lag {fmt_days(g.loc[g['ma'].dt.year >= 2020, 'lag'])}")
        print(f"  {'':16s} approved n={len(a)}: submission to conclusion median {a['span'].median():.0f} (mean {a['span'].mean():.0f}); "
              f"medians Infarmed {a['d_inf'].median():.0f}, company {a['d_co'].median():.0f}, negotiation {a['d_neg'].median():.0f} "
              f"(mean {a['d_neg'].mean():.0f})")
        o = g[g["outcome"].isin(["Arquivado", "Indeferido"])]
        print(f"  {'':16s} withdrawn {(g['outcome'] == 'Arquivado').sum()}, refused {(g['outcome'] == 'Indeferido').sum()}, open {(g['outcome'] == 'open').sum()}")
    a_l = c.loc[c["large"] & (c["outcome"] == "Deferido")]
    a_o = c.loc[~c["large"] & (c["outcome"] == "Deferido")]
    f_l = c.loc[c["large"] & (c["ma"].dt.year >= 2020), "lag"].dropna().to_numpy(float)
    f_o = c.loc[~c["large"] & (c["ma"].dt.year >= 2020), "lag"].dropna().to_numpy(float)
    print("  permutation tests (difference of medians, two-sided):")
    print(f"    filing lag, authorised 2020 or later: p={perm_test(f_l, f_o, rng):.4f}")
    print(f"    negotiation days, approved: p={perm_test(a_l['d_neg'].to_numpy(float), a_o['d_neg'].to_numpy(float), rng):.3f}")
    print(f"    submission to conclusion, approved: p={perm_test(a_l['span'].to_numpy(float), a_o['span'].to_numpy(float), rng):.3f}")
    print(f"    Infarmed days, approved: p={perm_test(a_l['d_inf'].to_numpy(float), a_o['d_inf'].to_numpy(float), rng):.3f}")
    for lab, m in (("orphan", c["orphan"]), ("not orphan", ~c["orphan"]), ("oncology", c["oncology"]),
                   ("not oncology", ~c["oncology"])):
        g = c[m]
        a = g[g["outcome"] == "Deferido"]
        print(f"  {lab:16s} filing lag {fmt_days(g['lag'])}; approved n={len(a)}, submission to conclusion median "
              f"{a['span'].median():.0f}, negotiation median {a['d_neg'].median():.0f}")

    # ------------------------------------------------------------------------------------- 7
    section("7. The arithmetic the article uses")
    gap = p["mean_days"] - avg_all
    full = next(m for m in model_rows if m["model"] == "size + distance + AIC + price")
    sd = next(m for m in model_rows if m["model"] == "size + distance + AIC")
    print(f"Portugal's gap to the EU27 average: {gap:.0f} days")
    print(f"  size, distance and AIC put Portugal at {sd['pt_fit']:.0f} days ({sd['pt_fit'] - sd['mean']:+.0f} against "
          f"their sample mean {sd['mean']:.0f})")
    print(f"  adding the price level puts it at {full['pt_fit']:.0f} days ({full['pt_fit'] - full['mean']:+.0f} against "
          f"the sample mean {full['mean']:.0f}); unexplained {full['pt_resid']:+.0f}")
    mn = figure["tracker_means"]
    print(f"Approved new active substances (tracker), mean days: filing {mn['lag']:.0f}, Infarmed {mn['infarmed']:.0f}, "
          f"company {mn['company']:.0f}, negotiation {mn['negotiation']:.0f}")
    print(f"  if negotiation took the legal 42 days: {mn['negotiation'] - LEGAL_NEGOTIATION_DAYS:.0f} days less on average")
    print(f"  if Infarmed's time were 75 days: {mn['infarmed'] - LEGAL_DECISION_DAYS:.0f} days less on average")

    (HERE / "figure_data.json").write_text(json.dumps(figure, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
