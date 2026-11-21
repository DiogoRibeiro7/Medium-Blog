"""Reanalysis of Miller (2008), "Women's Suffrage, Political Responsiveness, and Child Survival
in American History", with the data Miller published.

    python fetch.py --out data      # once; GitHub Actions keeps the snapshot in data/raw
    python suffrage.py              # writes results/summary.txt and the CSVs figures.py reads

Question: Miller's design compares states before and after their suffrage laws with state and
year fixed effects and state-specific linear trends, over 1900-1936. Every state has suffrage
from 1920, so the years 1920-1936 contain no untreated state. How much of the 8-15% fall in
child deaths survives estimators that use only clean comparisons (states that had not yet
enfranchised women), and a comparison of child deaths with adult deaths in the same state?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import dates
import did

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
RES = HERE / "results"
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 499
LAST = 1919
OUT: list[str] = []


def say(line: str = "") -> None:
    OUT.append(line)


def f3(x: float) -> str:
    return f"{x:+.3f}"


def pct(x: float) -> str:
    return f"{100 * (np.exp(x) - 1):+.1f}%"


def ci(draws) -> tuple[float, float]:
    if np.all(np.isnan(draws)):
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- data

AGES = {"under 1": ["all_under_1"], "1-4": ["all_1_4"], "5-9": ["all_5_9"],
        "10-14": ["all_10_14"], "15-19": ["all_15_19"],
        "1-19": ["all_1_4", "all_5_9", "all_10_14", "all_15_19"],
        "20-24": ["all_20_24"],
        "25-64": [f"all_{a}_{a + 4}" for a in range(25, 65, 5)]}
CAUSES = {"diarrhoea under 2": "diarrhea_under_2", "diphtheria": "diphtheria",
          "meningitis": "meningitis", "scarlet fever": "scarlet_fever", "measles": "measles",
          "whooping cough": "whooping_cough", "typhoid": "typhoid",
          "cancer": "cancer_tumor", "heart and circulation": "circulatory_disease",
          "accidents and violence": "accidents_violence"}


def load_mortality(coding: str = "miller") -> pd.DataFrame:
    m = pd.read_stata(RAW / "mortality_harmonized.dta", convert_categoricals=False)
    m["year"] = m.year.astype(int)
    m["g"] = m.state.map(lambda s: dates.suffrage_year(s, coding))
    new = {}
    for name, cols in AGES.items():
        new[f"ln {name}"] = np.log(m[cols].sum(axis=1))
    for name in ["under 1", "1-4", "5-9", "10-14", "15-19", "1-19", "20-24"]:
        new[f"dd {name}"] = new[f"ln {name}"] - new["ln 25-64"]
    for name, col in CAUSES.items():
        new[f"ln {name}"] = np.log(m[col].where(m[col] > 0))
        new[f"dd {name}"] = new[f"ln {name}"] - new["ln 25-64"]
    return pd.concat([m, pd.DataFrame(new, index=m.index)], axis=1)


def load_finance() -> pd.DataFrame:
    pf = pd.read_stata(RAW / "public_finance.dta", convert_categoricals=False)
    pf = pf.drop_duplicates(["fips_state", "fips_place", "year"])  # Quincy, MA twice in 1923
    pf["year"] = pf.year.astype(int)
    # the file's "id" is the city's population rank in that year, not a city identifier
    pf["rank"] = pf.id.astype(int)
    pf["city_name"] = pf.city.str.strip()
    pf["city"] = pf.fips_state.astype(int).astype(str) + "-" + pf.fips_place.astype(int).astype(str)
    pf["g"] = pf.state.map(dates.suffrage_year)
    for name, col in [("total", "e_total"), ("health and sanitation", "e_health_sanit"),
                      ("health", "e_health"), ("sanitation", "e_sanitation"),
                      ("charities, hospitals, corrections", "e_charities")]:
        pf[f"ln {name}"] = np.log(pf[col].where(pf[col] > 0))
    pf["health share"] = pf.e_health_sanit / pf.e_total
    return pf


# --------------------------------------------------------------------------- sections

def section_data(m: pd.DataFrame, pf: pd.DataFrame) -> None:
    say("== Data")
    g = pd.Series({s: dates.suffrage_year(s) for s in dates.STATES})
    say(f"states: {len(g)}; suffrage before 1920: {(g < 1920).sum()} "
        f"(full {len(dates.FULL)}, presidential {len(dates.PRESIDENTIAL)}, primary {len(dates.PRIMARY)})")
    by_cohort = g[(g >= 1900) & (g < 1920)].groupby(g).apply(lambda s: " ".join(sorted(s.index)))
    for year, states in by_cohort.items():
        say(f"  {year}: {states}")
    say(f"mortality: {len(m)} state-years, {m.state.nunique()} states, {m.year.min()}-{m.year.max()}; "
        f"states reporting in 1900: {(m.year == 1900).sum()}, in 1919: {(m.year == 1919).sum()}, "
        f"in 1936: {(m.year == 1936).sum()}")
    clean = m[(m.year <= LAST)]
    say(f"state-years up to 1919: {len(clean)}; of them treated: {(clean.year >= clean.g).sum()}; "
        f"states always treated in the data: "
        f"{' '.join(sorted(m.loc[m.groupby('state').year.transform('min') >= m.g, 'state'].unique()))}")
    cohorts = sorted(m.loc[(m.g <= LAST) & (m.groupby('state').year.transform('min') < m.g), 'g'].unique())
    say(f"cohorts with a pre-suffrage year in the mortality data: {' '.join(map(str, cohorts))}")
    for g0 in cohorts:
        say(f"  {g0}: {' '.join(sorted(m.loc[(m.g == g0) & (m.groupby('state').year.transform('min') < m.g), 'state'].unique()))}")
    share = m.loc[m.year >= 1920].shape[0] / len(m)
    say(f"share of state-years in 1920-1936, when every state has suffrage: {share:.3f}")
    say(f"municipal finance: {len(pf)} city-years, {pf.city.nunique()} cities in {pf.state.nunique()} states, "
        f"{pf.year.min()}-{pf.year.max()} (no 1914, no 1920)")
    med = pf.groupby("year")["health share"].median()
    say(f"median health-and-sanitation share of total spending: 1905 {med[1905]:.3f}, "
        f"1906 {med[1906]:.3f}, 1919 {med[1919]:.3f}, 1930 {med[1930]:.3f}  "
        f"(1905 total spending is not total spending; it is dropped below)")
    say()


SPECS = [("A", "Miller: TWFE + state trends, 1900-1936", dict(trend="state")),
         ("B", "TWFE, no trends, 1900-1936", dict()),
         ("C", "TWFE + state trends, 1900-1919", dict(trend="state", window=LAST)),
         ("D", "TWFE, no trends, 1900-1919", dict(window=LAST))]


def all_specs(m: pd.DataFrame, y: str) -> dict:
    out = {}
    for key, _, kw in SPECS:
        b, se, n, G = did.twfe(m, y, unit="state", cluster="state", **kw)
        out[key] = (b, se, n, G)
    out["E"] = did.impute(m, "state", y, last=LAST)[0]
    out["F"] = did.impute(m, "state", y, last=LAST, trend=True)[0]
    out["G"] = did.overall(did.att_gt(m, "state", y, last=LAST))
    return out


def section_mortality(m: pd.DataFrame, rows: list) -> None:
    say("== Mortality: seven estimators per outcome (log deaths; effect of a suffrage law)")
    for key, label, _ in SPECS:
        say(f"  {key} {label}  [SE clustered by state]")
    say("  E imputation (unit and year effects fitted on untreated state-years), 1900-1919")
    say("  F imputation with a linear trend per state fitted on untreated state-years, 1900-1919")
    say("  G Callaway-Sant'Anna, not-yet-treated controls, base year g-1, 1900-1919")
    outcomes = [f"ln {a}" for a in ["under 1", "1-4", "5-9", "10-14", "15-19", "1-19", "20-24", "25-64"]]
    outcomes += [f"dd {a}" for a in ["under 1", "1-4", "5-9", "10-14", "15-19", "1-19"]]
    for y in outcomes:
        r = all_specs(m, y)
        tw = "  ".join(f"{k} {f3(r[k][0])} ({r[k][1]:.3f})" for k in "ABCD")
        say(f"{y:10s} {tw}  E {f3(r['E'])}  F {f3(r['F'])}  G {f3(r['G'])}")
        rows.append(dict(outcome=y, **{k: r[k][0] for k in "ABCD"},
                         **{f"se_{k}": r[k][1] for k in "ABCD"}, E=r["E"], F=r["F"], G=r["G"]))
    say("(ln = log deaths; dd = log deaths at that age minus log deaths at 25-64 in the same state-year)")
    a = [r["A"] for r in rows if r["outcome"] in ("ln 1-4", "ln 5-9", "ln 10-14", "ln 15-19")]
    say(f"spec A, ages 1-4 to 15-19: {pct(max(a))} to {pct(min(a))} (Miller reports 8-15%)")
    say()


def section_causes(m: pd.DataFrame, rows: list) -> None:
    say("== Causes of death, all ages (ln: log deaths; dd: relative to deaths at 25-64)")
    for name in CAUSES:
        y = f"ln {name}"
        r = all_specs(m, y)
        say(f"{name:24s} A {f3(r['A'][0])} ({r['A'][1]:.3f})  B {f3(r['B'][0])} ({r['B'][1]:.3f})  "
            f"E {f3(r['E'])}  G {f3(r['G'])}")
        rows.append(dict(outcome=y, **{k: r[k][0] for k in "ABCD"},
                         **{f"se_{k}": r[k][1] for k in "ABCD"}, E=r["E"], F=r["F"], G=r["G"]))
    say(f"relative to adult deaths, with {REPS} bootstrap draws:")
    for name in ["diarrhoea under 2", "diphtheria", "meningitis", "scarlet fever", "cancer",
                 "heart and circulation"]:
        y = f"dd {name}"
        a = did.twfe(m, y, trend="state")

        def fn(d, y=y):
            return [did.overall(did.att_gt(d, "state", y, last=LAST)), did.impute(d, "state", y, last=LAST)[0]]
        point = fn(m)
        draws = did.bootstrap(m, fn, reps=REPS, seed=5)
        (l0, h0), (l1, h1) = ci(draws[:, 0]), ci(draws[:, 1])
        say(f"  {name:22s} A {f3(a[0])} ({a[1]:.3f})  CS {f3(point[0])} [{l0:+.3f}, {h0:+.3f}]  "
            f"imputation {f3(point[1])} [{l1:+.3f}, {h1:+.3f}]")
    say()


def section_weights(m: pd.DataFrame) -> None:
    say("== Where the two-way fixed-effects estimate comes from (ln 1-19)")
    say("estimate = sum of w_it * effect_it over treated state-years; the w sum to one")
    for key, label, kw in SPECS[:2]:
        w = did.twfe_weights(m, "ln 1-19", unit="state", **kw)
        late = w.year >= 1920
        gross = w.w.abs().sum()
        say(f"{key} {label}: treated state-years {len(w)} ({late.sum()} of them in 1920-1936)")
        say(f"   net weight on 1920-1936 {w.loc[late, 'w'].sum():+.3f} (zero by construction: every state "
            f"is treated, so year effects absorb the dummy); share of absolute weight on 1920-1936 "
            f"{w.loc[late, 'w'].abs().sum() / gross:.3f}")
        say(f"   negative weights: {(w.w < 0).sum()} state-years, total {w.loc[w.w < 0, 'w'].sum():+.3f}; "
            f"before 1920 {((w.w < 0) & ~late).sum()} state-years, total {w.loc[(w.w < 0) & ~late, 'w'].sum():+.3f}")
        early = w[~late]
        say(f"   weight on pre-1920 treated cells by event time: " +
            " ".join(f"e{e} {v:+.3f}" for e, v in early.groupby("e").w.sum().items()))
        w.assign(spec=key).to_csv(RES / f"twfe_weights_{key}.csv", index=False, float_format="%.5f")
    say()


def section_window(m: pd.DataFrame) -> None:
    say("== Miller's specification as the sample is extended past 1919 (every added year is fully treated)")
    rows = []
    for end in range(1919, 1937):
        r = dict(end=end)
        for y in ["ln 1-19", "dd 1-19"]:
            for key, kw in [("A", dict(trend="state")), ("B", dict())]:
                b, se, *_ = did.twfe(m, y, window=end, **kw)
                r[f"{key} {y}"], r[f"se {key} {y}"] = b, se
        rows.append(r)
    w = pd.DataFrame(rows)
    w.to_csv(RES / "window.csv", index=False, float_format="%.4f")
    for y in ["ln 1-19", "dd 1-19"]:
        say(f"{y}, A: " + " ".join(f"{int(r.end)} {r[f'A {y}']:+.3f}" for _, r in w.iterrows()
                                 if r.end in (1919, 1921, 1923, 1925, 1927, 1929, 1931, 1933, 1936)))
        say(f"{y}, B: " + " ".join(f"{int(r.end)} {r[f'B {y}']:+.3f}" for _, r in w.iterrows()
                                 if r.end in (1919, 1921, 1923, 1925, 1927, 1929, 1931, 1933, 1936)))
    say()


def section_gap(m: pd.DataFrame) -> None:
    say("== The 1920 switch seen directly: states enfranchised in 1920 minus states enfranchised by 1914")
    lo, hi = 1910, 1925
    n = m[m.year.between(lo, hi)].groupby("state").year.nunique()
    bal = n.index[n == hi - lo + 1]
    G = m.groupby("state").g.first()
    early = sorted(s for s in bal if G[s] <= 1914)
    late = sorted(s for s in bal if G[s] == 1920)
    say(f"states reporting every year {lo}-{hi}: by 1914 {' '.join(early)}; in 1920 {' '.join(late)}")
    rows = []
    for y in ["ln 1-19", "dd 1-19", "ln 25-64"]:
        W = m.pivot_table(index="state", columns="year", values=y)
        gap = (W.loc[late].mean() - W.loc[early].mean()).loc[lo:hi]
        gap = gap - gap.loc[1919]
        steps = gap.diff().loc[1911:1919]
        say(f"{y}: " + " ".join(f"{t} {v:+.2f}" for t, v in gap.items()))
        say(f"   year-to-year changes 1911-1919: sd {steps.std():.3f}, largest {steps.abs().max():.3f}; "
            f"1919 to 1920 {gap.loc[1920] - gap.loc[1919]:+.3f}; 1919 to 1921 {gap.loc[1921] - gap.loc[1919]:+.3f}")
        rows += [dict(outcome=y, year=t, gap=v) for t, v in gap.items()]
    pd.DataFrame(rows).to_csv(RES / "gap_1920_vs_early.csv", index=False, float_format="%.4f")
    say()


def section_registration(m: pd.DataFrame) -> None:
    say("== Deaths are counts, not rates: growth after a state joins the registration area")
    m = m.sort_values(["state", "year"]).copy()
    m["entry"] = m.groupby("state").year.transform("min")
    m["k"] = m.year - m.entry
    parts = []
    for y in ["ln 25-64", "ln 1-19", "dd 1-19"]:
        m["dy"] = m.groupby("state")[y].diff()
        bench = m[m.entry == 1900].groupby("year").dy.mean()
        ex = (m.dy - m.year.map(bench))[(m.entry > 1900) & m.k.between(1, 10)]
        parts.append(f"{y} {ex.mean():+.4f} per year (n={ex.notna().sum()})")
    say("excess annual growth over the ten states reporting since 1900, first ten years after entry: "
        + "; ".join(parts))
    say()


def boot_stats(m: pd.DataFrame, y: str, last: int = LAST):
    def fn(d):
        a = did.att_gt(d, "state", y, last=last)
        ev = did.event(a, -6, 5)
        pre = ev.loc[[e for e in ev.index if e <= -2], "att"].mean()
        return [did.overall(a), pre, did.impute(d, "state", y, last=last)[0],
                did.impute(d, "state", y, last=last, trend=True)[0]] + \
            [ev.att.get(e, np.nan) for e in range(-6, 6)]
    return fn


def section_clean(m: pd.DataFrame) -> None:
    say(f"== Clean comparisons with {REPS} state-cluster bootstrap draws (percentile 95% intervals)")
    events = []
    for y in ["ln 1-19", "dd 1-19", "ln 25-64", "ln 1-4", "dd 1-4"]:
        point = boot_stats(m, y)(m)
        draws = did.bootstrap(m, boot_stats(m, y), reps=REPS, seed=7)
        names = ["CS overall", "CS mean pre (e<=-2)", "imputation", "imputation + trends"]
        parts = []
        for i, nm in enumerate(names):
            lo, hi = ci(draws[:, i])
            parts.append(f"{nm} {f3(point[i])} [{lo:+.3f}, {hi:+.3f}]")
        say(f"{y}: " + "; ".join(parts))
        if y in ("ln 1-19", "dd 1-19"):
            a = did.att_gt(m, "state", y, last=LAST)
            ev = did.event(a, -6, 5)
            for j, e in enumerate(range(-6, 6)):
                lo, hi = ci(draws[:, 4 + j])
                if e == -1:
                    lo = hi = 0.0
                events.append(dict(outcome=y, e=e, att=0.0 if e == -1 else point[4 + j], lo=lo, hi=hi,
                                   n_treat=int(ev.n_treat.get(e, 0)), cohorts=int(ev.cohorts.get(e, 0))))
    ev = pd.DataFrame(events)
    ev.to_csv(RES / "event_study.csv", index=False, float_format="%.4f")
    say("event study, ages 1-19 (CS, base e=-1):")
    for y in ["ln 1-19", "dd 1-19"]:
        e = ev[ev.outcome == y]
        say(f"  {y}: " + " ".join(f"e{int(r.e)} {r.att:+.3f}" for r in e.itertuples()))
    e = ev[ev.outcome == "ln 1-19"]
    say("  treated states per event time: " + " ".join(f"e{int(r.e)} {r.n_treat}" for r in e.itertuples()))
    say()


def section_flu(m: pd.DataFrame) -> None:
    say("== The 1918 influenza: stop the clean window in 1917")
    for y in ["ln 1-19", "dd 1-19"]:
        point = boot_stats(m, y, last=1917)(m)
        draws = did.bootstrap(m, boot_stats(m, y, last=1917), reps=REPS, seed=11)
        lo, hi = ci(draws[:, 0])
        lo2, hi2 = ci(draws[:, 2])
        say(f"{y}: CS overall {f3(point[0])} [{lo:+.3f}, {hi:+.3f}]; imputation {f3(point[2])} "
            f"[{lo2:+.3f}, {hi2:+.3f}]")
    spike = m[m.year.between(1916, 1919)].groupby("year")[["influenza", "pneumonia"]].sum()
    say("influenza + pneumonia deaths in reporting states: " +
        ", ".join(f"{y} {int(r.influenza + r.pneumonia):,}" for y, r in spike.iterrows()))
    say()


def section_anticipation(m: pd.DataFrame, pf: pd.DataFrame) -> None:
    say("== Allowing one year of anticipation (base year g-2; controls not treated by t+1)")
    for y in ["ln 1-19", "dd 1-19"]:
        a0 = did.overall(did.att_gt(m, "state", y, last=LAST))
        a1 = did.overall(did.att_gt(m, "state", y, last=LAST, anticipation=1))
        say(f"{y}: CS {f3(a0)}; with anticipation {f3(a1)}")
    d = pf[pf.year >= 1906]
    for name in ["total", "health and sanitation"]:
        y = f"ln {name}"
        a0 = did.overall(did.att_gt(d, "city", y, last=LAST))
        a1 = did.overall(did.att_gt(d, "city", y, last=LAST, anticipation=1))
        say(f"spending, {name}: CS {f3(a0)}; with anticipation {f3(a1)}")
    say()


def section_coding() -> None:
    say("== Coding the suffrage dates (ages 1-19)")
    for coding in ["miller", "full_only", "lag"]:
        m = load_mortality(coding)
        parts = []
        for y in ["ln 1-19", "dd 1-19"]:
            b, se, *_ = did.twfe(m, y, trend="state")
            g = did.overall(did.att_gt(m, "state", y, last=LAST))
            e = did.impute(m, "state", y, last=LAST)[0]
            parts.append(f"{y}: A {f3(b)} ({se:.3f}) imputation {f3(e)} CS {f3(g)}")
        say(f"{coding:9s} " + "; ".join(parts))
    say()


def section_leave_one_out(m: pd.DataFrame) -> None:
    say("== Leave one early-suffrage state out (ages 1-19)")
    early = sorted(m.loc[(m.g <= LAST) & (m.groupby("state").year.transform("min") < m.g), "state"].unique())
    for y in ["ln 1-19", "dd 1-19"]:
        vals_a, vals_g = [], []
        for s in early:
            d = m[m.state != s]
            vals_a.append(did.twfe(d, y, trend="state")[0])
            vals_g.append(did.overall(did.att_gt(d, "state", y, last=LAST)))
        ia, ig = int(np.argmax(np.abs(vals_a))), int(np.argmax(np.abs(vals_g)))
        say(f"{y}: spec A ranges {min(vals_a):+.3f} to {max(vals_a):+.3f}; "
            f"CS ranges {min(vals_g):+.3f} to {max(vals_g):+.3f} "
            f"(largest CS when dropping {early[ig]}, largest A when dropping {early[ia]})")
    say()


def section_finance(pf: pd.DataFrame) -> None:
    say("== Municipal spending (city fixed effects, year fixed effects, state trends; SE clustered by state)")
    miller = {"total": (0.077, 0.031), "health and sanitation": (0.063, 0.036),
              "charities, hospitals, corrections": (0.451, 0.115)}
    say("Miller, Online Appendix Table 8, column 1: total +0.077 (0.031), health and sanitation "
        "+0.063 (0.036), charities, hospitals and corrections +0.451 (0.115)")
    for name in ["total", "health and sanitation", "health", "sanitation", "charities, hospitals, corrections"]:
        y = f"ln {name}"
        a = did.twfe(pf, y, unit="city", trend="state", cluster="state")
        a06 = did.twfe(pf, y, unit="city", trend="state", cluster="state", first=1906)
        b06 = did.twfe(pf, y, unit="city", cluster="state", first=1906)
        r = did.twfe(pf, y, unit="rank", trend="state", cluster="state")
        ref = f" [Miller {miller[name][0]:+.3f} ({miller[name][1]:.3f})]" if name in miller else ""
        say(f"{name:34s} A {f3(a[0])} ({a[1]:.3f}) n={a[2]}{ref}")
        say(f"{'':34s} A from 1906 {f3(a06[0])} ({a06[1]:.3f}); B from 1906 {f3(b06[0])} ({b06[1]:.3f}); "
            f"A with the rank column as city id {f3(r[0])} ({r[1]:.3f})")
    say(f"the column 'id' is the population rank of the city in that year: New York is 1 in every year; "
        f"rank 4 is {pf.loc[(pf['rank'] == 4) & (pf.year == 1905), 'city_name'].iloc[0]} in 1905 and "
        f"{pf.loc[(pf['rank'] == 4) & (pf.year == 1930), 'city_name'].iloc[0]} in 1930; "
        f"cities are identified here by state and place FIPS codes ({pf.city.nunique()} cities)")
    say("clean comparisons 1906-1919 (bootstrap over states):")
    d = pf[pf.year >= 1906]
    for name in ["total", "health and sanitation", "health", "sanitation"]:
        y = f"ln {name}"

        def fn(x, y=y):
            a = did.att_gt(x, "city", y, last=LAST)
            ev = did.event(a, -6, 5)
            return [did.overall(a), did.impute(x, "city", y, last=LAST)[0],
                    ev.loc[[e for e in ev.index if e <= -2], "att"].mean()]
        point = fn(d)
        draws = did.bootstrap(d, fn, reps=REPS, seed=3, cluster="state", relabel=("state", "city"))
        (l0, h0), (l1, h1), (l2, h2) = ci(draws[:, 0]), ci(draws[:, 1]), ci(draws[:, 2])
        say(f"  {name:22s} CS {f3(point[0])} [{l0:+.3f}, {h0:+.3f}]  imputation {f3(point[1])} "
            f"[{l1:+.3f}, {h1:+.3f}]  CS mean pre (e<=-2) {f3(point[2])} [{l2:+.3f}, {h2:+.3f}]")
        if name == "health and sanitation":
            ev = did.event(did.att_gt(d, "city", y, last=LAST), -6, 5)
            say("  event study, health and sanitation (cohorts in brackets): " +
                " ".join(f"e{e} {r.att:+.3f}({int(r.cohorts)})" for e, r in ev.iterrows()))
            ev.to_csv(RES / "event_study_health_spending.csv", float_format="%.4f")
    say()


def main() -> int:
    if not (RAW / "mortality_harmonized.dta").exists():
        print("data/raw is empty: run fetch.py first", file=sys.stderr)
        return 1
    RES.mkdir(exist_ok=True)
    m = load_mortality()
    pf = load_finance()
    section_data(m, pf)
    rows: list = []
    section_mortality(m, rows)
    section_causes(m, rows)
    pd.DataFrame(rows).to_csv(RES / "estimates.csv", index=False, float_format="%.4f")
    section_weights(m)
    section_window(m)
    section_gap(m)
    section_registration(m)
    section_clean(m)
    section_flu(m)
    section_coding()
    section_leave_one_out(m)
    section_finance(pf)
    section_anticipation(m, pf)
    (RES / "summary.txt").write_text("\n".join(OUT) + "\n")
    print("\n".join(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
