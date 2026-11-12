"""Chinese industries from the China Industrial Productivity (CIP) database, 4.0 (2023 release), RIETI.

The workbooks are read as published from ``data/raw/cip/``. Manufacturing is CIP industries 6 to 24 (19 industries).
Real value added is a Tornqvist aggregate of the CIP's real value-added growth rates by industry (growth accounts,
sheet ``g(VA)``) weighted by nominal value added, expressed in 2000 yuan; the capital stock is the sum of the CIP's
real net capital stocks in 2000 yuan; employment and hours are sums.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.optimize import brentq

RAW = Path(__file__).parent / "data" / "raw" / "cip"
DATA = Path(__file__).parent / "data"
MANUF = list(range(6, 25))
ALL = list(range(1, 38))
GROUPS = {
    "Agriculture": [1], "Mining": [2, 3, 4, 5], "Manufacturing": MANUF, "Utilities": [25], "Construction": [26],
    "Transport": [29], "Real estate": [32], "Public administration and services": [34, 35, 36],
    "Other services": [27, 28, 30, 31, 33, 37],
}
ACRONYM = {6: "F&B", 7: "TBC", 8: "TEX", 9: "WEA", 10: "LEA", 11: "W&F", 12: "P&P", 13: "PET", 14: "CHE", 15: "R&P",
           16: "BUI", 17: "MET", 18: "MEP", 19: "MCH", 20: "ELE", 21: "ICT", 22: "INS", 23: "TRS", 24: "OTH"}
NAME = {6: "Food, beverages", 7: "Tobacco", 8: "Textiles", 9: "Apparel", 10: "Leather, footwear", 11: "Wood, furniture",
        12: "Paper, printing", 13: "Petroleum, coal products", 14: "Chemicals", 15: "Rubber, plastics",
        16: "Building materials", 17: "Basic metals", 18: "Metal products", 19: "Machinery", 20: "Electrical equipment",
        21: "Electronics, telecom equipment", 22: "Instruments", 23: "Transport equipment", 24: "Other manufacturing"}
DELTA = 0.08          # depreciation rate at which the CIP manufacturing stock implies a smooth investment deflator
DELTAS = (0.06, 0.08, 0.10)


def available() -> bool:
    return (RAW / "CIP_4.0_(2023)_2.2.xlsx").exists()


def _table(fname: str, sheet: str = "Sheet1") -> pd.DataFrame:
    """First block of a CIP sheet: industries (CIP code, or 'Total') by year."""
    wb = openpyxl.load_workbook(RAW / fname, read_only=True, data_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    h = next(i for i, r in enumerate(rows) if r and str(r[0]).strip().lower() == "cip code")
    years = [c for c in rows[h] if isinstance(c, int)]
    first = next(j for j, c in enumerate(rows[h]) if isinstance(c, int))
    out = {}
    for r in rows[h + 1:]:
        if r is None or (r[0] is None and r[1] is None) or isinstance(r[0], str):
            break
        out[r[0] if r[0] is not None else "Total"] = [np.nan if v is None else float(v) for v in r[first:first + len(years)]]
    return pd.DataFrame(out, index=years).T


def load() -> dict[str, pd.DataFrame]:
    g4 = "CIP_4.0_(2023)_4.xlsx"
    return {
        "K": _table("CIP_4.0_(2023)_2.2.xlsx"), "I": _table("CIP_4.0_(2023)_2.1.xlsx"),
        "E": _table("CIP_4.0_(2023)_3.1.xlsx"), "H": _table("CIP_4.0_(2023)_3.2.xlsx"),
        "VA": _table(g4, "Nominal VA").drop(index="Total"), "gVA": _table(g4, "g(VA)").drop(index="Total"),
        "KC": _table(g4, "KC").drop(index="Total"), "VA12": _table("CIP_4.0_(2023)_1.2.xlsx"),
    }


def real_va(d: dict[str, pd.DataFrame], codes: list[int]) -> pd.Series:
    """Tornqvist aggregate of real value added, in 2000 yuan."""
    w, g = d["VA"].loc[codes], d["gVA"].loc[codes]
    sh = w / w.sum()
    years = [y for y in w.columns if y - 1 in w.columns]
    gm = pd.Series({y: float(((sh[y] + sh[y - 1]) / 2 * g[y]).sum()) for y in years})
    idx = np.exp(gm.cumsum())
    idx.loc[w.columns[0]] = 1.0
    idx = idx.sort_index()
    return idx / idx.loc[2000] * w[2000].sum()


def aggregate(d: dict[str, pd.DataFrame], codes: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"Y": real_va(d, codes), "K": d["K"].loc[codes].sum(), "E": d["E"].loc[codes].sum(),
                         "H": d["H"].loc[codes].sum(), "VA": d["VA"].loc[codes].sum(), "KC": d["KC"].loc[codes].sum(),
                         "I": d["I"].loc[codes].sum()})


def g(s: pd.Series, a: int, b: int) -> float:
    return 100 * (np.log(s.loc[b]) - np.log(s.loc[a])) / (b - a)


def f2(x: float) -> str:
    return f"{round(float(x), 2) + 0.0:.2f}"


def nbs_fai_shares() -> dict[int, float]:
    n = pd.read_csv(DATA / "nbs_published.csv")
    v = n.pivot_table(index="year", columns="indicator", values="value", aggfunc="first")
    return {int(y): float(v.loc[y, "fai_manufacturing"] / v.loc[y, "fai_total"]) for y in v.index
            if not np.isnan(v.loc[y].get("fai_manufacturing", np.nan))}


def capital_path(m: pd.DataFrame, i_total: pd.Series, shares: dict[int, float], delta: float) -> pd.Series:
    """Manufacturing capital 2013-2017 if manufacturing had received the given shares of total investment.

    Real investment uses the deflator implied by the CIP manufacturing stock at the same depreciation rate.
    """
    p = m.I / (m.K - (1 - delta) * m.K.shift(1))
    k = {2013: float(m.K[2013])}
    for y in range(2014, 2018):
        k[y] = (1 - delta) * k[y - 1] + shares[y] * float(i_total[y]) / float(p[y])
    return pd.Series(k)


def analyse(say, section, res: Path) -> None:
    d = load()
    m = aggregate(d, MANUF)
    tot = aggregate(d, ALL)
    oecd = pd.read_csv(DATA / "oecd_t1.csv")
    gfcf = oecd[(oecd.REF_AREA == "CHN") & (oecd.PRICE_BASE == "V") & (oecd.TRANSACTION == "P51G")].set_index(
        "TIME_PERIOD").OBS_VALUE
    gdp = oecd[(oecd.REF_AREA == "CHN") & (oecd.PRICE_BASE == "V") & (oecd.TRANSACTION == "B1GQ")].set_index(
        "TIME_PERIOD").OBS_VALUE

    section("CIP 4.0: coverage and checks")
    say("CIP years; industries; manufacturing industries", f"{d['K'].columns.min()}-{d['K'].columns.max()}; {len(ALL)}; {len(MANUF)}")
    say("CIP total investment / OECD gross fixed capital formation (China): 2007; 2013; 2017",
        "; ".join(f"{tot.I[y] / gfcf[y]:.3f}" for y in (2007, 2013, 2017)))
    say("CIP total nominal value added / OECD GDP (China): 2007; 2013; 2017",
        "; ".join(f"{tot.VA[y] / gdp[y]:.3f}" for y in (2007, 2013, 2017)))
    say("Max |growth-accounts nominal VA - file 1.2 VA| / VA, manufacturing industries",
        f"{float((d['VA'].loc[MANUF] - d['VA12'].loc[MANUF, d['VA'].columns]).abs().div(d['VA'].loc[MANUF]).max().max()):.6f}")

    section("CIP 4.0: Chinese manufacturing (19 industries), value added and capital in 2000 yuan")
    periods = [(1995, 2001), (2001, 2007), (1995, 2007), (2007, 2013), (2013, 2017), (2007, 2017)]
    rows = []
    for a, b in periods:
        rows.append({"period": f"{a}-{b}", "y": g(m.Y, a, b), "k": g(m.K, a, b), "e": g(m.E, a, b), "h": g(m.H, a, b),
                     "y_per_worker": g(m.Y / m.E, a, b), "k_per_worker": g(m.K / m.E, a, b),
                     "y_per_hour": g(m.Y / m.H, a, b), "k_per_hour": g(m.K / m.H, a, b), "y_per_k": g(m.Y / m.K, a, b)})
        r = rows[-1]
        say(f"Manufacturing {a}-{b}, average annual log growth, percent: value added; capital; employment; hours",
            f"{f2(r['y'])}; {f2(r['k'])}; {f2(r['e'])}; {f2(r['h'])}")
        say(f"Manufacturing {a}-{b}: output per worker; capital per worker; output per unit of capital | per hour: output; capital",
            f"{f2(r['y_per_worker'])}; {f2(r['k_per_worker'])}; {f2(r['y_per_k'])} | {f2(r['y_per_hour'])}; {f2(r['k_per_hour'])}")
    pd.DataFrame(rows).to_csv(res / "cip_manufacturing_periods.csv", index=False, float_format="%.4f")
    yk = m.Y / m.K
    annual = 100 * np.log(yk).diff()
    say("Manufacturing years 1996-2017 in which output per unit of capital fell, and by how much (percent)",
        ", ".join(f"{y} {f2(v)}" for y, v in annual.loc[1996:].items() if v < 0))
    say("Manufacturing capital-output ratio, 2000 yuan: 1995; 2001; 2007; 2013; 2017",
        "; ".join(f2(m.K[y] / m.Y[y]) for y in (1995, 2001, 2007, 2013, 2017)))
    say("Manufacturing employment, millions: 1995; 2007; 2013; 2017",
        "; ".join(f"{m.E[y] / 1000:.1f}" for y in (1995, 2007, 2013, 2017)))
    cs = m.KC / m.VA
    say("Manufacturing capital compensation / value added: 2007; 2013; 2017", "; ".join(f"{cs[y]:.3f}" for y in (2007, 2013, 2017)))
    ir = m.I / m.VA
    say("Manufacturing investment / nominal value added: 2007; 2010; 2013; 2015; 2017",
        "; ".join(f"{ir[y]:.3f}" for y in (2007, 2010, 2013, 2015, 2017)))
    say("Manufacturing investment, trillion current yuan: 2007; 2010; 2013; 2014; 2015; 2016; 2017",
        "; ".join(f"{m.I[y] / 1e6:.2f}" for y in (2007, 2010, 2013, 2014, 2015, 2016, 2017)))
    for a, b in [(2007, 2013), (2013, 2017)]:
        v = (cs[a] + cs[b]) / 2
        deep = v * g(m.K / m.H, a, b)
        say(f"Manufacturing {a}-{b}: output per hour growth = capital deepening (capital share {v:.3f} x capital per hour) + remainder",
            f"{f2(g(m.Y / m.H, a, b))} = {f2(deep)} + {f2(g(m.Y / m.H, a, b) - deep)}")
    m.assign(Y_per_K=yk).to_csv(res / "cip_manufacturing_annual.csv", float_format="%.4f")

    section("CIP 4.0: where China's investment went (sector groups)")
    rows = []
    for name, codes in list(GROUPS.items()) + [("Total economy", ALL)]:
        s = aggregate(d, codes)
        r = {"group": name}
        for a, b in [(1995, 2007), (2007, 2013), (2013, 2017)]:
            r[f"y_per_k_{a}_{b}"] = g(s.Y / s.K, a, b)
        for y in (2007, 2013, 2017):
            r[f"inv_share_{y}"] = s.I[y] / tot.I[y]
            r[f"k_share_{y}"] = s.K[y] / tot.K[y]
            r[f"va_share_{y}"] = s.VA[y] / tot.VA[y]
        rows.append(r)
    t = pd.DataFrame(rows).set_index("group")
    t.to_csv(res / "cip_sectors.csv", float_format="%.4f")
    for name in t.index:
        r = t.loc[name]
        say(f"{name}: output per unit of capital growth 1995-2007; 2007-2013; 2013-2017 | share of investment 2007; 2013; 2017 | share of value added 2017",
            f"{f2(r.y_per_k_1995_2007)}; {f2(r.y_per_k_2007_2013)}; {f2(r.y_per_k_2013_2017)} | "
            f"{r.inv_share_2007:.3f}; {r.inv_share_2013:.3f}; {r.inv_share_2017:.3f} | {r.va_share_2017:.3f}")
    shares = pd.DataFrame({name: d["I"].loc[codes].sum() / tot.I for name, codes in GROUPS.items()})
    shares.index.name = "year"
    shares.to_csv(res / "cip_investment_shares.csv", float_format="%.5f")
    noncore = [c for c in ALL if c not in MANUF]
    s = aggregate(d, noncore)
    pd.DataFrame({"china_manufacturing_cip": m.Y / m.K, "china_outside_manufacturing_cip": s.Y / s.K,
                  "china_total_cip": tot.Y / tot.K}).rename_axis("year").to_csv(res / "cip_output_per_capital.csv",
                                                                                float_format="%.6f")
    say("Economy outside manufacturing: output per unit of capital growth 2007-2013; 2013-2017",
        f"{f2(g(s.Y / s.K, 2007, 2013))}; {f2(g(s.Y / s.K, 2013, 2017))}")
    infra = GROUPS["Utilities"] + GROUPS["Transport"] + GROUPS["Real estate"] + GROUPS["Public administration and services"]
    say("Share of China's investment in utilities, transport, real estate and public administration/services: 2007; 2013; 2017",
        "; ".join(f"{d['I'].loc[infra, y].sum() / tot.I[y]:.3f}" for y in (2007, 2013, 2017)))

    section("CIP 4.0: the 19 manufacturing industries, 2013-2017")
    rows = []
    for c in MANUF:
        s = aggregate(d, [c])
        rows.append({"code": c, "industry": NAME[c], "acronym": ACRONYM[c],
                     "y_per_k_2007_2013": g(s.Y / s.K, 2007, 2013), "y_per_k_2013_2017": g(s.Y / s.K, 2013, 2017),
                     "k_per_worker_2013_2017": g(s.K / s.E, 2013, 2017), "y_per_worker_2013_2017": g(s.Y / s.E, 2013, 2017),
                     "va_share_2013": s.VA[2013] / m.VA[2013], "k_share_2013": s.K[2013] / m.K[2013],
                     "inv_change_2013_2017": s.I[2017] / s.I[2013] - 1})
    ind = pd.DataFrame(rows).set_index("code")
    ind.to_csv(res / "cip_industries.csv", float_format="%.4f")
    fell = ind[ind.y_per_k_2013_2017 < 0]
    say("Manufacturing industries with falling output per unit of capital 2013-2017 (of 19); their share of manufacturing value added 2013",
        f"{len(fell)} ({', '.join(fell.industry)}); {fell.va_share_2013.sum():.3f}")
    say("Manufacturing industries with falling output per unit of capital 2007-2013 (of 19)",
        f"{int((ind.y_per_k_2007_2013 < 0).sum())}")
    for c in (8, 9, 10, 14, 16, 17, 19, 20, 21, 23):
        r = ind.loc[c]
        say(f"  {r.industry}: output per unit of capital 2007-2013; 2013-2017 | capital per worker; output per worker 2013-2017 | investment 2017/2013 - 1",
            f"{f2(r.y_per_k_2007_2013)}; {f2(r.y_per_k_2013_2017)} | {f2(r.k_per_worker_2013_2017)}; {f2(r.y_per_worker_2013_2017)} | {r.inv_change_2013_2017:.2f}")

    section("How much depends on the allocation of investment to manufacturing (scenarios for 2014-2017)")
    shares_cip = (m.I / tot.I)
    say("Manufacturing share of China's investment in CIP: 2007; 2010; 2013; 2014; 2015; 2016; 2017",
        "; ".join(f"{shares_cip[y]:.3f}" for y in (2007, 2010, 2013, 2014, 2015, 2016, 2017)))
    fai = nbs_fai_shares()
    say("Manufacturing share of NBS fixed-asset investment (excluding rural households): 2013; 2017",
        f"{fai[2013]:.3f}; {fai[2017]:.3f}")
    say("Ratio 2017/2013 of manufacturing's investment share: CIP (GFCF allocation); NBS fixed-asset investment",
        f"{shares_cip[2017] / shares_cip[2013]:.2f}; {fai[2017] / fai[2013]:.2f}")
    for delta in DELTAS:
        p = m.I / (m.K - (1 - delta) * m.K.shift(1))
        say(f"Implied manufacturing investment deflator at depreciation {delta:.2f} (2000=1): 1995; 2005; 2013; 2017",
            "; ".join(f"{p[y]:.2f}" for y in (1995, 2005, 2013, 2017)))
    s13 = float(shares_cip[2013])
    fai_path = {y: s13 * (fai[2013] + (fai[2017] - fai[2013]) * (y - 2013) / 4) / fai[2013] for y in range(2014, 2018)}
    scen = {"CIP as published": {y: float(shares_cip[y]) for y in range(2014, 2018)},
            "share held at 2013 level": {y: s13 for y in range(2014, 2018)},
            "share falls as in NBS fixed-asset investment": fai_path}
    grid = []
    yg = g(m.Y, 2013, 2017)
    for delta in DELTAS:
        for name, sh in scen.items():
            k = capital_path(m, tot.I, sh, delta)
            yk_g = yg - 100 * (np.log(k[2017]) - np.log(k[2013])) / 4
            say(f"Scenario '{name}', depreciation {delta:.2f}: average share 2014-2017; output per unit of capital growth 2013-2017; capital per worker growth",
                f"{np.mean(list(sh.values())):.3f}; {f2(yk_g)}; {f2(100 * (np.log(k[2017] / m.E[2017]) - np.log(k[2013] / m.E[2013])) / 4)}")
        target = float(m.K[2013]) * np.exp(4 * yg / 100)

        def k17(share: float, delta: float = delta) -> float:
            return float(capital_path(m, tot.I, {y: share for y in range(2014, 2018)}, delta)[2017])

        s_star = brentq(lambda s: k17(s) - target, 0.01, 0.9)
        say(f"Break-even constant share 2014-2017 at which manufacturing output per unit of capital would not have risen, depreciation {delta:.2f}",
            f"{s_star:.3f}")
        for share in np.round(np.arange(0.05, 0.401, 0.01), 2):
            k = capital_path(m, tot.I, {y: share for y in range(2014, 2018)}, delta)
            grid.append({"delta": delta, "share": share, "y_per_k_growth": yg - 100 * (np.log(k[2017]) - np.log(k[2013])) / 4})
    pd.DataFrame(grid).to_csv(res / "cip_share_grid.csv", index=False, float_format="%.4f")
