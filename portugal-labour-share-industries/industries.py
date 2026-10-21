"""Portugal's labour share decomposed by industry, 1995-2024.

Companion code for the Medium follow-up on the Portuguese labour share. Every
number quoted in the article is written to ``results/summary.txt``; the
tables behind the figures are written to ``results/*.csv``.

Run from this folder::

    python industries.py

Inputs are the Eurostat snapshot in ``data/`` (see ``fetch.py``) and, for
the economy-wide mixed income used in section 2, the snapshot of the first
article in ``../portugal-labour-share/data/``.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIRST_ARTICLE_DATA = HERE.parent / "portugal-labour-share" / "data"
RESULTS = HERE / "results"

FIRST_YEAR = 1995
LAST_YEAR = 2024
SEED = 20261021
N_PERM = 9999
GEOS = ("PT", "ES", "IT", "EL")
WINDOWS = ((1995, 2024), (2010, 2013), (2016, 2024))

# The 64 industries of the A64 breakdown (U, extraterritorial organisations,
# has zero value added and is dropped).
A64 = (
    "A01 A02 A03 B C10-C12 C13-C15 C16 C17 C18 C19 C20 C21 C22 C23 C24 C25 C26 C27 C28 "
    "C29 C30 C31_C32 C33 D E36 E37-E39 F G45 G46 G47 H49 H50 H51 H52 H53 I J58 J59_J60 "
    "J61 J62_J63 K64 K65 K66 L M69_M70 M71 M72 M73 M74_M75 N77 N78 N79 N80-N82 O P Q86 "
    "Q87_Q88 R90-R92 R93 S94 S95 S96 T U"
).split()

# Industries whose factor-cost value added is negative in some year are merged
# with a neighbour, because a share computed on a near-zero or negative
# denominator makes the shift-share terms explode.
MERGES = {"C19": "C19_C20", "C20": "C19_C20", "H51": "H51_H52", "H52": "H51_H52"}
DROPPED = ("U",)

A10_OF_SECTION = {
    "A": "A", "B": "B-E", "C": "B-E", "D": "B-E", "E": "B-E", "F": "F", "G": "G-I", "H": "G-I",
    "I": "G-I", "J": "J", "K": "K", "L": "L", "M": "M_N", "N": "M_N", "O": "O-Q", "P": "O-Q",
    "Q": "O-Q", "R": "R-U", "S": "R-U", "T": "R-U", "U": "R-U",
}
SECTION_NAMES = {
    "A": "agriculture, forestry, fishing", "B": "mining", "C": "manufacturing",
    "D": "electricity and gas", "E": "water and waste", "F": "construction",
    "G": "wholesale and retail trade", "H": "transport and storage",
    "I": "accommodation and food", "J": "information and communication",
    "K": "finance and insurance", "L": "real estate", "M": "professional and technical",
    "N": "administrative and support", "O": "public administration", "P": "education",
    "Q": "health and social work", "R": "arts and recreation", "S": "other services",
    "T": "households as employers",
}
# Sections removed in the "market economy" scope: real estate (dominated by the
# imputed rent of owner-occupied housing), the mostly non-market public
# administration, education and health, and households as employers (whose
# value added is, by construction, their employees' pay).
NON_MARKET_OR_HOUSING = ("L", "O", "P", "Q", "T")

LEVELS = ("a10", "a21", "a64")
NUMERATORS = ("raw", "adjusted")
SCOPES = ("economy", "market")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def section_of(code: str) -> str:
    """NACE section (one letter) of an industry code."""
    return code[0]


def group_of(code: str, level: str) -> str:
    """Group an industry code at the requested level of detail."""
    if level == "a64":
        return code
    if level == "a21":
        return section_of(code)
    if level == "a10":
        return A10_OF_SECTION[section_of(code)]
    raise ValueError(f"Unknown level {level!r}")


def read_long(name: str) -> pd.DataFrame:
    """Read one snapshot file restricted to the sample years."""
    frame = pd.read_csv(DATA / f"{name}.csv")
    return frame[(frame["year"] >= FIRST_YEAR) & (frame["year"] <= LAST_YEAR)]


def load_industries(geo: str) -> pd.DataFrame:
    """Return the A64 panel for one country, with unstable industries merged.

    Columns: ``code``, ``year``, ``B1G``, ``D1``, ``D11``, ``D29X39``,
    ``EMP``, ``SAL`` (thousand persons), ``EMP_HW``, ``SAL_HW`` (thousand
    hours, NaN where not published) and ``Y``, factor-cost value added
    (B.1g less other taxes net of subsidies on production).
    """
    acc = read_long("nama_10_a64")
    acc = acc[(acc["geo"] == geo) & acc["nace_r2"].isin(A64)]
    acc = acc.pivot_table(index=["nace_r2", "year"], columns="na_item", values="value", dropna=False)
    emp = read_long("nama_10_a64_e")
    emp = emp[(emp["geo"] == geo) & emp["nace_r2"].isin(A64)].copy()
    emp["column"] = emp["na_item"].str[:3] + np.where(emp["unit"] == "THS_HW", "_HW", "")
    emp = emp.pivot_table(index=["nace_r2", "year"], columns="column", values="value", dropna=False)
    emp = emp.rename(columns={"EMP": "EMP", "SAL": "SAL"})
    panel = acc.join(emp).reset_index().rename(columns={"nace_r2": "code"})
    panel = panel[~panel["code"].isin(DROPPED)]
    panel["code"] = panel["code"].replace(MERGES)
    value_cols = ["B1G", "D1", "D11", "D29X39", "EMP", "SAL", "EMP_HW", "SAL_HW"]
    panel = panel.groupby(["code", "year"])[value_cols].sum(min_count=1).reset_index()
    for col in ("B1G", "D1", "D11", "D29X39", "EMP", "SAL"):
        if panel[col].isna().any():
            raise ValueError(f"{geo}: missing {col} in the A64 panel")
    panel["Y"] = panel["B1G"] - panel["D29X39"]
    return panel.sort_values(["code", "year"]).reset_index(drop=True)


def load_a10_hours(geo: str) -> pd.DataFrame:
    """Return A10 employment in hours (complete for every country)."""
    emp = read_long("nama_10_a10_e")
    emp = emp[(emp["geo"] == geo) & (emp["unit"] == "THS_HW") & (emp["nace_r2"] != "TOTAL") & (emp["nace_r2"] != "C")]
    return emp.pivot_table(index=["nace_r2", "year"], columns="na_item", values="value").reset_index()


def data_checks(geo: str) -> dict[str, float]:
    """Check that the A64 partition adds up to the published total."""
    acc = read_long("nama_10_a64")
    acc = acc[acc["geo"] == geo]
    wide = acc.pivot_table(index=["nace_r2", "year"], columns="na_item", values="value")
    parts = wide.loc[[c for c in A64 if c in wide.index.get_level_values(0)]].groupby(level="year").sum()
    total = wide.loc["TOTAL"]
    panel = load_industries(geo)
    return {
        "max_gap_B1G_pct": float(((parts["B1G"] - total["B1G"]) / total["B1G"]).abs().max() * 100),
        "max_gap_D1_pct": float(((parts["D1"] - total["D1"]) / total["D1"]).abs().max() * 100),
        "industries_after_merges": float(panel["code"].nunique()),
        "non_positive_value_added_cells": float((panel["Y"] <= 0).sum()),
    }


# ---------------------------------------------------------------------------
# Labour income of the self-employed
# ---------------------------------------------------------------------------


def imputed_labour(panel: pd.DataFrame, level: str, cap: bool = True) -> pd.Series:
    """Labour income of the self-employed, each credited with their group's pay per employee.

    The imputation is done within each group at ``level`` ("total" credits
    every self-employed person with the economy-wide average). With ``cap``,
    the imputed income is limited to the group's value added less its
    compensation of employees (floored at zero), so imputation never takes a
    group's labour share above 100% or above its compensation share.

    Returns
    -------
    pd.Series
        Imputed income by year, million euro.
    """
    frame = panel.copy()
    frame["g"] = "total" if level == "total" else frame["code"].map(lambda c: group_of(c, level))
    g = frame.groupby(["g", "year"])[["D1", "EMP", "SAL", "Y"]].sum()
    pay = g["D1"] / g["SAL"].where(g["SAL"] > 0)
    imp = (pay * (g["EMP"] - g["SAL"])).fillna(0.0)
    if cap:
        imp = np.minimum(imp, (g["Y"] - g["D1"]).clip(lower=0.0))
    return imp.groupby(level="year").sum()


def imputed_labour_hours_a10(geo: str, panel: pd.DataFrame) -> pd.Series:
    """Per-hour imputation within each A10 branch (A10 hours are complete)."""
    hours = load_a10_hours(geo)
    frame = panel.copy()
    frame["g"] = frame["code"].map(lambda c: group_of(c, "a10"))
    g = frame.groupby(["g", "year"])[["D1", "Y"]].sum().reset_index()
    g = g.merge(hours.rename(columns={"nace_r2": "g"}), on=["g", "year"], how="left")
    imp = (g["D1"] / g["SAL_DC"] * (g["EMP_DC"] - g["SAL_DC"])).fillna(0.0)
    imp = np.minimum(imp, (g["Y"] - g["D1"]).clip(lower=0.0))
    return imp.groupby(g["year"]).sum()


def cap_binding(panel: pd.DataFrame) -> pd.DataFrame:
    """Industry-years where the A64 per-person imputation exceeds operating surplus and mixed income."""
    pay = panel["D1"] / panel["SAL"].where(panel["SAL"] > 0)
    imp = (pay * (panel["EMP"] - panel["SAL"])).fillna(0.0)
    binding = panel[imp > (panel["Y"] - panel["D1"])]
    return binding[["code", "year"]]


def labour_by_industry(panel: pd.DataFrame, numerator: str) -> pd.Series:
    """Labour income by A64 industry and year (self-employed imputed at A64, capped)."""
    if numerator == "raw":
        return panel.set_index(["code", "year"])["D1"]
    pay = panel["D1"] / panel["SAL"].where(panel["SAL"] > 0)
    imp = (pay * (panel["EMP"] - panel["SAL"])).fillna(0.0)
    imp = np.minimum(imp, (panel["Y"] - panel["D1"]).clip(lower=0.0))
    return (panel["D1"] + imp).set_axis(pd.MultiIndex.from_frame(panel[["code", "year"]]))


# ---------------------------------------------------------------------------
# Shift-share
# ---------------------------------------------------------------------------


def grouped(panel: pd.DataFrame, numerator: str, level: str, scope: str) -> pd.DataFrame:
    """Labour income ``L`` and value added ``Y`` by group and year."""
    frame = panel[["code", "year", "Y"]].copy()
    frame["L"] = labour_by_industry(panel, numerator).to_numpy()
    if scope == "market":
        frame = frame[~frame["code"].map(section_of).isin(NON_MARKET_OR_HOUSING)]
    elif scope != "economy":
        raise ValueError(f"Unknown scope {scope!r}")
    frame["g"] = frame["code"].map(lambda c: group_of(c, level))
    return frame.groupby(["g", "year"])[["L", "Y"]].sum().reset_index()


def shift_share(frame: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Exact midpoint decomposition of the change in the aggregate share.

    With weights w = Y_i / sum Y and shares s = L_i / Y_i,
    d(sum w s) = sum w_bar ds (within) + sum dw s_bar (between).

    Returns
    -------
    pd.DataFrame
        One row per group with weights, shares and the two contributions, in
        percentage points.
    """
    a = frame[frame["year"] == start].set_index("g")
    b = frame[frame["year"] == end].set_index("g")
    w0, w1 = a["Y"] / a["Y"].sum(), b["Y"] / b["Y"].sum()
    s0, s1 = a["L"] / a["Y"], b["L"] / b["Y"]
    out = pd.DataFrame({
        "w_start": 100 * w0, "w_end": 100 * w1, "s_start": 100 * s0, "s_end": 100 * s1,
        "within": 100 * (w0 + w1) / 2 * (s1 - s0),
        "between": 100 * (w1 - w0) * (s0 + s1) / 2,
    })
    total = 100 * (b["L"].sum() / b["Y"].sum() - a["L"].sum() / a["Y"].sum())
    if abs(out["within"].sum() + out["between"].sum() - total) > 1e-9:
        raise AssertionError("Shift-share does not add up.")
    return out


def chained(frame: pd.DataFrame, start: int, end: int) -> tuple[float, float]:
    """Sum of the year-on-year decompositions between two years."""
    within = between = 0.0
    for year in range(start, end):
        step = shift_share(frame, year, year + 1)
        within += step["within"].sum()
        between += step["between"].sum()
    return within, between


# ---------------------------------------------------------------------------
# Pay and the within-industry change
# ---------------------------------------------------------------------------


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation."""
    w = w / w.sum()
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    return float(cov / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2)))


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks."""
    return pd.Series(x).rank().to_numpy()


def pay_and_share(panel: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Pay per employee, value added per person employed and share change by A64 industry."""
    a = panel[panel["year"] == start].set_index("code")
    b = panel[panel["year"] == end].set_index("code")
    return pd.DataFrame({
        "pay_start": a["D1"] / a["SAL"],
        "share_start": 100 * a["D1"] / a["Y"],
        "share_change": 100 * (b["D1"] / b["Y"] - a["D1"] / a["Y"]),
        "pay_growth": 100 * ((b["D1"] / b["SAL"]) / (a["D1"] / a["SAL"]) - 1),
        "va_per_worker_growth": 100 * ((b["Y"] / b["EMP"]) / (a["Y"] / a["EMP"]) - 1),
        "employees_start": a["SAL"],
        "va_start": a["Y"],
        "va_end": b["Y"],
        "section": [section_of(c) for c in a.index],
    })


def permutation_p(x: np.ndarray, y: np.ndarray, w: np.ndarray, label: str) -> float:
    """Two-sided permutation p-value for the weighted rank correlation."""
    rng = np.random.default_rng([SEED, zlib.crc32(label.encode("utf-8"))])
    rx, ry = rank(x), rank(y)
    observed = abs(weighted_corr(rx, ry, w))
    hits = 0
    for _ in range(N_PERM):
        perm = rng.permutation(len(y))
        if abs(weighted_corr(rx, ry[perm], w[perm])) >= observed:
            hits += 1
    return (1 + hits) / (N_PERM + 1)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class Report:
    """Collect lines for ``results/summary.txt``."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, title: str) -> None:
        self.lines += ["", title, "-" * len(title)]

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines).lstrip() + "\n", encoding="utf-8")


def share(series_l: pd.Series, series_y: pd.Series) -> pd.Series:
    """Labour share in percent."""
    return 100 * series_l / series_y


def main() -> None:
    """Run the analysis and write the results."""
    RESULTS.mkdir(exist_ok=True)
    rep = Report()
    rep.add("Portugal's labour share by industry")
    rep.add(f"Sample {FIRST_YEAR}-{LAST_YEAR}, annual, current prices, factor-cost value added. Seed {SEED}.")

    panels = {geo: load_industries(geo) for geo in GEOS}
    pt = panels["PT"]

    # 0. Checks ---------------------------------------------------------------
    rep.section("0. Data checks")
    for geo in GEOS:
        checks = data_checks(geo)
        rep.add(
            f"{geo}: A64 sum vs total max gap B1G {checks['max_gap_B1G_pct']:.3f}%, D1 {checks['max_gap_D1_pct']:.3f}%; "
            f"{int(checks['industries_after_merges'])} industries after merges; "
            f"non-positive value added cells {int(checks['non_positive_value_added_cells'])}"
        )
    tot = pt.groupby("year")[["D1", "Y", "EMP", "SAL"]].sum()
    raw_total = share(tot["D1"], tot["Y"])
    rep.add(f"PT compensation share of factor-cost value added: {raw_total.loc[FIRST_YEAR]:.2f}% ({FIRST_YEAR}) -> {raw_total.loc[LAST_YEAR]:.2f}% ({LAST_YEAR})")
    rep.add(f"merged: {', '.join(sorted(set(MERGES.values())))}; dropped: {', '.join(DROPPED)}")
    gdp = pd.read_csv(FIRST_ARTICLE_DATA / "nama_10_gdp.csv")
    gdp = gdp[(gdp["geo"] == "PT") & (gdp["year"] <= LAST_YEAR)].set_index("year")
    national = 100 * gdp["D1"] / (gdp["D1"] + gdp["B2A3G"])
    rep.add(f"PT max |industry-built share - national accounts share|: {(raw_total - national).abs().max():.3f} pp")
    negatives = []
    for geo in GEOS:
        acc = read_long("nama_10_a64")
        acc = acc[(acc["geo"] == geo) & acc["nace_r2"].isin(list(MERGES))]
        w = acc.pivot_table(index=["nace_r2", "year"], columns="na_item", values="value")
        neg = w[(w["B1G"] - w["D29X39"]) <= 0]
        negatives += [f"{geo} {c} {y}" for c, y in neg.index]
    rep.add(f"non-positive factor-cost value added before merging: {', '.join(negatives)}")

    # 1. Self-employment by branch -------------------------------------------
    rep.section("1. Self-employment by A10 branch, Portugal (thousand persons; pay per employee, thousand euro)")
    a10 = pt.assign(g=pt["code"].map(lambda c: group_of(c, "a10"))).groupby(["g", "year"])[["D1", "EMP", "SAL"]].sum()
    for year in (FIRST_YEAR, LAST_YEAR):
        sub = a10.xs(year, level="year")
        se = sub["EMP"] - sub["SAL"]
        pay = sub["D1"] / sub["SAL"]
        rep.add(f"{year}: self-employed total {se.sum():.0f}; " + "; ".join(
            f"{g} {se[g]:.0f} (pay {pay[g]:.1f})" for g in se.sort_values(ascending=False).index[:5]))
        rep.add(f"      economy-wide pay per employee {sub['D1'].sum() / sub['SAL'].sum():.1f}; agriculture share of self-employed {100 * se['A'] / se.sum():.1f}%")

    # 2. Where the imputation is done ----------------------------------------
    rep.section("2. Self-employed labour income imputed economy-wide or by industry (factor cost)")
    nasa = pd.read_csv(FIRST_ARTICLE_DATA / "nasa_10_nf_tr_PT.csv").set_index("year")
    imputation_rows = []
    for geo in GEOS:
        panel = panels[geo]
        t = panel.groupby("year")[["D1", "Y"]].sum()
        versions = {
            "raw": t["D1"] * 0,
            "economy_per_person": imputed_labour(panel, "total", cap=False),
            "a10_per_person": imputed_labour(panel, "a10"),
            "a64_per_person": imputed_labour(panel, "a64"),
            "a10_per_hour": imputed_labour_hours_a10(geo, panel),
        }
        parts = []
        for name, imp in versions.items():
            s = share(t["D1"] + imp, t["Y"])
            imputation_rows += [{"geo": geo, "rule": name, "year": int(y), "share": v} for y, v in s.items()]
            parts.append(f"{name} {s.loc[FIRST_YEAR]:.1f}->{s.loc[LAST_YEAR]:.1f} ({s.loc[LAST_YEAR] - s.loc[FIRST_YEAR]:+.2f})")
        rep.add(f"{geo}: " + "; ".join(parts))
        if geo == "PT":
            for name in ("economy_per_person", "a10_per_person", "a64_per_person"):
                alpha = versions[name] / nasa["B3G_S1"].loc[FIRST_YEAR:LAST_YEAR]
                rep.add(f"    PT implied labour fraction of mixed income, {name}: {alpha.loc[FIRST_YEAR]:.2f} ({FIRST_YEAR}) -> {alpha.loc[LAST_YEAR]:.2f} ({LAST_YEAR}), range {alpha.min():.2f}-{alpha.max():.2f}")
            uncapped = share(t["D1"] + imputed_labour(panel, "a64", cap=False), t["Y"])
            rep.add(f"    PT a64 per person without the cap: {uncapped.loc[FIRST_YEAR]:.1f}->{uncapped.loc[LAST_YEAR]:.1f} ({uncapped.loc[LAST_YEAR] - uncapped.loc[FIRST_YEAR]:+.2f})")
            binding = cap_binding(panel)
            rep.add(f"    PT cap binds in {len(binding)} of {len(panel)} industry-years, {binding['code'].nunique()} industries; agriculture (A01) years: {int((binding['code'] == 'A01').sum())}")
    pd.DataFrame(imputation_rows).to_csv(RESULTS / "imputation_levels.csv", index=False, float_format="%.4f")

    # 3. Shift-share grid -------------------------------------------------------
    rep.section("3. Within and between, Portugal (pp; endpoint midpoint decomposition, and chained year by year)")
    grid_rows = []
    for scope in SCOPES:
        for numerator in NUMERATORS:
            for level in LEVELS:
                frame = grouped(pt, numerator, level, scope)
                for start, end in WINDOWS:
                    ss = shift_share(frame, start, end)
                    cw, cb = chained(frame, start, end)
                    total = ss["within"].sum() + ss["between"].sum()
                    grid_rows.append({
                        "scope": scope, "numerator": numerator, "level": level, "start": start, "end": end,
                        "total": total, "within": ss["within"].sum(), "between": ss["between"].sum(),
                        "within_chained": cw, "between_chained": cb,
                        "groups": len(ss),
                    })
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(RESULTS / "shift_share_grid.csv", index=False, float_format="%.4f")
    for (start, end), sub in grid.groupby(["start", "end"]):
        rep.add(f"{start}->{end}:")
        for r in sub.itertuples():
            rep.add(
                f"  {r.scope:<8}{r.numerator:<9}{r.level}  total {r.total:+.2f} = within {r.within:+.2f} + between {r.between:+.2f}"
                f"  (within {100 * r.within / r.total:.0f}%; chained within {r.within_chained:+.2f}, between {r.between_chained:+.2f})"
            )

    # 4. Contributions 2016-2024 ----------------------------------------------
    rep.section("4. Contributions to 2016->2024, Portugal, compensation share, whole economy")
    sec = shift_share(grouped(pt, "raw", "a21", "economy"), 2016, 2024)
    sec["total"] = sec["within"] + sec["between"]
    sec = sec.sort_values("total")
    sec.to_csv(RESULTS / "contributions_2016_2024_sections.csv", float_format="%.4f")
    for g, r in sec.iterrows():
        rep.add(
            f"  {g} {SECTION_NAMES[g]:<32} weight {r.w_start:5.2f}->{r.w_end:5.2f}  share {r.s_start:5.1f}->{r.s_end:5.1f}  "
            f"within {r['within']:+.2f}  between {r['between']:+.2f}  total {r['total']:+.2f}"
        )
    ind = shift_share(grouped(pt, "raw", "a64", "economy"), 2016, 2024)
    ind["total"] = ind["within"] + ind["between"]
    ind.to_csv(RESULTS / "contributions_2016_2024_industries.csv", float_format="%.4f")
    rep.add("  largest A64 within contributions: " + "; ".join(
        f"{g} {r.within:+.2f} (share {r.s_start:.1f}->{r.s_end:.1f})" for g, r in ind.sort_values("within", ascending=False).head(5).iterrows()))
    rep.add("  most negative A64 within contributions: " + "; ".join(
        f"{g} {r.within:+.2f} (share {r.s_start:.1f}->{r.s_end:.1f})" for g, r in ind.sort_values("within").head(3).iterrows()))
    rep.add("  largest A64 between contributions: " + "; ".join(
        f"{g} {r['between']:+.2f} (weight {r.w_start:.2f}->{r.w_end:.2f})" for g, r in ind.sort_values("between", ascending=False).head(4).iterrows()))
    rep.add("  most negative A64 between contributions: " + "; ".join(
        f"{g} {r['between']:+.2f} (weight {r.w_start:.2f}->{r.w_end:.2f})" for g, r in ind.sort_values("between").head(3).iterrows()))

    # 5. Pay level and the within change --------------------------------------
    rep.section("5. Within-industry change against initial pay, A64, 2016->2024")
    pay_rows = []
    for geo in GEOS:
        ps = pay_and_share(panels[geo], 2016, 2024)
        ps["geo"] = geo
        pay_rows.append(ps.reset_index())
        x, y, w = ps["pay_start"].to_numpy(), ps["share_change"].to_numpy(), ps["va_start"].to_numpy()
        rho = weighted_corr(rank(x), rank(y), w)
        p = permutation_p(x, y, w, f"pay:{geo}")
        line = f"{geo}: weighted rank correlation of 2016 pay per employee with share change {rho:+.2f} (permutation p {p:.4f}, {len(ps)} industries)"
        if geo == "PT":
            emp_w = ps["employees_start"]
            order = ps["pay_start"].sort_values().index
            cum = emp_w.loc[order].cumsum() / emp_w.sum()
            terc = pd.Series(np.where(cum <= 1 / 3, "low", np.where(cum <= 2 / 3, "middle", "high")), index=order)
            within = ind["within"]
            for t in ("low", "middle", "high"):
                codes = terc[terc == t].index
                sub = ps.loc[codes]
                pay_g = 100 * ((1 + sub["pay_growth"] / 100) * sub["employees_start"]).sum() / sub["employees_start"].sum() - 100
                rep.add(
                    f"    PT {t:<6} pay third ({len(codes)} industries, pay 2016 {sub['pay_start'].min():.1f}-{sub['pay_start'].max():.1f}k): "
                    f"within contribution {within.reindex(codes).sum():+.2f} pp; "
                    f"pay per employee growth (employee-weighted) {pay_g:.0f}%; "
                    f"value-added-weighted share change {np.average(sub['share_change'], weights=sub['va_start']):+.1f} pp"
                )
            agg = pt[pt["year"].isin([2016, 2024])].groupby("year")[["D1", "SAL", "Y", "EMP"]].sum()
            rep.add(
                f"    PT economy: pay per employee {100 * ((agg.D1 / agg.SAL).loc[2024] / (agg.D1 / agg.SAL).loc[2016] - 1):.0f}%, "
                f"value added per person employed {100 * ((agg.Y / agg.EMP).loc[2024] / (agg.Y / agg.EMP).loc[2016] - 1):.0f}% (nominal, 2016->2024)"
            )
        rep.add(line)
    pd.concat(pay_rows).to_csv(RESULTS / "pay_and_share_2016_2024.csv", index=False, float_format="%.4f")

    # 6. 2010-2013 -------------------------------------------------------------
    rep.section("6. Contributions to 2010->2013, Portugal, compensation share, whole economy, A21")
    sec13 = shift_share(grouped(pt, "raw", "a21", "economy"), 2010, 2013)
    sec13["total"] = sec13["within"] + sec13["between"]
    sec13.to_csv(RESULTS / "contributions_2010_2013_sections.csv", float_format="%.4f")
    for g, r in sec13.sort_values("total").head(6).iterrows():
        rep.add(
            f"  {g} {SECTION_NAMES[g]:<32} weight {r.w_start:5.2f}->{r.w_end:5.2f}  share {r.s_start:5.1f}->{r.s_end:5.1f}  "
            f"within {r['within']:+.2f}  between {r['between']:+.2f}"
        )
    opq = sec13.loc[["O", "P", "Q"]]
    rep.add(f"  O+P+Q: within {opq['within'].sum():+.2f}, between {opq['between'].sum():+.2f}, total {opq['total'].sum():+.2f} of {sec13['total'].sum():+.2f}")
    f = sec13.loc["F"]
    rep.add(f"  F construction: within {f['within']:+.2f}, between {f['between']:+.2f}, total {f['total']:+.2f}")

    # 7. Peers --------------------------------------------------------------------
    rep.section("7. Peers, compensation share, whole economy (pp)")
    peer_rows = []
    for geo in GEOS:
        for level in ("a10", "a64"):
            frame = grouped(panels[geo], "raw", level, "economy")
            for start, end in WINDOWS:
                ss = shift_share(frame, start, end)
                peer_rows.append({"geo": geo, "level": level, "start": start, "end": end,
                                  "total": ss["within"].sum() + ss["between"].sum(),
                                  "within": ss["within"].sum(), "between": ss["between"].sum()})
    peers = pd.DataFrame(peer_rows)
    peers.to_csv(RESULTS / "peers.csv", index=False, float_format="%.4f")
    for (start, end), sub in peers.groupby(["start", "end"]):
        rep.add(f"{start}->{end}: " + "; ".join(
            f"{r.geo} {r.level} {r.total:+.2f} (within {r.within:+.2f}, between {r.between:+.2f})" for r in sub.itertuples()))

    # 8. Selected industries -----------------------------------------------------
    rep.section("8. Selected Portuguese industries (million euro; thousand employees; pay per employee, thousand euro)")
    for code in ("G46", "G47", "K64", "J62_J63"):
        sub = pt[pt["code"] == code].set_index("year")
        parts = []
        for year in (2016, 2022, 2023, 2024):
            r = sub.loc[year]
            parts.append(f"{year}: value added {r.Y:.0f}, compensation {r.D1:.0f}, employees {r.SAL:.0f}, "
                         f"share {100 * r.D1 / r.Y:.1f}%, pay {r.D1 / r.SAL:.1f}")
        rep.add(f"{code}: " + " | ".join(parts))
        window = sub.loc[2016:LAST_YEAR]
        shares = 100 * window["D1"] / window["Y"]
        rep.add(f"    {code} share range 2016-{LAST_YEAR}: {shares.min():.1f}-{shares.max():.1f}%")

    rep.write(RESULTS / "summary.txt")
    print("\n".join(rep.lines))


if __name__ == "__main__":
    main()
