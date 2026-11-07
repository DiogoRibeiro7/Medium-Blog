"""Build the annual product panel and describe how products enter the single-supplier screen.

    python panel.py

Reads the parts written by ``fetch_panel.py`` (``data/parts``) and writes:

- ``results/panel.csv.gz``: one row per HS 6-digit product and year, 2002-2025, with the three
  screen criteria, the flag, the largest and second-largest outside suppliers and their shares,
  unit values and the share of China;
- ``results/panel_flows.csv``: products flagged, entries, exits and durable entries by year;
- ``results/panel_events.csv``: one row per durable entry, with the year each criterion was first
  met in the run-up and the criterion met last;
- ``results/panel_event_paths.csv``: medians and quartiles of the run-up, by year relative to entry;
- ``results/panel_summary.txt``.

The screen is the single-supplier variant of the European Commission's 2021 screen: the
largest outside supplier provides more than 40% of extra-EU imports (instead of a Herfindahl
index above 0.4), extra-EU imports are more than half of all EU imports, and extra-EU imports
exceed total EU exports. A durable entry in year E is a product flagged in E and E+1 after
three years unflagged (E-3, E-2, E-1), so that products hovering at a threshold are not
counted every time they cross it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "parts"
RESULTS = HERE / "results"
TOP_SHARE, EXTRA_SHARE, RATIO = 0.4, 0.5, 1.0
CRITERIA = ("c_top", "c_extra", "c_ratio")
CLEAN_BEFORE = 3
EVENT_WINDOW = range(-8, 4)


def read_parts(kind: str) -> pd.DataFrame:
    files = sorted(DATA.glob(f"{kind}_*.csv.gz"))
    if not files:
        raise SystemExit(f"no {kind} parts in {DATA}; run fetch_panel.py (GitHub Actions) first")
    d = pd.concat([pd.read_csv(f, dtype={"product": str, "partner": str}, keep_default_na=False, na_values=[""]) for f in files], ignore_index=True)
    return d.drop_duplicates()


def leaders(p: pd.DataFrame) -> pd.DataFrame:
    """Largest and second-largest named outside supplier for each product and year."""
    named = p[p["partner"] != "ZZ"].sort_values(["product", "year", "value"], ascending=[True, True, False])
    named = named.assign(rank=named.groupby(["product", "year"]).cumcount() + 1)
    total = p.groupby(["product", "year"], as_index=False).agg(total=("value", "sum"), total_qty=("quantity", lambda q: q.sum(min_count=1)))
    first = named[named["rank"] == 1][["product", "year", "partner", "value", "quantity"]].rename(columns={"partner": "top_partner", "value": "top_value", "quantity": "top_qty"})
    second = named[named["rank"] == 2][["product", "year", "partner", "value"]].rename(columns={"partner": "second_partner", "value": "second_value"})
    china = named[named["partner"] == "CN"][["product", "year", "value"]].rename(columns={"value": "cn_value"})
    out = total.merge(first, on=["product", "year"], how="left").merge(second, on=["product", "year"], how="left").merge(china, on=["product", "year"], how="left")
    out[["second_value", "cn_value"]] = out[["second_value", "cn_value"]].fillna(0.0)
    out["top_share"] = out["top_value"] / out["total"]
    out["second_share"] = out["second_value"] / out["total"]
    out["cn_share"] = out["cn_value"] / out["total"]
    # unit value of the largest supplier relative to all other outside suppliers (euros per 100 kg)
    rest_v, rest_q = out["total"] - out["top_value"], out["total_qty"] - out["top_qty"]
    ok = (out["top_qty"] > 0) & (rest_q > 0) & (rest_v > 0)
    out["rel_unit_value"] = np.where(ok, np.log((out["top_value"] / out["top_qty"]) / (rest_v / rest_q).where(ok)), np.nan)
    return out


def build() -> pd.DataFrame:
    t = read_parts("totals")
    s = read_parts("summary")
    p = read_parts("partners")
    for c in ("imports_extra", "imports_intra", "exports_extra", "exports_intra"):
        if c not in t:
            t[c] = np.nan
    d = t.merge(s, on=["product", "year"], how="left").merge(leaders(p), on=["product", "year"], how="left")
    d = d[d["imports_extra"].fillna(0) > 0].copy()
    d[["imports_intra", "exports_extra", "exports_intra"]] = d[["imports_intra", "exports_extra", "exports_intra"]].fillna(0.0)
    d["extra_share"] = d["imports_extra"] / (d["imports_extra"] + d["imports_intra"])
    exports = d["exports_extra"] + d["exports_intra"]
    d["log_ratio"] = np.log(d["imports_extra"] / exports.where(exports > 0))
    d["log_ratio"] = d["log_ratio"].fillna(np.log(1e3))  # no exports at all: far above the threshold
    d["c_top"] = d["top_share"] > TOP_SHARE
    d["c_extra"] = d["extra_share"] > EXTRA_SHARE
    d["c_ratio"] = d["log_ratio"] > np.log(RATIO)
    d["flagged"] = d["c_top"] & d["c_extra"] & d["c_ratio"]
    d["top_value_flagged"] = np.where(d["flagged"], d["top_value"], 0.0)
    return d.sort_values(["product", "year"]).reset_index(drop=True)


def complete(d: pd.DataFrame) -> pd.DataFrame:
    """Square the panel: one row per product and every year in which its code exists in the trade data.

    Missing rows inside a code's life (no extra-EU imports that year) are kept as unflagged.
    """
    life = d.groupby("product")["year"].agg(["min", "max"])
    idx = [(p, y) for p, (a, b) in life.iterrows() for y in range(a, b + 1)]
    sq = pd.DataFrame(idx, columns=["product", "year"]).merge(d, on=["product", "year"], how="left")
    for c in ("flagged", *CRITERIA):
        sq[c] = sq[c].astype("boolean").fillna(False).astype(bool)
    return sq


def events(d: pd.DataFrame) -> pd.DataFrame:
    """Durable entries: flagged in E and E+1, unflagged in the three years before."""
    d = d.sort_values(["product", "year"])
    g = d.groupby("product")["flagged"]
    lead1 = g.shift(-1)
    lags = [g.shift(k) for k in range(1, CLEAN_BEFORE + 1)]
    ok = d["flagged"] & (lead1 == True)  # noqa: E712 - NaN at the end of a code's life
    for lag in lags:
        ok &= lag == False  # noqa: E712 - NaN at the start of a code's life is not a clean year
    e = d.loc[ok, ["product", "year", "top_partner", "top_share", "top_value", "imports_extra", "extra_share", "log_ratio"]].rename(columns={"year": "entry"})
    return e.reset_index(drop=True)


def run_up(d: pd.DataFrame, e: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each durable entry: the path of the entering supplier's share and of the other two criteria."""
    pv = read_parts("partners")
    tot = pv.groupby(["product", "year"])["value"].transform("sum")
    pv = pv.assign(share=pv["value"] / tot)
    pv = pv[pv["partner"] != "ZZ"].set_index(["product", "partner"]).sort_index()
    rows, paths = [], []
    by_prod = {k: g.set_index("year") for k, g in d.groupby("product")}
    for r in e.itertuples(index=False):
        g = by_prod[r.product]
        try:
            sup = pv.loc[(r.product, r.top_partner)].set_index("year")["share"]
        except KeyError:
            sup = pd.Series(dtype=float)
        share = sup.reindex(g.index).fillna(0.0)
        first = {}
        for c in CRITERIA:
            # start of the unbroken run of years, ending at entry, in which the criterion was met
            met = g[c].loc[: r.entry][::-1]
            run = met.astype(int).cumprod().astype(bool)
            first[c] = int(met.index[run].min())
        rec = {"product": r.product, "entry": r.entry, "supplier": r.top_partner, "censored_start": int(g.index.min()), **{f"since_{c}": first[c] for c in CRITERIA}}
        latest = max(first.values())
        last = [c for c in CRITERIA if first[c] == latest]
        rec["last"] = last[0] if len(last) == 1 else "joint"
        # how long the entering supplier had already been the largest outside supplier
        top = g["top_partner"].loc[: r.entry][::-1]
        same = (top == r.top_partner).astype(int).cumprod().astype(bool)
        rec["supplier_top_since"] = int(top.index[same].min())
        rec["years_supplier_largest"] = r.entry - rec["supplier_top_since"]
        for k in (5, 3, 1, 0):
            y = r.entry - k
            rec[f"share_m{k}" if k else "share_0"] = float(share.loc[y]) if y in share.index else np.nan
        v = g["imports_extra"]
        rec["value_0"] = float(v.loc[r.entry])
        rec["value_m3"] = float(v.loc[r.entry - 3]) if (r.entry - 3) in v.index else np.nan
        rows.append(rec)
        for k in EVENT_WINDOW:
            y = r.entry + k
            if y in g.index:
                paths.append({"product": r.product, "entry": r.entry, "rel": k, "supplier_share": float(share.loc[y]), "extra_share": float(g.loc[y, "extra_share"]), "log_ratio": float(g.loc[y, "log_ratio"]), "flagged": bool(g.loc[y, "flagged"])})
    ev = pd.DataFrame(rows)
    pa = pd.DataFrame(paths)
    q = pa.groupby("rel").agg(
        n=("product", "size"),
        share_q25=("supplier_share", lambda x: x.quantile(0.25)),
        share_med=("supplier_share", "median"),
        share_q75=("supplier_share", lambda x: x.quantile(0.75)),
        extra_med=("extra_share", "median"),
        ratio_med=("log_ratio", lambda x: float(np.exp(x.median()))),
        flagged=("flagged", "mean"),
    ).reset_index()
    return ev, q


def flows(d: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["product", "year"])
    prev = d.groupby("product")["flagged"].shift(1)
    d = d.assign(enter=d["flagged"] & (prev == False), leave=~d["flagged"] & (prev == True))  # noqa: E712
    f = d.groupby("year").agg(products=("product", "size"), flagged=("flagged", "sum"), entries=("enter", "sum"), exits=("leave", "sum"), top_value_flagged=("top_value_flagged", "sum")).reset_index()
    f = f.merge(e.groupby("entry").size().rename("durable_entries"), left_on="year", right_index=True, how="left").fillna({"durable_entries": 0})
    return f


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    d = complete(build())
    d.to_csv(RESULTS / "panel.csv.gz", index=False, float_format="%.6g", compression={"method": "gzip", "mtime": 0})
    e = events(d)
    ev, q = run_up(d, e)
    fl = flows(d, e)
    fl.to_csv(RESULTS / "panel_flows.csv", index=False, float_format="%.6g")
    ev.to_csv(RESULTS / "panel_events.csv", index=False, float_format="%.6g")
    q.to_csv(RESULTS / "panel_event_paths.csv", index=False, float_format="%.4f")

    years = sorted(d["year"].unique())
    life = d.groupby("product")["year"].agg(["min", "max"])
    lines = [
        f"Annual panel {years[0]}-{years[-1]}: {d['product'].nunique()} HS 6-digit codes, {len(d)} product-years",
        f"Codes present in every year: {int(((life['min'] == years[0]) & (life['max'] == years[-1])).sum())}",
        "",
        "Single-supplier screen by year (flagged products, entries, exits, durable entries, EUR bn from the largest supplier):",
    ]
    for r in fl.itertuples(index=False):
        lines.append(f"  {r.year}: {int(r.flagged)} flagged, {int(r.entries)} in, {int(r.exits)} out, {int(r.durable_entries)} durable, EUR {r.top_value_flagged / 1e9:.1f} bn")
    lines += ["", f"Durable entries (flagged in E and E+1 after {CLEAN_BEFORE} clean years), E = {int(e['entry'].min()) if len(e) else '-'}-{int(e['entry'].max()) if len(e) else '-'}: {len(e)}"]
    if len(ev):
        lines.append(f"  largest supplier at entry: " + ", ".join(f"{k} {v}" for k, v in ev["supplier"].value_counts().head(8).items()))
        lines.append("  criterion met last: " + ", ".join(f"{k} {v} ({v / len(ev):.0%})" for k, v in ev["last"].value_counts().items()))
        for c in CRITERIA:
            lines.append(f"  years {c} already met before entry: median {ev[f'since_{c}'].rsub(ev['entry']).median():.0f}, share >= 3 years {(ev['entry'] - ev[f'since_{c}'] >= 3).mean():.0%}")
        lines.append(f"  years the entering supplier was already the largest: median {ev['years_supplier_largest'].median():.0f}, share >= 3 years {(ev['years_supplier_largest'] >= 3).mean():.0%}")
        lines.append(f"  supplier criterion met at least one year before entry: {(ev['entry'] - ev['since_c_top'] >= 1).mean():.0%}; entering supplier's share >= 40% three years before: {(ev['share_m3'] >= 0.4).mean():.0%}")
        big = ev[ev["value_0"] >= 1e7]
        lines.append(f"  entries with extra-EU imports >= EUR 10 m at entry: {len(big)}; criterion met last: " + ", ".join(f"{k} {v / len(big):.0%}" for k, v in big["last"].value_counts().items()) + f"; supplier largest for a median {big['years_supplier_largest'].median():.0f} years; supplier criterion met >= 1 year before {(big['entry'] - big['since_c_top'] >= 1).mean():.0%}")
        for k in ("m5", "m3", "m1", "0"):
            col = f"share_{k}"
            lines.append(f"  entering supplier's share at {k.replace('m', '-')}: median {ev[col].median():.3f} (n {ev[col].notna().sum()})")
        g = ev.dropna(subset=["value_m3"])
        g = g[g["value_m3"] > 0]
        lines.append(f"  extra-EU imports at entry vs three years before: median ratio {np.median(g['value_0'] / g['value_m3']):.2f} (n {len(g)})")
    lines += ["", "Run-up, medians by year relative to entry (entering supplier's share, extra-EU share, imports/exports, share flagged):"]
    for r in q.itertuples(index=False):
        lines.append(f"  {r.rel:+d}: n {r.n}, share {r.share_med:.3f} [{r.share_q25:.3f}, {r.share_q75:.3f}], extra {r.extra_med:.3f}, ratio {r.ratio_med:.2f}, flagged {r.flagged:.2f}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "panel_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
