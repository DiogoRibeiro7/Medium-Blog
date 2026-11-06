"""How much did Europe deindustrialise, and in what sense? (AMECO, 1960-2025)

    python decompose.py

Manufacturing's share of the economy is measured three ways, per country and year:

- employment share   s_E = manufacturing employment / employment in all domestic industries
- nominal share      s_N = manufacturing gross value added / total gross value added, current prices
- real share         s_R = the same at 2020 prices

They are tied by two identities. With relative price p = s_N / s_R (the manufacturing price
level relative to the whole economy, 2020 = 1) and relative productivity q = s_R / s_E
(output per worker in manufacturing relative to the whole economy):

    log s_N = log s_R + log p
    log s_E = log s_R - log q

So a fall in the nominal share splits exactly into a fall in real output share and a fall in
relative prices, and a fall in the employment share into a fall in real output share and a rise
in relative productivity. Writes ``results/shares_long.csv``, ``results/decomposition.csv`` and
``results/decompose_summary.txt``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"

LAST = 2025  # AMECO carries forecasts beyond this
# AMECO country codes -> ISO2 and own currency suffix
COUNTRIES = {
    "aut": ("AT", "ats"), "bel": ("BE", "bef"), "bgr": ("BG", "bgn"), "cyp": ("CY", "cyp"), "cze": ("CZ", "czk"),
    "deu": ("DE", "dem"), "dnk": ("DK", "dkk"), "esp": ("ES", "esp"), "est": ("EE", "eek"), "fin": ("FI", "fim"),
    "fra": ("FR", "frf"), "grc": ("EL", "grd"), "hrv": ("HR", "hrk"), "hun": ("HU", "huf"), "irl": ("IE", "iep"),
    "ita": ("IT", "itl"), "ltu": ("LT", "ltl"), "lux": ("LU", "luf"), "lva": ("LV", "lvl"), "mlt": ("MT", "mtl"),
    "nld": ("NL", "nlg"), "pol": ("PL", "pln"), "prt": ("PT", "pte"), "rom": ("RO", "ron"), "svk": ("SK", "skk"),
    "svn": ("SI", "sit"), "swe": ("SE", "sek"), "gbr": ("UK", "gbp"), "usa": ("US", "usd"), "jpn": ("JP", "jpy"),
}
AGGREGATES = {"eu27": "EU27", "eu15": "EU15"}
NAMES = {"EU27": "EU-27", "EU15": "EU-15", "UK": "United Kingdom", "US": "United States", "JP": "Japan"}


def load_ameco() -> pd.DataFrame:
    a = pd.read_csv(DATA / "ameco_long.csv", low_memory=False)
    dims = a.dimensions.map(json.loads)
    a["ageo"] = dims.map(lambda d: d.get("geo"))
    a["unit"] = dims.map(lambda d: d.get("unit"))
    a["year"] = pd.to_numeric(a.period.astype(str).str[:4], errors="coerce")
    a["value"] = pd.to_numeric(a.value, errors="coerce")
    return a[a.value.notna() & (a.year <= LAST)]


def pick(a: pd.DataFrame, dataset: str, ageo: str, units: list[str]) -> pd.Series:
    s = a[(a.dataset == dataset) & (a.ageo == ageo)]
    for u in units:
        t = s[s.unit == u]
        if len(t):
            return t.groupby("year").value.mean()
    return pd.Series(dtype=float)


def country_shares(a: pd.DataFrame, ageo: str, cur: str) -> pd.DataFrame:
    own = f"mrd-euro-{cur}" if cur not in ("dkk", "sek", "gbp", "usd", "jpy", "pln", "czk", "huf", "ron", "bgn") else f"mrd-{cur}"
    units_nom = [own, f"mrd-{cur}", "mrd-ecu-eur"]
    units_real = [own, f"mrd-{cur}"]
    emp_m = pick(a, "NETM", ageo, ["1000-persons"])
    emp_t = pick(a, "NETD", ageo, ["1000-persons"])
    nom_m = pick(a, "UVGM", ageo, units_nom)
    nom_t = pick(a, "UVG0", ageo, units_nom)
    real_m = pick(a, "OVGM", ageo, units_real)
    real_t = pick(a, "OVG0", ageo, units_real)
    d = pd.DataFrame({"emp_m": emp_m, "emp_t": emp_t, "nom_m": nom_m, "nom_t": nom_t, "real_m": real_m, "real_t": real_t})
    return d


def aggregate_shares(a: pd.DataFrame, ageo: str) -> pd.DataFrame:
    agg = "mrd-ecu-eur-standard-aggregation"
    wm = "mrd-ecu-eur-weighted-mean-of-t-t-1-national-growth-rates-weights-t-1-current-prices-in-ecu-eur"
    d = pd.DataFrame(
        {
            "emp_m": pick(a, "NETM", ageo, ["1000-persons"]),
            "emp_t": pick(a, "NETD", ageo, ["1000-persons"]),
            "nom_m": pick(a, "UVGM", ageo, [agg, wm]),
            "nom_t": pick(a, "UVG0", ageo, [agg, wm]),
            "real_m": pick(a, "OVGM", ageo, [wm]),
            "real_t": pick(a, "OVG0", ageo, [wm]),
        }
    )
    return d


def finish(d: pd.DataFrame, geo: str) -> pd.DataFrame:
    d = d.copy()
    d["emp_share"] = d.emp_m / d.emp_t
    d["nom_share"] = d.nom_m / d.nom_t
    d["real_share"] = d.real_m / d.real_t
    # real shares are in 2020 prices; rescale so that the real and nominal shares coincide in 2020
    if 2020 in d.index and np.isfinite(d.loc[2020, ["nom_share", "real_share"]]).all():
        d["real_share"] *= d.loc[2020, "nom_share"] / d.loc[2020, "real_share"]
    d["rel_price"] = d.nom_share / d.real_share
    d["rel_prod"] = d.real_share / d.emp_share
    d.index.name = "year"
    d = d.reset_index()
    d.insert(0, "geo", geo)
    return d


def build() -> pd.DataFrame:
    a = load_ameco()
    parts = []
    for ageo, (geo, cur) in COUNTRIES.items():
        d = country_shares(a, ageo, cur)
        if geo == "DE":
            west = country_shares(a, "d-w", cur)
            d = pd.concat([west[west.index < 1991], d[d.index >= 1991]])
        parts.append(finish(d, geo))
    for ageo, geo in AGGREGATES.items():
        parts.append(finish(aggregate_shares(a, ageo), geo))
    return pd.concat(parts, ignore_index=True)


def decompose(s: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    rows = []
    for geo, g in s.groupby("geo"):
        g = g.set_index("year")
        if start not in g.index or end not in g.index:
            continue
        a, b = g.loc[start], g.loc[end]
        if not np.isfinite([a.emp_share, b.emp_share, a.real_share, b.real_share, a.nom_share, b.nom_share]).all():
            continue
        rows.append(
            {
                "geo": geo,
                "start": start,
                "end": end,
                "emp_share_start": a.emp_share,
                "emp_share_end": b.emp_share,
                "nom_share_start": a.nom_share,
                "nom_share_end": b.nom_share,
                "real_share_start": a.real_share,
                "real_share_end": b.real_share,
                "dlog_emp_share": np.log(b.emp_share / a.emp_share),
                "dlog_nom_share": np.log(b.nom_share / a.nom_share),
                "dlog_real_share": np.log(b.real_share / a.real_share),
                "dlog_rel_price": np.log(b.rel_price / a.rel_price),
                "dlog_rel_prod": np.log(b.rel_prod / a.rel_prod),
                "dlog_emp_level": np.log(b.emp_m / a.emp_m),
                "dlog_real_output": np.log(b.real_m / a.real_m),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RES.mkdir(exist_ok=True)
    s = build()
    s.to_csv(RES / "shares_long.csv", index=False, float_format="%.6g")
    decs = pd.concat([decompose(s, 1995, 2024), decompose(s, 1970, 2024), decompose(s, 2000, 2024)], ignore_index=True)
    decs.to_csv(RES / "decomposition.csv", index=False, float_format="%.6g")

    L = []
    say = L.append
    say("Manufacturing shares (employment, nominal value added, real value added at 2020 prices) and their decomposition")
    say("log s_N = log s_R + log p ; log s_E = log s_R - log q (p relative price, q relative productivity)\n")
    for start in (1995, 1970, 2000):
        say(f"From {start} to 2024 (log points; shares in percent)")
        d = decs[decs.start == start].sort_values("dlog_emp_share")
        for _, r in d.iterrows():
            say(
                f"  {r.geo:5s} emp share {100 * r.emp_share_start:5.1f} -> {100 * r.emp_share_end:5.1f} ({r.dlog_emp_share:+.2f}) = real share {r.dlog_real_share:+.2f} - rel. productivity {r.dlog_rel_prod:+.2f}; "
                f"nominal share {100 * r.nom_share_start:5.1f} -> {100 * r.nom_share_end:5.1f} ({r.dlog_nom_share:+.2f}) = real {r.dlog_real_share:+.2f} + rel. price {r.dlog_rel_price:+.2f}; "
                f"manufacturing jobs {r.dlog_emp_level:+.2f}, real output {r.dlog_real_output:+.2f}"
            )
        say("")
    # peaks of the employment share
    say("Peak manufacturing employment share in the data")
    for geo, g in s.groupby("geo"):
        g = g.dropna(subset=["emp_share"])
        if g.empty:
            continue
        i = g.emp_share.idxmax()
        say(f"  {geo:5s} peak {100 * g.loc[i, 'emp_share']:.1f}% in {int(g.loc[i, 'year'])} (data from {int(g.year.min())}); 2024 {100 * g.set_index('year').emp_share.get(2024, np.nan):.1f}%")
    text = "\n".join(L) + "\n"
    (RES / "decompose_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
