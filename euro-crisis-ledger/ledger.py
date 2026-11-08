"""Assemble the ledger: interest, central-bank income, official lending and private creditors.

    python ledger.py

Inputs: ``results/interest_totals.csv`` (``interest.py``), the BIS consolidated banking
statistics, quarterly government debt by instrument (Eurostat) and ``documented.csv``, which
records every figure taken from an official statement or a study, with its source.

The ledger keeps three kinds of entry apart:

- **transfers measured from data**: interest saved or paid because of the crisis (model);
- **transfers documented by their payers or recipients**: central-bank income on Greek bonds,
  the 2012 private-sector haircut;
- **positions, not transfers**: loans outstanding and the share of them each creditor country
  carries at its contribution key, which are costs only if they are not repaid.

Writes ``results/ledger_*.csv`` and ``results/ledger_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

CORE = ("DE", "FR", "NL", "FI", "AT")
PROGRAMME = ("EL", "IE", "PT", "CY", "ES")
BIS_DEBTORS = {"GR": "EL", "IE": "IE", "PT": "PT", "CY": "CY", "ES": "ES", "IT": "IT"}
BIS_REPORTERS = ("5A", "DE", "FR", "NL", "AT", "BE", "IT", "ES", "GB", "US", "CH", "JP")
QUARTERS = ("2008-Q1", "2009-Q4", "2010-Q2", "2011-Q4", "2012-Q2", "2013-Q4", "2015-Q4", "2019-Q4", "2025-Q1")


def documented() -> pd.DataFrame:
    return pd.read_csv(HERE / "documented.csv", keep_default_na=False, na_values=[""])


def doc(d: pd.DataFrame, item: str, country: str | None = None) -> float:
    s = d[(d["item"] == item) & ((d["country"] == country) if country else True)]
    return float(s["value"].iloc[0])


def usd_per_eur() -> pd.Series:
    f = pd.read_csv(DATA / "fred_long_rates.csv", keep_default_na=False, na_values=[""])
    f = f[f["country"] == "USD_PER_EUR"].copy()
    f["q"] = pd.PeriodIndex(pd.to_datetime(f["date"]), freq="Q").astype(str).str.replace("Q", "-Q")
    return f.groupby("q")["value"].mean()


def bank_claims() -> pd.DataFrame:
    """Foreign claims, ultimate risk basis, of banks by nationality on programme countries; EUR bn."""
    b = pd.read_csv(DATA / "bis_cbs.csv.gz", dtype=str, keep_default_na=False, na_values=[""])
    b = b[(b["CBS_BASIS"] == "U") & (b["L_POSITION"] == "C") & b["L_CP_SECTOR"].isin(["A", "O", "B", "R"]) & b["L_REP_CTY"].isin(BIS_REPORTERS)]
    b = b.assign(usd_mn=pd.to_numeric(b["OBS_VALUE"], errors="coerce"))
    fx = usd_per_eur()
    b["eur_bn"] = b["usd_mn"] / b["TIME_PERIOD"].map(fx) / 1e3
    b["debtor"] = b["L_CP_COUNTRY"].map(BIS_DEBTORS)
    return b.rename(columns={"L_REP_CTY": "reporter", "L_CP_SECTOR": "sector", "TIME_PERIOD": "quarter"})[["reporter", "debtor", "sector", "quarter", "eur_bn"]].dropna()


def debt_by_instrument() -> pd.DataFrame | None:
    p = DATA / "debt_quarterly.csv"
    if not p.exists():
        return None
    q = pd.read_csv(p, keep_default_na=False, na_values=[""])
    return q


def exposures(d: pd.DataFrame) -> pd.DataFrame:
    """Each core country's share of official loans still outstanding, at its contribution key."""
    efsf_esm_outstanding = (
        doc(d, "efsf_esm_greece_outstanding") + doc(d, "efsf_ireland_outstanding") + doc(d, "efsf_portugal_outstanding")
        + doc(d, "esm_spain_outstanding") + doc(d, "esm_cyprus_outstanding")
    )
    glf_out = doc(d, "glf_outstanding") - doc(d, "glf_early_2026")
    glf_share_de = doc(d, "glf_germany", "DE") / doc(d, "glf_total")
    rows = []
    for c in CORE:
        k_lo = min(doc(d, "efsf_key", c), doc(d, "esm_key", c)) / 100
        k_hi = max(doc(d, "efsf_key", c), doc(d, "esm_key", c)) / 100
        # GLF lenders' shares follow the ECB capital key; Germany's documented share scales the others
        glf_c = glf_out * glf_share_de * (doc(d, "efsf_key", c) / doc(d, "efsf_key", "DE"))
        rows.append({"country": c, "key_low": k_lo, "key_high": k_hi, "efsf_esm_low": efsf_esm_outstanding * k_lo, "efsf_esm_high": efsf_esm_outstanding * k_hi, "glf": glf_c})
    out = pd.DataFrame(rows)
    out.attrs["efsf_esm_outstanding"] = efsf_esm_outstanding
    out.attrs["glf_outstanding"] = glf_out
    return out


def main() -> int:
    d = documented()
    tot = pd.read_csv(RESULTS / "interest_totals.csv", keep_default_na=False, na_values=[""])
    main_tot = tot[tot["rho_variant"] == "pooled"]
    claims = bank_claims()
    claims.to_csv(RESULTS / "ledger_bank_claims.csv", index=False, float_format="%.4f")
    ex = exposures(d)
    ex.to_csv(RESULTS / "ledger_exposures.csv", index=False, float_format="%.4f")

    rows = []
    for c in CORE + PROGRAMME:
        r = {"country": c}
        for cf in ("no_fragmentation", "safe_asset"):
            for m in ("market_effect", "actual_effect", "actual_effect_full", "market_effect_to_2014"):
                s = main_tot[(main_tot["geo"] == c) & (main_tot["cf"] == cf) & (main_tot["method"] == m)]
                r[f"{m}_{cf}"] = float(s["eur_bn"].iloc[0])
        rng = tot[(tot["geo"] == c) & (tot["method"] == "market_effect")]["eur_bn"]
        r["market_effect_min"], r["market_effect_max"] = float(rng.min()), float(rng.max())
        rows.append(r)
    led = pd.DataFrame(rows)
    led.to_csv(RESULTS / "ledger_interest.csv", index=False, float_format="%.4f")

    lines = ["Interest account, cumulative 2010-2025, EUR bn (negative = saved, positive = paid): market repricing / actual against window counterfactual / actual against no-crisis counterfactual:"]
    for r in led.itertuples(index=False):
        lines.append(
            f"  {r.country}: no-fragmentation {r.market_effect_no_fragmentation:+.1f} / {r.actual_effect_no_fragmentation:+.1f} / {r.actual_effect_full_no_fragmentation:+.1f}; "
            f"safe-asset {r.market_effect_safe_asset:+.1f} / {r.actual_effect_safe_asset:+.1f} / {r.actual_effect_full_safe_asset:+.1f}; "
            f"market repricing across all rho variants and both counterfactuals {r.market_effect_min:+.1f} to {r.market_effect_max:+.1f}"
        )
    core = led[led["country"].isin(CORE)]
    lines.append(f"  core total, market repricing: no-fragmentation {core.market_effect_no_fragmentation.sum():+.1f}, safe-asset {core.market_effect_safe_asset.sum():+.1f}")
    prog = led[led["country"].isin(PROGRAMME)]
    lines.append(f"  programme total: market repricing {prog.market_effect_no_fragmentation.sum():+.1f} / {prog.market_effect_safe_asset.sum():+.1f}; actual against window counterfactual {prog.actual_effect_no_fragmentation.sum():+.1f} / {prog.actual_effect_safe_asset.sum():+.1f}; actual against no-crisis counterfactual {prog.actual_effect_full_no_fragmentation.sum():+.1f} / {prog.actual_effect_full_safe_asset.sum():+.1f}")

    lines += ["", "Documented central-bank income on Greek bonds (EUR bn):"]
    for item in ("smp_interest_ecb_greek", "smp_anfa_returned", "smp_bundesbank", "smp_anfa_france_retained"):
        s = d[d["item"] == item].iloc[0]
        lines.append(f"  {item} ({s.country}, {s.as_of}): {s.value}")

    lines += ["", f"Official loans outstanding (EUR bn): EFSF and ESM {ex.attrs['efsf_esm_outstanding']:.1f}; GLF after the June 2026 repayment about {ex.attrs['glf_outstanding']:.1f}", "Share carried by each core country at its contribution key (EFSF/ESM range) and GLF share, EUR bn:"]
    for r in ex.itertuples(index=False):
        lines.append(f"  {r.country}: key {r.key_low:.3f}-{r.key_high:.3f}, EFSF/ESM {r.efsf_esm_low:.1f}-{r.efsf_esm_high:.1f}, GLF {r.glf:.1f}")
    lines.append("  Realised losses to creditor governments on EFSF, ESM, EFSM or GLF loans to date: none recorded (all programmes repaying; Greece repaid the IMF in 2022 and part of the GLF early)")

    lines += ["", "Foreign claims of banks on the official sector of programme countries, ultimate risk basis, EUR bn (BIS):"]
    o = claims[claims["sector"] == "O"]
    for debtor in ("EL", "IE", "PT", "CY", "ES", "IT"):
        for rep in ("5A", "DE", "FR"):
            s = o[(o["debtor"] == debtor) & (o["reporter"] == rep)].set_index("quarter")["eur_bn"]
            if s.empty:
                continue
            peak_q = s.loc["2008-Q1":"2011-Q4"].idxmax() if len(s.loc["2008-Q1":"2011-Q4"]) else None
            vals = "; ".join(f"{q} {s[q]:.1f}" for q in QUARTERS if q in s.index)
            lines.append(f"  {debtor} by {'all reporting banks' if rep == '5A' else rep + ' banks'}: peak {peak_q} {s[peak_q]:.1f}; {vals}")
    a = claims[(claims["sector"] == "A") & (claims["reporter"] == "5A")]
    lines += ["", "Foreign claims of all reporting banks on all sectors (EUR bn), 2009-Q4 / 2012-Q2 / 2025-Q1:"]
    for debtor in ("EL", "IE", "PT", "CY", "ES", "IT"):
        s = a[a["debtor"] == debtor].set_index("quarter")["eur_bn"]
        lines.append(f"  {debtor}: " + " / ".join(f"{s.get(q, np.nan):.1f}" for q in ("2009-Q4", "2012-Q2", "2025-Q1")))

    q = debt_by_instrument()
    if q is not None and len(q):
        q = q[q["na_item"].isin(["GD", "F4", "F3"])].pivot_table(index=["geo", "time"], columns="na_item", values="value").reset_index()
        q["loan_share"] = q["F4"] / q["GD"]
        q.to_csv(RESULTS / "ledger_debt_instruments.csv", index=False, float_format="%.4f")
        lines += ["", "Loans as a share of general government debt (Eurostat quarterly), Q4 of 2009 / 2011 / 2012 / 2015 / 2018, and the latest quarter:"]
        for g in PROGRAMME:
            s = q[q["geo"] == g].set_index("time")["loan_share"]
            last = s.index.max() if len(s) else None
            vals = [f"{s.get(f'{y}-Q4', np.nan):.2f}" for y in (2009, 2011, 2012, 2015, 2018)]
            lines.append(f"  {g}: " + " / ".join(vals) + (f" / {s[last]:.2f} ({last})" if last else ""))

    lines += ["", "Documented private-sector involvement (Greece, 2012):"]
    for item in ("psi_exchanged", "psi_transfer", "esmt_total", "esmt_old_debt", "esmt_interest", "esmt_banks", "esmt_budget", "iwh_germany", "iwh_germany_updated"):
        s = d[d["item"] == item].iloc[0]
        lines.append(f"  {item}: {s.value} {s.unit} ({s.as_of}) - {s.source}")
    text = "\n".join(lines) + "\n"
    (RESULTS / "ledger_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
