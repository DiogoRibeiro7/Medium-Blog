"""If the ECB cannot control it: four channels from France to the rest of the EU.

    python channels.py

1. Other sovereigns. Monthly changes in 10-year spreads to Germany, 2010-01 onward. France's own
   shock is the residual of its spread change on the first principal component of the other
   countries' standardised changes. For each other country j, its spread change is regressed on
   France's own shock and on the first component of the countries other than France and j. The
   slope is the basis-point response of j to a one-basis-point France-specific move. Intervals
   are from a moving-block bootstrap (blocks of 3 months, 2,000 draws).
2. Banks. EBA 2025 transparency exercise, June 2025: banks' direct exposures to the French
   central government (carrying amount net of short positions, all maturities), by the home
   country of the bank, against the banks' CET1 capital. A mark-to-market loss from a uniform
   rise in French yields is computed on the fair-valued part only (held for trading, designated
   at fair value, fair value through other comprehensive income, less short positions), with the
   modified duration of a par bond at the midpoint of each maturity bucket.
3. Eurosystem balance sheet. French public-sector securities held under the PSPP and the PEPP
   (cumulative net purchases at book value, ECB files), against French general government debt;
   the loss-sharing split of those holdings; and the TARGET balance of the Banque de France.
4. EU budget and joint debt. France's share of EU-27 gross national income, applied to the
   proposed fixed annual amount for the repayment of NextGenerationEU in 2028-2034.

Writes ``results/channels_*.csv`` and ``results/channels_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA, NAMES, RESULTS, monthly_spreads, read, write

SEED = 20261109
N_BOOT = 2000
BLOCK = 3
OTHERS = ("IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE")
PERIODS = {"2013-2019": ("2013-01", "2019-12"), "2020-2026": ("2020-01", "2026-12"), "June 2024 on": ("2024-06", "2026-12")}

# EBA codes (TR_Metadata.xlsx, sheet Country) and items (tr_sov.csv)
EBA_FRANCE = 9
EBA_TOTAL_MATURITY = 8
ITEM_GROSS = 2520810     # total gross carrying amount of non-derivative financial assets
ITEM_NET = 2520811       # total carrying amount of non-derivative financial assets, net of short positions
ITEM_FV = {2520812: "held for trading", 2520813: "designated at fair value", 2520814: "FVOCI"}
ITEM_AC = 2520815        # amortised cost
ITEM_CET1 = 2520102      # CET1 capital (transitional), tr_oth.csv
EBA_PERIOD = 202506
MATURITY_MID = {1: 0.125, 2: 0.625, 3: 1.5, 4: 2.5, 5: 4.0, 6: 7.5, 7: 15.0}
YIELD_FOR_DURATION = 3.5
SHOCK_BP = 100

# ECB capital key from 1 January 2024 (% of subscribed capital); euro area of 21 from January 2026
KEY_FR = 16.3575
KEY_NON_EURO_2024 = 18.2319
KEY_BG = 0.9783
PSPP_SHARED_GOVT = 10 / 90  # ECB's own 10% of purchases, all in government bonds, against the NCBs' 80%
NGEU_ANNUAL_BN = 24.0  # proposed fixed annual amount for NGEU repayment, 2028-2034, current prices
TPI_SIZES_BN = (100, 250, 500)


def pc1(z: np.ndarray) -> np.ndarray:
    z = (z - z.mean(0)) / z.std(0)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    f = u[:, 0] * s[0]
    return f if np.corrcoef(f, z.mean(1))[0, 1] >= 0 else -f


def ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]


def spillover_betas(ch: pd.DataFrame) -> dict[str, float]:
    others = [c for c in OTHERS if c in ch]
    f_ex_fr = pc1(ch[others].to_numpy())
    b = ols(ch["FR"].to_numpy(), f_ex_fr[:, None])
    u = ch["FR"].to_numpy() - np.column_stack([np.ones(len(ch)), f_ex_fr]) @ b
    out = {"fr_own_share": float(u.var() / ch["FR"].var())}
    for j in others:
        rest = [c for c in others if c != j]
        f = pc1(ch[rest].to_numpy())
        beta = ols(ch[j].to_numpy(), np.column_stack([u, f]))
        out[j] = float(beta[1])
    return out


def calibration(n: int = 400, reps: int = 40) -> dict[str, float]:
    """Estimator on simulated data with known responses (Italy 0.4, Belgium 0.2, others 0).

    Every country loads on a common factor (loading 1 + N(0, 0.3)); France also carries its own
    shock of the same variance; idiosyncratic noise has standard deviation 0.3.
    """
    true = {"IT": 0.4, "ES": 0.0, "BE": 0.2, "AT": 0.0, "NL": 0.0, "FI": 0.0, "PT": 0.0, "IE": 0.0}
    est = []
    for seed in range(reps):
        rng = np.random.default_rng(SEED + seed)
        common, own = rng.normal(size=n), rng.normal(size=n)
        cols = {"FR": common + own}
        for c, b in true.items():
            cols[c] = (1 + 0.3 * rng.normal()) * common + b * own + 0.3 * rng.normal(size=n)
        est.append(spillover_betas(pd.DataFrame(cols)))
    m = pd.DataFrame(est).mean()
    zero = [c for c, b in true.items() if b == 0]
    return dict(IT=float(m["IT"]), BE=float(m["BE"]), zero_mean=float(m[zero].mean()), zero_min=float(m[zero].min()), zero_max=float(m[zero].max()))


def spillover() -> tuple[pd.DataFrame, list[str]]:
    s = monthly_spreads()
    ch = s[["FR", *OTHERS]].diff().dropna()
    rng = np.random.default_rng(SEED)
    rows, lines = [], []
    for label, (a, b) in PERIODS.items():
        c = ch.loc[a:b]
        est = spillover_betas(c)
        n = len(c)
        draws = []
        for _ in range(N_BOOT):
            starts = rng.integers(0, n - BLOCK + 1, size=int(np.ceil(n / BLOCK)))
            idx = np.concatenate([np.arange(k, k + BLOCK) for k in starts])[:n]
            draws.append(spillover_betas(c.iloc[idx].reset_index(drop=True)))
        D = pd.DataFrame(draws)
        for j in OTHERS:
            rows.append(dict(period=label, country=j, beta=est[j], lo=D[j].quantile(.05), hi=D[j].quantile(.95), months=n))
        lines.append(f"  {label} ({n} months; France's own shock is {est['fr_own_share']:.0%} of the variance of its spread changes): "
                     + "; ".join(f"{NAMES[j]} {est[j]:+.2f} [{D[j].quantile(.05):+.2f}, {D[j].quantile(.95):+.2f}]" for j in OTHERS))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "channels_spillover.csv", index=False, float_format="%.4f")
    recent = df[df.period == "June 2024 on"].set_index("country")
    full = df[df.period == "2013-2019"].set_index("country")
    lines.append(f"  A 100 bp France-specific widening, June 2024 on: response {', '.join(f'{NAMES[j]} {100 * recent.loc[j, 'beta']:+.0f} bp' for j in OTHERS)}")
    lines.append(f"  Same shock with 2013-2019 slopes: {', '.join(f'{NAMES[j]} {100 * full.loc[j, 'beta']:+.0f} bp' for j in OTHERS)}")
    cal = calibration()
    lines.append(f"  Calibration on simulated data with known responses (Italy 0.40, Belgium 0.20, six others 0): estimates Italy {cal['IT']:.2f}, Belgium {cal['BE']:.2f}, "
                 f"the six others {cal['zero_mean']:+.2f} on average ({cal['zero_min']:+.2f} to {cal['zero_max']:+.2f}); the estimator leans slightly negative, so values within about 0.1 of zero mean no response")
    yr = s.loc["2025-12":"2026-09"]
    lines.append("  Change in spreads, December 2025 to September 2026 (pp): " + ", ".join(f"{NAMES[c]} {yr[c].iloc[-1] - yr[c].iloc[0]:+.2f}" for c in ("FR", *OTHERS)))
    return df, lines


def duration(t: float, y: float = YIELD_FOR_DURATION) -> float:
    """Modified duration of a par bond with annual coupon y% and maturity t years."""
    r = y / 100
    if t <= 1:
        return t / (1 + r)
    mac = (1 + r) / r - (1 + r) / (r * (1 + r) ** t)  # Macaulay duration of a par bond (closed form)
    return mac / (1 + r)


def banks() -> tuple[pd.DataFrame, list[str]]:
    sov = read("eba_tr_sov.csv.gz")
    sov = sov[(sov["Period"] == EBA_PERIOD) & (sov["Country"] == EBA_FRANCE)]
    oth = read("eba_tr_oth.csv.gz")
    cet1 = oth[(oth["Period"] == EBA_PERIOD) & (oth["Item"] == ITEM_CET1)].groupby(["LEI_Code", "NSA"])["Amount"].sum().reset_index()
    net = sov[(sov["Item"] == ITEM_NET) & (sov["Maturity"] == EBA_TOTAL_MATURITY)].groupby(["LEI_Code", "NSA"])["Amount"].sum()
    # fair-valued holdings net of short positions, by maturity bucket: FV portfolios minus (gross - net)
    bucket = sov[sov["Maturity"].isin(MATURITY_MID)].pivot_table(index=["LEI_Code", "NSA", "Maturity"], columns="Item", values="Amount", aggfunc="sum").fillna(0)
    fvnet = bucket[[i for i in ITEM_FV if i in bucket]].sum(axis=1) - (bucket.get(ITEM_GROSS, 0) - bucket.get(ITEM_NET, 0))
    fvnet = fvnet.rename("fv").reset_index()
    fvnet["dur"] = fvnet["Maturity"].map(lambda m: duration(MATURITY_MID[m]))
    fvnet["loss"] = fvnet["fv"] * fvnet["dur"] * SHOCK_BP / 10_000
    fv_amt = fvnet.groupby(["LEI_Code", "NSA"])["fv"].sum()
    loss = fvnet.groupby(["LEI_Code", "NSA"])["loss"].sum()
    ac = sov[(sov["Item"] == ITEM_AC) & sov["Maturity"].isin(MATURITY_MID)].groupby(["LEI_Code", "NSA"])["Amount"].sum()
    b = cet1.set_index(["LEI_Code", "NSA"]).rename(columns={"Amount": "cet1"})
    b = b.join(net.rename("exposure"), how="left").join(fv_amt.rename("fair_value"), how="left").join(loss.rename("loss_100bp"), how="left").join(ac.rename("amortised_cost"), how="left").fillna(0)
    by = b.groupby("NSA")[["cet1", "exposure", "fair_value", "amortised_cost", "loss_100bp"]].sum() / 1000  # EUR bn
    by["banks"] = b.groupby("NSA").size()
    by["exp_pct_cet1"] = 100 * by["exposure"] / by["cet1"]
    by["loss_pct_cet1"] = 100 * by["loss_100bp"] / by["cet1"]
    by = by.sort_values("exposure", ascending=False)
    by.to_csv(RESULTS / "channels_banks.csv", float_format="%.3f")
    tot = by[["cet1", "exposure", "fair_value", "amortised_cost", "loss_100bp"]].sum()
    fr = by.loc["FR"]
    nonfr = tot - fr[["cet1", "exposure", "fair_value", "amortised_cost", "loss_100bp"]]
    top = by.drop(index="FR").head(6)
    lines = [
        f"  {int(by['banks'].sum()) - int('OT' in by.index)} banks plus the EBA's aggregate of other banks, {len(by) - int('OT' in by.index)} countries, June 2025. Direct exposure to the French central government (net of short positions): EUR {tot.exposure:.0f} bn, of which French banks {fr.exposure:.0f} bn ({fr.exposure / tot.exposure:.0%})",
        f"  French banks: exposure {fr.exposure:.0f} bn = {fr.exp_pct_cet1:.0f}% of their CET1 ({fr.cet1:.0f} bn); fair-valued {fr.fair_value:.0f} bn; loss on a {SHOCK_BP} bp rise {fr.loss_100bp:.1f} bn = {fr.loss_pct_cet1:.1f}% of CET1",
        f"  Banks outside France: exposure {nonfr.exposure:.0f} bn = {100 * nonfr.exposure / nonfr.cet1:.1f}% of their CET1 ({nonfr.cet1:.0f} bn); fair-valued {nonfr.fair_value:.0f} bn; loss on a {SHOCK_BP} bp rise {nonfr.loss_100bp:.1f} bn = {100 * nonfr.loss_100bp / nonfr.cet1:.2f}% of CET1",
        "  Largest holders outside France (exposure EUR bn; % of CET1; 100 bp loss % of CET1): " + "; ".join(f"{NAMES.get(c, c)} {r.exposure:.1f}, {r.exp_pct_cet1:.1f}%, {r.loss_pct_cet1:.2f}%" for c, r in top.iterrows()),
        f"  Fair-valued share of the exposure: all banks {tot.fair_value / tot.exposure:.0%}; French banks {fr.fair_value / fr.exposure:.0%}",
        f"  Durations used (years, par bond at {YIELD_FOR_DURATION}%): " + ", ".join(f"{MATURITY_MID[m]}y {duration(MATURITY_MID[m]):.2f}" for m in MATURITY_MID),
    ]
    return by, lines


def eurosystem() -> tuple[pd.DataFrame, list[str]]:
    def holdings(name: str, cumulative_col: str) -> pd.Series:
        raw = pd.read_csv(DATA / name, header=2, nrows=20, dtype=str)  # first block: monthly net purchases
        raw = raw.rename(columns={raw.columns[0]: "country"}).set_index("country")
        months = [c for c in raw.columns if c.lower() != cumulative_col.lower() and not c.startswith("Unnamed")]
        fr = raw.loc["France", months].astype(str).str.replace(",", "").astype(float)
        idx = pd.to_datetime(months, format="%d/%m/%Y", errors="coerce")
        if idx.isna().all():
            idx = pd.to_datetime(months, format="%b-%y")
        fr.index = idx.to_period("M")
        total = float(str(raw.loc["France", [c for c in raw.columns if c.lower() == cumulative_col.lower()][0]]).replace(",", ""))
        cum = fr.cumsum()
        assert abs(cum.iloc[-1] - total) < 5, (name, cum.iloc[-1], total)
        return cum / 1000  # EUR bn
    pspp = holdings("ecb_PSPP_breakdown_history.csv", "cumulative net purchases")
    pepp = holdings("ecb_PEPP_public_sector_securities_breakdown_history.csv", "Cumulative")
    h = pd.concat([pspp.rename("pspp"), pepp.rename("pepp")], axis=1).ffill().fillna(0)
    h["total"] = h["pspp"] + h["pepp"]
    debt = read("debt_quarterly.csv")
    debt = debt[debt["geo"] == "FR"].set_index("time")["value"] / 1000
    last_q = debt.index.max()
    h.to_csv(RESULTS / "channels_eurosystem.csv", float_format="%.3f")
    peak_m = h["total"].idxmax()
    last = h.index.max()
    fr_key = KEY_FR / (100 - KEY_NON_EURO_2024 + KEY_BG)
    shared = PSPP_SHARED_GOVT
    non_fr = shared * (1 - fr_key)
    tot_now = float(h.loc[last, "total"])
    t = read("target_balances.csv")
    t["geo"] = t["REF_AREA"]
    t["OBS_VALUE"] = pd.to_numeric(t["OBS_VALUE"], errors="coerce")
    tb = t.pivot_table(index="TIME_PERIOD", columns="geo", values="OBS_VALUE") / 1000
    tb.index = pd.PeriodIndex(tb.index, freq="M")
    tb.to_csv(RESULTS / "channels_target.csv", float_format="%.3f")
    gdp = read("gdp.csv")
    gdp = gdp.pivot_table(index="time", columns="geo", values="value") / 1000
    def tpct(c: str, m: str) -> float:
        return 100 * float(tb.loc[m, c]) / float(gdp.loc[int(m[:4]) if int(m[:4]) in gdp.index else gdp.index.max(), "EL" if c == "GR" else c])
    lines = [
        f"  French public-sector securities held by the Eurosystem (book value, EUR bn): PSPP {h.loc[last, 'pspp']:.0f} + PEPP {h.loc[last, 'pepp']:.0f} = {tot_now:.0f} at {last}; peak {h['total'].max():.0f} at {peak_m}",
        f"  French general government debt {last_q}: EUR {debt.loc[last_q]:.0f} bn; Eurosystem programme holdings = {100 * tot_now / debt.loc[last_q]:.1f}% of it (peak-month share against debt of the same quarter: {100 * h['total'].max() / debt.loc[str(peak_m.asfreq('Q')).replace('Q', '-Q')]:.1f}%)",
        f"  Run-off since the peak: EUR {h['total'].max() - tot_now:.0f} bn; last 12 months {float(h['total'].iloc[-13] - tot_now):.0f} bn",
        f"  Loss sharing (PSPP rules, applied also to PEPP): ECB-held share of government bonds {shared:.1%}; Banque de France share of the euro-area capital key {fr_key:.2%}; "
        f"share of a loss on French bonds borne outside France {non_fr:.1%} (EUR {non_fr * tot_now:.0f} bn of the {tot_now:.0f} bn)",
        "  TPI purchases (no loss-sharing rule published): share borne outside France if shared by capital key "
        f"{1 - fr_key:.1%}, if on PSPP rules {non_fr:.1%}; for purchases of " + ", ".join(f"EUR {s} bn: {s * (1 - fr_key):.0f} vs {s * non_fr:.0f} bn" for s in TPI_SIZES_BN),
        "  TARGET balances (EUR bn, % of GDP of the year): " + "; ".join(
            f"{NAMES['EL' if c == 'GR' else c]} {m}: {tb.loc[m, c]:+.0f} ({tpct(c, m):+.1f}%)" for c, m in (("FR", "2024-05"), ("FR", str(tb.index.max())), ("IT", "2011-06"), ("IT", "2012-08"), ("ES", "2011-06"), ("ES", "2012-08"))),
        f"  France's TARGET balance (end-month values swing at quarter and year ends): 12-month average to May 2024 {tb.loc['2023-06':'2024-05', 'FR'].mean():+.0f} bn; 12-month average to {tb.index.max()} {tb.loc[str(tb.index.max() - 11):, 'FR'].mean():+.0f} bn; "
        f"September 2024 {tb.loc['2024-09', 'FR']:+.0f}, September 2025 {tb.loc['2025-09', 'FR']:+.0f}, September 2026 {tb.loc['2026-09', 'FR']:+.0f}; June to September 2026 {float(tb.loc['2026-09', 'FR'] - tb.loc['2026-06', 'FR']):+.0f} bn",
        f"  Italy's change June 2011 to August 2012: {float(tb.loc['2012-08', 'IT'] - tb.loc['2011-06', 'IT']):+.0f} bn; Spain's: {float(tb.loc['2012-08', 'ES'] - tb.loc['2011-06', 'ES']):+.0f} bn",
    ]
    return h, lines


def eu_budget() -> tuple[pd.DataFrame, list[str]]:
    g = read("gni.csv")
    g = g.pivot_table(index="time", columns="geo", values="value")
    y = int(g["FR"].dropna().index.max())
    members = [c for c in g.columns if c != "EU27_2020"]
    total = g.loc[y, "EU27_2020"] if "EU27_2020" in g and pd.notna(g.loc[y, "EU27_2020"]) else g.loc[y, members].sum()
    share = g.loc[y, members] / total
    share = share.sort_values(ascending=False)
    share.rename("gni_share").to_csv(RESULTS / "channels_gni_share.csv", float_format="%.5f")
    fr = float(share["FR"])
    lines = [
        f"  EU-27 GNI {y}: EUR {total / 1000:.0f} bn; shares: " + ", ".join(f"{NAMES.get(c, {'PL': 'Poland'}.get(c, c))} {share[c]:.1%}" for c in share.index[:6]),
        f"  France's GNI share applied to the proposed EUR {NGEU_ANNUAL_BN:.0f} bn a year for NGEU repayment, 2028-2034: EUR {fr * NGEU_ANNUAL_BN:.1f} bn a year (EUR {7 * fr * NGEU_ANNUAL_BN:.0f} bn over seven years)",
        f"  The other 26 members: EUR {(1 - fr) * NGEU_ANNUAL_BN:.1f} bn a year; if France's share had to be covered by the others pro rata, their bill would rise by {fr / (1 - fr):.1%}",
    ]
    return share.to_frame(), lines


def run() -> str:
    lines = ["1. Other sovereigns: basis-point response to a 1 bp France-specific spread move (90% bootstrap interval)"]
    _, l1 = spillover()
    lines += l1
    lines += ["", "2. Banks (EBA 2025 transparency exercise, June 2025)"]
    _, l2 = banks()
    lines += l2
    lines += ["", "3. Eurosystem balance sheet"]
    _, l3 = eurosystem()
    lines += l3
    lines += ["", "4. EU budget and joint debt"]
    _, l4 = eu_budget()
    lines += l4
    return write("channels_summary.txt", lines)


if __name__ == "__main__":
    print(run())
