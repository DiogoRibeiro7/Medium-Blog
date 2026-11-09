"""Is France 2026 a 2011-style contagion? Co-movement of euro-area spreads, monthly.

    python contagion.py

Spreads are 10-year yields minus Germany's, monthly averages (Eurostat and ECB convergence
series). For a window of monthly spread changes:

- the common factor of the other countries is the first principal component of their
  standardised changes (Italy, Spain, Belgium, Austria, the Netherlands, Finland, Portugal,
  Ireland; Greece is left out because its 2010-2012 moves would dominate any factor);
- R^2 of a country's changes on that factor (leave-one-out: the country is never in its own
  factor) measures how much of its spread movement is shared;
- the cumulative change over the window splits into a common part (fitted) and a specific part;
- intervals come from a moving-block bootstrap of months (blocks of 3, 2,000 draws).

Windows: 24 months to the French peak of 2011 (2010-01 to 2011-12) and the latest 24 months,
plus a rolling 24-month series from 2001 for the figure. The same leave-one-out R^2 is computed
for every country in the latest window, as a placebo: if France is not unusual, its R^2 should
sit among the others'.

Writes ``results/contagion_*.csv`` and ``results/contagion_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import NAMES, RESULTS, monthly_yields, write

OTHERS = ("IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE")
PANEL = ("FR",) + OTHERS
WINDOW = 24
CRISIS = ("2010-01", "2011-12")
BLOCK, DRAWS, SEED = 3, 2000, 20111109


def spreads() -> pd.DataFrame:
    m = monthly_yields()
    return m.drop(columns="DE").sub(m["DE"], axis=0)


def factor(changes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """First principal component of standardised changes, signed so that it rises with spreads.

    Returns the factor of demeaned changes (for R^2), the same weights applied to changes that
    are scaled but not demeaned (so that a common widening shows up in the factor's sum), and
    the share of variance of the first component."""
    sd = changes.std(ddof=1)
    z = (changes - changes.mean()) / sd
    ev, vec = np.linalg.eigh(np.corrcoef(z.to_numpy().T))
    w = vec[:, -1] * np.sign(vec[:, -1].sum())
    return z.to_numpy() @ w, (changes / sd).to_numpy() @ w, float(ev[-1] / ev.sum())


def shared(changes: pd.DataFrame, target: str) -> dict:
    """R^2 of ``target`` on the common factor of the other panel countries, and the split of its
    cumulative change into common and specific parts."""
    others = [c for c in PANEL if c != target]
    f, f_raw, share = factor(changes[others])
    y = changes[target].to_numpy()
    X = np.column_stack([np.ones(len(f)), f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.var(y - X @ b) / np.var(y)
    # split of the cumulative change: loading on the raw factor (no intercept) times its sum
    lam = float(f_raw @ y / (f_raw @ f_raw))
    common = lam * float(f_raw.sum())
    return dict(r2=float(r2), beta=float(b[1]), share_pc1=share, cum=float(y.sum()), common=common, specific=float(y.sum() - common),
                sd=float(np.std(y, ddof=1)), sd_specific=float(np.std(y - X @ b, ddof=1)))


def block_bootstrap(changes: pd.DataFrame, target: str, rng: np.random.Generator) -> np.ndarray:
    n = len(changes)
    starts = np.arange(n - BLOCK + 1)
    out = np.empty(DRAWS)
    for k in range(DRAWS):
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, size=int(np.ceil(n / BLOCK)))])[:n]
        out[k] = shared(changes.iloc[idx], target)["r2"]
    return out


def run() -> str:
    s = spreads()
    ds = s.diff().dropna(subset=list(PANEL))
    latest = ds.index.max()
    recent = (str(latest - WINDOW + 1), str(latest))
    rng = np.random.default_rng(SEED)
    lines = [f"Monthly spreads to Germany, {ds.index.min()} to {latest}; windows of {WINDOW} months; factor from {', '.join(OTHERS)} (leave-one-out)"]

    rows, boot = [], {}
    for name, (a, b) in (("2011", CRISIS), ("latest", recent)):
        w = ds.loc[a:b, list(PANEL)]
        for c in PANEL:
            r = shared(w, c)
            r.update(window=name, start=a, end=b, geo=c)
            if c in ("FR", "IT", "ES"):
                bs = block_bootstrap(w, c, rng)
                boot[(name, c)] = bs
                r.update(r2_lo=float(np.percentile(bs, 5)), r2_hi=float(np.percentile(bs, 95)))
            rows.append(r)
    ep = pd.DataFrame(rows)
    ep.to_csv(RESULTS / "contagion_episodes.csv", index=False)
    for name in ("2011", "latest"):
        f = ep[ep.window == name].set_index("geo")
        a, b = f.iloc[0][["start", "end"]]
        fr = f.loc["FR"]
        lines.append("")
        lines.append(f"Window {name} ({a} to {b}): first factor of the others explains {fr.share_pc1:.2f} of their standardised variance")
        lines.append(f"  France: R^2 on the others' factor {fr.r2:.2f} (90% block-bootstrap interval {fr.r2_lo:.2f}-{fr.r2_hi:.2f}); cumulative spread change {fr.cum:+.2f} pp, of which common {fr.common:+.2f} and specific {fr.specific:+.2f}; sd of monthly changes {100 * fr.sd:.1f} bp, of the specific part {100 * fr.sd_specific:.1f} bp")
        for c in ("IT", "ES"):
            r = f.loc[c]
            lines.append(f"  {NAMES[c]}: R^2 {r.r2:.2f} ({r.r2_lo:.2f}-{r.r2_hi:.2f}); cumulative change {r.cum:+.2f} pp (common {r.common:+.2f}, specific {r.specific:+.2f})")
        order = f["r2"].sort_values()
        lines.append("  leave-one-out R^2, all panel countries (lowest first): " + ", ".join(f"{c} {v:.2f}" for c, v in order.items()))
        lines.append(f"  France ranks {int((order.index == 'FR').nonzero()[0][0]) + 1} of {len(order)} from the bottom")
        w = ds.loc[a:b]
        lines.append("  correlation of monthly changes with France: " + ", ".join(f"{c} {w['FR'].corr(w[c]):.2f}" for c in OTHERS))
        bb = np.polyfit(w["FR"], w["IT"], 1)[0]
        lines.append(f"  Italy's spread change per 1 pp change in France's: {bb:.2f}")
    d = boot[("2011", "FR")] - boot[("latest", "FR")]
    lines.append("")
    lines.append(f"France's R^2, 2011 window minus latest window: {ep[(ep.window == '2011') & (ep.geo == 'FR')].r2.iloc[0] - ep[(ep.window == 'latest') & (ep.geo == 'FR')].r2.iloc[0]:+.2f} (90% interval of the bootstrap difference {np.percentile(d, 5):+.2f} to {np.percentile(d, 95):+.2f}; share of draws above zero {np.mean(d > 0):.3f})")

    # rolling series
    roll = []
    for end in ds.index[WINDOW - 1 :]:
        w = ds.loc[end - WINDOW + 1 : end, list(PANEL)]
        if len(w) < WINDOW or w.isna().any().any():
            continue
        r = shared(w, "FR")
        roll.append(dict(end=str(end), r2_fr=r["r2"], sd_fr=r["sd"], sd_specific_fr=r["sd_specific"], share_pc1=r["share_pc1"], corr_it=w["FR"].corr(w["IT"]), corr_es=w["FR"].corr(w["ES"]), r2_it=shared(w, "IT")["r2"]))
    R = pd.DataFrame(roll)
    R.to_csv(RESULTS / "contagion_rolling.csv", index=False)
    calm = R[(R.end >= "2015-01") & (R.end <= "2019-12")]
    lines.append(f"Rolling {WINDOW}-month R^2 of France on the others' factor: windows ending 2011 {R[R.end.str.startswith('2011')].r2_fr.mean():.2f} on average; ending 2015-2019 {calm.r2_fr.mean():.2f} (range {calm.r2_fr.min():.2f}-{calm.r2_fr.max():.2f}); latest {R.r2_fr.iloc[-1]:.2f}; lowest since 2001: {R.r2_fr.min():.2f} (window ending {R.loc[R.r2_fr.idxmin(), 'end']}, when the sd of France's monthly spread changes was {100 * R.loc[R.r2_fr.idxmin(), 'sd_fr']:.1f} bp)")
    lines.append(f"  sd of France's monthly spread changes: windows ending 2011 {100 * R[R.end.str.startswith('2011')].sd_fr.mean():.1f} bp; 2015-2019 {100 * calm.sd_fr.mean():.1f}; latest {100 * R.sd_fr.iloc[-1]:.1f}; of the part not shared with the others: 2011 {100 * R[R.end.str.startswith('2011')].sd_specific_fr.mean():.1f}, 2015-2019 {100 * calm.sd_specific_fr.mean():.1f}, latest {100 * R.sd_specific_fr.iloc[-1]:.1f}")
    pre = R[(R.end >= "2005-01") & (R.end <= "2007-12")]
    lines.append(f"  windows ending 2005-2007: R^2 {pre.r2_fr.mean():.2f}, sd {100 * pre.sd_fr.mean():.1f} bp (spreads near zero, changes mostly noise)")

    # levels and changes over the twelve months to each peak
    lines.append("")
    peaks = {"2011": s["FR"].loc["2010-01":"2012-12"].idxmax(), "latest": latest}
    ch = []
    for name, p in peaks.items():
        before = p - 12
        lv = s.loc[p, list(PANEL)].sort_values(ascending=False)
        lines.append(f"Spreads at {name} reference month {p} (pp, widest first): " + ", ".join(f"{c} {v:.2f}" for c, v in lv.items()) + f"; France ranks {list(lv.index).index('FR') + 1} of {len(lv)}; Greece (not in the panel) {s.loc[p, 'EL']:.2f}")
        dd = (s.loc[p, list(PANEL)] - s.loc[before, list(PANEL)]).sort_values(ascending=False)
        lines.append(f"  change over the 12 months from {before}: " + ", ".join(f"{c} {v:+.2f}" for c, v in dd.items()) + f"; countries widening by more than 0.10 pp: {int((dd > 0.10).sum())} of {len(dd)}")
        for c in PANEL:
            ch.append(dict(episode=name, month=str(p), geo=c, level=float(s.loc[p, c]), change_12m=float(s.loc[p, c] - s.loc[before, c])))
    pd.DataFrame(ch).to_csv(RESULTS / "contagion_changes.csv", index=False)
    lines.append("")
    for start in (f"{latest.year}-01", f"{latest.year}-06"):
        dd = (s.loc[latest, list(PANEL)] - s.loc[pd.Period(start, "M"), list(PANEL)]).sort_values(ascending=False)
        lines.append(f"Change in spreads from {start} to {latest} (pp): " + ", ".join(f"{c} {v:+.2f}" for c, v in dd.items()))
    w = ds.loc[f"{latest.year}-01":str(latest)]
    lines.append(f"Correlation of monthly changes with France, {latest.year}-01 to {latest} ({len(w)} months): " + ", ".join(f"{c} {w['FR'].corr(w[c]):.2f}" for c in OTHERS))
    return write("contagion_summary.txt", lines)


if __name__ == "__main__":
    print(run())
