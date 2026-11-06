"""Worked example: the never-recover share of EU industry busts, by depth band.

    python application.py

Uses the episodes of the companion analysis ``../recovery-after-busts`` (EU-27 industries,
1995-2025, troughs in the single-market and euro stages, true peak observed). For each capacity
measure, version and depth band it reports what the data identify without a latency model (the
Kaplan-Meier plateau, an upper bound for the never-recover share), the Maller-Zhou statistic,
cure-model estimates with lognormal and Weibull latency, and the two model-based diagnostics from
the simulation study (the gap between latency families and the cure / no-cure extrapolation gap).
Writes ``results/application.csv`` and ``results/application_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models import fit, km_at, km_plateau, maller_zhou_alpha

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
EPISODES = HERE.parent / "recovery-after-busts" / "results" / "episodes_all.csv"

BANDS = [0.0, 0.3, 0.5, 0.8, np.inf]
LABELS = ["<0.30", "0.30-0.50", "0.50-0.80", ">0.80"]


def load() -> pd.DataFrame:
    d = pd.read_csv(EPISODES)
    d = d[d.main & d.stage.isin(["single market", "euro"])].copy()
    d["band"] = pd.cut(d.depth, BANDS, labels=LABELS, right=False)
    return d


def analyse(g: pd.DataFrame) -> dict:
    t = np.maximum(g.t_full.to_numpy(float), 0.5)
    e = g.full_recovered.to_numpy(int)
    ln, wb = fit(t, e, "lognormal", True), fit(t, e, "weibull", True)
    ln0, wb0 = fit(t, e, "lognormal", False), fit(t, e, "weibull", False)
    best_cure = min((ln, wb), key=lambda f: f.aic)
    best_none = min((ln0, wb0), key=lambda f: f.aic)
    horizon = 2 * t.max()
    s10, s20 = km_at(t, e, [10.0, 20.0])
    # a stable upper bound for pi: Kaplan-Meier at the last time with at least 10 busts still at risk
    t_bound = float(np.sort(t)[-10]) if len(t) >= 10 else float(t.min())
    bound = float(km_at(t, e, [t_bound])[0])
    return {
        "n": len(g),
        "events": int(e.sum()),
        "max_follow_up": float(t.max()),
        "at_risk_10": int((t >= 10).sum()),
        "at_risk_20": int((t >= 20).sum()),
        "km_10": float(s10),
        "km_20": float(s20),
        "km_plateau": km_plateau(t, e),
        "km_bound": bound,
        "t_bound": t_bound,
        "mz_alpha": maller_zhou_alpha(t, e),
        "pi_ln": ln.pi,
        "pi_ln_lo": ln.pi_lo,
        "pi_ln_hi": ln.pi_hi,
        "pi_wb": wb.pi,
        "pi_wb_lo": wb.pi_lo,
        "pi_wb_hi": wb.pi_hi,
        "aic_cure": best_cure.aic,
        "aic_none": best_none.aic,
        "cure_preferred": int(best_cure.aic < best_none.aic),
        "family_gap": abs(ln.pi - wb.pi),
        "extrap_gap": float(best_cure.survival(horizon)[0] - best_none.survival(horizon)[0]),
        "horizon": horizon,
        "pi_best": best_cure.pi,
        "passes_two_checks": int(abs(ln.pi - wb.pi) < 0.05 and best_cure.aic < best_none.aic),
    }


def main() -> None:
    RES.mkdir(exist_ok=True)
    d = load()
    rows = []
    for (m, v), g in d.groupby(["measure", "version"]):
        for lab in LABELS:
            sub = g[g.band == lab]
            rows.append({"measure": m, "version": v, "band": lab, **analyse(sub)})
    A = pd.DataFrame(rows)
    A.to_csv(RES / "application.csv", index=False, float_format="%.6g")
    L = ["EU industry busts (single-market and euro stages), never-recover share by depth band", ""]
    for (m, v), g in A.groupby(["measure", "version"], sort=False):
        L.append(f"{m} / {v}")
        for _, r in g.iterrows():
            L.append(
                f"  {r.band:9s} n={r.n:4d} recovered={r.events:4d} max follow-up {r.max_follow_up:.0f} y, at risk 10 y {r.at_risk_10}, 20 y {r.at_risk_20}; "
                f"KM 10 y {r.km_10:.2f}, upper bound (KM at {r.t_bound:.0f} y, 10 at risk) {r.km_bound:.2f}, KM plateau {r.km_plateau:.2f}; MZ alpha {r.mz_alpha:.3f}; "
                f"cure lognormal {r.pi_ln:.2f} [{r.pi_ln_lo:.2f}, {r.pi_ln_hi:.2f}], cure Weibull {r.pi_wb:.2f} [{r.pi_wb_lo:.2f}, {r.pi_wb_hi:.2f}]; "
                f"family gap {r.family_gap:.2f}; cure preferred by AIC {bool(r.cure_preferred)} ({r.aic_cure:.1f} vs {r.aic_none:.1f}); extrapolation gap at {r.horizon:.0f} y {r.extrap_gap:+.2f}; "
                f"passes both checks: {bool(r.passes_two_checks)}"
            )
    text = "\n".join(L) + "\n"
    (RES / "application_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
