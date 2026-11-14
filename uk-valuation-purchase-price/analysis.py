"""Every number used in the article, from the committed aggregates.

    python analysis.py            # writes results/summary.txt and prints it

Reads ``data/build_log.txt`` and the tables written by ``model.py`` under ``results/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"
SPLITS = ["2008-2012", "2017-2019", "2022-2025"]
MODELS = ["M0", "M1", "M2"]
BANDS = ["below -10%", "-10% to 0", "0 to +10%", "+10% to +25%", "above +25%"]
EVENT_MODELS = ["M0-dist", "M1-dist", "M2-dist", "M2-dist-h", "C2"]


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def build_log() -> dict:
    out = {}
    for line in (DATA / "build_log.txt").read_text().splitlines():
        k, v = line.split("\t")
        out[k] = v
    return out


def main() -> int:
    L: list[str] = []
    w = L.append
    w("VALUATION MODELS AND THE PURCHASE PRICE: every number used in the article")
    w("=" * 76)
    b = build_log()
    run = json.loads((RES / "model_run.json").read_text())
    w("")
    w("[0] Data")
    w(f"Standard sales with a usable address: {int(b['standard_sales']):,}; consecutive pairs "
      f"{int(b['raw_pairs']):,}; kept after filters {int(b['clean_pairs']):,}")
    w(f"Postcode areas with an index: {b['areas_in_index']}; pairs in those areas "
      f"{int(b['pairs_in_indexed_areas']):,}; index smoothness lambda {b['index_lambda']}")
    w(f"Pairs used by the models: {run['n_pairs']:,}, of which first sale not a new build "
      f"{run['n_pairs_not_new']:,}; resale years {run['years'][0]}-{run['years'][1]}")
    for s in run["splits"]:
        w(f"  test {s['test_years']}: train {s['train_years']} ({s['n_train']:,} pairs, "
          f"{s['n_train_used']:,} used for the mean models, {s['n_train_quantile']:,} for quantiles "
          f"and classes), calibrate {s['cal_year']} ({s['n_cal']:,}), test {s['n_test']:,}")
        w(f"      share with m < 0: train {pct(s['train_share_m_negative'])}, test "
          f"{pct(s['test_share_m_negative'])}; m in [-10%, 0): train "
          f"{pct(s['train_share_m_minus10_0'])}, test {pct(s['test_share_m_minus10_0'])}")

    idx = pd.read_csv(DATA / "area_index.csv").pivot(index="t", columns="area", values="log_index")

    def t(y, m):
        return (y - 1995) * 12 + m - 1

    def falls(a0, a1, b1):
        out = {}
        for a in idx.columns:
            s_ = idx[a]
            pk = s_.loc[a0:a1].idxmax()
            out[a] = float(np.expm1(s_.loc[pk:b1].min() - s_.loc[pk]))
        return pd.Series(out)

    growth = (idx.loc[t(2006, 12)] - idx.loc[t(1996, 1)]) / (131 / 12)
    bust = falls(t(2006, 1), t(2008, 12), t(2013, 12))
    stall = falls(t(2015, 1), t(2018, 12), t(2020, 12))
    recent = falls(t(2021, 6), t(2023, 12), t(2025, 12))
    london = ["E", "EC", "N", "NW", "SE", "SW", "W", "WC"]
    w("")
    w("[0b] Area index")
    w(f"  1996-01 to 2006-12, annual log growth across areas: median {pct(growth.median())}, "
      f"range {pct(growth.min())} to {pct(growth.max())}")
    w(f"  peak (2006-2008) to trough (to 2013): median area {pct(bust.median())}, range "
      f"{pct(bust.max())} to {pct(bust.min())}")
    w(f"  peak (2015-2018) to trough (to 2020): London areas {pct(stall[london].max())} to "
      f"{pct(stall[london].min())}; other areas median {pct(stall.drop(london).median())}")
    w(f"  peak (2021-06 to 2023) to trough (to 2025): median area {pct(recent.median())}, range "
      f"{pct(recent.max())} to {pct(recent.min())}")

    met = pd.read_csv(RES / "metrics.csv")
    w("")
    w("[1] Point accuracy (d = log predicted - log actual; MdAE = median |d|; PE10 = share within 10%)")
    for sp in SPLITS:
        w(f"  test {sp}")
        for g in ["all"] + BANDS:
            gt = "all" if g == "all" else "mband"
            sub = met[(met.split == sp) & (met.group_type == gt) & (met.group == g)].set_index("model")
            if sub.empty:
                continue
            n = int(sub.n.iloc[0])
            parts = [f"{m}: MdAE {pct(sub.loc[m, 'mdae'], 2)} mean {pct(sub.loc[m, 'mean_err'], 2)} "
                     f"PE10 {pct(sub.loc[m, 'pe10'])}" for m in MODELS]
            w(f"    {g:>13} n={n:>9,}  " + " | ".join(parts))
    for gt, label in [("year", "year of resale"), ("hold", "holding band"),
                      ("first_sale", "first sale")]:
        w(f"  by {label}: MdAE and mean error")
        sub = met[met.group_type == gt]
        for sp in SPLITS:
            for g in sorted(sub[sub.split == sp].group.astype(str).unique()):
                r = sub[(sub.split == sp) & (sub.group.astype(str) == g)].set_index("model")
                w(f"    {sp} {g:>10} n={int(r.n.iloc[0]):>9,}  " + " | ".join(
                    f"{m}: {pct(r.loc[m, 'mdae'], 2)} ({pct(r.loc[m, 'mean_err'], 2)})" for m in MODELS))
    w("  M2 against M1, change in MdAE (percentage points) by region and band of m")
    rm = met[met.group_type == "region_mband"].copy()
    for sp in SPLITS:
        sub = rm[rm.split == sp].pivot_table(index="group", columns="model", values="mdae")
        nn = rm[(rm.split == sp) & (rm.model == "M2")].set_index("group").n
        for g, row in sub.iterrows():
            if nn.get(g, 0) >= 5000:
                w(f"    {sp} {g:>36}: n={int(nn[g]):>8,} M1 {pct(row.M1, 2)} M2 {pct(row.M2, 2)} "
                  f"change {100 * (row.M2 - row.M1):+.2f}")

    pdp = pd.read_csv(RES / "pdp_m.csv")
    w("")
    w("[2] Partial dependence of M2's correction on m (indexed value held fixed), in %")
    for sp in SPLITS:
        s = pdp[pdp.split == sp].set_index("m").mean_pred
        if s.empty:
            continue
        base = s.loc[0.0]
        pts = [-0.30, -0.20, -0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
        w(f"  {sp}: " + ", ".join(f"m={p:+.2f}: {100 * (s.loc[p] - base):+.2f}" for p in pts))
    imp = pd.read_csv(RES / "importance.csv")
    w("  share of M2's gain from each feature")
    for sp in SPLITS:
        s = imp[(imp.split == sp) & (imp.model == "M2")].sort_values("gain_share", ascending=False)
        w(f"    {sp}: " + ", ".join(f"{r.feature} {pct(r.gain_share)}" for r in s.itertuples()))

    cov = pd.read_csv(RES / "coverage.csv")
    w("")
    w("[3] 80% intervals: coverage (below / above) and mean width in log points")
    for sp in SPLITS:
        w(f"  test {sp}")
        for g in ["all"] + BANDS:
            gt = "all" if g == "all" else "mband"
            sub = cov[(cov.split == sp) & (cov.group_type == gt) & (cov.group == g)]
            if sub.empty:
                continue
            parts = []
            for m in MODELS:
                for meth in ["global", "mondrian"]:
                    r = sub[(sub.model == m) & (sub.method == meth)].iloc[0]
                    parts.append(f"{m}/{meth[0]} {pct(r.coverage)} ({pct(r.below)}/{pct(r.above)}) "
                                 f"w{r.mean_width:.3f}")
            w(f"    {g:>13}: " + " | ".join(parts))

    ev = pd.read_csv(RES / "events.csv")
    w("")
    w("[4] Where the sale lands relative to the purchase price: predicted against observed")
    for sp in SPLITS:
        w(f"  test {sp}")
        for g in ["all"] + BANDS:
            gt = "all" if g == "all" else "mband"
            sub = ev[(ev.split == sp) & (ev.group_type == gt) & (ev.group == g)].set_index("model")
            if sub.empty:
                continue
            o = sub.iloc[0]
            w(f"    {g:>13} n={int(o.n):>9,} observed: loss {pct(o.obs_loss)}, within 2% below "
              f"{pct(o.obs_loss_within_2, 2)}, exact {pct(o.obs_exact, 2)}, within 2% above "
              f"{pct(o.obs_gain_within_2, 2)}")
            for m in EVENT_MODELS:
                r = sub.loc[m]
                w(f"    {'':>13} {m:>8}: loss {pct(r.pred_loss)} (Brier {r.brier_loss:.4f}), within 2% "
                  f"below {pct(r.pred_loss_within_2, 2)} (Brier {r.brier_loss_within_2:.5f}), exact "
                  f"{pct(r.pred_exact, 2)} (Brier {r.brier_exact:.5f}), within 2% above "
                  f"{pct(r.pred_gain_within_2, 2)}")

    cur = pd.read_csv(RES / "curve_observed.csv")
    w("")
    w("[5] Observed outcomes by 1% bin of m (pairs not first sold new)")
    for per in ["all", "2008-2012", "2022-2025"]:
        sub = cur[(cur.period == per) & (cur.bin.between(-15, 14))]
        w(f"  {per}")
        for r in sub.itertuples():
            w(f"    m in [{r.bin:+d}%, {r.bin + 1:+d}%): n={int(r.n):>8,} mean r {100 * r.mean_r:+.2f} "
              f"median r {100 * r.r_p50:+.2f} p10 {100 * r.r_p10:+.1f} p90 {100 * r.r_p90:+.1f} "
              f"loss {pct(r.share_nominal_loss)} below2 {pct(r.share_within_2pct_below, 2)} "
              f"exact {pct(r.share_exact, 2)} above2 {pct(r.share_within_2pct_above, 2)} "
              f"mean hold {r.mean_hold_y:.1f}y")

    dis = pd.read_csv(RES / "discontinuity.csv")
    w("")
    w("[6] Jump and change of slope in M0's error at m = c (holding band x year effects removed)")
    for per in ["all", "2008-2012", "2022-2025"]:
        for bw in sorted(dis.bw.unique()):
            sub = dis[(dis.period == per) & (dis.bw == bw)]
            w(f"  {per}, bandwidth {bw:.2f}: " + "; ".join(
                f"c={r.cut:+.2f} jump {100 * r.jump:+.2f} ({100 * r.jump_se:.2f}) kink "
                f"{r.kink:+.3f} ({r.kink_se:.3f})" for r in sub.itertuples() if np.isfinite(r.jump)))

    text = "\n".join(L) + "\n"
    (RES / "summary.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
