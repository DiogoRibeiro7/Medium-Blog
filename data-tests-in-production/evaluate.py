"""Run every check on the real deliveries and on the faulted ones; write the results.

    python evaluate.py --profiles profiles --out results

Settings fixed before the real profiles were computed (see README): change checks compare a
delivery with the trailing ``WINDOW`` = 12 deliveries and fire at a robust |z| above ``K`` = 5;
tolerant table checks allow 1% of rows in violation; the volume floor is 1,000,000 rows.

Outputs
  results/real_alarms.csv       every check on every real delivery from 2020-01 (fired, value, detail)
  results/fault_detection.csv   every check on every faulted delivery, and whether the same check
                                fired on the clean delivery of that month
  results/k_sweep.csv           change checks at other thresholds: alarms on real deliveries,
                                detection of the silent faults
  results/summary.txt           the numbers quoted in the article
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feed.bundle import read
from feed.checks import CHANGE, change_scores, run_table
from feed.faults import FAULTS

WINDOW, K = 12, 5.0
FIRST_EVAL = "2020-01"
K_GRID = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)


def load(folder: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


def history(real: dict[str, dict], month: str) -> list[dict]:
    ms = sorted(m for m in real if m < month)
    return [real[m] for m in ms[-WINDOW:]]


def rows_for(p, hist, family_tag: str) -> list[dict]:
    out = []
    for name, r in run_table(p).items():
        out.append(dict(family="table", check=name, fired=r.fired, value=r.value, detail=r.detail))
    scores = change_scores(p, hist)
    for name in CHANGE:
        z, detail = scores.get(name, (0.0, ""))
        out.append(dict(family="change", check=name, fired=z > K, value=z, detail=detail))
    for r in out:
        r["source"] = family_tag
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="profiles")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root, out = Path(a.profiles), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (root / "real.jsonl.gz").exists():
        real, faults = read(root)
    else:
        real = load(root / "real")
        faults = {f: load(root / "fault" / f) for f in FAULTS if (root / "fault" / f).exists()}

    # real deliveries
    real_rows = []
    for m in sorted(real):
        if m < FIRST_EVAL:
            continue
        for r in rows_for(real[m], history(real, m), "real"):
            real_rows.append(dict(month=m, **r))
    R = pd.DataFrame(real_rows)
    R.to_csv(out / "real_alarms.csv", index=False, float_format="%.6g")

    # faulted deliveries
    clean_fired = {(r["month"], r["check"]): r["fired"] for r in real_rows}
    clean_value = {(r["month"], r["check"]): r["value"] for r in real_rows}
    fault_rows = []
    for f, profs in faults.items():
        for m, p in sorted(profs.items()):
            for r in rows_for(p, history(real, m), f):
                fault_rows.append(dict(fault=f, kind=FAULTS[f].kind, month=m, clean_fired=clean_fired[(m, r["check"])],
                                       clean_value=clean_value[(m, r["check"])], **r))
    F = pd.DataFrame(fault_rows)
    F["detected"] = F.fired & ~F.clean_fired
    F.to_csv(out / "fault_detection.csv.gz", index=False, float_format="%.6g",
             compression={"method": "gzip", "mtime": 0})

    # thresholds: change checks at other k
    sweep = []
    Rc, Fc = R[R.family == "change"], F[F.family == "change"]
    for k in K_GRID:
        fired_real = Rc.assign(f=Rc.value > k).groupby("month").f.any()
        rate_check = (Rc.value > k).groupby(Rc.check).mean()
        x = Fc.assign(f=(Fc.value > k), c=(Fc.clean_value > k))
        x["det"] = x.f & ~x.c
        by_case = x.groupby(["fault", "kind", "month"]).det.any().reset_index()
        silent = by_case[by_case.kind == "silent"]
        sweep.append(dict(k=k, real_months_with_alarm=float(fired_real.mean()), mean_alarms_per_month=float(
            (Rc.value > k).groupby(Rc.month).sum().mean()), worst_check_rate=float(rate_check.max()),
            silent_detected=float(silent.det.mean()), all_faults_detected=float(by_case.det.mean())))
    pd.DataFrame(sweep).to_csv(out / "k_sweep.csv", index=False, float_format="%.4f")

    # summary
    L = [f"Yellow-taxi monthly files: {len(real)} real deliveries profiled ({min(real)} to {max(real)}); "
         f"checks evaluated from {FIRST_EVAL} ({R.month.nunique()} deliveries); "
         f"{sum(len(v) for v in faults.values())} faulted deliveries ({len(faults)} faults).",
         f"Change checks: trailing {WINDOW} deliveries, robust |z| > {K:g}. Tolerant table checks: 1% of rows. "
         "Volume floor: 1,000,000 rows.", ""]
    L.append("=== Real deliveries: share of deliveries on which each check fired ===")
    for fam in ("table", "change"):
        x = R[R.family == fam].groupby("check").fired.mean().sort_values(ascending=False)
        L.append(f"[{fam}] " + "; ".join(f"{c} {100 * v:.0f}%" for c, v in x.items()))
    per_month = R.groupby(["month", "family"]).fired.sum().unstack()
    L.append(f"Mean alarms per delivery: table {per_month['table'].mean():.1f}, "
             f"change {per_month['change'].mean():.1f}; "
             f"deliveries with no table alarm: {int((per_month['table'] == 0).sum())}, with no change alarm: "
             f"{int((per_month['change'] == 0).sum())} of {len(per_month)}")
    L.append("")
    L.append("=== Real deliveries: every change alarm (month: check, |z|, metric) ===")
    for m, x in R[(R.family == "change") & R.fired].groupby("month"):
        L.append(f"  {m}: " + "; ".join(f"{r.check} {r.value:.1f} ({r.detail})" for r in x.itertuples()))
    L.append("")
    L.append("=== Faults: share of faulted deliveries detected (fired on the fault and not on the clean month) ===")
    det = F.groupby(["fault", "family"]).apply(lambda d: d.groupby("month").detected.any().mean(),
                                               include_groups=False).unstack()
    for f in FAULTS:
        if f in det.index:
            best = (F[(F.fault == f) & F.detected].groupby("check").month.nunique().sort_values(ascending=False))
            top = ", ".join(f"{c} {n}" for c, n in best.head(4).items())
            L.append(f"  {f:>16} ({FAULTS[f].kind}): table {100 * det.loc[f, 'table']:.0f}%, change "
                     f"{100 * det.loc[f, 'change']:.0f}%  | checks that caught it (months): {top}")
    n_months = F.month.nunique()
    L.append(f"  ({n_months} base deliveries per fault)")
    for kind in ("silent", "loud", "schema"):
        x = F[F.kind == kind]
        t = x[x.family == "table"].groupby(["fault", "month"]).detected.any().mean()
        c = x[x.family == "change"].groupby(["fault", "month"]).detected.any().mean()
        L.append(f"  all {kind} faults: table {100 * t:.0f}%, change {100 * c:.0f}%")
    L.append("")
    L.append("=== Change checks at other thresholds (real deliveries / faulted deliveries) ===")
    for s in sweep:
        L.append(f"  k {s['k']:>4g}: deliveries with an alarm {100 * s['real_months_with_alarm']:.0f}%, mean alarms "
                 f"{s['mean_alarms_per_month']:.2f}, worst check {100 * s['worst_check_rate']:.0f}%; silent faults "
                 f"detected {100 * s['silent_detected']:.0f}%, all faults {100 * s['all_faults_detected']:.0f}%")
    (out / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
