"""Regime split and forward-in-time test of the p95 rule (PLAN.md).

    python evaluate.py --ratios ../second-count-prediction/results/corrected/ratios.csv --out results

Input: the monthly log(delivered / outside count) of the eight pairs of the second-count-prediction study, with
the Citi Bike files as counted with the corrected rule. Writes results/summary.txt, results/pairs.json,
results/windows.csv and results/curves.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from feed import detect as D
from feed import regimes as R

MOTIVATING = "citibike / DOT table"  # the pair the conjecture was read from: reported, not a test
DROP = {("TLC fhv / monthly report", "2019-01")}  # that FHV file still holds the high-volume trips (Part 4)


def load(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row["pair"], row["month"]) not in DROP:
                out.setdefault(row["pair"], {})[row["month"]] = float(row["log_ratio"])
    return out


def pct(x: float | None) -> str:
    return "not reached by 50%" if x is None else f"{100 * x:.2f}%"


def analyse(r: dict[str, float]) -> dict:
    full_pred = R.rule(R.deviations(r))
    sp = R.split(r)
    seg = R.segment(r, sp["month"]) if sp["month"] else r
    seg_pred = R.rule(R.deviations(seg))
    cur = D.curve(seg)
    obs = D.observed_l90(cur)
    return {
        "months": len(r),
        "split": sp,
        "regime_change": sp["reduction"] >= R.REGIME,
        "segment_evaluated": len(D.scores(seg)),
        "pred_full": full_pred,
        "pred_segment": seg_pred,
        "observed_segment": obs,
        "within_segment_rule": D.within_factor_two(seg_pred, obs),
        "within_full_rule": D.within_factor_two(full_pred, obs),
        "err_segment_rule": D.log_error(seg_pred, obs),
        "err_full_rule": D.log_error(full_pred, obs),
        "curve": cur,
        "windows": R.forward_windows(r),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ratios", type=Path, default=Path("../second-count-prediction/results/corrected/ratios.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    series = load(a.ratios)
    res = {name: analyse(r) for name, r in series.items()}
    tests = [n for n in res if n != MOTIVATING]

    L = ["Regime split and forward-in-time test of the p95 rule. Settings and hypotheses: PLAN.md.", ""]
    for name, x in res.items():
        sp = x["split"]
        tag = "motivating pair, reported" if name == MOTIVATING else "test"
        L.append(f"{name}  [{tag}]")
        L.append(
            f"  {x['months']} months; split at {sp['month']} ({sp['left']} before, {sp['right']} from), "
            f"absolute deviation reduced by {100 * sp['reduction']:.1f}%; regime change: "
            f"{'yes' if x['regime_change'] else 'no'}"
        )
        L.append(
            f"  segment medians of r: {100 * math.expm1(sp['median_left']):+.2f}% before, "
            f"{100 * math.expm1(sp['median_right']):+.2f}% from the split; evaluated months in the later segment "
            f"{x['segment_evaluated']}"
        )
        L.append(
            f"  later segment: segment rule {pct(x['pred_segment'])}, full-history rule {pct(x['pred_full'])}, "
            f"observed {pct(x['observed_segment'])}"
        )
        L.append(
            f"  within a factor of two: segment rule {'yes' if x['within_segment_rule'] else 'no'}, full-history "
            f"rule {'yes' if x['within_full_rule'] else 'no'}; |log(obs/pred)|: segment {x['err_segment_rule']:.3f}, "
            f"full {x['err_full_rule']:.3f}"
        )
        for w in x["windows"]:
            L.append(
                f"    window {w['start']}: trailing {pct(w['pred_trailing'])}, expanding {pct(w['pred_expanding'])}, "
                f"observed over the next {w['n_ahead']} months {pct(w['observed'])}"
            )
        L.append("")

    h1 = all(res[n]["within_segment_rule"] for n in tests)
    L.append(
        f"H1 (segment rule within a factor of two on the later segment, all {len(tests)} test pairs): "
        f"{'supported' if h1 else 'not supported'} ({sum(res[n]['within_segment_rule'] for n in tests)} of "
        f"{len(tests)})"
    )
    reg = [n for n in tests if res[n]["regime_change"]]
    if reg:
        h2 = all(res[n]["err_segment_rule"] < res[n]["err_full_rule"] for n in reg)
        better = sum(res[n]["err_segment_rule"] < res[n]["err_full_rule"] for n in reg)
        L.append(
            f"H2 (segment rule closer than the full-history rule in every test pair with a regime change): "
            f"{'supported' if h2 else 'not supported'} ({better} of {len(reg)}: {', '.join(reg)})"
        )
    else:
        L.append("H2: no test pair has a regime change by the registered criterion; not testable")
    wins = [(n, w) for n in tests for w in res[n]["windows"]]
    within = [D.within_factor_two(w["pred_trailing"], w["observed"]) for _, w in wins]
    within_exp = [D.within_factor_two(w["pred_expanding"], w["observed"]) for _, w in wins]
    errs = [
        (D.log_error(w["pred_trailing"], w["observed"]), D.log_error(w["pred_expanding"], w["observed"]))
        for _, w in wins
    ]
    differ = [(t, e) for t, e in errs if abs(t - e) > 1e-12]
    closer = sum(t < e for t, e in differ)
    L.append(
        f"H3a (trailing rule within a factor of two in at least two thirds of the {len(wins)} forward windows of the "
        f"test pairs): {'supported' if within and sum(within) >= 2 * len(within) / 3 else 'not supported'} "
        f"({sum(within)} of {len(wins)}; expanding rule {sum(within_exp)} of {len(wins)})"
    )
    L.append(
        f"H3b (trailing rule closer than the expanding rule in more than half of the windows where they differ): "
        f"{'supported' if differ and closer > len(differ) / 2 else 'not supported'} ({closer} of {len(differ)})"
    )
    text = "\n".join(L) + "\n"
    (a.out / "summary.txt").write_text(text, encoding="utf-8")
    pairs = {n: {k: v for k, v in x.items() if k != "curve"} | {"test": n != MOTIVATING} for n, x in res.items()}
    (a.out / "pairs.json").write_text(json.dumps(pairs, indent=1), encoding="utf-8")
    with open(a.out / "windows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["pair", "start", "n_trail", "n_expanding", "n_ahead", "pred_trailing", "pred_expanding", "observed"]
        )
        for n, x in res.items():
            for v in x["windows"]:
                w.writerow(
                    [
                        n,
                        v["start"],
                        v["n_trail"],
                        v["n_expanding"],
                        v["n_ahead"],
                        v["pred_trailing"],
                        v["pred_expanding"],
                        "" if v["observed"] is None else v["observed"],
                    ]
                )
    with open(a.out / "curves.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "loss", "detected_later_segment"])
        for n, x in res.items():
            for p, v in x["curve"].items():
                w.writerow([n, p, f"{v:.4f}"])
    print(text)


if __name__ == "__main__":
    main()
