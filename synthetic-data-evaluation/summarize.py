"""Write results/summary.txt: every number the article quotes, in one place."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import theory

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def _f(x, d=3):
    return f"{x:.{d}f}"


def axes_section(lines: list[str]) -> None:
    a = json.loads((RES / "axes.json").read_text())
    lines.append("== Part 1: one synthetic copy of the Adult training third ==")
    lines.append(f"train rows {a['n_train']}, holdout rows {a['n_holdout']}")
    lines.append(f"TRTR (real train -> real holdout): AUC {_f(a['trtr']['auc'])}, Brier {_f(a['trtr']['brier'], 4)}")
    tc = a["train_consistency"]
    lines.append(
        "real train rules: education mismatch {}, female husbands {}, male wives {}, out of range {}".format(
            _f(tc["education_mismatch"], 4), _f(tc["female_husband"], 4), _f(tc["male_wife"], 4), _f(tc["out_of_range"], 4)
        )
    )
    lines.append("real train point masses: " + ", ".join(f"{k} {v:.4f}" for k, v in a["train_point_masses"].items()))
    terms = a["terms"]
    lines.append("real-train coefficients: " + ", ".join(f"{t} {b:.4f} (se {s:.4f})" for t, b, s in zip(terms, a["real_coef"], a["real_se"])))
    lines.append("")
    for name, g in a["generators"].items():
        lines.append(f"-- {name}")
        lines.append(
            f"   fidelity: mean marginal distance {_f(g['marginal_mean'], 4)}, "
            f"max {_f(max(g['marginal'].values()), 4)} ({max(g['marginal'], key=g['marginal'].get)}), "
            f"mean pairwise TVD {_f(g['pairwise_tvd_mean'], 4)}"
        )
        lines.append(
            f"   discriminator AUC {_f(g['discriminator_auc'])} (pair-aware folds {_f(g['discriminator_auc_paired'])}), "
            f"pMSE {_f(g['pmse'], 4)} (pair-aware {_f(g['pmse_paired'], 4)})"
        )
        c = g["consistency"]
        lines.append(
            f"   rules: education mismatch {_f(c['education_mismatch'], 4)}, female husbands {_f(c['female_husband'], 4)}, "
            f"male wives {_f(c['male_wife'], 4)}, out of range {_f(c['out_of_range'], 4)}"
        )
        lines.append("   point masses: " + ", ".join(f"{k} {v:.4f}" for k, v in g["point_masses"].items()))
        lines.append(f"   utility: TSTR AUC {_f(g['tstr']['auc'])}, Brier {_f(g['tstr']['brier'], 4)}")
        lines.append(
            "   coefficients (difference in real-data SEs): "
            + ", ".join(f"{t} {z:+.1f}" for t, z in zip(terms[1:], g["coef_diff_in_real_se"][1:]))
        )
        lines.append(
            "   coefficients (synthetic estimate): "
            + ", ".join(f"{t} {b:.4f}" for t, b in zip(terms[1:], g["coef"][1:]))
        )
        lines.append(
            "   CI overlap with real: "
            + ", ".join(f"{t} {o:.2f}" for t, o in zip(terms[1:], g["ci_overlap"][1:]))
            + f" (mean {np.mean(g['ci_overlap'][1:]):.2f})"
        )
        lines.append(
            f"   privacy: exact copies {_f(g['exact_copies'], 4)}, median DCR {_f(g['median_dcr'])}, "
            f"DCR share {_f(g['dcr_share'])}, MIA AUC {_f(g['mia_auc'])}, MIA TPR at 1% FPR {_f(g['mia_tpr_at_1pct'])}"
        )
    lines.append("")


def mechanism_section(lines: list[str]) -> None:
    d = json.loads((RES / "coverage_mechanism.json").read_text())
    terms = d["terms"]
    lines.append(f"== Part 2a: known mechanism, n = {d['n']}, {d['reps']} replications ==")
    lines.append("coverage per term: " + ", ".join(terms[1:]) + "  [Monte Carlo SE about 0.005 near 0.95]")
    rows = {tuple(r["key"]): r for r in d["rows"]}

    def cov(r):
        c = r["coverage"][1:]
        return " ".join(f"{x:.3f}" for x in c) + f"   mean {np.mean(c):.3f}"

    lines.append(f"real data: {cov(rows[('real',)])}")
    lines.append("")
    lines.append("-- size study: one synthetic data set of k*n rows analysed as real")
    for gen in ("parametric", "cart", "copula", "smote", "independent"):
        for mode in ("simple", "proper"):
            for k in (1, 2, 5, 10, 20):
                r = rows[("size", gen, mode, k)]
                extra = ""
                if gen == "parametric":
                    extra = f"   theory {theory.size_coverage(k, mode == 'proper'):.3f}"
                lines.append(f"{gen:11s} {mode:6s} k={k:2d}: {cov(r)}{extra}")
    lines.append("")
    lines.append("-- rules: m synthetic data sets of n rows")
    for gen in ("parametric", "cart", "copula", "smote", "independent"):
        for mode in ("simple", "proper"):
            for rule in ("naive", "simple", "proper", "raghunathan"):
                for m in (1, 2, 5, 10, 20):
                    key = ("rules", gen, mode, rule, m)
                    if key not in rows:
                        continue
                    r = rows[key]
                    extra = ""
                    if gen == "parametric" and rule != "raghunathan":
                        extra = f"   theory {theory.coverage(rule, 1.0, m, mode == 'proper'):.3f}"
                    if rule == "raghunathan":
                        lens = np.array(r["mean_length"][1:]) / np.array(rows[("real",)]["mean_length"][1:])
                        extra += f"   fallback {np.mean(r['fallback_rate'][1:]):.3f}, mean length / real {np.round(lens, 2).tolist()}"
                    lines.append(f"{gen:11s} {mode:6s} {rule:11s} m={m:2d}: {cov(r)}{extra}")
        bias = np.array(rows[("rules", gen, "simple", "naive", 20)]["bias"][1:])
        real_sd = np.array(rows[("real",)]["sd"][1:])
        lines.append(
            f"{gen:11s} bias of the m=20 average (simple): "
            + " ".join(f"{b:+.3f}" for b in bias)
            + "   in real-data SDs: "
            + " ".join(f"{b:+.2f}" for b in bias / real_sd)
        )
    lines.append("")


def adult_section(lines: list[str]) -> None:
    d = json.loads((RES / "coverage_adult.json").read_text())
    terms = d["terms"]
    lines.append(
        f"== Part 2b: Adult as population ({d['population']} rows), n = {d['n']}, m = {d['m']}, {d['reps']} replications =="
    )
    lines.append("population coefficients: " + ", ".join(f"{t} {b:.4f}" for t, b in zip(terms, d["truth"])))
    lines.append("coverage per term: " + ", ".join(terms[1:]) + "  [Monte Carlo SE about 0.007 near 0.95]")
    rows = {tuple(r["key"]): r for r in d["rows"]}
    real = rows[("real",)]
    lines.append("real data: " + " ".join(f"{c:.3f}" for c in real["coverage"][1:]))
    for gen in ("independent", "copula", "cart", "smote"):
        for mode in ("simple", "proper"):
            for rule, m in (("naive", 1), ("naive", d["m"]), ("simple", d["m"]), ("proper", d["m"]), ("raghunathan", d["m"])):
                r = rows[(gen, mode, rule, m)]
                lens = np.array(r["mean_length"][1:]) / np.array(real["mean_length"][1:])
                extra = f"   length/real {np.round(lens, 2).tolist()}"
                if rule == "raghunathan":
                    extra += f"   fallback {np.mean(r['fallback_rate'][1:]):.3f}"
                lines.append(
                    f"{gen:11s} {mode:6s} {rule:11s} m={m}: "
                    + " ".join(f"{c:.3f}" for c in r["coverage"][1:])
                    + f"   mean {np.mean(r['coverage'][1:]):.3f}{extra}"
                )
            r = rows[(gen, mode, "naive", d["m"])]
            lines.append(
                f"{gen:11s} {mode:6s} bias of the average, in real-data SDs: "
                + " ".join(f"{b:+.2f}" for b in r["bias_in_real_sd"][1:])
            )
    lines.append("")


def main() -> None:
    lines: list[str] = []
    axes_section(lines)
    mechanism_section(lines)
    adult_section(lines)
    (RES / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
