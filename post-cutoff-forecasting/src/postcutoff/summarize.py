"""Turning the stored scores into the tables the article quotes.

Uncertainty comes from a circular moving-block bootstrap over forecast origins within each
window (blocks of consecutive days, so that the dependence between neighbouring origins,
whose forecast periods overlap, is kept). All models are resampled with the same origins,
so differences between models are paired.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .config import Config
from .metrics import summarise
from .origins import window_status


def weights_dates(cfg: Config) -> dict:
    """Last commit date of each foundation model's weights file, from the run manifest."""
    path = cfg.results_dir / "run_manifest.json"
    if not path.exists():
        return {}
    models = json.loads(path.read_text(encoding="utf-8")).get("models", {})
    out = {}
    for name, prov in models.items():
        date = prov.get("weights_last_commit")
        if date and name in cfg.models:
            kind = cfg.models[name].kind
            out[kind] = max(out.get(kind, date), date)
    return out


HORIZONS = {"day-ahead": [1], "week-ahead": [1, 2, 3, 4, 5, 6, 7]}


def load_scores(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.results_dir / "scores.csv")


def block_counts(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """How many times each of n ordered origins is drawn, for each replication."""
    if n == 0:
        return np.zeros((reps, 0))
    block = max(1, min(block, n))
    k = math.ceil(n / block)
    starts = rng.integers(0, n, size=(reps, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    idx = idx.reshape(reps, -1)[:, :n]
    counts = np.zeros((reps, n))
    np.add.at(counts, (np.repeat(np.arange(reps), n), idx.ravel()), 1.0)
    return counts


def per_origin(scores: pd.DataFrame, window: str, days, models, column: str) -> np.ndarray:
    """Matrix (origins, models) of a summed score column, origins in time order."""
    sub = scores[(scores["window"] == window) & (scores["lead_day"].isin(days))]
    table = sub.pivot_table(index="origin", columns="model", values=column, aggfunc="sum").sort_index()
    missing = [m for m in models if m not in table.columns]
    if missing:
        raise ValueError(f"window {window}: no scores for {missing}")
    if table[models].isna().any().any():
        raise ValueError(f"window {window}: some models were not scored on every origin")
    return table[models].to_numpy()


def interval(samples: np.ndarray) -> list[float]:
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return [float(lo), float(hi)]


class Bootstrap:
    """Bootstrap draws of per-window totals, reused for every derived quantity."""

    def __init__(self, scores, cfg: Config, models, column: str, days, block_days: int | None = None):
        self.models = list(models)
        rng = np.random.default_rng(cfg.bootstrap_seed)
        block = block_days or cfg.bootstrap_block_days
        self.point, self.draws = {}, {}
        for w in cfg.windows:
            m = per_origin(scores, w.name, days, self.models, column)
            counts = block_counts(m.shape[0], block, cfg.bootstrap_replications, rng)
            self.point[w.name] = m.sum(axis=0)
            self.draws[w.name] = counts @ m  # (reps, models)

    def totals(self, windows, draws: bool):
        src = self.draws if draws else self.point
        return sum(src[w] for w in windows)

    def ratio(self, windows, model: str, reference: str):
        i, j = self.models.index(model), self.models.index(reference)
        p = self.totals(windows, False)
        d = self.totals(windows, True)
        return float(p[i] / p[j]), d[:, i] / d[:, j]


def relative_table(boot: Bootstrap, cfg: Config, reference: str, group_sets: dict) -> list[dict]:
    rows = []
    for label, windows in group_sets.items():
        for model in boot.models:
            point, draws = boot.ratio(windows, model, reference)
            rows.append(
                {
                    "windows": label,
                    "model": model,
                    "reference": reference,
                    "ratio": point,
                    "ci95": interval(draws),
                }
            )
    return rows


def change_table(boot: Bootstrap, before, after, reference: str) -> list[dict]:
    """How each model's error relative to the reference changes from ``before`` to ``after``.

    The change is the ratio (after / before) of relative errors, minus one, in percent.
    A positive number means the model lost ground on the reference. Windows are resampled
    independently of each other.
    """
    rows = []
    for model in boot.models:
        if model == reference:
            continue
        pb, db = boot.ratio(before, model, reference)
        pa, da = boot.ratio(after, model, reference)
        rows.append(
            {
                "model": model,
                "reference": reference,
                "before_windows": list(before),
                "after_windows": list(after),
                "before": pb,
                "after": pa,
                "change_pct": 100.0 * (pa / pb - 1.0),
                "ci95": interval(100.0 * (da / db - 1.0)),
            }
        )
    return rows


def metric_table(scores: pd.DataFrame, cfg: Config, models) -> list[dict]:
    rows = []
    groups = {w.name: [w.name] for w in cfg.windows}
    groups["before (pooled)"] = list(cfg.before)
    for label, windows in groups.items():
        for horizon, days in HORIZONS.items():
            for model in models:
                sub = scores[
                    (scores["window"].isin(windows)) & (scores["lead_day"].isin(days)) & (scores["model"] == model)
                ]
                if sub.empty:
                    continue
                rows.append({"windows": label, "horizon": horizon, "model": model, **summarise(sub)})
    return rows


def lead_day_table(scores: pd.DataFrame, cfg: Config, models) -> list[dict]:
    rows = []
    for w in cfg.windows:
        for model in models:
            for day in range(1, cfg.horizon // 24 + 1):
                sub = scores[(scores["window"] == w.name) & (scores["model"] == model) & (scores["lead_day"] == day)]
                if sub.empty:
                    continue
                rows.append(
                    {
                        "window": w.name,
                        "model": model,
                        "lead_day": day,
                        "mae": float(sub["sum_abs_err"].sum() / sub["n_hours"].sum()),
                    }
                )
    return rows


def holiday_table(scores: pd.DataFrame, cfg: Config, models, reference: str) -> list[dict]:
    """Error on lead days that are mostly a public holiday, against days with no holiday hour."""
    rows = []
    groups = {"before (pooled)": list(cfg.before), "after": list(cfg.after)}
    for label, windows in groups.items():
        sub = scores[scores["window"].isin(windows)]
        # Lead days are UTC days; a Lisbon holiday in summer time starts at 23:00 UTC the day
        # before, so a lead day counts as a holiday only when most of its hours are.
        holiday = sub["n_holiday_hours"] >= 12
        for kind, mask in (("holiday", holiday), ("other", sub["n_holiday_hours"] == 0)):
            part = sub[mask]
            ref = part[part["model"] == reference]
            ref_mae = ref["sum_abs_err"].sum() / ref["n_hours"].sum() if len(ref) else float("nan")
            for model in models:
                p = part[part["model"] == model]
                if p.empty:
                    continue
                mae = p["sum_abs_err"].sum() / p["n_hours"].sum()
                rows.append(
                    {
                        "windows": label,
                        "days": kind,
                        "model": model,
                        "lead_days": int(len(p)),
                        "mae": float(mae),
                        "relative_to_reference": float(mae / ref_mae),
                    }
                )
    return rows


def summarize(cfg: Config) -> dict:
    scores = load_scores(cfg)
    present = list(dict.fromkeys(scores["model"]))
    models = [m for m in cfg.models if m in present]
    reference = cfg.reference_model
    classical = [m for m in models if cfg.models[m].kind == "statsforecast"]
    foundation = [m for m in models if cfg.models[m].is_foundation]
    best_classical = "mstl_ets" if "mstl_ets" in models else None

    group_sets = {w.name: [w.name] for w in cfg.windows}
    group_sets["before (pooled)"] = list(cfg.before)
    group_sets["after (pooled)"] = list(cfg.after)

    out = {
        "reference_model": reference,
        "models": {m: cfg.models[m].label for m in models},
        "classical": classical,
        "foundation": foundation,
        "windows": [w.name for w in cfg.windows],
        "before": list(cfg.before),
        "after": list(cfg.after),
        "window_status": window_status(cfg, weights_dates(cfg)).to_dict("records"),
        "bootstrap": {
            "replications": cfg.bootstrap_replications,
            "block_days": cfg.bootstrap_block_days,
            "seed": cfg.bootstrap_seed,
        },
        "metrics": metric_table(scores, cfg, models),
        "lead_days": lead_day_table(scores, cfg, models),
        "holidays": holiday_table(scores, cfg, models, reference),
        "relative_mae": {},
        "relative_wql": {},
        "change": {},
        "block_sensitivity": {},
    }
    for horizon, days in HORIZONS.items():
        boot_mae = Bootstrap(scores, cfg, models, "sum_abs_err", days)
        boot_wql = Bootstrap(scores, cfg, models, "sum_wql_num", days)
        out["relative_mae"][horizon] = {"vs_reference": relative_table(boot_mae, cfg, reference, group_sets)}
        out["relative_wql"][horizon] = {"vs_reference": relative_table(boot_wql, cfg, reference, group_sets)}
        main, placebo = (cfg.before, cfg.after), (cfg.placebo_before, cfg.placebo_after)
        out["change"][horizon] = {"mae_vs_reference": change_table(boot_mae, *main, reference)}
        if best_classical:
            out["relative_mae"][horizon]["vs_" + best_classical] = relative_table(
                boot_mae, cfg, best_classical, group_sets
            )
            out["change"][horizon]["mae_vs_" + best_classical] = change_table(boot_mae, *main, best_classical)
        out["change"][horizon]["wql_vs_reference"] = change_table(boot_wql, *main, reference)
        if placebo[0] and placebo[1]:
            # Two windows that every model could have seen: the year-to-year change that
            # owes nothing to a release.
            out["change"][horizon]["placebo_mae_vs_reference"] = change_table(boot_mae, *placebo, reference)
            if best_classical:
                out["change"][horizon]["placebo_mae_vs_" + best_classical] = change_table(
                    boot_mae, *placebo, best_classical
                )
        out["block_sensitivity"][horizon] = []
        for block in (7, 14, 28, 56):
            b = Bootstrap(scores, cfg, models, "sum_abs_err", days, block_days=block)
            for r in change_table(b, *main, reference):
                out["block_sensitivity"][horizon].append(
                    {"model": r["model"], "block_days": block, "change_pct": r["change_pct"], "ci95": r["ci95"]}
                )

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (cfg.results_dir / "RESULTS.md").write_text(render_markdown(out, cfg), encoding="utf-8")
    pd.DataFrame(out["metrics"]).to_csv(
        cfg.results_dir / "metrics.csv", index=False, float_format="%.6g", lineterminator="\n"
    )
    return out


# --------------------------------------------------------------------------- report


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def render_markdown(out: dict, cfg: Config) -> str:
    label = out["models"]
    lines = [
        "# Results",
        "",
        "Generated by `postcutoff summarize` from `results/scores.csv`. Do not edit by hand.",
        "",
        f"Reference model: {label[out['reference_model']]}. Windows the foundation models could have seen: "
        f"{', '.join(out['before'])}. Window after every release: {', '.join(out['after'])}.",
        f"Intervals are 95% percentile intervals from a circular block bootstrap over origins "
        f"({out['bootstrap']['replications']} replications, blocks of {out['bootstrap']['block_days']} days).",
        "",
        "## Window status by model",
        "",
        "| Window | " + " | ".join(sorted({r["model"] for r in out["window_status"]})) + " |",
        "|---|" + "---|" * len({r["model"] for r in out["window_status"]}),
    ]
    status = pd.DataFrame(out["window_status"]).pivot(index="window", columns="model", values="status")
    for w in out["windows"]:
        lines.append(f"| {w} | " + " | ".join(status.loc[w, c] for c in sorted(status.columns)) + " |")

    for horizon in HORIZONS:
        lines += ["", f"## {horizon.capitalize()}: MAE relative to {label[out['reference_model']]}", ""]
        rel = pd.DataFrame(out["relative_mae"][horizon]["vs_reference"])
        groups = list(dict.fromkeys(rel["windows"]))
        lines.append("| Model | " + " | ".join(groups) + " |")
        lines.append("|---|" + "---|" * len(groups))
        for model in out["models"]:
            cells = []
            for g in groups:
                r = rel[(rel["model"] == model) & (rel["windows"] == g)].iloc[0]
                cells.append(f"{r['ratio']:.3f} {_fmt_ci(r['ci95'])}")
            lines.append(f"| {label[model]} | " + " | ".join(cells) + " |")

        lines += ["", f"### {horizon.capitalize()}: change between windows", ""]
        for key, rows in out["change"][horizon].items():
            metric = "WQL" if key.startswith("wql") else "MAE"
            ref = rows[0]["reference"] if rows else ""
            kind = "Placebo (two windows before every release)" if key.startswith("placebo") else "Main contrast"
            if rows:
                lines += [
                    f"{kind}: {metric} relative to {label[ref]}, "
                    f"{' + '.join(rows[0]['before_windows'])} to {' + '.join(rows[0]['after_windows'])}.",
                    "",
                ]
            lines.append("| Model | Before | After | Change (%) | 95% interval |")
            lines.append("|---|---|---|---|---|")
            for r in rows:
                lines.append(
                    f"| {label[r['model']]} | {r['before']:.3f} | {r['after']:.3f} | {r['change_pct']:+.1f} | "
                    f"[{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}] |"
                )
            lines.append("")

        lines += [
            f"### {horizon.capitalize()}: interval for the main contrast by bootstrap block length",
            "",
            "| Model | " + " | ".join(f"{b} days" for b in (7, 14, 28, 56)) + " |",
            "|---|---|---|---|---|",
        ]
        sens = pd.DataFrame(out["block_sensitivity"][horizon])
        for model in dict.fromkeys(sens["model"]):
            cells = [
                f"[{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]"
                for _, r in sens[sens["model"] == model].sort_values("block_days").iterrows()
            ]
            lines.append(f"| {label[model]} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Accuracy by window", ""]
    m = pd.DataFrame(out["metrics"])
    for horizon in HORIZONS:
        lines += [f"### {horizon.capitalize()}", ""]
        lines.append("| Window | Model | Origins | MAE (MW) | MAPE (%) | MASE | WQL | 80% coverage | Bias (MW) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in m[m["horizon"] == horizon].iterrows():
            lines.append(
                f"| {r['windows']} | {label[r['model']]} | {r['origins']} | {r['mae']:.1f} | {r['mape']:.2f} | "
                f"{r['mase']:.3f} | {r['wql']:.4f} | {r['coverage80']:.3f} | {r['bias']:+.1f} |"
            )
        lines.append("")

    lines += [
        "## Holidays",
        "",
        "Lead days (UTC) with at least 12 hours of a national public holiday, against lead days with none.",
        "",
    ]
    lines.append("| Windows | Days | Model | Lead days | MAE (MW) | Relative to reference |")
    lines.append("|---|---|---|---|---|---|")
    for r in out["holidays"]:
        lines.append(
            f"| {r['windows']} | {r['days']} | {label[r['model']]} | {r['lead_days']} | {r['mae']:.1f} | "
            f"{r['relative_to_reference']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)
