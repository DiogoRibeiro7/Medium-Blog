"""The registered test (PLAN.md): 4 forecasts x 5 shift regimes on 52 test weeks of NYC 311 calls.

python evaluate.py --sources sources --out results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from feed import erlang as E
from feed import forecast as F
from feed import schedule as S
from feed.calls import Calls, evaluation_weeks, excluded_weeks, load, structure

AHTS = (300.0, 180.0, 480.0)  # seconds; the first is the main analysis
KS = (1, 3)
BOOT, BOOT_SEED = 2000, 11
MEASURES = ("excess", "shortfall", "need", "paid", "calls", "answered")


def weeks_used(c: Calls) -> list:
    ex = set(excluded_weeks(c))
    return [w for w in evaluation_weeks() if w not in ex]


def forecasts_for(c: Calls, weeks: list, settings: dict | None = None) -> dict[str, list[np.ndarray]]:
    model, hol = F.fit_gbm(c, evaluation_weeks()[0] - np.timedelta64(7, "D"), settings)
    out = {name: [] for name in F.FORECASTS}
    for w in weeks:
        for name in ("two_weeks", "mean4", "oracle"):
            out[name].append(F.simple(c, w, name))
        out["gbm"].append(F.gbm_forecast(model, hol, c, w))
    return out


def run(c: Calls, ahts=AHTS, settings: dict | None = None, boot: int = BOOT) -> dict:
    weeks = weeks_used(c)
    fc = forecasts_for(c, weeks, settings)
    tot: dict[tuple, dict[str, np.ndarray]] = {}
    example = {}
    calls_week = np.array([x.sum() for x in fc["oracle"]])
    median_week = int(np.argsort(calls_week)[len(calls_week) // 2])
    for aht in ahts:
        need_actual = [E.requirement(x, aht) for x in fc["oracle"]]
        for name in F.FORECASTS:
            for regime in S.REGIMES:
                m = {k: np.zeros(len(weeks)) for k in MEASURES}
                for i, (f, act, need_a) in enumerate(zip(fc[name], fc["oracle"], need_actual, strict=True)):
                    need_f = E.requirement(f, aht)
                    on, paid = S.staffing(regime, need_f)
                    m["excess"][i] = np.maximum(on - need_a, 0).sum()
                    m["shortfall"][i] = np.maximum(need_a - on, 0).sum()
                    m["need"][i] = need_a.sum()
                    m["paid"][i] = paid
                    m["calls"][i] = act.sum()
                    m["answered"][i] = (act * E.service_level(act, on, aht)).sum()
                    if aht == ahts[0] and i == median_week and name in ("oracle", "gbm"):
                        example[(name, regime)] = on
                tot[(aht, name, regime)] = m
    shadow = S.shadow_prices("R4", E.requirement(fc["oracle"][median_week], ahts[0]))
    rng = np.random.default_rng(BOOT_SEED)
    W = rng.multinomial(len(weeks), np.full(len(weeks), 1 / len(weeks)), size=boot).astype(np.float64)
    return {
        "weeks": weeks,
        "fc": fc,
        "tot": tot,
        "W": W,
        "median_week": median_week,
        "example": example,
        "shadow": shadow,
        "ahts": ahts,
    }


def cost(res: dict, aht: float, name: str, regime: str, k: float, boot: bool = False) -> np.ndarray | float:
    m = res["tot"][(aht, name, regime)]
    if boot:
        W = res["W"]
        return (W @ m["excess"] + k * (W @ m["shortfall"])) / (W @ m["need"])
    return float((m["excess"].sum() + k * m["shortfall"].sum()) / m["need"].sum())


def constraint_cost(res, aht, regime, k, boot=False):
    return cost(res, aht, "oracle", regime, k, boot) - cost(res, aht, "oracle", "R0", k, boot)


def forecast_cost(res, aht, name, regime, k, boot=False):
    return cost(res, aht, name, regime, k, boot) - cost(res, aht, "oracle", regime, k, boot)


def comparisons(res: dict, aht: float) -> list[dict]:
    out = []

    def add(h: str, label: str, k: int, fn) -> None:
        est = float(fn(False))
        lo, hi = (float(x) for x in np.percentile(fn(True), [2.5, 97.5]))
        out.append(
            {
                "aht": aht,
                "hypothesis": h,
                "comparison": label,
                "k": k,
                "estimate": est,
                "lo": lo,
                "hi": hi,
                "established": lo > 0,
            }
        )

    for k in KS:
        for regime in ("R3", "R4"):
            for name in ("mean4", "gbm"):
                add(
                    "H1",
                    f"{regime}: constraint cost - forecast cost of {name}",
                    k,
                    lambda b, regime=regime, name=name, k=k: (
                        constraint_cost(res, aht, regime, k, b) - forecast_cost(res, aht, name, regime, k, b)
                    ),
                )
        add(
            "H2",
            "two_weeks -> gbm saving under R0 - the same under R4",
            k,
            lambda b, k=k: (
                (cost(res, aht, "two_weeks", "R0", k, b) - cost(res, aht, "gbm", "R0", k, b))
                - (cost(res, aht, "two_weeks", "R4", k, b) - cost(res, aht, "gbm", "R4", k, b))
            ),
        )
        for regime in ("R1", "R2"):
            add(
                "H3",
                f"{regime}: forecast cost of gbm - constraint cost",
                k,
                lambda b, regime=regime, k=k: (
                    forecast_cost(res, aht, "gbm", regime, k, b) - constraint_cost(res, aht, regime, k, b)
                ),
            )
    return out


def verdicts(comp: list[dict]) -> dict:
    out = {}
    for aht in sorted({x["aht"] for x in comp}):
        for h in ("H1", "H2", "H3"):
            sel = [x for x in comp if x["aht"] == aht and x["hypothesis"] == h]
            out[f"{h}@{aht:g}"] = {
                "n": len(sel),
                "established": sum(x["established"] for x in sel),
                "positive": sum(x["estimate"] > 0 for x in sel),
                "supported": all(x["established"] for x in sel),
            }
    return out


def cells(res: dict) -> list[dict]:
    rows = []
    for (aht, name, regime), m in res["tot"].items():
        for k in KS:
            b = cost(res, aht, name, regime, k, True)
            lo, hi = np.percentile(b, [2.5, 97.5])
            rows.append(
                {
                    "aht": aht,
                    "forecast": name,
                    "regime": regime,
                    "k": k,
                    "cost": cost(res, aht, name, regime, k),
                    "lo": float(lo),
                    "hi": float(hi),
                    "constraint_cost": constraint_cost(res, aht, regime, k),
                    "forecast_cost": forecast_cost(res, aht, name, regime, k),
                    "excess_share": float(m["excess"].sum() / m["need"].sum()),
                    "shortfall_share": float(m["shortfall"].sum() / m["need"].sum()),
                    "paid_over_need": float(m["paid"].sum() / m["need"].sum()),
                    "service_level": float(m["answered"].sum() / m["calls"].sum()),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) and k not in ("aht",) else v) for k, v in r.items()})


def write(c: Calls, res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    struct = structure(c)
    (out / "structure.txt").write_text("\n".join(struct) + "\n", encoding="utf-8")
    rows = cells(res)
    write_csv(out / "cells.csv", rows)
    comp = [x for aht in res["ahts"] for x in comparisons(res, aht)]
    write_csv(out / "comparisons.csv", comp)
    ver = verdicts(comp)
    actual = np.concatenate(res["fc"]["oracle"])
    wape = {n: float(np.abs(np.concatenate(res["fc"][n]) - actual).sum() / actual.sum()) for n in F.FORECASTS}
    weekly = []
    for (aht, name, regime), m in res["tot"].items():
        for i, w in enumerate(res["weeks"]):
            weekly.append(
                {
                    "aht": aht,
                    "forecast": name,
                    "regime": regime,
                    "week": str(w.date()),
                    **{k: float(m[k][i]) for k in MEASURES},
                }
            )
    write_csv(out / "weekly.csv", weekly)
    mw = res["median_week"]
    aht0 = res["ahts"][0]
    act = res["fc"]["oracle"][mw]
    ex = []
    for h in range(168):
        r = {
            "hour": h,
            "calls": float(act[h]),
            "need": int(E.requirement(act[h : h + 1], aht0)[0]),
            "gbm_calls": float(res["fc"]["gbm"][mw][h]),
            "shadow_R4": float(res["shadow"][h]),
        }
        for (name, regime), on in res["example"].items():
            r[f"{name}_{regime}"] = float(on[h])
        ex.append(r)
    write_csv(out / "example_week.csv", ex)

    L = ["Shift constraints against forecast quality, NYC 311 agent-handled calls. Settings: PLAN.md.", ""]
    L += struct + [""]
    L.append(f"test weeks used: {len(res['weeks'])}; median week by calls: {res['weeks'][mw].date()}")
    L.append("WAPE of the forecasts over the test weeks: " + ", ".join(f"{n} {v:.4f}" for n, v in wape.items()))
    L.append("")
    for aht in res["ahts"]:
        for k in KS:
            L.append(
                f"AHT {aht / 60:g} min, k = {k}: cost per actual agent-hour needed (constraint cost of the "
                f"regime; forecast cost)"
            )
            for regime in S.REGIMES:
                parts = []
                for name in F.FORECASTS:
                    x = next(
                        r for r in rows if (r["aht"], r["forecast"], r["regime"], r["k"]) == (aht, name, regime, k)
                    )
                    parts.append(f"{name} {x['cost']:.4f} ({x['forecast_cost']:+.4f})")
                cc = constraint_cost(res, aht, regime, k)
                L.append(f"  {regime} {S.LABELS[regime]} [constraint {cc:.4f}]: " + "; ".join(parts))
            L.append("")
        L.append(f"AHT {aht / 60:g} min: paid hours / needed hours and share of calls answered within 30 s")
        for regime in S.REGIMES:
            parts = []
            for name in F.FORECASTS:
                x = next(r for r in rows if (r["aht"], r["forecast"], r["regime"], r["k"]) == (aht, name, regime, 1))
                parts.append(f"{name} {x['paid_over_need']:.3f} / {100 * x['service_level']:.1f}%")
            L.append(f"  {regime}: " + "; ".join(parts))
        L.append("")
    for aht in res["ahts"]:
        tag = "main" if aht == res["ahts"][0] else "sensitivity"
        for h, txt in (
            ("H1", "under R3 and R4 the constraint cost exceeds the forecast cost of mean4 and gbm"),
            ("H2", "the two_weeks -> gbm saving is smaller under R4 than under R0"),
            ("H3", "under R1 and R2 the forecast cost of gbm exceeds the constraint cost"),
        ):
            v = ver[f"{h}@{aht:g}"]
            L.append(
                f"{h}, AHT {aht / 60:g} min ({tag}): {'supported' if v['supported'] else 'not supported'} "
                f"({v['established']} of {v['n']} established, {v['positive']} in the stated direction) - {txt}"
            )
            if aht == res["ahts"][0]:
                for x in (x for x in comp if x["aht"] == aht and x["hypothesis"] == h):
                    L.append(
                        f"    k {x['k']}: {x['comparison']}: {x['estimate']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"
                    )
    L.append("")
    sp = res["shadow"]
    top = np.argsort(-sp)[:10]
    L.append("Shadow prices, R4 with the oracle, median week (paid hours per extra agent needed in that hour):")
    L.append("  largest: " + ", ".join(f"day {h // 24} {h % 24:02d}:00 {sp[h]:.2f}" for h in top))
    by_hod = np.array([sp[h::24].sum() for h in range(24)])
    L.append("  summed by hour of day: " + ", ".join(f"{h}:{v:.1f}" for h, v in enumerate(by_hod)))
    L.append(f"  hours with a positive shadow price: {int((sp > 1e-9).sum())} of 168")
    (out / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "verdicts.json").write_text(json.dumps({"verdicts": ver, "wape": wape}, indent=1), encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", type=Path, default=Path("sources"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    c = load(a.sources / "hourly_calls.csv")
    res = run(c)
    write(c, res, a.out)


if __name__ == "__main__":
    main()
