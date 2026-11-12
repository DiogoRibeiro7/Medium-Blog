"""Run both experiments of PLAN.md and write the results.

    python evaluate.py --profiles profiles --sources sources --out results

E1, dictionaries: every rule of every contract on every delivery of yellow, FHV and high-volume FHV.
E2, random losses: the history-based detectors and the outside count on yellow, green, FHV and high-volume FHV.

Writes results/summary.txt, results/rules_by_delivery.csv.gz, results/losses.csv and results/scores.csv.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from feed import counts as C
from feed.contracts import CONTRACTS, DICTIONARY_DATE
from feed.rules import judge

E2_FEEDS = ("yellow", "green", "fhv", "fhvhv")
SILENT_PART3 = 0.82  # Part 3: share of silent-fault deliveries the change checks caught
PERMUTATIONS, SEED = 10_000, 20261112


def read_profiles(root: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for f in sorted(root.glob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                p = json.loads(line)
                out.setdefault(p["feed"], {})[p["month"]] = p
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        for f in sorted(d.glob("*.json")):
            p = json.loads(f.read_text(encoding="utf-8"))
            out.setdefault(p["feed"], {})[p["month"]] = p
    return out


def bundle(root: Path) -> None:
    """Pack profiles/<feed>/*.json into profiles/<feed>.jsonl.gz."""
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        files = sorted(d.glob("*.json"))
        with gzip.open(root / f"{d.name}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for f in files:
                fh.write(f.read_text(encoding="utf-8").strip() + "\n")
        missing = d / "missing.txt"
        if missing.exists():
            (root / f"{d.name}.missing.txt").write_text(missing.read_text(encoding="utf-8"), encoding="utf-8")
        for f in d.iterdir():
            f.unlink()
        d.rmdir()


def months_between(a: str, b: str) -> int:
    ya, ma = map(int, a.split("-"))
    yb, mb = map(int, b.split("-"))
    return (yb - ya) * 12 + (mb - ma)


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    # average ranks for ties
    for arr, r in ((np.asarray(x), rx), (np.asarray(y), ry)):
        for v in np.unique(arr):
            idx = np.where(arr == v)[0]
            r[idx] = r[idx].mean()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(x, y, rho: float, rng: np.random.Generator) -> float:
    if np.isnan(rho):
        return float("nan")
    y = np.asarray(y)
    hits = sum(spearman(x, rng.permutation(y)) >= rho for _ in range(PERMUTATIONS))
    return (hits + 1) / (PERMUTATIONS + 1)


def e1(profiles: dict[str, dict[str, dict]], out: Path, L: list[str]) -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    L += ["E1. Rules written from the dictionaries of 18 March 2025, on every delivery", ""]
    verdicts = []
    for feed in CONTRACTS:
        ps = profiles.get(feed, {})
        ms = sorted(ps)
        if not ms:
            L.append(f"{feed}: no profiles")
            continue
        judged = {m: judge(ps[m]) for m in ms}
        for m in ms:
            for r in judged[m]:
                rows.append([feed, m, r.kind, r.variant, int(r.fired), f"{r.share:.6g}", r.detail])
        names = [r.name for r in judged[ms[0]]]
        L.append(f"{feed}: {len(ms)} deliveries, {ms[0]} to {ms[-1]}, {len(names) // 2} rules x 2 variants")
        for variant in ("strict", "tolerant"):
            rate = {
                nm: np.mean([next(r for r in judged[m] if r.name == nm).fired for m in ms])
                for nm in names
                if nm.endswith(f"({variant})")
            }
            always = [nm for nm, v in rate.items() if v == 1.0]
            never = [nm for nm, v in rate.items() if v == 0.0]
            L.append(
                f"  {variant}: fire on every delivery {len(always)} of {len(rate)} "
                f"({len(always) / len(rate):.0%}); never fire {len(never)}"
            )
            for nm, v in rate.items():
                L.append(f"    {nm:<28} fired on {v:6.1%} of deliveries")
            if variant == "strict":
                verdicts.append((feed, len(always) / len(rate)))
        alarms = np.array([sum(r.fired for r in judged[m]) for m in ms], float)
        dist = np.array([abs(months_between(DICTIONARY_DATE, m)) for m in ms], float)
        rho = spearman(dist, alarms)
        p = perm_p(dist, alarms, rho, rng)
        after = [m for m in ms if m > DICTIONARY_DATE]
        L.append(
            f"  alarms per delivery (both variants): mean {alarms.mean():.2f}; "
            f"after the dictionary ({len(after)} deliveries) "
            f"{np.mean([alarms[ms.index(m)] for m in after]) if after else float('nan'):.2f}"
        )
        L.append(
            f"  Spearman rho, months from {DICTIONARY_DATE} vs alarms: {rho:+.3f} (one-sided permutation p {p:.4f})"
        )
        verdicts.append((feed + " rho", rho, p))
        L.append("")
    strict_share = [v for v in verdicts if len(v) == 2]
    rhos = [v for v in verdicts if len(v) == 3]
    h1a = all(s > 0.5 for _, s in strict_share) and len(strict_share) == 3
    h1b = sum(1 for _, r, p in rhos if r > 0 and p < 0.05) >= 2
    L.append(
        f"H1a (more than half of the strict rules fire on every delivery, in each of the three feeds): "
        f"{'supported' if h1a else 'not supported'} "
        f"({', '.join(f'{f} {s:.0%}' for f, s in strict_share)})"
    )
    L.append(
        f"H1b (alarms grow with distance from the dictionary date, rho > 0 at p < 0.05 in at least two feeds): "
        f"{'supported' if h1b else 'not supported'}"
    )
    L.append("")
    with gzip.open(out / "rules_by_delivery.csv.gz", "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feed", "month", "kind", "variant", "fired", "share", "detail"])
        w.writerows(rows)


def e2(profiles: dict[str, dict[str, dict]], sources: Path, out: Path, L: list[str]) -> None:
    rows = C.rows_from_profiles(profiles)
    report = C.read_report(sources / "data_reports_monthly.csv")
    L += ["E2. Random losses of rows: the delivery's own history against a count from outside it", ""]
    res, sc = [], []
    h2a, h2b = [], []
    for feed in E2_FEEDS:
        r, rep = rows.get(feed, {}), report.get(feed, {})
        clean = C.scores(r, rep)
        n_eval = {d: len(v) for d, v in clean.items()}
        ratio = {m: r[m] / rep[m] - 1 for m in r if m in rep and m >= C.FIRST_EVAL}
        rv = np.array(list(ratio.values()))
        L.append(f"{feed}: deliveries evaluated {n_eval}")
        if len(rv):
            L.append(
                f"  file rows / report count - 1: median {np.median(rv):+.4%}, "
                f"5th-95th percentile {np.quantile(rv, 0.05):+.4%} to {np.quantile(rv, 0.95):+.4%}, "
                f"max |.| {np.abs(rv).max():.4%}"
            )
        for m, v in sorted(ratio.items()):
            sc.append([feed, m, "ratio", f"{v:.6g}"])
        thr = {d: (C.threshold(d), C.matched_threshold(clean[d])) for d in C.DETECTORS}
        for d in C.DETECTORS:
            fired = np.mean([v > thr[d][0] for v in clean[d].values()]) if clean[d] else float("nan")
            L.append(f"  {d:<26} clean alarms at the fixed line {fired:6.1%}; matched threshold {thr[d][1]:.2f}")
            for m, v in clean[d].items():
                sc.append([feed, m, d, f"{v:.6g}"])
        for loss in C.LOSSES:
            faulted = C.scores(r, rep, loss)
            line = []
            best_delivery = 0.0
            for d in C.DETECTORS:
                for label, t in (("fixed", thr[d][0]), ("matched", thr[d][1])):
                    ms = sorted(set(faulted[d]) & set(clean[d]))
                    det = np.mean([faulted[d][m] > t and clean[d][m] <= t for m in ms]) if ms else float("nan")
                    res.append([feed, loss, d, label, f"{t:.4g}", len(ms), f"{det:.4f}"])
                    if label == "matched":
                        line.append(f"{d} {det:.0%}")
                        if d in ("volume", "seasonal"):
                            best_delivery = max(best_delivery, det)
                        if d == "outside count":
                            outside = det
            L.append(f"  loss {loss:>4.0%}, matched budget: " + "; ".join(line))
            if loss in (0.05, 0.10, 0.20):
                h2a.append((feed, loss, best_delivery))
            h2b.append((feed, loss, outside, best_delivery))
        L.append("")
    a_ok = all(b < SILENT_PART3 for _, _, b in h2a)
    b_ok = all(o >= b for _, _, o, b in h2b) and all(o >= 0.9 for f, loss, o, _ in h2b if loss == 0.05)
    L.append(
        f"H2a (best delivery-only detector catches under {SILENT_PART3:.0%} of 5%, 10% and 20% losses at the "
        f"matched budget, every feed): {'supported' if a_ok else 'not supported'}"
    )
    for f, loss, b in h2a:
        L.append(f"    {f} {loss:.0%}: best delivery-only {b:.0%}")
    L.append(
        f"H2b (outside count at least as good at every loss, and at least 90% at 5%, every feed): "
        f"{'supported' if b_ok else 'not supported'}"
    )
    for f, loss, o, b in h2b:
        L.append(f"    {f} {loss:.0%}: outside count {o:.0%} vs best delivery-only {b:.0%}")
    with (out / "losses.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feed", "loss", "detector", "threshold", "value", "deliveries", "detected"])
        w.writerows(res)
    with (out / "scores.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feed", "month", "detector", "value"])
        w.writerows(sc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="profiles")
    ap.add_argument("--sources", default="sources")
    ap.add_argument("--out", default="results")
    ap.add_argument("--bundle", action="store_true", help="pack profiles/<feed>/ into profiles/<feed>.jsonl.gz first")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.bundle:
        bundle(Path(a.profiles))
    L = ["Part 4 of Code Quality for Data Work: what a delivery cannot vouch for. Settings: PLAN.md.", ""]
    profiles = read_profiles(Path(a.profiles))
    e1(profiles, out, L)
    e2(profiles, Path(a.sources), out, L)
    (out / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
