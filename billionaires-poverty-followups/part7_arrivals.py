"""Part 7: what are the large arrivals on the Forbes list made of?

Part 4 conjectured that most arrivals far above the $1 billion line are new
valuations of fortunes that already existed (stock-market listings, funding
rounds) or divisions of existing fortunes among heirs, rather than fortunes
created within the year. This script tests what the Forbes lists themselves can
test, using the link counts from ``fetch_forbes_arrivals.py``:

* each entrant is put in one category, in this order: the same person relisted
  under a new key; a return to the list; an heir of a family member who left at
  this or the previous transition; a division from a family member on the list
  whose fortune fell; joining family members already on the list; a co-owner of
  a company label whose other owners entered with them or rose; a familyList
  link only; nothing found;
* family links are compared with a placebo, the same family link to someone who
  leaves two or three transitions after the entry, and company rises with the
  rate at which any fortune on the list rose that much in the same year;
* Forbes's own self-made flag checks the links from outside: heirs should
  mostly be classified as not self-made and co-owners as self-made;
* departures far above the line are followed to the entries and rises of their
  families.

Every number quoted in the Part 7 article is computed here and written to
``results/part7/numbers.json``; ``--check`` recomputes everything and fails if
any number differs. Figures go to ``figures/part7/``.

Usage::

    python part7_arrivals.py           # print, write results/part7 and figures/part7
    python part7_arrivals.py --check   # fail if any number differs from numbers.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from part1_poverty_lines import AQUA, BASELINE, BLUE, INK, MUTED, ORANGE, SECONDARY, _compare, _frame, _style  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results" / "part7"
FIGURES = HERE / "figures" / "part7"

FIRST, LAST = 2010, 2024        # transitions: list t to list t+1, labelled t as in Part 4
FAR = 1500                      # Part 4's "far above the line": entering at $1.5 billion or more
VERY_FAR = 3000
N_BOOT = 400
YELLOW, PALE_ORANGE = "#eda100", "#f9d2c2"
SOURCE = ("Data: Forbes annual billionaire lists 2010-2025, linked year to year; only aggregate counts are stored. "
          "Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/billionaires-poverty-followups")

RULES = ("strict", "label", "rare", "loose")     # family-matching rules, as in fetch_forbes_arrivals.py
MAIN = ("label", "rare")                         # the rule used to classify: either of the two middle rules
HEIR, FELL, INC, PLACEBO = 0, 1, 2, 3            # bit offsets within a rule (departures: HEIR = a relative enters,
                                                 # FELL = a relative on the list rose)
E_WINDOW, E_CO_ENTRY, E_CO_RISE, E_CO_INC, E_GENERIC, E_RELISTED = 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21
X_WINDOW, X_NEXT, X_RETURNS, X_RELISTED = 1 << 16, 1 << 17, 1 << 18, 1 << 19


def mask(rules: Sequence[str], k: int) -> int:
    return sum(1 << (4 * RULES.index(r) + k) for r in rules)


CATEGORIES = (("relisted", "Same person, new key"), ("returning", "Return to the list"),
              ("heir", "Heir of a family member who left"), ("division", "Division from a relative whose fortune fell"),
              ("family", "Joins relatives already listed"), ("company", "Co-owner of a revalued company"),
              ("inherited", "Inherited, no relative on the list"), ("selfmade", "Self-made, no link"),
              ("unknown", "No link, self-made status unknown"))
LINKED = ("relisted", "returning", "heir", "division", "family", "company")
EXISTING = ("returning", "heir", "division", "family", "company", "inherited")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(data: Path = DATA) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for man, name, key in (("forbes_arrivals_manifest.json", "forbes_arrivals.csv.gz", "arrivals_sha256"),
                           ("forbes_tenure_manifest.json", "forbes_tenure.csv.gz", "tenure_sha256")):
        m = json.loads((data / man).read_text(encoding="utf-8"))
        if _sha(data / name) != m[key]:
            raise ValueError(f"{name} does not match {man}")
        out[name.split(".")[0]] = pd.read_csv(data / name, keep_default_na=False,
                                              dtype={"key": str, "prev": str, "tenure": str, "age": str})
        out[man.split(".")[0]] = m
    out["forbes_flows"] = pd.read_csv(data / "forbes_flows.csv.gz", keep_default_na=False)
    out["forbes_flows_manifest"] = json.loads((data / "forbes_flows_manifest.json").read_text(encoding="utf-8"))
    return out


def check_against_part4(arr: pd.DataFrame, flows: pd.DataFrame, changed: set[int]) -> dict[str, Any]:
    """Entries and exits by transition and band must equal Part 4's flows exactly. Part 4 tabulates flows by country,
    so it leaves out people with no recorded citizenship; they are compared separately and reported."""
    out: dict[str, Any] = {"changed_lists": sorted(changed)}
    for table, flow in (("entry", "entry"), ("exit", "exit")):
        t = arr[arr.table == table]
        a = t[t.region != "unknown"].groupby(["year", "band_lo_musd"]).n.sum()
        b = flows[flows.flow == flow].groupby(["year", "band_lo_musd"]).n.sum()
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(0)
        bad = int((j.a != j.b).sum())
        if bad:
            raise ValueError(f"{table}: {bad} cells differ from Part 4's flows")
        out[table] = {"people": int(t.n.sum()), "known_citizenship": int(j.a.sum()), "part4_people": int(j.b.sum()),
                      "unknown_citizenship": int(t[t.region == "unknown"].n.sum()), "cells": int(len(j)), "mismatches": 0}
    e = arr[arr.table == "entry"]
    out["entry_far"] = int(e[e.band_lo_musd >= FAR].n.sum())
    out["entry_far_known"] = int(e[(e.band_lo_musd >= FAR) & (e.region != "unknown")].n.sum())
    return out


def changed_lists(arrivals_manifest: dict[str, Any], flows_manifest: dict[str, Any]) -> set[int]:
    old = {y: e.get("sha256") for y, e in flows_manifest["years"].items()}
    return {int(y) for y, e in arrivals_manifest["years"].items() if e.get("sha256") != old.get(y)}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(links: int, tenure: str, self_made: str = "unknown", rules: Sequence[str] = MAIN) -> str:
    """One category per entrant, in a fixed order of precedence. familyList turned out to be a yes/no field in the
    Forbes data, so it links no one and is not used."""
    if links & E_RELISTED:
        return "relisted"
    if tenure == "returning":
        return "returning"
    if links & mask(rules, HEIR):
        return "heir"
    if links & mask(rules, FELL):
        return "division"
    if links & mask(rules, INC):
        return "family"
    if links & (E_CO_ENTRY | E_CO_RISE):
        return "company"
    return {"no": "inherited", "yes": "selfmade"}.get(self_made, "unknown")


def entries(arr: pd.DataFrame, rules: Sequence[str] = MAIN) -> pd.DataFrame:
    e = arr[arr.table == "entry"].copy()
    e["links"] = e["links"].astype(int)
    e["category"] = [classify(f, t, m, rules) for f, t, m in zip(e["links"], e.tenure, e.self_made)]
    e["far"] = e.band_lo_musd >= FAR
    return e


def _shares(d: pd.DataFrame) -> dict[str, Any]:
    n = int(d.n.sum())
    by = d.groupby("category").n.sum()
    return {"n": n, **{c: {"n": int(by.get(c, 0)), "share": float(by.get(c, 0) / n) if n else float("nan")}
                       for c, _ in CATEGORIES}}


def composition(e: pd.DataFrame) -> dict[str, Any]:
    groups = (("near", e.band_lo_musd < FAR), ("far", e.far), ("very_far", e.band_lo_musd >= VERY_FAR),
              ("1500_3000", (e.band_lo_musd >= FAR) & (e.band_lo_musd < VERY_FAR)),
              ("3000_10000", (e.band_lo_musd >= VERY_FAR) & (e.band_lo_musd < 10000)), ("10000_up", e.band_lo_musd >= 10000))
    out = {g: _shares(e[s]) for g, s in groups}
    for g, sel in groups:
        out[g]["linked_share"] = sum(out[g][c]["share"] for c in LINKED)
        out[g]["relatives_share"] = sum(out[g][c]["share"] for c in ("heir", "division", "family"))
        out[g]["existing_share"] = sum(out[g][c]["share"] for c in EXISTING)
        co = e[sel & (e.category == "company")]
        out[g]["company_rise"] = int(co[co["links"] & E_CO_RISE > 0].n.sum())
        out[g]["company_entry_only"] = int(co[co["links"] & E_CO_RISE == 0].n.sum())
        known = e[sel & (e.self_made != "unknown")]
        out[g]["self_made_share"] = float(known[known.self_made == "yes"].n.sum() / known.n.sum())
    return out


def by_region(e: pd.DataFrame) -> dict[str, Any]:
    f = e[e.far]
    out = {}
    for r in ("United States", "China", "India", "Europe", "Other"):
        d = f[f.region == r]
        s = _shares(d)
        out[r] = {"n": s["n"], **{c: s[c]["share"] for c, _ in CATEGORIES},
                  "family_any": sum(s[c]["share"] for c in ("heir", "division", "family")),
                  "existing": sum(s[c]["share"] for c in EXISTING)}
    return out


def by_year(e: pd.DataFrame) -> dict[str, Any]:
    f = e[e.far]
    out = {}
    for y in sorted(f.year.unique()):
        s = _shares(f[f.year == y])
        out[str(int(y))] = {"n": s["n"], **{c: s[c]["n"] for c, _ in CATEGORIES}}
    return out


# ---------------------------------------------------------------------------
# Are the links real? Placebo and Forbes's self-made flag
# ---------------------------------------------------------------------------

def _bin_ci(k: float, n: float) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    se = math.sqrt(max(p * (1 - p), 1e-12) / n)
    return [p - 1.96 * se, p + 1.96 * se]


def window_sizes(arr: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ex = arr[arr.table == "exit"].groupby("year").n.sum()
    en = arr[arr.table == "entry"].groupby("year").n.sum()
    return ex, en


def placebo(e: pd.DataFrame, arr: pd.DataFrame) -> dict[str, Any]:
    """Entrants linked to a relative who left at the entry transition or the one before, against the same link to a
    relative who leaves two or three transitions later. The placebo is scaled by the number of departures in the two
    windows, year by year, since chance links grow with the number of people they can match. Forbes's self-made flag
    is a second check: an heir link that is real should pick out people Forbes classifies as not self-made."""
    ex, _ = window_sizes(arr)
    w = e[(e["links"] & E_WINDOW > 0) & (e["links"] & E_RELISTED == 0) & (e.year >= FIRST + 1)]
    scale = {y: (ex.get(y, 0) + ex.get(y - 1, 0)) / (ex.get(y + 2, 0) + ex.get(y + 3, 0)) for y in w.year.unique()}
    out: dict[str, Any] = {"years": [int(y) for y in sorted(w.year.unique())],
                           "scale_mean": float(np.average([scale[y] for y in w.year], weights=w.n))}
    for rule_name, rules in (("strict", ("strict",)), ("label", ("label",)), ("rare", ("rare",)), ("main", MAIN),
                             ("loose", ("loose",))):
        out[rule_name] = {}
        for g, sel in (("far", w.far), ("near", ~w.far)):
            d = w[sel]
            n = float(d.n.sum())
            if n == 0:
                continue
            hit = d["links"] & mask(rules, HEIR) > 0
            plc = d["links"] & mask(rules, PLACEBO) > 0
            obs = float(d[hit].n.sum())
            exp = float((d[plc].n * d[plc].year.map(scale)).sum())
            linked = d[hit & (d.self_made != "unknown")]
            out[rule_name][g] = {"n": int(n), "observed": obs / n, "placebo_scaled": exp / n, "excess": (obs - exp) / n,
                                 "ratio": obs / exp if exp else float("inf"), "ci95_observed": _bin_ci(obs, n),
                                 "not_self_made_linked": float(linked[linked.self_made == "no"].n.sum() / linked.n.sum())
                                 if linked.n.sum() else float("nan")}
    known = w[(w.self_made != "unknown") & w.far]
    out["not_self_made_all_far"] = float(known[known.self_made == "no"].n.sum() / known.n.sum()) if len(known) else float("nan")
    return out


def co_owner_rises(e: pd.DataFrame, tenure: pd.DataFrame) -> dict[str, Any]:
    """Entrants with a co-owner already on the list: how often did a co-owner rise by 0.1 or more, against the share of
    all stayers that rose that much in the same year (the chance an unrelated co-owner would)."""
    lc = tenure[(tenure.table == "logchange") & tenure.year.between(FIRST, LAST)]
    lc = lc.assign(lo=lc.key.astype(int))
    rose = lc[lc.lo >= 100].groupby("year").n.sum() / lc.groupby("year").n.sum()
    d = e[(e["links"] & E_CO_INC > 0) & (e["links"] & E_RELISTED == 0)]
    out = {}
    for name, sel in (("far", d.far), ("near", ~d.far), ("all", d.far | ~d.far)):
        x = d[sel]
        n = float(x.n.sum())
        obs = float(x[x["links"] & E_CO_RISE > 0].n.sum())
        exp = float((x.n * x.year.map(rose)).sum())
        out[name] = {"n": int(n), "observed": obs / n if n else float("nan"), "expected_one_owner": exp / n if n else float("nan"),
                     "ratio": obs / exp if exp else float("nan")}
    out["stayer_rise_share_mean"] = float(rose.mean())
    out["co_inc_share_far"] = float(d[d.far].n.sum() / e[e.far & (e["links"] & E_RELISTED == 0)].n.sum())
    return out


def self_made(e: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for name, sel in (("far", e.far), ("near", ~e.far)):
        d = e[sel & (e.self_made != "unknown")]
        out[name] = {}
        for c, _ in CATEGORIES:
            x = d[d.category == c]
            n = int(x.n.sum())
            if n:
                out[name][c] = {"n": n, "self_made": float(x[x.self_made == "yes"].n.sum() / n)}
    return out


def ages(e: pd.DataFrame) -> dict[str, Any]:
    f = e[e.far & (e.age != "unknown")]
    out = {}
    for c in ("heir", "division", "family", "company", "inherited", "selfmade", "returning"):
        x = f[f.category == c]
        n = float(x.n.sum())
        out[c] = {"n": int(n), "under_50": float(x[x.age == "<50"].n.sum() / n) if n else float("nan"),
                  "85_plus": float(x[x.age == "85+"].n.sum() / n) if n else float("nan")}
    return out


# ---------------------------------------------------------------------------
# Departures far above the line: where do the fortunes go?
# ---------------------------------------------------------------------------

def departures(arr: pd.DataFrame) -> dict[str, Any]:
    """Departures followed by a relative entering at the same or the next transition, or a relative on the list
    rising by 0.1 or more, against the placebo of a relative who entered three or four transitions earlier (scaled by
    the number of entrants in the two windows)."""
    x = arr[arr.table == "exit"].copy()
    x["links"] = x["links"].astype(int)
    x = x[(x["links"] & X_RELISTED == 0) & (x["links"] & X_NEXT > 0)]
    _, en = window_sizes(arr)
    out: dict[str, Any] = {}
    groups = (("very_far", x.band_lo_musd >= VERY_FAR), ("far", x.band_lo_musd >= FAR), ("near", x.band_lo_musd < FAR),
              ("very_far_85", (x.band_lo_musd >= VERY_FAR) & (x.age == "85+")))
    for g, sel in groups:
        d = x[sel]
        n = float(d.n.sum())
        out[g] = {"n": int(n), "age_85_plus": float(d[d.age == "85+"].n.sum() / n),
                  "returns_later": float(d[d["links"] & X_RETURNS > 0].n.sum() / n)}
        for rule_name, rules in (("strict", ("strict",)), ("main", MAIN), ("loose", ("loose",))):
            heir = d["links"] & mask(rules, HEIR) > 0
            rise = d["links"] & mask(rules, FELL) > 0
            out[g][rule_name] = {"heir_entry": float(d[heir].n.sum() / n), "relative_rise": float(d[rise].n.sum() / n),
                                 "either": float(d[heir | rise].n.sum() / n)}
    w = x[x["links"] & X_WINDOW > 0]
    scale = {y: (en.get(y, 0) + en.get(y + 1, 0)) / (en.get(y - 3, 0) + en.get(y - 4, 0)) for y in w.year.unique()}
    for g, sel in (("very_far", w.band_lo_musd >= VERY_FAR), ("near", w.band_lo_musd < FAR)):
        d = w[sel]
        n = float(d.n.sum())
        out["placebo_" + g] = {"n": int(n), "years": [int(y) for y in sorted(d.year.unique())]}
        for rule_name, rules in (("strict", ("strict",)), ("label", ("label",)), ("rare", ("rare",)), ("main", MAIN),
                                 ("loose", ("loose",))):
            hit = d["links"] & mask(rules, HEIR) > 0
            plc = d["links"] & mask(rules, PLACEBO) > 0
            obs = float(d[hit].n.sum())
            exp = float((d[plc].n * d[plc].year.map(scale)).sum())
            out["placebo_" + g][rule_name] = {"observed": obs / n, "placebo_scaled": exp / n, "excess": (obs - exp) / n,
                                             "ci95_observed": _bin_ci(obs, n)}
    return out


# ---------------------------------------------------------------------------
# Against Part 4's crossing model
# ---------------------------------------------------------------------------

def against_model(comp: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    far = comp["far"]
    linked = sum(far[c]["n"] for c in LINKED)
    existing = sum(far[c]["n"] for c in EXISTING)
    out = {"observed_far": far["n"], "model_far": ref.get("model_far"), "linked_far": linked, "existing_far": existing,
           "selfmade_unlinked_far": far["selfmade"]["n"], "unknown_unlinked_far": far["unknown"]["n"]}
    if ref.get("model_far"):
        out["excess_over_model"] = far["n"] - ref["model_far"]
        out["linked_over_excess"] = linked / out["excess_over_model"]
        out["existing_over_excess"] = existing / out["excess_over_model"]
        out["selfmade_unlinked_minus_model"] = far["selfmade"]["n"] - ref["model_far"]
        out["selfmade_unlinked_minus_model_share"] = out["selfmade_unlinked_minus_model"] / far["n"]
    return out


def bootstrap(e: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    years = sorted(e.year.unique())
    f = e[e.far]
    tot = {y: f[f.year == y].n.sum() for y in years}
    cat = {y: f[f.year == y].groupby("category").n.sum() for y in years}
    res: dict[str, list[float]] = {"linked": [], "relatives": [], "existing": [], "returning": [], "company": [],
                                   "inherited": []}
    for _ in range(n_boot):
        draw = rng.choice(years, size=len(years), replace=True)
        n = sum(tot[y] for y in draw)
        c = pd.concat([cat[y] for y in draw]).groupby(level=0).sum()
        res["linked"].append(sum(c.get(k, 0) for k in LINKED) / n)
        res["relatives"].append(sum(c.get(k, 0) for k in ("heir", "division", "family")) / n)
        res["returning"].append(c.get("returning", 0) / n)
        res["company"].append(c.get("company", 0) / n)
        res["inherited"].append(c.get("inherited", 0) / n)
        res["existing"].append(sum(c.get(k, 0) for k in EXISTING) / n)
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]  # noqa: E731
    return {"n_boot": n_boot, **{k + "_ci95": q(v) for k, v in res.items()}}


def relisting_2023(arr: pd.DataFrame) -> dict[str, Any]:
    """Heir and division links among large arrivals at each transition, under the strict and main rules: Forbes
    overhauled how it lists shared family wealth on its 2023 list (the 2022 transition here)."""
    out: dict[str, Any] = {}
    for name, rules in (("strict", ("strict",)), ("main", MAIN)):
        e = entries(arr, rules)
        f = e[e.far & e.category.isin(["heir", "division"])]
        by = f.groupby("year").n.sum()
        out[name] = {"by_year": {str(int(y)): int(v) for y, v in by.items()}, "total": int(by.sum()),
                     "y2022": int(by.get(2022, 0)), "y2022_share": float(by.get(2022, 0) / by.sum()),
                     "y2022_europe": int(f[(f.year == 2022) & (f.region == "Europe")].n.sum()),
                     "mean_other_years": float(by.drop(2022, errors="ignore").mean())}
    return out


def _py(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    return x.item() if isinstance(x, np.generic) else x


def compute(inp: dict[str, Any], n_boot: int = N_BOOT) -> dict[str, Any]:
    arr, flows, tenure, man = inp["forbes_arrivals"], inp["forbes_flows"], inp["forbes_tenure"], inp["forbes_arrivals_manifest"]
    ref: dict[str, Any] = {}
    p4 = HERE / "results" / "part4" / "numbers.json"
    if p4.exists():
        n4 = json.loads(p4.read_text(encoding="utf-8"))
        ref = _part4_refs(n4)
    e = entries(arr)
    comp = composition(e)
    out = {
        "fetch": {y: {"n": v.get("n_persons"), "same_as_part6": v.get("same_as_part6")} for y, v in man["years"].items()},
        "fields": {y: v.get("fields") for y, v in man["years"].items()},
        "generic_labels": man.get("generic_labels"),
        "part4_refs": ref,
        "check_part4": check_against_part4(arr, flows, changed_lists(man, inp["forbes_flows_manifest"])),
        "composition": comp,
        "by_region": by_region(e),
        "by_year": by_year(e),
        "placebo": placebo(e, arr),
        "rule_sensitivity": {r: {k: composition(entries(arr, rr))["far"][k] for k in ("relatives_share", "existing_share")}
                             for r, rr in (("strict", ("strict",)), ("main", MAIN), ("loose", ("loose",)))},
        "co_owner": co_owner_rises(e, tenure),
        "self_made": self_made(e),
        "ages": ages(e),
        "departures": departures(arr),
        "against_model": against_model(comp, ref),
        "stock_family": stock_family(arr),
        "relisting_2023": relisting_2023(arr),
        "bootstrap": bootstrap(e, n_boot),
    }
    return _py(out)


def _part4_refs(n4: dict[str, Any]) -> dict[str, Any]:
    m, c = n4["model"], n4["churn"]["entries_returning"]
    return {"model_far": m["entries_total_pred"] - m["entries_near_total_pred"],
            "observed_far": m["entries_total_obs"] - m["entries_near_total_obs"],
            "returning_far_share_from_2012": c["far_share"], "returning_near_share_from_2012": c["near_share"]}


def stock_family(arr: pd.DataFrame) -> dict[str, Any]:
    s = arr[arr.table == "stock"].copy()
    s["links"] = s["links"].astype(int)
    out = {}
    for name, sel in (("all", s.year >= FIRST), ("far", s.band_lo_musd >= FAR), ("near", s.band_lo_musd < FAR)):
        d = s[sel]
        n = float(d.n.sum())
        out[name] = {"n": int(n), **{r: float(d[d["links"] & (1 << k) > 0].n.sum() / n) for k, r in enumerate(RULES)}}
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _r(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_r(v) for v in x] if len(x) <= 12 else f"[{len(x)} items]"
    return x


def summary_lines(n: dict[str, Any]) -> list[str]:
    out = [f"Fetch: {n['fetch']}", f"Generic labels: {n['generic_labels']}", f"Check against Part 4: {n['check_part4']}",
           f"Part 4 refs: {_r(n['part4_refs'])}"]
    y0 = sorted(n["fields"])[0]
    out.append(f"Fields {y0}: {n['fields'][y0]}")
    for g, d in n["composition"].items():
        out.append(f"Composition {g}: n {d['n']} linked {d['linked_share']:.4f} existing {d['existing_share']:.4f} | "
                   + "; ".join(f"{c} {d[c]['n']} ({d[c]['share']:.3f})" for c, _ in CATEGORIES))
    for k in ("by_region", "placebo", "rule_sensitivity", "co_owner", "self_made", "ages", "departures", "against_model",
              "stock_family", "relisting_2023", "bootstrap"):
        out.append(f"{k}: {_r(n[k])}")
    out.append(f"by_year: {n['by_year']}")
    return out


FIGURE_FILES = ("fig1_departures_and_heirs.png", "fig2_what_arrivals_are.png")
RULE_LABELS = (("strict", "Surname +\ncompany"), ("label", "Surname +\nany label"), ("rare", "Rare\nsurname"),
               ("main", "Either of\nthe two"), ("loose", "Surname\nonly"))


def frame(fig: plt.Figure, title: str, subtitle: str, **kw: Any) -> None:
    _frame(fig, title, subtitle, source=SOURCE, **kw)


def fig_departures(n: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True)
    for ax, key, title in ((axes[0], "placebo_very_far", "Left from $3bn or more (mostly deaths)"),
                           (axes[1], "placebo_near", "Left from below $1.5bn (mostly falls)")):
        d = n["departures"][key]
        x = np.arange(len(RULE_LABELS))
        obs = np.array([100 * d[r]["observed"] for r, _ in RULE_LABELS])
        lo = np.array([100 * d[r]["ci95_observed"][0] for r, _ in RULE_LABELS])
        hi = np.array([100 * d[r]["ci95_observed"][1] for r, _ in RULE_LABELS])
        plc = np.array([100 * d[r]["placebo_scaled"] for r, _ in RULE_LABELS])
        ax.bar(x - 0.19, obs, width=0.36, color=BLUE, label="A relative enters within a year", zorder=2)
        ax.vlines(x - 0.19, np.clip(lo, 0, None), hi, color=INK, lw=1.1, zorder=3)
        ax.bar(x + 0.19, plc, width=0.36, color=PALE_ORANGE, edgecolor=ORANGE, lw=1.2,
               label="Expected by chance (placebo)", zorder=2)
        for xi, v, h in zip(x - 0.19, obs, hi):
            ax.text(xi, max(v, h) + 1.0, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
        for xi, v in zip(x + 0.19, plc):
            ax.text(xi, v + 1.6, f"{v:.0f}%", ha="center", fontsize=9, color=SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in RULE_LABELS], fontsize=8.6)
        ax.set_title(f"{title}, n = {d['n']:,}", fontsize=11, color=INK, loc="left")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 62)
    axes[0].set_ylabel("Share of departures")
    axes[0].legend(loc="upper left", fontsize=9.5)
    frame(fig, "A large fortune's departure is often followed by a relative's entry",
          "Departures 2014-2023 followed by a relative entering at the same or the next list, under five matching rules\n"
          "(all need the same country), against a placebo: a relative who entered three or four lists earlier, scaled.",
          top=0.80, bottom=0.17, left=0.07)
    fig.savefig(out, dpi=250)
    plt.close(fig)


GROUPS_FIG2 = (("returned", "Returned to the list", BLUE, ("relisted", "returning")),
               ("relative", "Relative on the list", ORANGE, ("heir", "division", "family")),
               ("company", "Co-owner of a revalued company", AQUA, ("company",)),
               ("inherited", "Inherited, no relative listed", YELLOW, ("inherited",)),
               ("selfmade", "Self-made, no link", BASELINE, ("selfmade", "unknown")))


def fig_composition(n: dict[str, Any], out: Path) -> None:
    rows = (("near", "$1bn-$1.5bn"), ("1500_3000", "$1.5bn-$3bn"), ("very_far", "$3bn and above"))
    fig, ax = plt.subplots(figsize=(11, 4.9))
    for i, (key, label) in enumerate(rows):
        c = n["composition"][key]
        left = 0.0
        for _, name, col, cats in GROUPS_FIG2:
            v = 100 * sum(c[k]["share"] for k in cats)
            ax.barh(i, v - 0.25, left=left, color=col, height=0.62, label=name if i == 0 else None)
            if v >= 4:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9.5,
                        color=INK if col in (YELLOW, BASELINE, AQUA) else "white")
            left += v
        ax.text(101, i, f"{c['n']:,}", va="center", fontsize=9.5, color=SECONDARY)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"Entered at {lab}" for _, lab in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%" if v <= 100 else ""))
    ax.grid(axis="y", visible=False)
    ax.text(101, -0.55, "people", fontsize=9, color=SECONDARY)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=5, fontsize=9, frameon=False)
    frame(fig, "Most large arrivals are existing fortunes; few are heirs of someone who just left",
          "People entering the Forbes list 2011-2025 by the net worth at entry. Relatives: same surname and country, and the\n"
          "same company label or a rare surname; about half of the heir links among large arrivals are expected by chance.",
          top=0.80, bottom=0.27, left=0.19, right=0.95)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def render(n: dict[str, Any], outdir: Path) -> None:
    _style()
    outdir.mkdir(parents=True, exist_ok=True)
    fig_departures(n, outdir / FIGURE_FILES[0])
    fig_composition(n, outdir / FIGURE_FILES[1])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    numbers = json.loads(json.dumps(compute(load_inputs())))
    lines_out = summary_lines(numbers)
    print("\n".join(lines_out))
    target = RESULTS / "numbers.json"
    if args.check:
        problems: list[str] = []
        _compare(numbers, json.loads(target.read_text(encoding="utf-8")), "numbers", problems)
        if problems:
            print("\nREPRODUCTION FAILED:\n" + "\n".join(problems[:50]), file=sys.stderr)
            return 1
        print("\nREPRODUCED: every number matches results/part7/numbers.json")
    else:
        RESULTS.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (RESULTS / "summary.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    if not args.no_figures:
        render(numbers, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
