"""The Medium article: every number comes from the committed results.

``collect_values`` reads ``docs/results/results.json`` and ``docs/panel/summary.json``
and formats each quoted number once. ``render`` fills the ``{{key}}`` placeholders of
``publication/article.template.md``; an unknown or unused key is an error.
``write_publication`` writes the values, the article and the figures; ``check``
compares the committed values and article with what the code produces.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")
MINUS = "−"


def _signed(x: float, digits: int) -> str:
    text = f"{x:+.{digits}f}"
    return text.replace("-", MINUS) if not text.startswith("+") else text


def _plain(x: float, digits: int) -> str:
    text = f"{abs(x):.{digits}f}"
    return (MINUS + text) if x < 0 and float(text) != 0 else text


def pct(log_points: float, digits: int = 1, signed: bool = True) -> str:
    """A change in log points as a percentage change."""
    value = 100 * (math.exp(log_points) - 1)
    return (_signed(value, digits) if signed else _plain(value, digits)) + "%"


def p_value(p: float) -> str:
    return "below 0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def p_value3(p: float) -> str:
    """Three decimals: the maize article quotes p-values near 0.05, which two would blur."""
    return "below 0.001" if p < 0.001 else f"{p:.3f}"


def collect_values(root: Path) -> dict[str, str]:
    r = json.loads((root / "docs" / "results" / "results.json").read_text(encoding="utf-8"))
    panel = json.loads((root / "docs" / "panel" / "summary.json").read_text(encoding="utf-8"))
    coef = {(c["model"], c["term"]): c for c in r["coefficients"]}
    s, e1, e2, e3, e5, d = r["sample"], r["E1"], r["E2"], r["E3"], r["E5"], r["decisions"]
    edd = coef[("M1", "edd")]
    dry = e2["dry_season"]
    sens = {(x["variant"], x["estimand"]): x for x in r["sensitivity"]}
    wb = sens[("water_balance", "E2")]
    ge36 = next(b for b in r["M3"] if b["term"] == "t_ge36")
    ph = r["post_hoc"]
    direct, optimum, no_spain = ph["E2_direct"], ph["optimum_mm"], ph["E1_without_spain"]
    shares, july, y2003 = ph["identifying_variation"], ph["july"], ph["E4_2003"]
    worst = [100 * (math.exp(w["observed"]) - 1) for w in y2003["worst_regions"]]
    heat_2003 = y2003["heat_by_country"]
    edd_q = _panel_quantiles(root, "edd_30", (0.25, 0.5, 0.75))
    v: dict[str, str] = {
        "n_obs": f"{s['n']:,}", "n_regions": str(s["regions"]),
        "n_countries": str(s["countries"]), "year_first": str(s["years"][0]),
        "year_last": str(s["years"][1]), "n_harvests": str(s["years"][1] - s["years"][0] + 1),
        "single_region_countries": str(len(panel["single_region_countries"])),
        "e5_p_twoway": p_value(e5["p_twoway"]), "e5_p_boot": p_value(e5["p_bootstrap"]),
        "e1_pct": pct(edd["estimate"]), "e1_low": pct(edd["low"]), "e1_high": pct(edd["high"]),
        "e1_holm_twoway": p_value(d["E1"]["p_twoway_holm"]),
        "e1_holm_boot": p_value(d["E1"]["p_bootstrap_holm"]),
        "e2_holm_twoway": p_value(d["E2"]["p_twoway_holm"]),
        "e2_holm_boot": p_value(d["E2"]["p_bootstrap_holm"]),
        "dry_mm": str(int(round(abs(dry["mean_p10_minus_p50"])))),
        "dry_pct": pct(dry["effect"]), "dry_low": pct(dry["low"]),
        "dry_high": pct(dry["high"]),
        "optimum_mm": str(int(round(e2["optimum_mm"], -1))),
        "optimum_low": str(int(round(optimum["low"], -1))),
        "optimum_high": str(int(round(optimum["high"], -1))),
        "median_mm": str(int(round(r["precipitation_curve"]["median_mm"], -1))),
        "e2_direct_twoway": p_value(direct["p_twoway"]),
        "e2_direct_boot": p_value(direct["p_bootstrap"]),
        "e2_direct_holm_twoway": p_value(direct["p_twoway_holm"]),
        "e2_direct_holm_boot": p_value(direct["p_bootstrap_holm"]),
        "wb_pct": pct(wb["estimate"]), "wb_low": pct(wb["low"]), "wb_high": pct(wb["high"]),
        "edd_p25": f"{edd_q[0]:.1f}", "edd_median": f"{edd_q[1]:.1f}",
        "edd_p75": f"{edd_q[2]:.1f}",
        "edd_share_es": _share(shares["edd"]["ES"]), "edd_share_it": _share(shares["edd"]["IT"]),
        "edd_share_ro": _share(shares["edd"]["RO"]),
        "e1_no_spain_pct": pct(no_spain["estimate"]), "e1_no_spain_low": pct(no_spain["low"]),
        "e1_no_spain_high": pct(no_spain["high"]),
        "ge36_hour_pct": pct(ge36["estimate"] / 24, 2),
        "ge36_hour_low": pct(ge36["low"] / 24, 2),
        "ge36_hour_high": pct(ge36["high"] / 24, 2),
        "ge36_share_es": _share(shares["t_ge36"]["ES"]),
        "jul_edd_share": _share(july["edd_30_share"]),
        "jul_ge36_share": _share(july["ge36_share"]),
        "jul_south_share": _share(july["ge36_july_south_share_of_all"]),
        "e3_holm_twoway": p_value(d["E3"]["p_twoway_holm"]),
        "e3_holm_boot": p_value(d["E3"]["p_bootstrap_holm"]),
        "e3_mde_pct": pct(e3["minimum_detectable"], signed=False),
        "e3_mde_holm_pct": pct(ph["E3_minimum_detectable_holm"], signed=False),
        **{f"{c.lower()}_2003_edd": f"{heat_2003[c]['edd_30']:.1f}" for c in ("FR", "RO", "HU")},
        **{f"{c.lower()}_other_edd": f"{heat_2003[c]['edd_30_other_years']:.1f}"
           for c in ("FR", "RO", "HU")},
        "ge36_boot_p": p_value(ph["M3_p_bootstrap"]["t_ge36"]),
        "boot_other_bins": " and ".join(
            _bin_range(b) for b, p in ph["M3_p_bootstrap"].items()
            if p < 0.05 and b != "t_ge36"),
        # falls, so unsigned: the claims test checks that all three are losses
        "worst_2003_least": f"{-max(worst):.0f}%", "worst_2003_most": f"{-min(worst):.0f}%",
        "n_sensitivity": str(len({x["variant"] for x in r["sensitivity"]})),
        "e1_area_weighted": pct(sens[("area_weighted", "E1")]["estimate"]),
        "e1_water_balance": pct(sens[("water_balance", "E1")]["estimate"]),
        "dry_apr_jul": pct(sens[("apr_jul", "E2")]["estimate"]),
        "dry_area_weighted": pct(sens[("area_weighted", "E2")]["estimate"]),
        "e1_variants_clear": str(sum(1 for x in r["sensitivity"]
                                     if x["estimand"] == "E1" and x["high"] < 0)),
    }
    for row in r["E4"]:
        y = row["year"]
        v[f"e4_{y}_obs"] = _signed(row["eu_yield_observed_pct"], 1) + "%"
        v[f"e4_{y}_pred"] = _signed(row["eu_yield_predicted_pct"], 1) + "%"
        v[f"e4_{y}_corr"] = f"{row['correlation']:.2f}"
    v["ge36_hours"] = f"{24 * _ge36_days(root):.1f}"
    if e1["p_twoway"] is None or e3["p_twoway"] is None:
        raise ValueError("Missing tests in the results.")
    amended = _amendment(root)
    if amended is not None:
        v.update(amendment_values(amended))
    maize = root / "docs" / "maize" / "results" / "results.json"
    if maize.exists():                       # the pointer to the maize article
        m = json.loads(maize.read_text(encoding="utf-8"))
        v["maize_e1_pct"], v["maize_e2_pct"] = pct(m["E1"]["estimate"]), pct(m["E2"]["estimate"])
    return v


def _amendment(root: Path) -> dict[str, Any] | None:
    path = root / "docs" / "amendment" / "results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _date(month_day: str) -> str:
    months = ("January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December")
    return f"{int(month_day[3:])} {months[int(month_day[:2]) - 1]}"


def amendment_values(a: Mapping[str, Any]) -> dict[str, str]:
    """The values of Amendment A1 (METHODS.md, section 8) that the article quotes."""
    d, s = a["decisions"], a["sample"]
    ge36 = next(b for b in a["M3"] if b["term"] == "t_ge36")
    data_e1 = next(x for x in a["decomposition"] if x["combination"] == "data")["E1"]
    out = {"a1_excluded": str(s["excluded_region_years"]),
           "a1_data_e1_low": pct(data_e1["low"]), "a1_data_e1_high": pct(data_e1["high"]),
           "a1_data_e1_boot": p_value(data_e1["p_bootstrap"]),
           "maturity_first": _date(s["maturity"]["first"]),
           "maturity_last": _date(s["maturity"]["last"]),
           "a1_e1_no_spain_pct": pct(a["leave_one_out"]["summary"]["E1"]["without_it"]),
           "a1_e5_p_twoway": p_value(a["E5"]["p_twoway"]),
           "a1_e5_p_boot": p_value(a["E5"]["p_bootstrap"]),
           "a1_ge36_hour_pct": pct(ge36["estimate"] / 24, 2),
           "a1_ge36_hour_low": pct(ge36["low"] / 24, 2),
           "a1_ge36_hour_high": pct(ge36["high"] / 24, 2)}
    for key in ("E1", "E2"):
        k = key.lower()
        out |= {f"a1_{k}_pct": pct(a[key]["estimate"]), f"a1_{k}_low": pct(a[key]["low"]),
                f"a1_{k}_high": pct(a[key]["high"])}
    for key in ("E1", "E2", "E3"):
        k = key.lower()
        out |= {f"a1_{k}_holm_twoway": p_value(d[key]["p_twoway_holm"]),
                f"a1_{k}_holm_boot": p_value(d[key]["p_bootstrap_holm"])}
    return out


def _share(x: float) -> str:
    return f"{100 * x:.0f}%"


def _bin_range(term: str) -> str:
    """t_09_12 -> 9–12 °C (bins between two edges only)."""
    low, high = term.removeprefix("t_").split("_")
    return f"{int(low)}–{int(high)} °C"


def _panel_quantiles(root: Path, column: str, qs: tuple[float, ...],
                     panel: str = "panel") -> list[float]:
    """Quantiles of a panel column over every panel region-season (linear interpolation):
    the wheat panel (docs/panel) or the maize panel (docs/maize)."""
    import csv

    import numpy as np

    with (root / "docs" / panel / "panel.csv").open(encoding="utf-8") as stream:
        values = [float(r[column]) for r in csv.DictReader(stream)]
    return [float(np.quantile(values, q)) for q in qs]


def _ge36_days(root: Path) -> float:
    """Mean time at 36 °C and above per panel region-season (1 March to 31 July), in
    units of 24 hours."""
    import csv

    with (root / "docs" / "panel" / "panel.csv").open(encoding="utf-8") as stream:
        keys = {(r["geo"], r["year"]) for r in csv.DictReader(stream)}
    with (root / "docs" / "panel" / "weather.csv").open(encoding="utf-8") as stream:
        days = [float(r["t_ge36"]) for r in csv.DictReader(stream)
                if r["window"] == "mar_jul" and r["weighting"] == "all"
                and (r["geo"], r["year"]) in keys]
    return sum(days) / len(days)


def render(template: str, values: Mapping[str, str]) -> str:
    used = set(PLACEHOLDER.findall(template))
    unknown = sorted(used - set(values))
    if unknown:
        raise KeyError(f"Unknown placeholders: {', '.join(unknown)}.")
    unused = sorted(set(values) - used)
    if unused:
        raise KeyError(f"Values the article does not use: {', '.join(unused)}.")
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


MAIZE_DIR = "publication/maize"


def _json(root: Path, *parts: str) -> Any:
    return json.loads(root.joinpath(*parts).read_text(encoding="utf-8"))


def maize_values(root: Path) -> dict[str, str]:
    """The values the maize article quotes (METHODS.md, section 9)."""
    r = _json(root, "docs", "maize", "results", "results.json")
    summary = _json(root, "docs", "maize", "summary.json")
    wheat = _json(root, "docs", "results", "results.json")
    wheat_a1 = _json(root, "docs", "amendment", "results.json")
    s, d, e1, e2, e5 = r["sample"], r["decisions"], r["E1"], r["E2"], r["E5"]
    bins = {b["term"]: b for b in r["M3"]}
    sens = {(x["variant"], x["estimand"]): x for x in r["sensitivity"]}
    edd_q = _panel_quantiles(root, "edd_30", (0.25, 0.5, 0.75), "maize")
    wheat_edd = next(c for c in wheat["coefficients"]
                     if c["model"] == "M1" and c["term"] == "edd")
    v = {
        "n_obs": f"{s['n']:,}", "n_regions": str(s["regions"]),
        "n_countries": str(s["countries"]), "year_first": str(s["years"][0]),
        "year_last": str(s["years"][1]), "n_harvests": str(s["years"][1] - s["years"][0] + 1),
        "planting_first": _date(summary["season"]["planting_first"]),
        "planting_last": _date(summary["season"]["planting_last"]),
        "maturity_first": _date(summary["season"]["maturity_first"]),
        "maturity_last": _date(summary["season"]["maturity_last"]),
        "n_excluded_rule": str(summary["excluded_by_reason"].get("data_rule", 0)),
        "edd_p25": f"{edd_q[0]:.1f}", "edd_median": f"{edd_q[1]:.1f}",
        "edd_p75": f"{edd_q[2]:.1f}",
        "e5_p_twoway": p_value3(e5["p_twoway"]), "e5_p_boot": p_value3(e5["p_bootstrap"]),
        "dry_mm": str(int(round(abs(e2["mean_p10_minus_p50_mm"])))),
        "h2730_pct": pct(bins["t_27_30"]["estimate"] / 24, 2),
        "h3336_pct": pct(bins["t_33_36"]["estimate"] / 24, 2),
        "ge36_pct": pct(bins["t_ge36"]["estimate"] / 24, 2),
        "ge36_low": pct(bins["t_ge36"]["low"] / 24, 2),
        "ge36_high": pct(bins["t_ge36"]["high"] / 24, 2),
        "ge36_boot_p": p_value3(bins["t_ge36"]["p_bootstrap"]),
        "wheat_a1_e1_pct": pct(wheat_a1["E1"]["estimate"]),
        "wheat_a1_e2_pct": pct(wheat_a1["E2"]["estimate"]),
        "wheat_e1_pct": pct(wheat_edd["estimate"]),
        "wheat_e2_pct": pct(wheat["E2"]["dry_season"]["effect"]),
        "e1_not_irrigated": pct(sens[("not_irrigated", "E1")]["estimate"]),
        "e1_area_weighted": pct(sens[("area_weighted", "E1")]["estimate"]),
        "e2_area_weighted": pct(sens[("area_weighted", "E2")]["estimate"]),
        "n_sensitivity": str(len({x["variant"] for x in r["sensitivity"]})),
    }
    for key, x in (("e1", e1), ("e2", e2)):
        v |= {f"{key}_pct": pct(x["estimate"]), f"{key}_low": pct(x["low"]),
              f"{key}_high": pct(x["high"]),
              f"{key}_holm_twoway": p_value3(d[key.upper()]["p_twoway_holm"]),
              f"{key}_holm_boot": p_value3(d[key.upper()]["p_bootstrap_holm"])}
    for row in r["E4"]:
        y = row["year"]
        v[f"e4_{y}_obs"] = _signed(row["eu_yield_observed_pct"], 1) + "%"
        v[f"e4_{y}_pred"] = _signed(row["eu_yield_predicted_pct"], 1) + "%"
        v[f"e4_{y}_corr"] = f"{row['correlation']:.2f}"
    return v


def outputs(root: Path) -> dict[str, str]:
    """The text files the publication step writes, by path relative to ``root``: the wheat
    article and, once its results exist, the maize article."""
    values = collect_values(root)
    template = (root / "publication" / "article.template.md").read_text(encoding="utf-8")
    header = ("<!-- Generated by eu-wheat publication from article.template.md and the "
              "committed results. Do not edit. -->\n")
    out = {"publication/values.json": json.dumps(values, indent=1, ensure_ascii=False) + "\n",
           "publication/article.md": header + render(template, values)}
    maize_template = root / MAIZE_DIR / "article.template.md"
    if maize_template.exists():
        maize = maize_values(root)
        out[f"{MAIZE_DIR}/values.json"] = json.dumps(maize, indent=1, ensure_ascii=False) + "\n"
        out[f"{MAIZE_DIR}/article.md"] = header + render(
            maize_template.read_text(encoding="utf-8"), maize)
    return out


def write_publication(root: Path, figures: bool = True) -> list[str]:
    written = []
    for name, text in outputs(root).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8", newline="\n")
        written.append(name)
    if figures:
        from eu_wheat_weather.figures import draw_all

        results: dict[str, Any] = json.loads(
            (root / "docs" / "results" / "results.json").read_text(encoding="utf-8"))
        written += [str(p.relative_to(root).as_posix())
                    for p in draw_all(results, root / "publication" / "figures",
                                      _amendment(root))]
        if (root / MAIZE_DIR / "article.template.md").exists():
            from eu_wheat_weather.figures import draw_maize

            written += [str(p.relative_to(root).as_posix()) for p in draw_maize(
                _json(root, "docs", "maize", "results", "results.json"),
                _json(root, "docs", "amendment", "results.json"), results,
                root / MAIZE_DIR / "figures")]
    return written


def check(root: Path) -> list[str]:
    """Generated text files that differ from what the code produces now."""
    return [name for name, text in outputs(root).items()
            if not (root / name).exists()
            or (root / name).read_text(encoding="utf-8") != text]
