"""Stage 2: rates corrected for what the dictionary gets wrong, era by era.

For each stratum s (era x dictionary tag x side of the floor) the share of Chair speeches
that are disciplinary interventions has a Beta posterior from the hand labels,

    theta_s ~ Beta(k_s + a, n_s - k_s + a),       a = 1/2 (Jeffreys),

and every quantity below is a function of the stratum totals N_s, which are known exactly
from the full corpus:

    true interventions   T_e  = sum_s N_s theta_s                  (s in era e)
    tagged speeches      D_e  = sum_{s tagged} N_s
    true positives       TP_e = sum_{s tagged} N_s theta_s
    precision            P_e  = TP_e / D_e
    recall               R_e  = TP_e / T_e
    naive rate           1000 D_e / M_e,   corrected rate   1000 T_e / M_e

Since D = T R / P, the naive change between two eras splits exactly into a change in the
true rate, a change in recall and a change in precision:

    log(naive_e / naive_r) = log(rate_e / rate_r) + log(R_e / R_r) - log(P_e / P_r).

The identity holds draw by draw, so its three terms carry their uncertainty with them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config
from .labels import final_labels

SIDES = ["Government", "Opposition", "Other"]


def _q(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "lo": float(np.quantile(x, 0.025)), "hi": float(np.quantile(x, 0.975))}


def stratum_table(cfg: Config, labels: pd.DataFrame | None = None, source: str = "") -> tuple[pd.DataFrame, str]:
    design = pd.read_csv(cfg.labels_dir / "design.csv", dtype={"id": str})
    if labels is None:
        labels, source = final_labels(cfg)
    lab = design.merge(labels, on="id", how="left")
    pop = pd.read_csv(cfg.stage1_dir / "population.csv")
    grp = (
        lab.groupby(["era", "d_tag", "side"])
        .agg(drawn=("id", "size"), n=("discipline", "count"), k=("discipline", "sum"))
        .reset_index()
    )
    t = pop.merge(grp, on=["era", "d_tag", "side"], how="left").fillna({"drawn": 0, "n": 0, "k": 0})
    t[["drawn", "n", "k"]] = t[["drawn", "n", "k"]].astype(int)
    t["missing_labels"] = (t["N"] > 0) & (t["n"] == 0)
    return t, source


def draw_theta(t: pd.DataFrame, draws: int, prior: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = t["k"].to_numpy() + prior
    b = t["n"].to_numpy() - t["k"].to_numpy() + prior
    return rng.beta(a[None, :], b[None, :], size=(draws, len(t)))


def estimate(cfg: Config) -> dict:
    est = cfg.estimate
    t, source = stratum_table(cfg)
    den = pd.read_csv(cfg.stage1_dir / "denominators.csv")
    m_side = den.set_index(["era", "side"])["members"]
    m_era = den.groupby("era")["members"].sum()
    theta = draw_theta(t, int(est["draws"]), float(est["prior"]), int(est["seed"]))
    weighted = theta * t["N"].to_numpy()[None, :]
    tagged = t["d_tag"].to_numpy() == 1

    eras = [e for e in cfg.era_names if t.loc[t["era"] == e, "N"].sum() > 0]
    per_era, draws_by_era = [], {}
    for e in eras:
        in_e = (t["era"] == e).to_numpy()
        T = weighted[:, in_e].sum(axis=1)
        TP = weighted[:, in_e & tagged].sum(axis=1)
        D = float(t.loc[in_e & tagged, "N"].sum())
        M = float(m_era[e])
        P = TP / D
        R = TP / T
        draws_by_era[e] = {"T": T, "P": P, "R": R, "rate": 1000 * T / M, "naive": 1000 * D / M}
        row = {
            "era": e,
            "partial": any(x.partial for x in cfg.eras if x.name == e),
            "member_speeches": int(M),
            "chair_speeches": int(t.loc[in_e, "N"].sum()),
            "tagged": int(D),
            "labelled": int(t.loc[in_e, "n"].sum()),
            "naive_rate": 1000 * D / M,
        }
        for key, val in (("interventions", T), ("rate", 1000 * T / M), ("precision", P), ("recall", R)):
            for stat, v in _q(val).items():
                row[f"{key}_{stat}"] = v
        per_era.append(row)
    per_era = pd.DataFrame(per_era)

    # Probability that each era had the lowest / the highest true rate.
    rates = np.column_stack([draws_by_era[e]["rate"] for e in eras])
    full = [e for e in eras if not any(x.partial for x in cfg.eras if x.name == e)]
    fi = [eras.index(e) for e in full]
    lowest = np.bincount(np.argmin(rates[:, fi], axis=1), minlength=len(fi)) / rates.shape[0]
    highest = np.bincount(np.argmax(rates[:, fi], axis=1), minlength=len(fi)) / rates.shape[0]
    naive_rank = per_era.set_index("era").loc[full, "naive_rate"]
    rank_tab = pd.DataFrame(
        {
            "era": full,
            "naive_rate": naive_rank.to_numpy(),
            "p_lowest": lowest,
            "p_highest": highest,
        }
    )
    per_era = per_era.merge(rank_tab[["era", "p_lowest", "p_highest"]], on="era", how="left")

    # Decomposition of the naive change against the reference era.
    ref = est["reference_era"]
    dec_rows = []
    r = draws_by_era[ref]
    for e in eras:
        x = draws_by_era[e]
        naive_log = float(np.log(x["naive"] / r["naive"]))
        true_log = np.log(x["rate"] / r["rate"])
        recall_log = np.log(x["R"] / r["R"])
        precision_log = -np.log(x["P"] / r["P"])
        resid = naive_log - (true_log + recall_log + precision_log)
        row = {
            "era": e,
            "reference": ref,
            "naive_log_ratio": naive_log,
            "max_abs_identity_error": float(np.abs(resid).max()),
        }
        for key, val in (("true", true_log), ("recall", recall_log), ("precision", precision_log)):
            for stat, v in _q(val).items():
                row[f"{key}_{stat}"] = v
        row["naive_ratio"] = float(np.exp(naive_log))
        for stat, v in _q(np.exp(true_log)).items():
            row[f"true_ratio_{stat}"] = v
        dec_rows.append(row)
    decomposition = pd.DataFrame(dec_rows)

    # Government and Opposition: the gap a dictionary study reports, and the corrected gap.
    side_rows = []
    for e in eras:
        res = {"era": e}
        for side in ("Government", "Opposition"):
            sel = ((t["era"] == e) & (t["side"] == side)).to_numpy()
            M = float(m_side.get((e, side), 0))
            D = float(t.loc[sel & tagged, "N"].sum())
            T = weighted[:, sel].sum(axis=1)
            TP = weighted[:, sel & tagged].sum(axis=1)
            res[f"{side}_members"] = int(M)
            res[f"{side}_naive"] = 1000 * D / M if M else np.nan
            rate = 1000 * T / M if M else np.full(T.shape, np.nan)
            res[f"_{side}_rate"] = rate
            for stat, v in _q(rate).items():
                res[f"{side}_rate_{stat}"] = v
            for stat, v in _q(TP / D if D else np.full(T.shape, np.nan)).items():
                res[f"{side}_precision_{stat}"] = v
            for stat, v in _q(TP / T).items():
                res[f"{side}_recall_{stat}"] = v
        res["naive_ratio_opp_gov"] = res["Opposition_naive"] / res["Government_naive"]
        ratio = res.pop("_Opposition_rate") / res.pop("_Government_rate")
        for stat, v in _q(ratio).items():
            res[f"ratio_opp_gov_{stat}"] = v
        res["p_ratio_above_1"] = float(np.mean(ratio > 1))
        side_rows.append(res)
    sides = pd.DataFrame(side_rows)

    # The whole period, as a single headline number.
    all_T = weighted.sum(axis=1)
    totals = {
        "tagged": int(t.loc[tagged, "N"].sum()),
        "interventions": _q(all_T),
        "precision": _q(weighted[:, tagged].sum(axis=1) / t.loc[tagged, "N"].sum()),
        "recall": _q(weighted[:, tagged].sum(axis=1) / all_T),
    }

    out = cfg.stage2_dir
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "strata.csv", index=False)
    per_era.to_csv(out / "corrected_by_era.csv", index=False)
    decomposition.to_csv(out / "decomposition.csv", index=False)
    sides.to_csv(out / "sides.csv", index=False)
    summary = {
        "label_source": source,
        "labelled_items": int(t["n"].sum()),
        "strata_without_labels": t.loc[t["missing_labels"], ["era", "d_tag", "side", "N"]].to_dict("records"),
        "draws": int(est["draws"]),
        "prior": float(est["prior"]),
        "reference_era": ref,
        "totals": totals,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def annotator_sensitivity(cfg: Config) -> pd.DataFrame:
    """The corrected numbers under each annotator's labels alone, next to the panel's.

    If the conclusions hold under every annotator, they do not hinge on whose reading of
    the codebook is taken as the reference.
    """
    from .labels import annotator_files, read_annotator

    est = cfg.estimate
    den = pd.read_csv(cfg.stage1_dir / "denominators.csv")
    m_era = den.groupby("era")["members"].sum()
    sets = [("panel", None)]
    for f in annotator_files(cfg):
        df = read_annotator(f)
        df = df[df["discipline"].astype(str).isin(["0", "1"])].copy()
        df["discipline"] = df["discipline"].astype(int)
        sets.append((f.stem.removeprefix("labels_"), df[["id", "discipline"]]))
    rows = []
    for name, labels in sets:
        t, _ = stratum_table(cfg, labels, name)
        theta = draw_theta(t, int(est["draws"]), float(est["prior"]), int(est["seed"]))
        weighted = theta * t["N"].to_numpy()[None, :]
        tagged = t["d_tag"].to_numpy() == 1
        ref = None
        for e in [x for x in cfg.era_names if t.loc[t["era"] == x, "N"].sum() > 0]:
            in_e = (t["era"] == e).to_numpy()
            T = weighted[:, in_e].sum(axis=1)
            TP = weighted[:, in_e & tagged].sum(axis=1)
            D = float(t.loc[in_e & tagged, "N"].sum())
            rate = 1000 * T / float(m_era[e])
            if e == est["reference_era"]:
                ref = rate
            row = {"labels": name, "era": e, "labelled": int(t.loc[in_e, "n"].sum())}
            for key, val in (("rate", rate), ("precision", TP / D), ("recall", TP / T)):
                for stat, v in _q(val).items():
                    row[f"{key}_{stat}"] = v
            row["_rate"] = rate
            rows.append(row)
        for r in rows:
            if r["labels"] == name and ref is not None:
                for stat, v in _q(r["_rate"] / ref).items():
                    r[f"ratio_to_reference_{stat}"] = v
    out = pd.DataFrame(rows).drop(columns="_rate")
    cfg.stage2_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.stage2_dir / "annotator_sensitivity.csv", index=False)
    return out
