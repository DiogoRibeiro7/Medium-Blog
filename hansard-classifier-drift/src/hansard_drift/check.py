"""Recompute key numbers by a second route and fail on any mismatch."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config


def check(cfg: Config) -> list[str]:
    problems: list[str] = []
    s1 = cfg.stage1_dir
    man = json.loads((s1 / "manifest.json").read_text(encoding="utf-8"))
    pop = pd.read_csv(s1 / "population.csv")
    den = pd.read_csv(s1 / "denominators.csv")
    naive = pd.read_csv(s1 / "naive_rates.csv")

    if int(pop["N"].sum()) != man["chair_speeches"]:
        problems.append(f"population sums to {pop['N'].sum()}, manifest says {man['chair_speeches']} Chair speeches")
    if int(pop.loc[pop["d_tag"] == 1, "N"].sum()) != man["discipline_tagged"]:
        problems.append("tagged population does not match the manifest")
    if int(den["members"].sum()) != man["member_speeches"]:
        problems.append("denominators do not sum to the Member speeches in the manifest")

    # Naive rates from the population table and the denominators.
    for era in cfg.era_names:
        tagged = pop[(pop["era"] == era) & (pop["d_tag"] == 1)]["N"].sum()
        members = den[den["era"] == era]["members"].sum()
        row = naive[(naive["construct"] == "discipline") & (naive["era"] == era) & (naive["side"] == "All")]
        if members and not row.empty and not np.isclose(row["rate_per_1000"].iloc[0], 1000 * tagged / members):
            problems.append(f"naive discipline rate for {era} does not match population / denominators")

    design_path = cfg.labels_dir / "design.csv"
    if design_path.exists():
        design = pd.read_csv(design_path, dtype={"id": str})
        if design["id"].duplicated().any():
            problems.append("duplicate ids in the validation sample")
        m = (
            design.groupby(["era", "d_tag", "side"])["N"]
            .first()
            .reset_index()
            .merge(pop, on=["era", "d_tag", "side"], suffixes=("_design", "_pop"))
        )
        bad = m[m["N_design"] != m["N_pop"]]
        if len(bad):
            problems.append(f"{len(bad)} strata where the design N differs from the population (sample is stale)")
        sample = pd.read_csv(cfg.labels_dir / "sample.csv", dtype=str, keep_default_na=False)
        if set(sample["id"]) != set(design["id"]):
            problems.append("sample.csv and design.csv list different speeches")

    s2 = cfg.stage2_dir
    if (s2 / "decomposition.csv").exists():
        dec = pd.read_csv(s2 / "decomposition.csv")
        worst = float(dec["max_abs_identity_error"].max())
        if worst > 1e-8:
            problems.append(f"decomposition identity off by {worst:g}")
        ce = pd.read_csv(s2 / "corrected_by_era.csv")
        for _, r in ce.iterrows():
            if not (r["precision_lo"] <= r["precision_median"] <= r["precision_hi"]):
                problems.append(f"precision interval out of order for {r['era']}")
            nr = naive[(naive["construct"] == "discipline") & (naive["era"] == r["era"]) & (naive["side"] == "All")]
            if not np.isclose(nr["rate_per_1000"].iloc[0], r["naive_rate"]):
                problems.append(f"stage 2 naive rate for {r['era']} differs from stage 1")
    return problems
