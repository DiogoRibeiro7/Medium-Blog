"""What would the old maps have produced in 2026?

For every state with two or more seats, the 2024 district results (old map)
are moved by the state's 2024-to-2026 swing in the two-party statewide House
vote, and the districts the Democrats would then carry are counted. The
difference between the seats Democrats actually won in 2026 and that count
is, in states that kept their map, the error of the uniform-swing projection;
in the nine states with a new map it is the projection error plus the effect
of the new lines. The redraw effect is the mean gap in redrawn states minus
the mean gap in states that kept their maps, with a permutation p-value.

The same projection run from 2016 to 2018 and from 2022 to 2024, for states
whose maps did not change, calibrates how far uniform swing misses in a
wave and in a normal year.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DER = HERE / "data" / "derived"
RES = HERE / "results"
RNG = np.random.default_rng(2026)
REDRAWN_2026 = {"TX": "R", "FL": "R", "NC": "R", "OH": "R", "TN": "R", "LA": "R",
                "AL": "R", "CA": "D", "UT": "Court"}
# states whose map changed between the base and the target year (excluded as controls)
CHANGED = {(2016, 2018): {"PA"}, (2022, 2024): {"AL", "LA", "NC", "NY", "GA"},
           (2024, 2026): set(REDRAWN_2026)}


def project(base_year: int, target_year: int) -> pd.DataFrame:
    dist = pd.read_csv(DER / "house_districts_imputed_1972_2024.csv")
    panel = pd.read_csv(DER / "state_panel.csv").set_index(["state_po", "year"])
    rows = []
    for st, g in dist[dist.year == base_year].groupby("state_po"):
        if (st, target_year) not in panel.index or len(g) < 2:
            continue
        v0 = panel.loc[(st, base_year), "v_raw"]
        v1 = panel.loc[(st, target_year), "v_raw"]
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        shares = g["v_imp_d"].dropna().to_numpy()
        if len(shares) != len(g):
            continue  # an independent-held seat; skip the state
        swing = v1 - v0
        pred = int(np.sum(shares + swing > 0.5))
        actual = int(panel.loc[(st, target_year), "dem_seats"])
        seats = int(panel.loc[(st, target_year), "dem_seats"] + panel.loc[(st, target_year), "rep_seats"])
        if seats != len(g):
            continue  # apportionment changed; not comparable
        status = ("redrawn" if st in CHANGED.get((base_year, target_year), set()) else "kept")
        rows.append(dict(state_po=st, base=base_year, target=target_year, seats=seats,
                         v_base=v0, v_target=v1, swing=swing, dem_seats_base=int((g.winner == "D").sum()),
                         projected_old_map=pred, actual=actual, gap=actual - pred, status=status,
                         drawer_new=REDRAWN_2026.get(st) if target_year == 2026 else None))
    return pd.DataFrame(rows)


def perm_test(d: pd.DataFrame, treated_mask: np.ndarray, n: int = 50000) -> tuple[float, float]:
    gaps = d["gap"].to_numpy(dtype=float)
    obs = gaps[treated_mask].mean() - gaps[~treated_mask].mean()
    k = int(treated_mask.sum())
    sims = np.empty(n)
    for i in range(n):
        idx = RNG.choice(len(gaps), size=k, replace=False)
        m = np.zeros(len(gaps), bool)
        m[idx] = True
        sims[i] = gaps[m].mean() - gaps[~m].mean()
    return float(obs), float(np.mean(np.abs(sims) >= abs(obs)))


def main() -> dict:
    out = {}
    cal = []
    for base, target in [(2016, 2018), (2022, 2024)]:
        d = project(base, target)
        d = d[d.status == "kept"]
        cal.append(d)
        out[f"calibration_{base}_{target}"] = dict(
            states=len(d), mean_gap=float(d.gap.mean()), mean_abs_gap=float(d.gap.abs().mean()),
            sd_gap=float(d.gap.std()), sum_gap=int(d.gap.sum()))
    pd.concat(cal).to_csv(RES / "old_map_projection_calibration.csv", index=False)

    d = project(2024, 2026)
    d.to_csv(RES / "old_map_projection_2026.csv", index=False)
    kept = d[d.status == "kept"]
    r_mask = d["drawer_new"].eq("R").to_numpy()
    pool = d[(d.status == "kept") | d["drawer_new"].eq("R")].reset_index(drop=True)
    obs, p = perm_test(pool, pool["drawer_new"].eq("R").to_numpy())
    out["redraw_2026"] = dict(
        states=len(d), kept=len(kept), kept_mean_gap=float(kept.gap.mean()),
        kept_sd_gap=float(kept.gap.std()), kept_sum_gap=int(kept.gap.sum()),
        r_states=d.loc[r_mask, "state_po"].tolist(),
        r_sum_gap=int(d.loc[r_mask, "gap"].sum()),
        r_mean_gap=float(d.loc[r_mask, "gap"].mean()),
        r_minus_kept_mean=obs, perm_p=p,
        r_effect_seats_total=float(d.loc[r_mask, "gap"].sum() - kept.gap.mean() * r_mask.sum()),
        ca_gap=int(d.loc[d.state_po == "CA", "gap"].iloc[0]) if (d.state_po == "CA").any() else None,
        ut_gap=int(d.loc[d.state_po == "UT", "gap"].iloc[0]) if (d.state_po == "UT").any() else None,
        by_state=d.round(4).to_dict(orient="records"),
    )
    (RES / "counterfactual_2026.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    o = main()
    for k, v in o.items():
        print(k, {a: b for a, b in v.items() if a != "by_state"})
    print(pd.DataFrame(o["redraw_2026"]["by_state"]).sort_values(["status", "gap"]).to_string())
