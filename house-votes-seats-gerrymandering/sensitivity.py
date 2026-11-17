"""Sensitivity of the causal estimates to choices made upstream.

  * the share imputed to unopposed winners (0.70, 0.75, 0.80)
  * the minimum delegation size (5 seats, 8 seats)
  * leave one state out of the post-1992 event study
Writes results/sensitivity.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import causal
import state_panel

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DER = HERE / "data" / "derived"


def panel_with_share(share: float) -> pd.DataFrame:
    d = pd.read_csv(DER / "house_districts_raw_1972_2024.csv")
    d = state_panel.impute_districts(d, share=share)
    rows = state_panel.state_rows_1972_2024(d)
    base = causal.load()
    m = base.drop(columns=["bias_us", "y_us"]).merge(
        rows[["year", "state_po", "bias_us"]], on=["year", "state_po"], how="left")
    m["y_us"] = 100 * m["bias_us"]
    return m


def main() -> dict:
    out = {}
    for share in [0.70, 0.75, 0.80]:
        p = panel_with_share(share)
        p = p[p.year <= 2024]
        t = causal.twfe(p[p.year >= 2002], "y_us", f"share {share}")
        ev = causal.stacked_events(p, "y_us")
        e = causal.pooled_post(ev[ev.t0 >= 1992])
        out[f"impute_{share}"] = dict(twfe_post2000_R=t["R"], twfe_post2000_D=t["D"],
                                      event_post1992_toward_R=e["dk_neg"]["coef"],
                                      event_post1992_toward_D=e["dk_pos"]["coef"])
    base = causal.load()
    base = base[base.year <= 2024]
    causal.MIN_SEATS = 8
    t8 = causal.twfe(base[base.year >= 2002], "y_us", "8+ seats")
    ev8 = causal.stacked_events(base, "y_us")
    e8 = causal.pooled_post(ev8[ev8.t0 >= 1992])
    out["min_seats_8"] = dict(twfe_post2000_R=t8["R"], twfe_post2000_D=t8["D"],
                              event_post1992_toward_R=e8["dk_neg"]["coef"],
                              event_post1992_toward_D=e8["dk_pos"]["coef"], n=t8["n"])
    causal.MIN_SEATS = 5
    ev = causal.stacked_events(base, "y_us")
    ev = ev[ev.t0 >= 1992]
    loo = []
    for st in sorted(ev.state_po.unique()):
        e = causal.pooled_post(ev[ev.state_po != st])
        loo.append(dict(state=st, toward_R=e["dk_neg"]["coef"], toward_D=e["dk_pos"]["coef"]))
    loo = pd.DataFrame(loo)
    out["leave_one_state_out_post1992"] = dict(
        toward_R_min=float(loo.toward_R.min()), toward_R_max=float(loo.toward_R.max()),
        toward_D_min=float(loo.toward_D.min()), toward_D_max=float(loo.toward_D.max()),
        toward_D_min_state=loo.loc[loo.toward_D.idxmin(), "state"],
        toward_R_min_state=loo.loc[loo.toward_R.idxmin(), "state"])
    (RES / "sensitivity.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print(json.dumps(main(), indent=1, default=float))
