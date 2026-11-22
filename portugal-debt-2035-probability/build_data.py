"""Rebuild the annual panel for Portugal from the raw Eurostat JSON-stat responses.

Output: data/panel.csv (not committed; rebuilt at each run).
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "data" / "panel.csv"


def jsonstat_to_frame(path: Path) -> pd.DataFrame:
    js = json.loads(path.read_text(encoding="utf-8"))
    dims = js["id"]
    codes = []
    for d in dims:
        idx = js["dimension"][d]["category"]["index"]
        codes.append(sorted(idx, key=idx.get) if isinstance(idx, dict) else list(idx))
    values = js["value"]
    rows = [
        dict(zip(dims, combo, strict=True), value=values[str(k)])
        for k, combo in enumerate(product(*codes))
        if str(k) in values and values[str(k)] is not None
    ]
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    edp_pc = jsonstat_to_frame(RAW / "edp_pc.json")
    edp_mio = jsonstat_to_frame(RAW / "edp_mio.json")
    gdp = jsonstat_to_frame(RAW / "gdp.json")
    yld = jsonstat_to_frame(RAW / "yld.json")

    a = edp_pc.pivot_table(index="time", columns="na_item", values="value").add_suffix("_pcgdp")
    b = edp_mio.pivot_table(index="time", columns="na_item", values="value").add_suffix("_meur")
    c = gdp.pivot_table(index="time", columns="unit", values="value").rename(
        columns={"CLV_PCH_PRE": "real_growth", "PD_PCH_PRE_EUR": "deflator_growth", "CP_MEUR": "gdp_meur"}
    )
    y = yld.pivot_table(index="time", values="value").rename(columns={"value": "yield_10y"})
    panel = a.join(b, how="outer").join(c, how="outer").join(y, how="outer")
    panel.index = panel.index.astype(int)
    panel.index.name = "year"
    return panel.sort_index().loc[1995:]


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    p = build()
    p.to_csv(OUT)
    print(f"wrote {OUT.relative_to(HERE)}: {p.index.min()}-{p.index.max()}, {p.shape[1]} columns")
