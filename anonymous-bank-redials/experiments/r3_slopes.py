"""R3. The interval statistics on the real table.

Slopes of the surplus of offered calls on abandonments at leads and lags of
one to three half hours, for all offered calls, for offered calls minus the
redials identified through customer IDs, and for the identified redials
alone; the controlled slope; the clustering statistics. Day-cluster
bootstrap standard deviations.
"""

from __future__ import annotations

import numpy as np

from common import load_table, parse, real_statistics, rng, save

LAGS = (-3, -2, -1, 1, 2, 3)


def main() -> None:
    args = parse(__doc__)
    reps = 100 if args.quick else 1000
    tab = load_table()
    out = {}
    for col in ("offered", "non_redials", "id_redials"):
        st = real_statistics(tab, col)
        g = rng("r3", len(col))
        idx = g.integers(0, st.n_days, size=(reps, st.n_days))
        res = {}
        for L in LAGS:
            boot = np.array([st.lag(L, i) for i in idx])
            res[f"lag_{L}"] = {"value": st.lag(L), "sd": float(boot.std())}
        for name in ("lag_minus_lead", "controlled", "acf1", "dispersion"):
            boot = np.array([st.get(name, i) for i in idx])
            res[name] = {"value": st.get(name), "sd": float(boot.std())}
        out[col] = res
        print(col, {k: (round(v["value"], 3), round(v["sd"], 3))
                    for k, v in res.items()})
    save("r3_slopes", {"series": out, "reps": reps})


if __name__ == "__main__":
    main()
