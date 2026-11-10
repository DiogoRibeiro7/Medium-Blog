"""Run every scheme for every model family and distribution on one shard of one domain.

    python run.py --domain web --shard 3 --of 16 --workers 4

For each series: window decisions every 91 days with the SLM design (Gaussian and Student-t
decisions), then for every family and every window needed (all window lengths L, 2L, ... and all
data) a fit at each re-selection origin and one-step forecasts for the next 91 days. Writes
``results/shards/<domain>_<shard>_periods.csv.gz`` (sum of log losses and day counts per series,
family, distribution, scheme and 91-day period) and ``..._decisions.csv``.
"""

from __future__ import annotations

import argparse
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

import common
from backward_eval import ARGaussian, ARStudentT
from episode import select
from models import FAMILIES, T_DF, design, fit_predict

SCHEMES = ("E", "B", "C0", "C1645", "P0.25", "AveW")
DISTS = ("gauss", "t")


def _dens(z, m, s, dist):
    w = (z - m) / s
    return (norm.pdf(w) if dist == "gauss" else student_t.pdf(w, T_DF)) / s


def run_series(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain, key, s = args
    D = design(s.to_numpy(), s.index)
    n = len(D.z)
    L = common.BLOCK
    first = 2 * L + common.GAP
    origins = np.arange(first, n)
    seed = zlib.crc32(key.encode()) % (2**31)
    sel = {}
    for dist in DISTS:
        factory = ARGaussian if dist == "gauss" else (lambda: ARStudentT(df=T_DF))
        sel[dist] = select(D.X_slm, D.z, origins, factory, L, common.MAX_BACK, common.Z_CRIT, common.GAP,
                           common.RESELECT, common.FOLDS, seed, rs=(common.R_CHECK,))
    recs = []
    for p0 in range(0, len(origins), common.RESELECT):
        o = int(origins[p0])
        rows = origins[p0 : p0 + common.RESELECT]
        zr = D.z[rows]
        starts = sorted({o - k * L for k in range(1, o // L + 1)} | {0})
        choice = {}
        for dist in DISTS:
            sl = sel[dist]
            sB, stopped = int(sl.start_B[p0]), bool(sl.stopped[p0])
            choice[dist] = dict(B=sB, stopped=stopped and sB > 0,
                                C0=int(sl.start_C(common.R_CHECK, common.C_PRIMARY)[p0]),
                                C1645=int(sl.start_C(common.R_CHECK, common.C_SECONDARY)[p0]))
        for fam in FAMILIES:
            fits = {a: fit_predict(fam, D, a, o, rows) for a in starts}
            for dist in DISTS:
                dens = {a: _dens(zr, f.mean if dist == "gauss" else f.mean_t, f.sd_g if dist == "gauss" else f.sd_t, dist)
                        for a, f in fits.items()}
                ch = choice[dist]
                dE, dB = dens[0], dens[ch["B"]]
                out = {"E": dE, "B": dB, "C0": dens[ch["C0"]], "C1645": dens[ch["C1645"]],
                       "P0.25": 0.25 * dB + 0.75 * dE if ch["stopped"] else dE,
                       "AveW": np.mean([dens[a] for a in starts], axis=0)}
                for k, d in out.items():
                    ll = -np.log(np.maximum(d, 1e-300))
                    recs.append((domain, key, fam, dist, k, f"{D.dates[o]:%Y-%m-%d}", len(rows), float(ll.sum()),
                                 float(ll[(D.dates[rows] < pd.Timestamp(common.COVID_SCORING[0])) |
                                          (D.dates[rows] > pd.Timestamp(common.COVID_SCORING[1]))].sum()),
                                 int(((D.dates[rows] < pd.Timestamp(common.COVID_SCORING[0])) |
                                      (D.dates[rows] > pd.Timestamp(common.COVID_SCORING[1]))).sum())))
    per = pd.DataFrame(recs, columns=["domain", "series", "family", "dist", "scheme", "period", "n", "sum_ll",
                                      "sum_ll_nocovid", "n_nocovid"])
    dec = []
    for dist in DISTS:
        for d in sel[dist].decisions:
            st = d.stopped_at is not None
            ze = d.z_end[common.R_CHECK]
            dec.append({"domain": domain, "series": key, "dist": dist, "date": f"{D.dates[d.origin]:%Y-%m-%d}",
                        "stopped_at": d.stopped_at if st else 0, "z_star": d.z_star if st else np.nan, "z_end": ze,
                        "switch_c0": bool(st and ze > common.C_PRIMARY),
                        "switch_c1645": bool(st and ze > common.C_SECONDARY)})
    return per, pd.DataFrame(dec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=common.DOMAINS, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="first N series of the shard only (testing)")
    args = ap.parse_args()
    data = common.load(args.domain)
    keys = sorted(data)[args.shard :: args.of]
    if args.limit:
        keys = keys[: args.limit]
    tasks = [(args.domain, k, data[k]) for k in keys]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(run_series, tasks))
    out = common.OUT / "shards"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.domain}_{args.shard:02d}"
    pd.concat([r[0] for r in res], ignore_index=True).to_csv(out / f"{tag}_periods.csv.gz", index=False,
                                                           float_format="%.6f", compression={"method": "gzip", "mtime": 0})
    pd.concat([r[1] for r in res], ignore_index=True).to_csv(out / f"{tag}_decisions.csv", index=False, float_format="%.4f")
    print(f"{tag}: {len(keys)} series")


if __name__ == "__main__":
    main()
