"""Longitudinal MSM/IPW versus g-computation benchmark. Python 3.12+."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, LinearRegression

SEED: int = 20261018
REPS: int = 160
N: int = 2200
TRUE_TOTAL: float = 3.36

def simulate(rng: np.random.Generator) -> dict[str, float]:
    """Compute sustained (1,1) vs (0,0) contrast from one simulated panel."""
    l0 = rng.standard_normal(N)
    a0 = rng.binomial(1, expit(0.8*l0))
    l1 = 0.8*l0 + 1.2*a0 + rng.standard_normal(N)
    a1 = rng.binomial(1, expit(0.8*l1))
    y = 1 + 0.5*l0 + 1.3*l1 + 0.7*a0 + 1.1*a1 + rng.standard_normal(N)
    p0 = LogisticRegression(C=1e6, max_iter=250).fit(l0[:,None], a0).predict_proba(l0[:,None])[:,1]
    x1 = np.column_stack((l0,a0,l1))
    p1 = LogisticRegression(C=1e6, max_iter=250).fit(x1,a1).predict_proba(x1)[:,1]
    p0 = np.clip(p0,0.01,0.99); p1 = np.clip(p1,0.01,0.99)
    joint = np.where(a0==1,p0,1-p0)*np.where(a1==1,p1,1-p1)
    w = 1/joint
    one = (a0==1)&(a1==1); zero = (a0==0)&(a1==0)
    ipw = float(np.average(y[one],weights=w[one])-np.average(y[zero],weights=w[zero]))
    outcome = LinearRegression().fit(np.column_stack((l0,a0,l1,a1)),y)
    evolution = LinearRegression().fit(np.column_stack((l0,a0)),l1)
    def g(a: int)->float:
        new_l1 = evolution.predict(np.column_stack((l0,np.full(N,a))))
        return float(outcome.predict(np.column_stack((l0,np.full(N,a),new_l1,np.full(N,a)))).mean())
    ess = lambda ww: float(ww.sum()**2/np.square(ww).sum())
    return dict(msm_ipw=ipw,g_formula=g(1)-g(0),
        direct_adjustment=float(outcome.coef_[1]+outcome.coef_[3]),
        ess_11=ess(w[one]),ess_00=ess(w[zero]))

def main()->None:
    """Save Monte Carlo effect errors and effective sample sizes."""
    rng = np.random.default_rng(SEED)
    observations = [simulate(rng) for _ in range(REPS)]
    rows = []
    for key in observations[0]:
        v = np.array([r[key] for r in observations])
        effect = key in ("msm_ipw","g_formula","direct_adjustment")
        rows.append(dict(metric=key,n=N,replications=REPS,
            true_total_effect=TRUE_TOTAL if effect else "",
            mean=float(v.mean()),sd=float(v.std(ddof=1)),
            bias=float(v.mean()-TRUE_TOTAL) if effect else "",
            rmse=float(np.sqrt(np.mean((v-TRUE_TOTAL)**2))) if effect else ""))
    with Path(__file__).with_name("results.csv").open("w",newline="",encoding="utf-8") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for row in rows: print(row)

if __name__=="__main__":main()
