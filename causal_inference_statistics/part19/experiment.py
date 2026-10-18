"""Monte Carlo investigation of trial-to-target transportability (Python 3.12+)."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

SEED=20261018
REPLICATIONS=400
N_TRIAL=1200
N_TARGET=3000
SCENARIOS=("measured_effect_modifier","omitted_effect_modifier","nonoverlap")

def simulate(rng:np.random.Generator,scenario:str)->tuple[float,float,float,float]:
    """Compute source effect, transported effect, supported-target effect, support fraction."""
    if scenario not in SCENARIOS:
        raise ValueError("Invalid scenario")
    x=rng.normal(0,1,N_TRIAL)
    target_x=rng.normal(1,1,N_TARGET)
    if scenario=="nonoverlap":
        x=rng.uniform(-1,1,N_TRIAL)
        target_x=rng.uniform(.5,2.5,N_TARGET)
    a=rng.binomial(1,.5,N_TRIAL)
    if scenario=="omitted_effect_modifier":
        # Source U ~ N(0,1); target U ~ N(1,1) by design, unobserved in the target.
        # The target E[1 + U] = 2 is an analytical truth, not estimated from U samples.
        u=rng.normal(0,1,N_TRIAL)
        y=.5*x+a*(1+u)+rng.normal(size=N_TRIAL)
    else:
        y=.5*x+a*(1+x)+rng.normal(size=N_TRIAL)
    design=np.column_stack((np.ones(N_TRIAL),x,a,a*x))
    beta=np.linalg.lstsq(design,y,rcond=None)[0]
    trial=float(beta[2]+beta[3]*x.mean())
    transported=float(beta[2]+beta[3]*target_x.mean())
    common=(target_x>=x.min())&(target_x<=x.max())
    if not common.any():
        raise ValueError("No observations within common support")
    restricted=float(beta[2]+beta[3]*target_x[common].mean())
    return trial,transported,restricted,float(common.mean())

def main()->None:
    """Reproduce mean, bias, RMSE and empirical support for distinct estimands."""
    rng=np.random.default_rng(SEED)
    rows=[]
    for scenario in SCENARIOS:
        runs=np.asarray([simulate(rng,scenario) for _ in range(REPLICATIONS)])
        for method in (("trial","transported","restricted") if scenario=="nonoverlap" else ("trial","transported")):
            index=("trial","transported","restricted").index(method)
            vals=runs[:,index]
            target=(1.75 if scenario=="nonoverlap" and method=="restricted"
                else 2.5 if scenario=="nonoverlap" and method=="transported"
                else 1.0 if method=="trial" else 2.0)
            rows.append(dict(scenario=scenario,method=method,n_trial=N_TRIAL,n_target=N_TARGET,
                replications=REPLICATIONS,target=target,mean=float(vals.mean()),
                bias=float(vals.mean()-target),rmse=float(np.sqrt(np.mean((vals-target)**2))),
                sd=float(vals.std(ddof=1)),target_support_fraction=float(runs[:,3].mean())))
    with Path(__file__).with_name("results.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows: print(row)

if __name__=="__main__":
    main()
