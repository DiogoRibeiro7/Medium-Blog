"""Negative controls and placebo tests under measured and hidden confounding.

Python >=3.12. Dependencies: NumPy. Generates deterministic Monte Carlo CSV.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

SEED: int = 20261018
REPS: int = 500
N: int = 1200
TRUE_EFFECT: float = 1.0
SCENARIOS: tuple[str,...] = ('no_confounding','shared_confounder','outcome_specific_confounder')

def coefficient(y: NDArray[np.float64], x: NDArray[np.float64]) -> tuple[float,float]:
    """OLS slope and classical estimated SE with intercept."""
    if y.ndim != 1 or x.shape != y.shape or y.size < 10:
        raise ValueError('Incompatible samples')
    design=np.column_stack((np.ones(y.size),x))
    beta=np.linalg.lstsq(design,y,rcond=None)[0]
    e=y-design@beta
    se=np.sqrt((e@e/(y.size-2))*np.linalg.inv(design.T@design)[1,1])
    return float(beta[1]),float(se)

def simulate(rng:np.random.Generator, scenario:str)->tuple[float,float,float,float,float,float]:
    """Return target and placebo coefficients, SEs, and negative-control results."""
    if scenario not in SCENARIOS:
        raise ValueError('Unknown scenario')
    u,v,ex,ey,en,ep=rng.normal(size=(6,N))
    # Third scenario: V confounds treatment and target outcome,
    # but the negative-control outcome is linked only to U.
    if scenario == 'no_confounding':
        a=ex
        y=TRUE_EFFECT*a+u+v+ey
    elif scenario == 'shared_confounder':
        a=u+ex
        y=TRUE_EFFECT*a+1.5*u+v+ey
    else:
        a=v+ex
        y=TRUE_EFFECT*a+u+1.5*v+ey
    negative_outcome=u+en
    placebo_exposure=ep
    b,se=coefficient(y,a)
    nc,ncse=coefficient(negative_outcome,a)
    place,pse=coefficient(y,placebo_exposure)
    return b,se,nc,ncse,place,pse

def main()->None:
    """Aggregate simulations, write CSV, and print benchmark."""
    rng=np.random.default_rng(SEED)
    rows=[]
    for scenario in SCENARIOS:
        arr=np.asarray([simulate(rng,scenario) for _ in range(REPS)])
        for label,col,true in (('target',0,TRUE_EFFECT),('negative_control_outcome',2,0.),('placebo_exposure',4,0.)):
            estimate=arr[:,col]; se=arr[:,col+1]
            rows.append(dict(scenario=scenario,test=label,n=N,replications=REPS,true_effect=true,
                mean=float(estimate.mean()),bias=float(estimate.mean()-true),
                rmse=float(np.sqrt(np.mean((estimate-true)**2))),
                mean_se=float(se.mean()),coverage=float(np.mean(np.abs(estimate-true)<=1.96*se)),
                reject_zero=float(np.mean(np.abs(estimate)>1.96*se))))
    output=Path(__file__).with_name('results.csv')
    with output.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for row in rows:print(row)

if __name__=='__main__':main()
