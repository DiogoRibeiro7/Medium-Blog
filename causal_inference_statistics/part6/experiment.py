"""Reproducible Difference-in-Differences identification experiment (Python 3.12+)."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

SEED: int = 20261016
REPLICATIONS: int = 500
N_PER_GROUP: int = 500
TRUE_EFFECT: float = -1.0
SCENARIOS: tuple[str, ...] = ('parallel', 'differential_post_trend', 'anticipation')


def difference_in_means(values: np.ndarray, treated: np.ndarray) -> tuple[float, float]:
    """Return difference of means and its heteroskedasticity-robust SE."""
    if values.ndim != 1 or treated.shape != values.shape or not np.isfinite(values).all():
        raise ValueError('values and treatment indicator must be equal-length finite vectors')
    a, b = values[treated], values[~treated]
    return float(a.mean()-b.mean()), float(np.sqrt(a.var(ddof=1)/a.size+b.var(ddof=1)/b.size))


def simulate(rng: np.random.Generator, scenario: str) -> tuple[float,float,float,float]:
    """Evaluate the post-period DID and a disjoint pre-period placebo."""
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario')
    n=2*N_PER_GROUP
    group=np.r_[np.zeros(N_PER_GROUP,dtype=bool),np.ones(N_PER_GROUP,dtype=bool)]
    individual_effect=rng.normal(size=n)
    errors=rng.normal(size=(n,4))
    time=np.array([0.,0.2,0.4,0.6])
    y=individual_effect[:,None]+time[None,:]+errors
    y[group,:]+=1.0  # permanent selection on levels, removed by DID
    if scenario=='differential_post_trend':
        y[group,3]+=0.75  # untreated counterfactual deviates only after intervention
    if scenario=='anticipation':
        y[group,2]+=-0.4  # pre-program response at baseline
    y[group,3]+=TRUE_EFFECT
    did, se=difference_in_means(y[:,3]-y[:,2],group)
    placebo,placebo_se=difference_in_means(y[:,1]-y[:,0],group)
    return did,se,placebo,placebo_se


def main()->None:
    """Produce Monte Carlo summary of effect, SE, coverage and pretrend tests."""
    rng=np.random.default_rng(SEED)
    records=[]
    for scenario in SCENARIOS:
        a=np.array([simulate(rng,scenario) for _ in range(REPLICATIONS)])
        estimate,se,pre,pre_se=a.T
        records.append(dict(scenario=scenario,replications=REPLICATIONS,n=2*N_PER_GROUP,
          mean_did=float(estimate.mean()),bias=float(estimate.mean()-TRUE_EFFECT),
          sd_did=float(estimate.std(ddof=1)),mean_se=float(se.mean()),
          coverage=float(np.mean(np.abs(estimate-TRUE_EFFECT)<=1.96*se)),
          mean_placebo=float(pre.mean()),placebo_rejection=float(np.mean(np.abs(pre)>1.96*pre_se))))
    path=Path(__file__).with_name('results.csv')
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for r in records:print(r)


if __name__=='__main__':main()
