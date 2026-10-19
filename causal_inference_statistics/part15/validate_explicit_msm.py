"""Validate a saturated longitudinal MSM against normalized regime IPW."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

SEED: int = 20261018
REPS: int = 160
N: int = 2200
TRUE: float = 3.36

def replicate(rng: np.random.Generator) -> tuple[float,float,float]:
    """Reproduce Part 15 data and compare algebraically equivalent estimates."""
    l0=rng.standard_normal(N)
    a0=rng.binomial(1,expit(.8*l0))
    l1=.8*l0+1.2*a0+rng.standard_normal(N)
    a1=rng.binomial(1,expit(.8*l1))
    y=1+.5*l0+1.3*l1+.7*a0+1.1*a1+rng.standard_normal(N)
    p0=LogisticRegression(C=1e6,max_iter=250).fit(l0[:,None],a0).predict_proba(l0[:,None])[:,1]
    p1=LogisticRegression(C=1e6,max_iter=250).fit(np.column_stack((l0,a0,l1)),a1).predict_proba(np.column_stack((l0,a0,l1)))[:,1]
    p0,p1=np.clip(p0,.01,.99),np.clip(p1,.01,.99)
    w=1/(np.where(a0==1,p0,1-p0)*np.where(a1==1,p1,1-p1))
    one=(a0==1)&(a1==1);zero=(a0==0)&(a1==0)
    ipw=float(np.average(y[one],weights=w[one])-np.average(y[zero],weights=w[zero]))
    x=np.column_stack((np.ones(N),a0,a1,a0*a1))
    sw=np.sqrt(w)
    beta=np.linalg.lstsq(x*sw[:,None],y*sw,rcond=None)[0]
    msm=float(beta[1]+beta[2]+beta[3])
    return ipw,msm,float(msm-ipw)

def main()->None:
    """Assert equivalence over all Monte Carlo repetitions."""
    rng=np.random.default_rng(SEED)
    z=np.asarray([replicate(rng) for _ in range(REPS)])
    if not np.allclose(z[:,0],z[:,1],rtol=0,atol=1e-10):
        raise AssertionError("IPW does not equal saturated MSM")
    rows=[]
    for i,name in enumerate(("regime_ipw","saturated_msm","difference")):
        a=z[:,i]
        rows.append(dict(metric=name,n=N,replications=REPS,mean=float(a.mean()),sd=float(a.std(ddof=1)),max_abs=float(np.max(np.abs(a))),rmse_to_truth=float(np.sqrt(np.mean((a-TRUE)**2))) if i<2 else ""))
    with Path(__file__).with_name("msm_validation_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:print(row)

if __name__=="__main__":main()
