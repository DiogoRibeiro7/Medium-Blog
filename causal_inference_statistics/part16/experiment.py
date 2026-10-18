"""Sensitivity of causal adjustment to an omitted common cause.

Python 3.12+, NumPy only. Reproduce tables with `python experiment.py`.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

SEED: int = 20261018
N: int = 2000
REPS: int = 400
TRUE_EFFECT: float = 1.0
STRENGTHS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float,float]:
    """Estimate coefficient and classical homoskedastic standard error."""
    if y.ndim != 1 or x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError('Incompatible outcome/design dimensions')
    b=np.linalg.lstsq(x,y,rcond=None)[0]
    e=y-x@b
    se=np.sqrt((e@e/(len(y)-x.shape[1])) * np.linalg.inv(x.T@x)[1,1])
    return float(b[1]),float(se)


def main()->None:
    """Compare an observed-only regression with oracle adjustment."""
    rng=np.random.default_rng(SEED)
    rows=[]
    for strength in STRENGTHS:
        estimates={'observed_only':[],'oracle_adjusted':[]}
        for _ in range(REPS):
            w,u,ev,ey=rng.standard_normal((4,N))
            x=.7*w + strength*u + ev
            y=TRUE_EFFECT*x+.8*w+1.5*u+ey
            estimates['observed_only'].append(ols(y,np.column_stack((np.ones(N),x,w))))
            estimates['oracle_adjusted'].append(ols(y,np.column_stack((np.ones(N),x,w,u))))
        for name, result in estimates.items():
            a=np.asarray(result); b,se=a[:,0],a[:,1]
            rows.append({'strength':strength,'model':name,'n':N,'replications':REPS,'true_effect':TRUE_EFFECT,
                'mean':float(b.mean()),'bias':float(b.mean()-TRUE_EFFECT),'rmse':float(np.sqrt(np.mean((b-TRUE_EFFECT)**2))),
                'mean_se':float(se.mean()),'coverage':float(np.mean(np.abs(b-TRUE_EFFECT)<=1.96*se)),
                'theoretical_observed_bias':1.5*strength/(1+strength**2) if name=='observed_only' else 0.0})
    with Path(__file__).with_name('results.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:print(row)

if __name__=='__main__':main()
