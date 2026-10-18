"""Independent nuisance-basis validation of partially linear DML.

Python 3.12+, NumPy. Uses 1000 independent replications per design.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

SEED: int = 20261018
REPS: int = 1000
N: int = 800
TAU: float = 1.0

def experiment(rng: np.random.Generator, nonlinear: bool) -> dict[str,float]:
    """Compare linear basis, correct nonlinear basis, and oracle nuisance means."""
    x = rng.standard_normal((N,3))
    if nonlinear:
        m = 1.4*np.sin(x[:,0])+0.9*(x[:,1]**2-1)+0.5*x[:,0]*x[:,2]
        g = 1.2*np.sin(x[:,0])+1.3*(x[:,1]**2-1)+0.8*x[:,0]*x[:,2]
    else:
        m = 1.4*x[:,0]+0.9*x[:,1]+0.5*x[:,2]
        g = 1.2*x[:,0]+1.3*x[:,1]+0.8*x[:,2]
    d = m+rng.standard_normal(N)
    y = TAU*d+g+rng.standard_normal(N)
    linear = np.column_stack((np.ones(N), x))
    flexible = np.column_stack((linear,np.sin(x[:,0]),x[:,1]**2-1,x[:,0]*x[:,2]))
    predictions: dict[str,float] = {}
    for name, z in (('linear_basis',linear),('correct_basis',flexible)):
        dh=np.empty(N);yh=np.empty(N)
        shuffled=rng.permutation(N)
        for holdout in np.array_split(shuffled,3):
            train=np.ones(N,dtype=bool);train[holdout]=False
            dh[holdout]=z[holdout]@np.linalg.lstsq(z[train],d[train],rcond=None)[0]
            yh[holdout]=z[holdout]@np.linalg.lstsq(z[train],y[train],rcond=None)[0]
        dr=d-dh;yr=y-yh
        predictions[name]=float(dr@yr/(dr@dr))
    dr=d-m;yr=y-(TAU*m+g)
    predictions['oracle']=float(dr@yr/(dr@dr))
    return predictions

def main()->None:
    """Persist Monte Carlo summaries for linear and nonlinear designs."""
    rng=np.random.default_rng(SEED); records=[]
    for nonlinear in (False,True):
        estimates={name:[] for name in ('linear_basis','correct_basis','oracle')}
        for _ in range(REPS):
            for name,value in experiment(rng,nonlinear).items():
                estimates[name].append(value)
        for name,values in estimates.items():
            a=np.asarray(values)
            records.append(dict(design='nonlinear' if nonlinear else 'linear',
                method=name,n=N,replications=REPS,truth=TAU,mean=float(a.mean()),
                bias=float(a.mean()-TAU),sd=float(a.std(ddof=1)),
                rmse=float(np.sqrt(np.mean((a-TAU)**2)))))
    with Path(__file__).with_name('validation_results.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    for row in records: print(row)

if __name__=='__main__': main()
