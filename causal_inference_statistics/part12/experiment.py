"""Monte Carlo evaluation of heterogeneous treatment effects and treatment allocation.

Python >=3.12. Requires NumPy; the simulation uses genuine held-out decisions.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

SEED: int = 20261018
REPETITIONS: int = 400
N_TRAIN: int = 2400
N_TEST: int = 1200
CAPACITY: float = 0.20


def generate(rng: np.random.Generator, n: int) -> tuple[np.ndarray, ...]:
    """Return feature, randomized treatment, outcome, baseline and true effect."""
    if not isinstance(n, int) or n < 10:
        raise ValueError('n must be an integer >= 10')
    x = rng.normal(size=n)
    a = rng.binomial(1, .5, size=n).astype(float)
    baseline = 2.0 - 1.5*x + .3*x*x
    tau = .5 + 1.2*x + .4*(x*x - 1)
    y = baseline + a*tau + rng.normal(size=n)
    return x, a, y, baseline, tau


def treatment_score(x: np.ndarray, a: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Fit centered polynomial treatment interaction model by least squares."""
    if degree not in (0, 1, 2):
        raise ValueError('degree must be 0, 1, or 2')
    phi = np.column_stack((np.ones(len(x)), x, x*x-1))
    interaction = phi[:, :degree+1]
    design = np.column_stack((phi, a[:,None]*interaction))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return beta[3:]


def score(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Evaluate estimated conditional average treatment effect."""
    phi = np.column_stack((np.ones(len(x)), x, x*x-1))
    return phi[:, :len(beta)] @ beta


def selected_mean(values: np.ndarray, rank: np.ndarray) -> float:
    """Return true mean treatment effect among highest-ranked 20%."""
    count = round(len(values)*CAPACITY)
    if count <= 0:
        raise ValueError('No selected observations')
    chosen = np.argsort(rank, kind='stable')[-count:]
    return float(values[chosen].mean())


def main() -> None:
    """Write reproducible Monte Carlo summaries and decision metrics."""
    rng = np.random.default_rng(SEED)
    results: dict[str,list[tuple[float,float]]] = {k:[] for k in ('constant', 'linear', 'quadratic')}
    decisions: dict[str,list[float]] = {k:[] for k in ('random', 'baseline_risk', 'constant', 'linear', 'quadratic', 'oracle')}
    group_effects: dict[str,list[float]] = {'low_x':[], 'high_x':[]}
    for _ in range(REPETITIONS):
        x,a,y,_,_ = generate(rng,N_TRAIN)
        xt,_,_,base,tau = generate(rng,N_TEST)
        for name,deg in (('constant',0),('linear',1),('quadratic',2)):
            beta=treatment_score(x,a,y,deg)
            pred=score(xt,beta)
            results[name].append((float(np.mean((pred-tau)**2)),float(np.mean(pred-tau))))
            # For a constant model, randomly break ties to represent no ability to rank.
            rank = rng.random(len(xt)) if deg==0 else pred
            decisions[name].append(selected_mean(tau,rank))
        decisions['random'].append(selected_mean(tau,rng.random(len(xt))))
        decisions['baseline_risk'].append(selected_mean(tau,base))
        decisions['oracle'].append(selected_mean(tau,tau))
        group_effects['low_x'].append(float(tau[xt<0].mean()))
        group_effects['high_x'].append(float(tau[xt>=0].mean()))
    records=[]
    for name,values in results.items():
        z=np.asarray(values)
        records.append({'type':'prediction','method':name,'mean':float(z[:,0].mean()),'sd':float(z[:,0].std(ddof=1)),'secondary':float(z[:,1].mean()),'replications':REPETITIONS})
    for name,values in decisions.items():
        z=np.asarray(values)
        records.append({'type':'policy','method':name,'mean':float(z.mean()),'sd':float(z.std(ddof=1)),'secondary':'','replications':REPETITIONS})
    for name,values in group_effects.items():
        z=np.asarray(values)
        records.append({'type':'group_effect','method':name,'mean':float(z.mean()),'sd':float(z.std(ddof=1)),'secondary':'','replications':REPETITIONS})
    output=Path(__file__).with_name('results.csv')
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for r in records:
        print(f"{r['type']:12} {r['method']:15} {r['mean']:.5f} sd={r['sd']:.5f} secondary={r['secondary']}")


if __name__=='__main__':
    main()
