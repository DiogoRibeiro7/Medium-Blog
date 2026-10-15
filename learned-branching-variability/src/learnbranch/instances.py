"""Random instance families used in the learned-branching literature.

Both generators follow the descriptions used by Gasse et al. (2019):

* set covering in the style of Balas and Ho (1980): a random 0/1 matrix with a
  fixed density, integer costs in [1, 100], every row covered by at least one
  column and every column used by at least two rows;
* capacitated facility location in the style of Cornuejols, Sridharan and
  Thizy (1991): customers and facilities on the unit square, transport cost
  proportional to distance times demand, fixed costs growing with the square
  root of capacity, and total capacity scaled to `ratio` times total demand.

Each generator returns the data needed to rebuild the model; `to_model` turns
it into a PySCIPOpt model. Keeping the data separate makes instances cheap to
store (as .npz) and lets the branching code read column statistics directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyscipopt as scip


@dataclass
class SetCover:
    cost: np.ndarray          # (ncols,)
    rows: list[np.ndarray]    # column indices covering each row

    @property
    def kind(self) -> str:
        return "setcover"


@dataclass
class Facility:
    fixed: np.ndarray         # (nfac,)
    capacity: np.ndarray      # (nfac,)
    demand: np.ndarray        # (ncust,)
    trans: np.ndarray         # (ncust, nfac) cost of serving all of customer i from j

    @property
    def kind(self) -> str:
        return "facility"


def set_cover(n_rows: int, n_cols: int, rng: np.random.Generator,
              density: float = 0.05, max_coef: int = 100) -> SetCover:
    nnz = int(n_rows * n_cols * density)
    # every column appears in at least two rows, every row has at least one column
    col_of = np.concatenate([np.arange(n_cols), np.arange(n_cols),
                             rng.integers(0, n_cols, nnz - 2 * n_cols)])
    row_of = rng.permutation(np.concatenate([np.arange(n_rows),
                                             rng.integers(0, n_rows, nnz - n_rows)]))
    pairs = np.unique(np.stack([row_of, col_of], 1), axis=0)
    rows = [pairs[pairs[:, 0] == r, 1] for r in range(n_rows)]
    for r in range(n_rows):                       # guard against an empty row
        if len(rows[r]) == 0:
            rows[r] = rng.integers(0, n_cols, 1)
    cost = rng.integers(1, max_coef + 1, n_cols).astype(float)
    return SetCover(cost=cost, rows=rows)


def facility(n_customers: int, n_facilities: int, rng: np.random.Generator,
             ratio: float = 5.0) -> Facility:
    cx, cy = rng.random(n_customers), rng.random(n_customers)
    fx, fy = rng.random(n_facilities), rng.random(n_facilities)
    demand = rng.integers(5, 36, n_customers).astype(float)
    capacity = rng.integers(10, 161, n_facilities).astype(float)
    fixed = (rng.integers(100, 111, n_facilities) * np.sqrt(capacity)
             + rng.integers(0, 91, n_facilities)).astype(int).astype(float)
    capacity = np.floor(capacity * ratio * demand.sum() / capacity.sum())
    dist = np.sqrt((cx[:, None] - fx[None, :]) ** 2 + (cy[:, None] - fy[None, :]) ** 2)
    trans = dist * 10 * demand[:, None]
    return Facility(fixed=fixed, capacity=capacity, demand=demand, trans=trans)


def to_model(inst, name: str = "instance") -> scip.Model:
    m = scip.Model(name)
    m.hideOutput()
    if isinstance(inst, SetCover):
        x = [m.addVar(f"x{j}", vtype="B", obj=float(c)) for j, c in enumerate(inst.cost)]
        for r, cols in enumerate(inst.rows):
            m.addCons(scip.quicksum(x[j] for j in cols) >= 1, name=f"cover{r}")
    elif isinstance(inst, Facility):
        nc, nf = inst.trans.shape
        y = [m.addVar(f"y{j}", vtype="B", obj=float(inst.fixed[j])) for j in range(nf)]
        x = [[m.addVar(f"x{i}_{j}", vtype="C", lb=0.0, ub=1.0, obj=float(inst.trans[i, j]))
              for j in range(nf)] for i in range(nc)]
        for i in range(nc):
            m.addCons(scip.quicksum(x[i][j] for j in range(nf)) == 1, name=f"demand{i}")
        for j in range(nf):
            m.addCons(scip.quicksum(inst.demand[i] * x[i][j] for i in range(nc))
                      <= inst.capacity[j] * y[j], name=f"cap{j}")
        m.addCons(scip.quicksum(inst.capacity[j] * y[j] for j in range(nf))
                  >= inst.demand.sum(), name="total")
        for i in range(nc):
            for j in range(nf):
                m.addCons(x[i][j] <= y[j], name=f"link{i}_{j}")
    else:
        raise TypeError(type(inst))
    return m


def save(inst, path) -> None:
    if isinstance(inst, SetCover):
        lens = np.array([len(r) for r in inst.rows])
        np.savez_compressed(path, kind="setcover", cost=inst.cost,
                            flat=np.concatenate(inst.rows), lens=lens)
    else:
        np.savez_compressed(path, kind="facility", fixed=inst.fixed,
                            capacity=inst.capacity, demand=inst.demand, trans=inst.trans)


def load(path):
    d = np.load(path)
    if str(d["kind"]) == "setcover":
        rows = np.split(d["flat"], np.cumsum(d["lens"])[:-1])
        return SetCover(cost=d["cost"], rows=rows)
    return Facility(fixed=d["fixed"], capacity=d["capacity"], demand=d["demand"],
                    trans=d["trans"])
