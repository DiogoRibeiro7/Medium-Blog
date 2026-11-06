"""Pooled versus unpooled: what the single market insures, and when it stops.

    python simulate.py

Writes ``results/summary.txt`` and the CSVs behind the model figures.

Model, per simulated year
-------------------------
Each member state i has a population share ``w_i``, a self-sufficiency ratio ``s_i``
(its share of EU cereal production over its share of population, so that the union as a
whole produces what it consumes at the reference price), and a supply shock ``z_i``
(lognormal; log sd and correlations estimated in ``evidence.py``).

Households in country i need one unit of a staple. Incomes are lognormal with the
country's Gini; the share who cannot afford their need at the reference price is set to
the country's share unable to afford a meal with meat, fish or a vegetarian equivalent
every second day (Eurostat ilc_mdes03). Above need there is a discretionary layer that
is cut first, so the aggregate price elasticity at the reference price is about -0.2.

Pooled (single market): one price clears the union. The rest of the world supplies or
takes up to ``M`` of EU consumption at the world price before the price has to move.

Unpooled (the counterfactual that isolates risk-sharing): normal trade flows are kept,
but each country's own production deviation stays at home, ``supply_i = 1 + s_i (z_i - 1)``,
and the same world buffer, scaled to its consumption, is available to it.

The difference between the two is the insurance the single market provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

X_DISC = 0.25  # discretionary layer per household at the reference price
ETA = 1.0  # its price elasticity
EPS_WORLD = 0.22  # elasticity used to turn a common supply shock into a world price move
M_WORLD = 0.05  # world buffer, share of consumption, at the world price
N_DRAWS = 40000
SEED = 20261106


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------


@dataclass
class CountryDemand:
    """Normalised demand d(p) (= 1 at p = 1) and the priced-out share, for one country."""

    b: float  # share priced out at the reference price
    sigma: float  # log-income sd (from the Gini)

    def __post_init__(self) -> None:
        # P(k y < 1) = b with median income 1  =>  ln k = -sigma * Phi^-1(b)
        self.lnk = -self.sigma * ndtri(self.b)
        self.scale = float(self._need(np.array([1.0]))[0] + X_DISC)
        lp = np.linspace(-6.0, 5.0, 12001)
        q = self.d(np.exp(lp))
        self._q, self._lp = q[::-1], lp[::-1]

    def _need(self, p):
        a = (np.log(p) - self.lnk) / self.sigma
        return 1.0 - ndtr(a) + np.exp(self.lnk - np.log(p) + self.sigma**2 / 2.0) * ndtr(a - self.sigma)

    def d(self, p):
        p = np.asarray(p, dtype=float)
        return (self._need(p) + X_DISC * p ** (-ETA)) / self.scale

    def inverse(self, q):
        return np.exp(np.interp(np.clip(q, self._q[0], self._q[-1]), self._q, self._lp))

    def priced_out(self, p):
        return ndtr((np.log(np.asarray(p, dtype=float)) - self.lnk) / self.sigma)


def gini_to_sigma(g: float) -> float:
    return float(np.sqrt(2.0) * ndtri((1.0 + g) / 2.0))


@dataclass
class Union:
    table: pd.DataFrame  # country_table.csv, indexed by geo
    corr: pd.DataFrame  # correlation of shocks (26 producers)

    def __post_init__(self) -> None:
        t = self.table.copy()
        t["shock_sd"] = t["shock_sd"].fillna(0.0)
        self.geo = list(t.index)
        self.w = t.pop_share.to_numpy()
        self.s = t.self_sufficiency.to_numpy()
        self.sd = t.shock_sd.to_numpy()
        self.b = t.meal_deprivation.to_numpy()
        self.dem = [CountryDemand(float(bi), gini_to_sigma(float(gi))) for bi, gi in zip(t.meal_deprivation, t.gini)]
        # EU demand table
        lp = np.linspace(-6.0, 5.0, 12001)
        q = sum(wi * di.d(np.exp(lp)) for wi, di in zip(self.w, self.dem))
        self._q, self._lp = q[::-1], lp[::-1]
        self.baseline_priced_out = float(np.sum(self.w * self.b))
        # full correlation over all members (Malta: no production, no shock)
        R = pd.DataFrame(np.eye(len(self.geo)), index=self.geo, columns=self.geo)
        prod = list(self.corr.index)
        R.loc[prod, prod] = self.corr.values
        self.R_emp = R.to_numpy()
        self.producers = np.array([g in prod for g in self.geo])

    # --- correlation family ------------------------------------------------
    def mean_corr(self, R: np.ndarray) -> float:
        P = R[np.ix_(self.producers, self.producers)]
        return float(P[np.triu_indices(P.shape[0], 1)].mean())

    def corr_at(self, target: float, shrink: float = 0.1) -> np.ndarray:
        """Correlation matrix with mean pairwise correlation ``target`` among producers.

        Start from the empirical matrix shrunk towards the identity, then move along a
        straight line towards the identity (lower targets) or towards the all-ones matrix
        (higher targets). Both paths stay positive semi-definite.
        """
        n = len(self.geo)
        Rs = (1 - shrink) * self.R_emp + shrink * np.eye(n)
        m0 = self.mean_corr(Rs)
        J = np.eye(n)
        J[np.ix_(self.producers, self.producers)] = 1.0
        if target <= m0:
            a = target / m0
            return (1 - a) * np.eye(n) + a * Rs
        a = (target - m0) / (1 - m0)
        return (1 - a) * Rs + a * J

    # --- draws ---------------------------------------------------------------
    def draw(self, R: np.ndarray, n: int, common_share: float = 0.0, seed: int = SEED, scale: float = 1.0):
        """Log supply shocks (n x countries) and the common factor.

        ``common_share`` moves that share of every producer's variance into one perfectly
        common factor; the rest keeps the correlation structure ``R``.
        """
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(R + 1e-10 * np.eye(len(R)))
        e = rng.standard_normal((n, len(R))) @ L.T
        f = rng.standard_normal(n)
        sd = self.sd * scale
        x = np.sqrt(1 - common_share) * e * sd + np.sqrt(common_share) * f[:, None] * sd
        return x, f

    # --- clearing --------------------------------------------------------------
    def _clear(self, supply, pw, demand_at, inverse, M):
        """Price given supply, world price and buffer (vectorised over draws)."""
        dw = demand_at(pw)
        gap = dw - supply
        p = pw.copy()
        hi = gap > M
        lo = gap < -M
        p[hi] = inverse(supply[hi] + M)
        p[lo] = inverse(supply[lo] - M)
        return p

    def eu_demand(self, p):
        return sum(wi * di.d(p) for wi, di in zip(self.w, self.dem))

    def eu_inverse(self, q):
        return np.exp(np.interp(np.clip(q, self._q[0], self._q[-1]), self._q, self._lp))

    def outcomes(self, x: np.ndarray, f: np.ndarray, common_share: float, M: float = M_WORLD, capacity: np.ndarray | None = None):
        """Prices and priced-out shares, pooled and unpooled, for each draw."""
        n = x.shape[0]
        z = np.exp(x)
        s = self.s if capacity is None else self.s * capacity
        # a common supply shock is common to the world too: the world price moves with it
        common = np.sqrt(common_share) * f * float(np.sum(self.w * self.s * self.sd))
        pw = np.exp(-common / EPS_WORLD) if common_share > 0 else np.ones(n)
        # pooled
        Q = (z * (self.w * s)).sum(axis=1)
        p_pool = self._clear(Q, pw, self.eu_demand, self.eu_inverse, M)
        po_pool = np.column_stack([d.priced_out(p_pool) for d in self.dem])
        # unpooled: own deviation stays home, normal trade kept
        p_un = np.empty_like(z)
        for i, d in enumerate(self.dem):
            supply = 1.0 + s[i] * z[:, i] - self.s[i]
            p_un[:, i] = self._clear(supply, pw, d.d, d.inverse, M)
        po_un = np.column_stack([d.priced_out(p_un[:, i]) for i, d in enumerate(self.dem)])
        return p_pool, po_pool, p_un, po_un

    def summarise(self, po: np.ndarray) -> dict:
        """Risk measures in percentage points of households priced out above the baseline.

        ``mean_excess_pp``: expected household-years priced out, per 100 households.
        ``national_cvar95_pp``: each country's mean excess in its own worst 5% of years,
        averaged with population weights: the bad-year risk a household faces where it lives.
        ``eu_cvar95_pp``: the union's aggregate excess in its worst 5% of years.
        """
        ex = po - self.b
        k = max(1, len(po) // 20)
        nat_cvar = np.sort(ex, axis=0)[-k:].mean(axis=0)
        eu = ex @ self.w
        return {
            "mean_excess_pp": 100 * float(eu.mean()),
            "national_cvar95_pp": 100 * float(nat_cvar @ self.w),
            "eu_cvar95_pp": 100 * float(np.sort(eu)[-k:].mean()),
            "prob_eu_excess_gt_1pp": float(np.mean(eu > 0.01)),
        }


def load() -> Union:
    t = pd.read_csv(RES / "country_table.csv", index_col=0)
    c = pd.read_csv(RES / "evidence_correlation.csv", index_col=0)
    return Union(t, c)


def correlation_sweep(u: Union, targets=None) -> pd.DataFrame:
    if targets is None:
        m_emp = u.mean_corr((1 - 0.1) * u.R_emp + 0.1 * np.eye(len(u.geo)))
        targets = np.sort(np.concatenate([np.round(np.arange(0.0, 1.0001, 0.05), 3), [m_emp]]))
    rows = []
    for m in targets:
        R = u.corr_at(float(m))
        x, f = u.draw(R, N_DRAWS)
        _, po_p, _, po_u = u.outcomes(x, f, 0.0)
        a, b = u.summarise(po_p), u.summarise(po_u)
        rows.append({"mean_corr": u.mean_corr(R), **{f"pooled_{k}": v for k, v in a.items()}, **{f"unpooled_{k}": v for k, v in b.items()}})
    df = pd.DataFrame(rows)
    df["insurance_mean_pp"] = df.unpooled_mean_excess_pp - df.pooled_mean_excess_pp
    df["insurance_tail_pp"] = df.unpooled_national_cvar95_pp - df.pooled_national_cvar95_pp
    df["insurance_tail_share"] = df.insurance_tail_pp / df.unpooled_national_cvar95_pp
    df["insurance_mean_share"] = df.insurance_mean_pp / df.unpooled_mean_excess_pp
    return df


def common_sweep(u: Union, R: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in np.round(np.arange(0.0, 1.0001, 0.1), 2):
        x, f = u.draw(R, N_DRAWS, common_share=float(c))
        _, po_p, _, po_u = u.outcomes(x, f, float(c))
        a, b = u.summarise(po_p), u.summarise(po_u)
        rows.append({"common_share": c, **{f"pooled_{k}": v for k, v in a.items()}, **{f"unpooled_{k}": v for k, v in b.items()}})
    df = pd.DataFrame(rows)
    df["insurance_mean_pp"] = df.unpooled_mean_excess_pp - df.pooled_mean_excess_pp
    df["insurance_tail_pp"] = df.unpooled_national_cvar95_pp - df.pooled_national_cvar95_pp
    df["insurance_tail_share"] = df.insurance_tail_pp / df.unpooled_national_cvar95_pp
    df["insurance_mean_share"] = df.insurance_mean_pp / df.unpooled_mean_excess_pp
    return df


def by_country(u: Union, R: np.ndarray) -> pd.DataFrame:
    x, f = u.draw(R, N_DRAWS)
    p_p, po_p, p_u, po_u = u.outcomes(x, f, 0.0)
    q = int(len(x) * 0.95)
    eu_excess = po_p @ u.w
    bad = np.argsort(eu_excess)[q:]
    rows = []
    for i, g in enumerate(u.geo):
        rows.append(
            {
                "geo": g,
                "pop_share": u.w[i],
                "self_sufficiency": u.s[i],
                "baseline": u.b[i],
                "pooled_mean_excess_pp": 100 * (po_p[:, i].mean() - u.b[i]),
                "unpooled_mean_excess_pp": 100 * (po_u[:, i].mean() - u.b[i]),
                "pooled_p95_excess_pp": 100 * (np.quantile(po_p[:, i], 0.95) - u.b[i]),
                "unpooled_p95_excess_pp": 100 * (np.quantile(po_u[:, i], 0.95) - u.b[i]),
                "pooled_in_eu_bad_years_pp": 100 * (po_p[bad, i].mean() - u.b[i]),
                "price_sd_pooled": float(np.std(np.log(p_p))),
                "price_sd_unpooled": float(np.std(np.log(p_u[:, i]))),
            }
        )
    return pd.DataFrame(rows).set_index("geo")


def ageing_scenarios(u: Union, R: np.ndarray, ages: pd.DataFrame, replace_share: float = 0.5) -> pd.DataFrame:
    """Anchored capacity lost where farm managers are old, before the same weather shocks.

    In each member state, a share (1 - replace_share) of the output produced on holdings
    whose manager is 65 or older (standard output, Farm Structure Survey) is not replaced
    when they retire.
    """
    col = "ge65_output_share"
    age = ages.reindex(u.geo)[col].fillna(ages.loc["EU27_2020", col]).to_numpy() / 100.0
    rows = []
    x, f = u.draw(R, N_DRAWS)
    for label, loss in (
        ("none", np.zeros_like(age)),
        ("unreplaced share of 65+ capacity, by country", age * (1 - replace_share)),
        ("same EU-wide loss, spread evenly", np.full_like(age, float(np.sum(u.w * u.s * age * (1 - replace_share)) / np.sum(u.w * u.s)))),
    ):
        cap = 1.0 - loss
        p_p, po_p, _, _ = u.outcomes(x, f, 0.0, capacity=cap)
        a = u.summarise(po_p)
        rows.append(
            {
                "scenario": label,
                "eu_capacity_loss": float(np.sum(u.w * u.s * loss)),
                "median_price": float(np.median(p_p)),
                "p95_price": float(np.quantile(p_p, 0.95)),
                **a,
            }
        )
    return pd.DataFrame(rows)


def buffer_sensitivity(u: Union, R: np.ndarray) -> pd.DataFrame:
    rows = []
    x, f = u.draw(R, N_DRAWS)
    for M in (0.0, 0.025, 0.05, 0.10):
        _, po_p, _, po_u = u.outcomes(x, f, 0.0, M=M)
        a, b = u.summarise(po_p), u.summarise(po_u)
        rows.append({"M": M, "pooled_tail": a["national_cvar95_pp"], "unpooled_tail": b["national_cvar95_pp"], "pooled_mean": a["mean_excess_pp"], "unpooled_mean": b["mean_excess_pp"]})
    return pd.DataFrame(rows)


def main() -> None:
    u = load()
    L: list[str] = []
    say = L.append
    say("== Calibration")
    say(f"member states {len(u.geo)}; producers with shocks {int(u.producers.sum())}")
    say(f"baseline share priced out (population-weighted meal deprivation): {u.baseline_priced_out:.4f}")
    el = (np.log(u.eu_demand(np.array([1.0001]))) - np.log(u.eu_demand(np.array([0.9999])))) / (np.log(1.0001) - np.log(0.9999))
    say(f"EU demand elasticity at the reference price: {float(el[0]):.3f}")
    say(f"anchored supply at the reference: {float(np.sum(u.w * u.s)):.4f} of consumption; world buffer M = {M_WORLD}")
    R0 = (1 - 0.1) * u.R_emp + 0.1 * np.eye(len(u.geo))
    m_emp = u.mean_corr(R0)
    say(f"empirical correlation shrunk 10% towards identity: mean pairwise {m_emp:.3f}")
    x, f = u.draw(R0, N_DRAWS)
    eu_log_supply = np.log((np.exp(x) * (u.w * u.s)).sum(axis=1))
    say(f"simulated EU log-supply sd {eu_log_supply.std():.4f} (data: EU aggregate shock sd 0.0594)")

    say("\n== Pooled vs unpooled at the empirical correlation")
    p_p, po_p, p_u, po_u = u.outcomes(x, f, 0.0)
    a, b = u.summarise(po_p), u.summarise(po_u)
    for k in a:
        say(f"  {k}: pooled {a[k]:.4f}, unpooled {b[k]:.4f}")
    say(f"  pooled EU price: sd of log {np.std(np.log(p_p)):.4f}; 95th percentile {np.quantile(p_p, 0.95):.3f}; share of years at the world price {np.mean(np.isclose(p_p, 1.0)):.3f}")

    say("\n== Correlation sweep (weather-type shocks only)")
    cs = correlation_sweep(u)
    cs.to_csv(RES / "model_correlation_sweep.csv", index=False)
    for _, r in cs.iterrows():
        say(
            f"  mean corr {r.mean_corr:.3f}: mean excess pooled {r.pooled_mean_excess_pp:.3f} pp, unpooled {r.unpooled_mean_excess_pp:.3f} pp, insurance {r.insurance_mean_pp:.3f} pp ({r.insurance_mean_share:.2f}); "
            f"national bad-year risk pooled {r.pooled_national_cvar95_pp:.3f} pp, unpooled {r.unpooled_national_cvar95_pp:.3f} pp, insurance {r.insurance_tail_pp:.3f} pp ({r.insurance_tail_share:.2f}); "
            f"EU worst-5% pooled {r.pooled_eu_cvar95_pp:.3f}, unpooled {r.unpooled_eu_cvar95_pp:.3f}"
        )

    say("\n== Common-shock sweep (share of each producer's variance moved to one common factor; world price moves with it)")
    cm = common_sweep(u, R0)
    cm.to_csv(RES / "model_common_sweep.csv", index=False)
    for _, r in cm.iterrows():
        say(
            f"  common share {r.common_share:.1f}: mean excess pooled {r.pooled_mean_excess_pp:.3f}, unpooled {r.unpooled_mean_excess_pp:.3f} (insurance share {r.insurance_mean_share:.2f}); "
            f"national bad-year risk pooled {r.pooled_national_cvar95_pp:.3f}, unpooled {r.unpooled_national_cvar95_pp:.3f} (insurance share {r.insurance_tail_share:.2f})"
        )

    say("\n== By member state, empirical correlation")
    bc = by_country(u, R0)
    bc.to_csv(RES / "model_by_country.csv")
    for g, r in bc.sort_values("unpooled_p95_excess_pp", ascending=False).iterrows():
        say(
            f"  {g}: self-sufficiency {r.self_sufficiency:.2f}, baseline {r.baseline:.3f}; 95th pct excess pooled {r.pooled_p95_excess_pp:.2f} pp, unpooled {r.unpooled_p95_excess_pp:.2f} pp; "
            f"in EU-wide bad years {r.pooled_in_eu_bad_years_pp:.2f} pp"
        )
    worse = bc[bc.pooled_p95_excess_pp > bc.unpooled_p95_excess_pp]
    say(f"  member states with a worse 95th percentile when pooled: {', '.join(worse.index)}")

    say("\n== Buffer sensitivity (empirical correlation)")
    bs = buffer_sensitivity(u, R0)
    bs.to_csv(RES / "model_buffer_sensitivity.csv", index=False)
    for _, r in bs.iterrows():
        say(f"  M {r.M:.3f}: national bad-year risk pooled {r.pooled_tail:.3f}, unpooled {r.unpooled_tail:.3f}; mean pooled {r.pooled_mean:.3f}, unpooled {r.unpooled_mean:.3f}")

    say("\n== Ageing: capacity not replaced where managers are 65+")
    ages = pd.read_csv(RES / "evidence_farm_output_ages.csv", index_col=0)
    ag = ageing_scenarios(u, R0, ages)
    ag.to_csv(RES / "model_ageing.csv", index=False)
    for _, r in ag.iterrows():
        say(
            f"  {r.scenario}: EU capacity loss {r.eu_capacity_loss:.4f}; median price {r.median_price:.3f}, 95th pct {r.p95_price:.3f}; "
            f"mean excess {r.mean_excess_pp:.3f} pp, national bad-year risk {r.national_cvar95_pp:.3f} pp"
        )
    text = "\n".join(L) + "\n"
    (RES / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
