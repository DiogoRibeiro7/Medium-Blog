"""The competing-risks estimators on cases with a known answer."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exits_core as ec  # noqa: E402


def simulate_fine_gray(n, beta1, beta2, p, rng, censor=3.0):
    """Data from the Fine and Gray (1999) simulation design with two covariates.

    P(cause 1 | x) = 1 - (1 - p)^exp(x beta1); given cause 1, T has the subdistribution of a
    unit-exponential baseline; given cause 2, T is exponential with rate exp(x beta2).
    """
    x = rng.normal(size=(n, 2))
    lp1 = np.exp(x @ np.asarray(beta1))
    p1 = 1 - (1 - p) ** lp1
    u = rng.uniform(size=n)
    one = u < p1
    t = np.empty(n)
    inner = 1 - (1 - (1 - u[one]) ** (1 / lp1[one])) / p
    t[one] = -np.log(inner)
    t[~one] = rng.exponential(1 / np.exp(x[~one] @ np.asarray(beta2)))
    c = rng.uniform(0, censor, size=n)
    cause = np.where(t <= c, np.where(one, "a", "b"), "censored")
    return np.minimum(t, c), cause.astype(object), x


def brute_force_score(time, cause, x, k, beta):
    """Fine-Gray score by explicit risk sets and weights, for checking the cumulative-sum version."""
    cens = cause == "censored"

    def g_left(t):
        g = 1.0
        for u in np.unique(time[cens]):
            if u >= t:
                break
            g *= 1 - np.sum(cens & (time == u)) / np.sum(time >= u)
        return g

    e = np.exp(x @ beta)
    score = np.zeros(x.shape[1])
    for tj in np.unique(time[cause == k]):
        w = np.where(time >= tj, 1.0, 0.0)
        comp = (time < tj) & (cause != k) & ~cens
        w[comp] = [g_left(tj) / g_left(ti) for ti in time[comp]]
        mean = (w * e) @ x / np.sum(w * e)
        dj = (time == tj) & (cause == k)
        score += x[dj].sum(axis=0) - dj.sum() * mean
    return score


class AalenJohansen(unittest.TestCase):
    def test_hand_example(self):
        res = ec.aalen_johansen([1, 2, 3, 4], ["a", "censored", "b", "a"], ["a", "b"], [0.5, 1, 3, 4, 9])
        np.testing.assert_allclose(res["a"], [0, 0.25, 0.25, 0.625, 0.625])
        np.testing.assert_allclose(res["b"], [0, 0, 0.375, 0.375, 0.375])

    def test_no_censoring_is_empirical_distribution(self):
        rng = np.random.default_rng(2)
        t = rng.exponential(size=300)
        c = np.where(rng.uniform(size=300) < 0.4, "a", "b").astype(object)
        grid = [0.2, 0.5, 1.0, 2.0]
        res = ec.aalen_johansen(t, c, ["a", "b"], grid)
        for k in ("a", "b"):
            np.testing.assert_allclose(res[k], [np.mean((t <= g) & (c == k)) for g in grid])

    def test_one_cause_is_one_minus_kaplan_meier(self):
        from lifelines import KaplanMeierFitter

        rng = np.random.default_rng(3)
        t = rng.exponential(size=400)
        cens = rng.exponential(1.5, size=400)
        obs = np.minimum(t, cens)
        cause = np.where(t <= cens, "a", "censored").astype(object)
        grid = [0.1, 0.5, 1.0, 2.0]
        km = KaplanMeierFitter().fit(obs, cause == "a")
        np.testing.assert_allclose(ec.aalen_johansen(obs, cause, ["a"], grid)["a"], 1 - km.predict(grid), atol=1e-12)

    def test_integer_weights_equal_replication(self):
        rng = np.random.default_rng(4)
        t = np.round(rng.exponential(size=60), 2)
        c = rng.choice(np.array(["a", "b", "censored"], dtype=object), size=60)
        w = rng.integers(1, 4, size=60)
        grid = [0.3, 0.8, 1.5]
        weighted = ec.aalen_johansen(t, c, ["a", "b"], grid, weights=w)
        replicated = ec.aalen_johansen(np.repeat(t, w), np.repeat(c, w), ["a", "b"], grid)
        pd.testing.assert_frame_equal(weighted, replicated)


class FineGray(unittest.TestCase):
    def test_without_competing_events_it_is_cox(self):
        from lifelines import CoxPHFitter

        rng = np.random.default_rng(5)
        x = rng.normal(size=(800, 2))
        t = rng.exponential(1 / np.exp(x @ [0.7, -0.4]))
        c = rng.exponential(2.0, size=800)
        cause = np.where(t <= c, "a", "censored").astype(object)
        obs = np.minimum(t, c)
        beta, _ = ec.fine_gray(obs, cause, x, "a")
        df = pd.DataFrame(x, columns=["x1", "x2"]).assign(t=obs, e=(cause == "a").astype(int))
        cox = CoxPHFitter().fit(df, "t", "e").params_.to_numpy()
        np.testing.assert_allclose(beta, cox, atol=1e-6)

    def test_solution_zeroes_the_brute_force_score(self):
        rng = np.random.default_rng(6)
        t, cause, x = simulate_fine_gray(150, [0.6, -0.3], [0.2, 0.4], 0.4, rng)
        beta, _ = ec.fine_gray(t, cause, x, "a")
        np.testing.assert_allclose(brute_force_score(t, cause, x, "a", beta), 0, atol=1e-7)

    def test_recovers_the_simulated_coefficients(self):
        rng = np.random.default_rng(7)
        t, cause, x = simulate_fine_gray(6000, [0.5, -0.5], [0.3, 0.3], 0.5, rng)
        beta, cov = ec.fine_gray(t, cause, x, "a")
        np.testing.assert_allclose(beta, [0.5, -0.5], atol=4 * np.sqrt(np.diag(cov)).max())


class Outcomes(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "entry": ["2018-01-01", "2018-01-01", "2018-01-01", "2018-01-01", ""],
                "censor": ["2024-01-01"] * 5,
                "foreign_acquisition": ["2017-06-01", "2020-01-01", "", "", ""],
                "flip": ["", "", "2019-01-01", "", ""],
                "uk_acquisition": ["", "", "", "", ""],
                "offshore": ["", "", "", "", ""],
                "closure": ["", "2019-01-01", "2021-01-01", "2030-01-01", ""],
            }
        )

    def test_first_event_after_entry(self):
        out = ec.outcomes(self.frame())
        self.assertEqual(len(out), 3)  # acquired before entry, and no entry, are dropped
        self.assertEqual(list(out["cause"]), ["closure", "flip", "censored"])
        np.testing.assert_allclose(out["time"], np.array([365, 365, 2191]) / 365.25)

    def test_merge(self):
        out = ec.outcomes(self.frame(), merge={"flip": "foreign_acquisition"})
        self.assertEqual(list(out["cause"]), ["closure", "foreign_acquisition", "censored"])

    def test_states_at(self):
        out = ec.outcomes(self.frame()).assign(next_allotment=["", "", "2019-06-01"])
        st = ec.states_at(out, 3, ec.CAUSES)
        self.assertEqual(list(st["state"]), ["closure", "flip", "independent, raised again"])
        st = ec.states_at(out, 7, ec.CAUSES)
        self.assertEqual(len(st), 0)


class Weights(unittest.TestCase):
    def test_balancing_weights_balance_the_covariate(self):
        rng = np.random.default_rng(8)
        z = rng.normal(size=4000)
        treat = (rng.uniform(size=4000) < 1 / (1 + np.exp(-z))).astype(int)
        df = pd.DataFrame({"z": z, "t": treat})
        w, _ = ec.balancing_weights(df, "t", ["z"], {}, trim=0.0)
        self.assertGreater(abs(ec.standardised_difference(df, "z", "t")), 0.5)
        self.assertLess(abs(ec.standardised_difference(df, "z", "t", w)), 0.05)


if __name__ == "__main__":
    unittest.main()
