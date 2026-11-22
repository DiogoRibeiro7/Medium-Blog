"""Loop drift when the transports are learned instead of written.

Fibres are embedding spaces: Production readings live in R^d, Incident readings in R^m with
m > d. The true push A (m x d, orthonormal columns) and pop B = A^T satisfy the law
B A = I_d, the embedding version of u;i = id. Adapters are fitted by least squares on n noisy
pairs each, and the loop statistic is the RMS relative drift ||B_hat A_hat - I||_F / sqrt(d),
which is the root-mean-square of |B_hat A_hat x - x| / |x| over isotropic x.

To first order, B_hat A_hat - I = B dA + dB A, and with Gaussian inputs and noise variance
sigma^2 the inverse-Wishart mean gives
    E ||B_hat A_hat - I||_F^2 / d  =  sigma^2 d (1/(n - d - 1) + 1/(n - m - 1)),
so drift falls like sigma sqrt(2d/n) and is never zero for finite n.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


def orthonormal(rng: np.random.Generator, m: int, d: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((m, d)))
    return q * np.sign(np.diag(r))


def fit(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least-squares W with Y ~ X W^T (rows are samples)."""
    return np.linalg.solve(X.T @ X, X.T @ Y).T


def analytic_rms(sigma: float, d: int, m: int, n: int) -> float:
    return float(sigma * np.sqrt(d * (1.0 / (n - d - 1) + 1.0 / (n - m - 1))))


def one_loop_ols(rng, A, B_true, n, sigma):
    """Fit both adapters on n simulated pairs and return the RMS relative loop drift."""
    m, d = A.shape
    X = rng.standard_normal((n, d))
    A_hat = fit(X, X @ A.T + sigma * rng.standard_normal((n, m)))
    Y = rng.standard_normal((n, m))
    B_hat = fit(Y, Y @ B_true.T + sigma * rng.standard_normal((n, d)))
    return float(np.linalg.norm(B_hat @ A_hat - np.eye(d)) / np.sqrt(d))


def bartlett(rng: np.random.Generator, p: int, n: int) -> np.ndarray:
    """Lower-triangular L with L L^T ~ Wishart_p(n, I) (Bartlett decomposition)."""
    L = np.tril(rng.standard_normal((p, p)), -1)
    L[np.diag_indices(p)] = np.sqrt(rng.chisquare(n - np.arange(p)))
    return L


def one_loop(rng, A, B_true, n, sigma):
    """The same draw as one_loop_ols, sampled exactly without forming the n samples.

    With Gaussian inputs, X^T X = L L^T is Wishart and, given X, X^T E = sigma L Z with Z
    standard normal, so the least-squares error is dA = sigma Z^T L^{-1}. The cost no longer
    grows with n."""
    m, d = A.shape
    LX = bartlett(rng, d, n)
    dA = sigma * solve_triangular(LX, rng.standard_normal((d, m)), lower=True, trans="T").T
    LY = bartlett(rng, m, n)
    dB = sigma * solve_triangular(LY, rng.standard_normal((m, d)), lower=True, trans="T").T
    return float(np.linalg.norm((B_true + dB) @ (A + dA) - np.eye(d)) / np.sqrt(d))


def sampler_check(seed: int, d=16, n=400, sigma=0.1, reps=3000) -> dict:
    """Brute-force least squares and the exact sampler give the same distribution."""
    rng = np.random.default_rng(seed)
    A = orthonormal(rng, 2 * d, d)
    a = np.array([one_loop_ols(rng, A, A.T, n, sigma) for _ in range(reps)])
    b = np.array([one_loop(rng, A, A.T, n, sigma) for _ in range(reps)])
    from scipy.stats import ks_2samp
    return {"d": d, "n": n, "reps": reps, "rms_ols": float(np.sqrt(np.mean(a ** 2))),
            "rms_exact": float(np.sqrt(np.mean(b ** 2))), "q95_ols": float(np.quantile(a, 0.95)),
            "q95_exact": float(np.quantile(b, 0.95)), "ks_p": float(ks_2samp(a, b).pvalue)}


def drift_curve(seed: int, dims=(8, 32, 128), sigma=0.1, reps=None) -> list[dict]:
    rng = np.random.default_rng(seed)
    reps = reps or {8: 400, 32: 200, 128: 60}
    out = []
    for d in dims:
        m = 2 * d
        A = orthonormal(rng, m, d)
        ns = [2 * m + 16] + [n for n in (512, 1024, 2048, 4096, 8192, 16384) if n > 2 * m + 16]
        for n in ns:
            vals = np.array([one_loop(rng, A, A.T, n, sigma) for _ in range(reps[d])])
            out.append({"d": d, "m": m, "n": n, "reps": reps[d],
                        "rms_sim": float(np.sqrt(np.mean(vals ** 2))),
                        "rms_analytic": analytic_rms(sigma, d, m, n),
                        "q05": float(np.quantile(vals, 0.05)), "q95": float(np.quantile(vals, 0.95)),
                        "min": float(vals.min())})
    return out


def strict_drift(seed: int, d: int = 32) -> float:
    rng = np.random.default_rng(seed)
    A = orthonormal(rng, 2 * d, d)
    return float(np.abs(A.T @ A - np.eye(d)).max())


def power_study(seed: int, d=32, sigma=0.1, ns=(256, 1024, 4096, 16384), null_reps=4000,
                alt_reps=1000, multiples=None, alpha=0.05, misjudged=(-0.10, -0.05, 0.05, 0.10)) -> dict:
    """The loop check as a test.

    Threshold: the null 95th percentile with the noise level known exactly (the best case).
    Alternative: a pop that breaks the law by delta, the RMS relative loop error it alone
    would produce; delta is set as a multiple of the null RMS drift at each n.
    Misjudged noise: the threshold a tester would set believing the noise is sigma(1 + e);
    to first order the null drift scales with sigma, so that threshold is (1 + e) times the
    true one, and its false-alarm rate is read off the same null draws.
    """
    rng = np.random.default_rng(seed)
    m = 2 * d
    A = orthonormal(rng, m, d)
    K = rng.standard_normal((d, d))
    K *= np.sqrt(d) / np.linalg.norm(K)
    multiples = multiples if multiples is not None else np.round(np.arange(0.0, 1.01, 0.1), 2)
    res = {"d": d, "m": m, "sigma": sigma, "alpha": alpha, "multiples": [float(x) for x in multiples],
           "by_n": []}
    for n in ns:
        null = np.array([one_loop(rng, A, A.T, n, sigma) for _ in range(null_reps)])
        thr = float(np.quantile(null, 1 - alpha))
        rms0 = float(np.sqrt(np.mean(null ** 2)))
        power, power_over = [], []
        for k in multiples:
            delta = k * rms0
            B1 = A.T + delta * K @ A.T
            vals = np.array([one_loop(rng, A, B1, n, sigma) for _ in range(alt_reps)])
            power.append(float(np.mean(vals > thr)))
            power_over.append(float(np.mean(vals > 1.10 * thr)))

        def k_at_80(pw):
            p = np.array(pw)
            j = int(np.argmax(p >= 0.8)) if np.any(p >= 0.8) else None
            if j is None:
                return float("nan")
            if j == 0:
                return float(multiples[0])
            return float(multiples[j - 1] + (0.8 - p[j - 1]) * (multiples[j] - multiples[j - 1]) / (p[j] - p[j - 1]))

        k80, k80_over = k_at_80(power), k_at_80(power_over)
        size_mis = {f"{e:+.0%}": float(np.mean(null > (1 + e) * thr)) for e in misjudged}
        res["by_n"].append({"n": n, "threshold": thr, "null_rms": rms0,
                            "analytic_rms": analytic_rms(sigma, d, m, n), "power": power,
                            "k80": k80, "delta80": k80 * rms0,
                            "power_noise_overestimated_10pct": power_over, "k80_noise_overestimated_10pct": k80_over, "size": float(np.mean(null > thr)),
                            "null_cv": float(np.std(null) / np.mean(null)), "size_misjudged": size_mis})
    return res
