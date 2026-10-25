"""A small linear mixed model fitted by profiled (RE)ML.

The parameterisation follows Bates, Maechler, Bolker and Walker (2015),
"Fitting Linear Mixed-Effects Models Using lme4", J. Stat. Softw. 67(1):

    y_j = X_j beta + Z_j b_j + e_j,   b_j ~ N(0, sigma^2 L L'),  e_j ~ N(0, sigma^2 I)

L is a q x q lower-triangular relative covariance factor. beta and sigma^2
are profiled out, so the optimiser only searches over the q(q+1)/2 entries
of L. Everything is computed from per-group cross-products, so a fit costs
O(J q^3) per likelihood evaluation regardless of the group sizes.

Only one grouping factor is supported (nested/crossed designs need lme4,
glmmTMB or MixedModels.jl). That is all the experiments in this folder need.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass
class LMMResult:
    beta: np.ndarray
    cov_beta: np.ndarray
    sigma2: float
    D: np.ndarray            # covariance matrix of the random effects
    blup: dict               # group label -> b_j
    loglik: float
    reml: bool
    n_groups: int
    n_obs: int

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov_beta))


class LMM:
    def __init__(self, reml: bool = True):
        self.reml = reml

    # ------------------------------------------------------------------ fit
    def fit(self, X, Z, y, groups) -> LMMResult:
        X = np.asarray(X, float)
        Z = np.asarray(Z, float)
        if Z.ndim == 1:
            Z = Z[:, None]
        y = np.asarray(y, float)
        labels, g = np.unique(np.asarray(groups), return_inverse=True)
        J, N, p, q = len(labels), len(y), X.shape[1], Z.shape[1]

        # per-group cross-products
        XtX = np.zeros((J, p, p)); XtZ = np.zeros((J, p, q)); ZtZ = np.zeros((J, q, q))
        Xty = np.zeros((J, p)); Zty = np.zeros((J, q)); yty = np.zeros(J)
        np.add.at(XtX, g, X[:, :, None] * X[:, None, :])
        np.add.at(XtZ, g, X[:, :, None] * Z[:, None, :])
        np.add.at(ZtZ, g, Z[:, :, None] * Z[:, None, :])
        np.add.at(Xty, g, X * y[:, None])
        np.add.at(Zty, g, Z * y[:, None])
        np.add.at(yty, g, y * y)
        self._cp = (XtX, XtZ, ZtZ, Xty, Zty, yty)
        self._dims = (J, N, p, q)
        tril = np.tril_indices(q)
        diag_pos = [i for i, (r, c) in enumerate(zip(*tril)) if r == c]

        def unpack(theta):
            L = np.zeros((q, q))
            L[tril] = theta
            return L

        def pieces(theta):
            L = unpack(theta)
            M = np.eye(q) + np.einsum("ai,jab,bk->jik", L, ZtZ, L)
            Minv = np.linalg.inv(M)
            _, logdetM = np.linalg.slogdet(M)
            # K_j = L M_j^{-1} L'  (q x q)
            K = np.einsum("ia,jab,kb->jik", L, Minv, L)
            ZtX = np.transpose(XtZ, (0, 2, 1))
            A = (XtX - np.einsum("jpa,jab,jbr->jpr", XtZ, K, ZtX)).sum(0)
            b = (Xty - np.einsum("jpa,jab,jb->jp", XtZ, K, Zty)).sum(0)
            c = (yty - np.einsum("ja,jab,jb->j", Zty, K, Zty)).sum()
            return A, b, c, logdetM.sum(), K

        def negll(theta):
            A, b, c, logdetM, _ = pieces(theta)
            try:
                beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return 1e300
            r = c - b @ beta
            if self.reml:
                s2 = r / (N - p)
                _, logdetA = np.linalg.slogdet(A)
                return 0.5 * ((N - p) * np.log(s2) + logdetM + logdetA + (N - p))
            s2 = r / N
            return 0.5 * (N * np.log(s2) + logdetM + N)

        k = len(tril[0])
        bounds = [(0.0, None) if i in diag_pos else (None, None) for i in range(k)]
        best = None
        for start in (1.0, 0.3, 3.0):
            th0 = np.zeros(k)
            th0[diag_pos] = start
            res = optimize.minimize(negll, th0, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        theta = best.x

        A, b, c, _, K = pieces(theta)
        beta = np.linalg.solve(A, b)
        r = c - b @ beta
        s2 = r / (N - p) if self.reml else r / N
        L = unpack(theta)
        ZtX = np.transpose(XtZ, (0, 2, 1))
        resid_proj = Zty - np.einsum("jap,p->ja", ZtX, beta)
        B = np.einsum("jab,jb->ja", K, resid_proj)
        const = 0.5 * ((N - p) if self.reml else N) * np.log(2 * np.pi)
        self.result = LMMResult(
            beta=beta, cov_beta=s2 * np.linalg.inv(A), sigma2=s2,
            D=s2 * L @ L.T, blup={lab: B[j] for j, lab in enumerate(labels)},
            loglik=-best.fun - const, reml=self.reml, n_groups=J, n_obs=N,
        )
        return self.result

    # -------------------------------------------------------------- predict
    def predict(self, X, Z, groups, use_blup: bool = True) -> np.ndarray:
        res = self.result
        X = np.asarray(X, float)
        Z = np.asarray(Z, float)
        if Z.ndim == 1:
            Z = Z[:, None]
        out = X @ res.beta
        if use_blup:
            zero = np.zeros(Z.shape[1])
            Bm = np.vstack([res.blup.get(gl, zero) for gl in np.asarray(groups)])
            out = out + (Z * Bm).sum(1)
        return out
