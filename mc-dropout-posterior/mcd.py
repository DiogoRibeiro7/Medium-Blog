"""Shared code for the MC dropout experiments.

Everything is written in NumPy so the experiments run on a laptop CPU and in CI
without a deep-learning framework:

- ``load_uci``: the UCI regression benchmark exactly as distributed with the
  MC dropout paper (Gal & Ghahramani 2016), with its published train/test
  splits and its hyperparameter grids, downloaded once from a pinned commit of
  github.com/yaringal/DropoutUncertaintyExps and cached in ``data/``.
- ``Nets``: M fully connected ReLU networks trained side by side with batched
  matrix products. Each member has its own training rows (so bootstrap
  replicates, inner validation splits and full-data fits share one loop), its
  own dropout rate and its own L2 weight decay. Dropout is applied before every
  dense layer, input included, as in the reference Keras code; training uses
  inverted dropout, MSE loss, Adam (Keras defaults) and minibatches of 128.
- ``gp_posterior``: an exact Gaussian-process posterior with fixed
  hyperparameters, used as a reference whose spread is known to shrink.
- small helpers: the MC dropout test log-likelihood as computed in the
  reference code, Gaussian log-likelihoods, risk-coverage curves, AUROC.
"""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).parent
DATA = HERE / "data"
# CI smoke runs redirect output so the committed results stay untouched.
RESULTS = Path(os.environ.get("MCD_RESULTS", HERE / "results"))
FIGURES = Path(os.environ.get("MCD_FIGURES", HERE / "figures"))
QUICK = os.environ.get("MCD_QUICK") == "1"  # tiny settings for a smoke run

UCI_COMMIT = "6eb4497628d12b0f300f4b4f6bdc386bebad565c"
UCI_URL = ("https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/"
           f"{UCI_COMMIT}/UCI_Datasets/{{name}}/data/{{file}}")

# Boston housing is left out: its variable B is built from the proportion of
# Black residents, and scikit-learn removed the data set for that reason.
UCI_SETS = ["yacht", "concrete", "energy", "wine-quality-red", "kin8nm",
            "power-plant", "naval-propulsion-plant", "protein-tertiary-structure"]
UCI_LABEL = {"yacht": "Yacht", "concrete": "Concrete", "energy": "Energy",
             "wine-quality-red": "Wine", "kin8nm": "Kin8nm",
             "power-plant": "Power plant", "naval-propulsion-plant": "Naval",
             "protein-tertiary-structure": "Protein"}

# Plot colours (validated categorical slots, fixed order) and ink.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def _fetch(name: str, file: str) -> Path:
    dest = DATA / name / file
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = UCI_URL.format(name=name, file=file)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                payload = r.read()
            dest.write_bytes(payload)
            return dest
        except Exception:  # pragma: no cover - network retry
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    return dest


def load_uci(name: str) -> dict:
    """Data, splits and the reference hyperparameter grids for one data set."""
    data = np.loadtxt(_fetch(name, "data.txt"))
    feat = np.loadtxt(_fetch(name, "index_features.txt"), ndmin=1).astype(int)
    targ = int(np.loadtxt(_fetch(name, "index_target.txt")))
    n_splits = int(np.loadtxt(_fetch(name, "n_splits.txt")))
    out = {
        "name": name,
        "X": data[:, feat],
        "y": data[:, targ],
        "n_splits": n_splits,
        "n_hidden": int(np.loadtxt(_fetch(name, "n_hidden.txt"))),
        "n_epochs": int(np.loadtxt(_fetch(name, "n_epochs.txt"))),
        "dropout_rates": np.loadtxt(_fetch(name, "dropout_rates.txt"), ndmin=1),
        "tau_values": np.loadtxt(_fetch(name, "tau_values.txt"), ndmin=1),
        "train": [], "test": [],
    }
    for k in range(n_splits):
        out["train"].append(np.loadtxt(_fetch(name, f"index_train_{k}.txt")).astype(int))
        out["test"].append(np.loadtxt(_fetch(name, f"index_test_{k}.txt")).astype(int))
    return out


# ----------------------------------------------------------------------------
# Batched networks
# ----------------------------------------------------------------------------

class Nets:
    """M one-output ReLU networks with the same layer sizes, trained together.

    Parameters
    ----------
    d : input dimension
    hidden : list of hidden-layer widths (e.g. [50])
    drop : (M,) dropout probabilities (probability of DROPPING a unit; 0 = none)
    reg : (M,) L2 coefficients added to the loss as reg * sum(W**2), kernels only
    seed : seed of this object's random generator
    """

    def __init__(self, d, hidden, drop, reg, seed=0):
        self.rng = np.random.default_rng(seed)
        self.drop = np.asarray(drop, float)
        self.M = len(self.drop)
        self.reg = np.broadcast_to(np.asarray(reg, float), (self.M,)).copy()
        self.keep = 1.0 - self.drop
        sizes = [d] + list(hidden) + [1]
        self.W, self.b = [], []
        for din, dout in zip(sizes[:-1], sizes[1:]):
            lim = np.sqrt(6.0 / (din + dout))  # Glorot uniform, Keras default
            self.W.append(self.rng.uniform(-lim, lim, (self.M, din, dout)))
            self.b.append(np.zeros((self.M, dout)))
        self.W = [w.astype(np.float32) for w in self.W]
        self.b = [b.astype(np.float32) for b in self.b]
        self.dm = np.flatnonzero(self.drop > 0)  # members that use dropout
        self.mx = self.sx = self.my = self.sy = None

    # -- helpers ---------------------------------------------------------
    def _masks(self, shapes):
        """Inverted-dropout masks, one per dense layer input (float32).

        Uniform 16-bit integers are compared with round(65536 q); the keep
        probability is therefore exact to within 1/65536.
        """
        out = []
        dm = self.dm
        thr = np.round(self.keep[dm] * 65536).astype(np.uint32)[:, None, None]
        scale = (1.0 / self.keep[dm]).astype(np.float32)[:, None, None]
        for shp in shapes:
            m = np.ones(shp, dtype=np.float32)
            if len(dm):
                sub = (len(dm),) + tuple(shp[1:])
                u = np.frombuffer(self.rng.bytes(2 * int(np.prod(sub))), dtype=np.uint16).reshape(sub)
                m[dm] = (u < thr) * scale
            out.append(m)
        return out

    def forward(self, X, masks):
        """X: (M, B, d) already normalised. Returns output (M, B) and cache."""
        h, cache = X, []
        L = len(self.W)
        for l in range(L):
            hin = h * masks[l]
            a = np.matmul(hin, self.W[l]) + self.b[l][:, None, :]
            cache.append((hin, a))
            h = np.maximum(a, 0.0) if l < L - 1 else a
        return h[..., 0], cache

    def loss_and_grads(self, X, y, masks, want_loss=True):
        """Per-member loss (MSE + L2) and gradients for given masks."""
        out, cache = self.forward(X, masks)
        B = X.shape[1]
        err = out - y
        loss = None
        if want_loss:
            loss = (np.mean(err.astype(np.float64) ** 2, axis=1)
                    + self.reg * sum(np.sum(W.astype(np.float64) ** 2, axis=(1, 2)) for W in self.W))
        g = (2.0 / B) * err[..., None]  # dL/d out, (M, B, 1)
        reg = self.reg.astype(X.dtype)[:, None, None]
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            hin, a = cache[l]
            if l < len(self.W) - 1:
                g = g * (a > 0)
            gW[l] = np.matmul(hin.transpose(0, 2, 1), g) + 2 * reg * self.W[l]
            gb[l] = g.sum(axis=1)
            if l > 0:
                g = np.matmul(g, self.W[l].transpose(0, 2, 1)) * masks[l]
        return loss, gW, gb

    # -- training --------------------------------------------------------
    def fit(self, X, y, pools, steps, batch=128, lr=1e-3, b1=0.9, b2=0.999, eps=1e-7,
            val=None, n_checks=100):
        """Train every member on its own rows of (X, y).

        pools : list of M integer index arrays into X (bootstrap pools may repeat
                rows). Each member normalises inputs and target with the mean and
                standard deviation of its own pool, as the reference code does.
        steps : number of Adam steps; each step draws a minibatch of ``batch``
                rows uniformly from each member's pool.
        val   : optional index array of validation rows shared by all members.
                If given, the squared error of each member's deterministic
                prediction (all units kept) on these rows is checked
                ``n_checks`` times during training, and each member ends with
                the weights of its best check (early stopping without a
                stopping rule: training always runs ``steps`` steps).
        """
        M = self.M
        self.mx = np.stack([X[p].mean(0) for p in pools])
        sx = np.stack([X[p].std(0) for p in pools])
        sx[sx == 0] = 1.0
        self.sx = sx
        self.my = np.array([y[p].mean() for p in pools])
        self.sy = np.array([y[p].std() for p in pools])
        lens = np.array([len(p) for p in pools])
        poolmat = np.zeros((M, lens.max()), dtype=np.int64)
        for m, p in enumerate(pools):
            poolmat[m, :len(p)] = p
        params = self.W + self.b
        mom = [np.zeros_like(p) for p in params]
        vel = [np.zeros_like(p) for p in params]
        rows = np.arange(M)[:, None]
        widths = [W.shape[1] for W in self.W]
        if val is not None:
            every = max(1, steps // n_checks)
            Xv = ((X[val][None, :, :] - self.mx[:, None, :]) / self.sx[:, None, :]).astype(np.float32)
            yv = (y[val][None, :] - self.my[:, None]) / self.sy[:, None]
            best_err = np.full(M, np.inf)
            best = [p.copy() for p in params]
            self.best_step = np.zeros(M, dtype=int)
        for t in range(1, steps + 1):
            j = (self.rng.random((M, batch)) * lens[:, None]).astype(np.int64)
            idx = poolmat[rows, j]
            Xb = ((X[idx] - self.mx[:, None, :]) / self.sx[:, None, :]).astype(np.float32)
            yb = ((y[idx] - self.my[:, None]) / self.sy[:, None]).astype(np.float32)
            masks = self._masks([(M, batch, w) for w in widths])
            _, gW, gb = self.loss_and_grads(Xb, yb, masks, want_loss=False)
            grads = gW + gb
            lr_t = np.float32(lr * np.sqrt(1 - b2 ** t) / (1 - b1 ** t))
            for p, g, mm, vv in zip(params, grads, mom, vel):
                mm *= np.float32(b1)
                mm += np.float32(1 - b1) * g
                vv *= np.float32(b2)
                vv += np.float32(1 - b2) * g * g
                p -= lr_t * mm / (np.sqrt(vv) + np.float32(eps))
            if val is not None and (t % every == 0 or t == steps):
                h = Xv
                for l in range(len(self.W)):
                    h = np.matmul(h, self.W[l]) + self.b[l][:, None, :]
                    if l < len(self.W) - 1:
                        h = np.maximum(h, 0.0)
                err = np.mean((h[..., 0] - yv) ** 2, axis=1)
                better = err < best_err
                if better.any():
                    best_err[better] = err[better]
                    self.best_step[better] = t
                    for bp, p in zip(best, params):
                        bp[better] = p[better]
        if val is not None:
            for bp, p in zip(best, params):
                p[...] = bp
        return self

    # -- prediction ------------------------------------------------------
    def predict(self, X, T=1, members=None, chunk=None, mc=True):
        """Predictions in the original target units.

        Returns an array (len(members), T, N). With mc=False (or for members
        with no dropout) the network is evaluated once with every unit kept
        (inverted dropout needs no rescaling) and the result is repeated T times
        only if T == 1; deterministic members always return T == 1 rows when
        called with mc=False.
        """
        members = range(self.M) if members is None else members
        N = X.shape[0]
        out = []
        for m in members:
            Xn = ((X - self.mx[m]) / self.sx[m]).astype(np.float32)
            Wm = [W[m] for W in self.W]
            bm = [b[m] for b in self.b]
            q = self.keep[m]
            if not mc or q == 1.0:
                h = Xn
                for l in range(len(Wm)):
                    h = h @ Wm[l] + bm[l]
                    if l < len(Wm) - 1:
                        h = np.maximum(h, 0.0)
                pred = h[:, 0][None, :]
                if mc and T > 1:
                    pred = np.repeat(pred, T, axis=0)
                out.append(pred * self.sy[m] + self.my[m])
                continue
            c = chunk or max(1, int(4e6 // (N * max(W.shape[0] for W in Wm))))
            samples = np.empty((T, N))
            for s in range(0, T, c):
                tc = min(c, T - s)
                h = np.broadcast_to(Xn, (tc, N, Xn.shape[1]))
                for l in range(len(Wm)):
                    u = np.frombuffer(self.rng.bytes(2 * int(np.prod(h.shape))),
                                      dtype=np.uint16).reshape(h.shape)
                    mask = (u < np.uint32(round(q * 65536))) * np.float32(1.0 / q)
                    h = (h * mask) @ Wm[l] + bm[l]
                    if l < len(Wm) - 1:
                        h = np.maximum(h, 0.0)
                samples[s:s + tc] = h[..., 0]
            out.append(samples * self.sy[m] + self.my[m])  # float64
        return np.stack(out)


def gal_weight_decay(drop, n, tau, lengthscale=1e-2):
    """L2 coefficient used in the reference code: l^2 (1 - p) / (2 N tau)."""
    return lengthscale ** 2 * (1.0 - np.asarray(drop)) / (2.0 * n * np.asarray(tau))


def steps_for(n_train, epochs, batch=128):
    """Adam steps equivalent to ``epochs`` passes of minibatches of 128."""
    return int(epochs * np.ceil(n_train / batch))


# ----------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------

def mc_loglik(y, samples, tau):
    """Per-point test log-likelihood of the MC dropout predictive density.

    The mixture of T Gaussians N(yhat_t, 1/tau), exactly as in the reference
    code (net.py, predict).
    """
    T = samples.shape[0]
    return (logsumexp(-0.5 * tau * (y[None, :] - samples) ** 2, axis=0) - np.log(T)
            - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau))


def gauss_loglik(y, mean, var):
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mean) ** 2 / var


def risk_coverage(abs_err, score, coverages):
    """RMSE on the fraction ``c`` of points with the lowest ``score``."""
    order = np.argsort(score, kind="stable")
    n = len(abs_err)
    out = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        out.append(np.sqrt(np.mean(abs_err[order[:k]] ** 2)))
    return np.array(out)


def auroc(score_pos, score_neg):
    """Probability that a random positive scores above a random negative."""
    s = np.concatenate([score_pos, score_neg])
    ranks = np.empty(len(s))
    order = np.argsort(s, kind="mergesort")
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n1, n0 = len(score_pos), len(score_neg)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def knn_distance(Xtr, Xte, k=10, chunk=2000):
    """Mean Euclidean distance to the k nearest training rows (standardised)."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    a2 = np.sum(A ** 2, 1)
    out = np.empty(len(B))
    for s in range(0, len(B), chunk):
        Bc = B[s:s + chunk]
        d2 = np.sum(Bc ** 2, 1)[:, None] + a2[None, :] - 2 * Bc @ A.T
        np.maximum(d2, 0, out=d2)
        part = np.partition(d2, k - 1, axis=1)[:, :k]
        out[s:s + chunk] = np.sqrt(part).mean(1)
    return out


# ----------------------------------------------------------------------------
# Gaussian process reference
# ----------------------------------------------------------------------------

def fit_gp_hypers(X, y, n_sub=1500, seed=0):
    """ARD squared-exponential + noise hyperparameters by marginal likelihood
    on a subsample (scikit-learn), on standardised inputs and target."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), size=min(n_sub, len(X)), replace=False)
    mx, sx = X.mean(0), X.std(0)
    sx[sx == 0] = 1
    my, sy = y.mean(), y.std()
    Xs, ys = (X[sub] - mx) / sx, (y[sub] - my) / sy
    d = X.shape[1]
    kern = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(np.ones(d), (1e-2, 1e3))
            + WhiteKernel(0.1, (1e-4, 1.0)))
    gp = GaussianProcessRegressor(kern, normalize_y=False, random_state=seed,
                                  n_restarts_optimizer=1).fit(Xs, ys)
    k = gp.kernel_
    return {"amp": float(k.k1.k1.constant_value),
            "ls": np.asarray(k.k1.k2.length_scale, float),
            "noise": float(k.k2.noise_level)}


def gp_posterior(Xtr, ytr, Xte, hyp, chunk=2000):
    """Posterior mean and latent standard deviation (original units), fixed
    hyperparameters, inputs/target standardised with the training rows."""
    from scipy.linalg import cho_factor, cho_solve, solve_triangular
    mx, sx = Xtr.mean(0), Xtr.std(0)
    sx[sx == 0] = 1
    my, sy = ytr.mean(), ytr.std()
    A = (Xtr - mx) / sx / hyp["ls"]
    B = (Xte - mx) / sx / hyp["ls"]
    a2 = np.sum(A ** 2, 1)
    K = hyp["amp"] * np.exp(-0.5 * np.maximum(a2[:, None] + a2[None, :] - 2 * A @ A.T, 0))
    K[np.diag_indices_from(K)] += hyp["noise"] + 1e-8
    cf = cho_factor(K, lower=True, overwrite_a=True)
    alpha = cho_solve(cf, (ytr - my) / sy)
    L = cf[0]
    mean = np.empty(len(B))
    sd = np.empty(len(B))
    for s in range(0, len(B), chunk):
        Bc = B[s:s + chunk]
        Ks = hyp["amp"] * np.exp(-0.5 * np.maximum(
            np.sum(Bc ** 2, 1)[:, None] + a2[None, :] - 2 * Bc @ A.T, 0))
        mean[s:s + chunk] = Ks @ alpha
        v = solve_triangular(L, Ks.T, lower=True, check_finite=False)
        sd[s:s + chunk] = np.sqrt(np.maximum(hyp["amp"] - np.sum(v ** 2, 0), 0))
    return mean * sy + my, sd * sy, hyp["noise"] * sy ** 2


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

class Report:
    """Collect lines, print them and write them to results/<name>.txt."""

    def __init__(self, name):
        self.name, self.lines = name, []

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self):
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{self.name}.txt").write_text("\n".join(self.lines) + "\n")


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
