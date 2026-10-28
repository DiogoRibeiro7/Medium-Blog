"""Checks of the NumPy implementation before any experiment is run.

1. Backpropagation matches finite differences for fixed dropout masks, with
   several members, two hidden layers and weight decay.
2. Training recovers a known smooth function on noiseless data, and
   checkpointing on validation rows returns weights from before overfitting.
3. Inverted dropout: the mean over masks of a linear network equals the
   all-units-kept prediction, and the variance matches (1 - q)/q sum x_j^2 w_j^2.
4. The exact GP posterior matches scikit-learn with the same fixed kernel.
5. The MC log-likelihood equals a direct mixture computation.
6. The rank AUROC equals the brute-force pairwise probability.
"""

import numpy as np

from mcd import Nets, auroc, gp_posterior, mc_loglik

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    ok &= bool(cond)


# 1. gradient check -----------------------------------------------------------
rng = np.random.default_rng(1)
M, B, d = 3, 7, 4
net = Nets(d, [5, 6], drop=[0.0, 0.2, 0.5], reg=[0.0, 1e-3, 1e-2], seed=3)
net.W = [w.astype(np.float64) for w in net.W]  # float64 for finite differences
net.b = [b.astype(np.float64) for b in net.b]
for b in net.b:  # non-zero biases keep pre-activations away from the ReLU kink
    b[:] = rng.normal(scale=0.5, size=b.shape)
X = rng.normal(size=(M, B, d))
y = rng.normal(size=(M, B))
masks = net._masks([(M, B, w.shape[1]) for w in net.W])
loss, gW, gb = net.loss_and_grads(X, y, masks)
worst = 0.0
for plist, glist in ((net.W, gW), (net.b, gb)):
    for P, G in zip(plist, glist):
        it = np.nditer(P, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = P[i]
            P[i] = old + 1e-6
            lp = net.loss_and_grads(X, y, masks)[0][i[0]]
            P[i] = old - 1e-6
            lm = net.loss_and_grads(X, y, masks)[0][i[0]]
            P[i] = old
            num = (lp - lm) / 2e-6
            worst = max(worst, abs(num - G[i]) / max(1.0, abs(num)))
check("backprop vs finite differences", worst < 1e-6, f"max rel err {worst:.2e}")

# 2. learns a smooth function --------------------------------------------------
Xf = rng.uniform(-2, 2, size=(2000, 2))
yf = np.sin(Xf[:, 0]) + 0.5 * Xf[:, 1] ** 2
net = Nets(2, [50], drop=[0.0, 0.05], reg=[0.0, 0.0], seed=4)
net.fit(Xf, yf, [np.arange(2000)] * 2, steps=6000)
pred = net.predict(Xf, mc=False)
rmse = np.sqrt(np.mean((pred - yf[None, None, :]) ** 2, axis=(1, 2)))
check("fits a smooth function", rmse[0] < 0.05 * yf.std() and rmse[1] < 0.15 * yf.std(), f"rmse {np.round(rmse, 4)} (sd {yf.std():.3f})")

# 2b. checkpointing keeps the best validation weights ------------------------------
Xs = rng.uniform(-2, 2, size=(160, 1))
ys = np.sin(Xs[:, 0]) + 0.5 * rng.normal(size=160)
tr_rows, va_rows = np.arange(60), np.arange(60, 160)
es = Nets(1, [200], drop=[0.0, 0.0], reg=[0.0, 0.0], seed=6)
es.fit(Xs, ys, [tr_rows] * 2, steps=8000, val=va_rows, n_checks=80)
pv = es.predict(Xs[va_rows], mc=False)[:, 0, :]
err_best = np.mean((pv - ys[va_rows]) ** 2, axis=1)
full = Nets(1, [200], drop=[0.0, 0.0], reg=[0.0, 0.0], seed=6)
full.fit(Xs, ys, [tr_rows] * 2, steps=8000)  # same trajectory, no checkpointing
err_last = np.mean((full.predict(Xs[va_rows], mc=False)[:, 0, :] - ys[va_rows]) ** 2, axis=1)
check("checkpointing keeps the best validation weights",
      np.all(es.best_step < 8000) and np.all(err_best <= err_last + 1e-9),
      f"best steps {es.best_step}; validation MSE {np.round(err_best, 3)} vs {np.round(err_last, 3)} at the end")

# 3. inverted dropout moments on a linear model ----------------------------------
q = 0.8
lin = Nets(3, [], drop=[1 - q], reg=[0.0], seed=5)
lin.W[0][0] = np.array([[1.5], [-2.0], [0.5]], dtype=np.float32)
lin.b[0][0] = 0.3
lin.mx, lin.sx = np.zeros((1, 3)), np.ones((1, 3))
lin.my, lin.sy = np.zeros(1), np.ones(1)
x = np.array([[0.7, -1.2, 2.0]])
s = lin.predict(x, T=400000)[0, :, 0]
exact_mean = (x @ lin.W[0][0]).item() + 0.3
exact_var = (1 - q) / q * np.sum(x[0] ** 2 * lin.W[0][0][:, 0] ** 2)
check("MC mean = weight-scaled prediction (linear)", abs(s.mean() - exact_mean) < 0.015,
      f"{s.mean():.4f} vs {exact_mean:.4f}")
check("MC variance = (1-q)/q sum x^2 w^2 (linear)", abs(s.var() / exact_var - 1) < 0.01,
      f"{s.var():.4f} vs {exact_var:.4f}")

# 4. GP posterior vs scikit-learn -----------------------------------------------
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

Xg = rng.normal(size=(300, 3))
yg = np.sin(Xg[:, 0]) + 0.1 * rng.normal(size=300)
Xt = rng.normal(size=(50, 3))
hyp = {"amp": 1.3, "ls": np.array([0.8, 1.5, 2.0]), "noise": 0.05}
mean, sd, _ = gp_posterior(Xg, yg, Xt, hyp)
mx, sx = Xg.mean(0), Xg.std(0)
my, sy = yg.mean(), yg.std()
kern = ConstantKernel(1.3, "fixed") * RBF(hyp["ls"], "fixed") + WhiteKernel(0.05, "fixed")
gp = GaussianProcessRegressor(kern, optimizer=None, alpha=1e-8).fit((Xg - mx) / sx, (yg - my) / sy)
m2, s2 = gp.predict((Xt - mx) / sx, return_std=True)
s2_latent = np.sqrt(np.maximum(s2 ** 2 - 0.05, 0))
check("GP mean matches scikit-learn", np.allclose(mean, m2 * sy + my, atol=1e-6),
      f"max diff {np.abs(mean - (m2 * sy + my)).max():.1e}")
check("GP latent sd matches scikit-learn", np.allclose(sd, s2_latent * sy, atol=1e-5),
      f"max diff {np.abs(sd - s2_latent * sy).max():.1e}")

# 5. MC log-likelihood -----------------------------------------------------------
samp = rng.normal(size=(20, 5))
yy = rng.normal(size=5)
tau = 2.0
direct = np.log(np.mean(np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (yy - samp) ** 2), axis=0))
check("MC log-likelihood", np.allclose(mc_loglik(yy, samp, tau), direct), "")

# 6. AUROC ------------------------------------------------------------------------
a, b = rng.normal(1, 1, 400), rng.normal(0, 1, 500)
brute = np.mean(a[:, None] > b[None, :])
check("AUROC", abs(auroc(a, b) - brute) < 1e-12, f"{auroc(a, b):.4f}")

print("all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
