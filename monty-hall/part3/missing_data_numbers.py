"""Every number quoted in Part 3 of the Monty Hall series.

Part 3 reads the host as a selection mechanism. The contents of the three
doors are the complete data, the opened door is the observed part, and the
rule that decides which door is opened is a missing-data mechanism in the
sense of Rubin (1976). The script

1. writes the Monty Hall problem as missing data, computes the ignorable and
   the full-likelihood answers, checks coarsening at random, and gives the
   identification region when the mechanism is unknown;
2. works the "broadcast" example: estimating how often switching wins from
   clips selected on the outcome;
3. repeats the analysis on real data with the same structure: the Coat
   shopping data of Schnabel et al. (2016), in which 290 people rated 24
   coats they chose themselves and 16 coats chosen for them at random.
   It measures the selection on the hidden rating, separates the part of
   the bias explained by which coats get rated from the part explained by
   who rates them, and shows what the selection does to the offline
   evaluation of four rating models.

Data: the Coat files are downloaded on first run from a pinned copy in the
AutoDebias repository (github.com/DongHande/AutoDebias), checked against
SHA-256 digests and cached in data/ (not committed). The original source is
https://www.cs.cornell.edu/~schnabts/mnar/

Run:  python missing_data_numbers.py > output.txt
"""

from __future__ import annotations

import hashlib
import math
import urllib.request
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy import linalg, optimize

SEED = 1976
HERE = Path(__file__).parent
DATA = HERE / "data"
MIRROR = "https://raw.githubusercontent.com/DongHande/AutoDebias/39101ad5a67194ef3828268102cf9b23487cb759/datasets/coat/"
FILES = {
    "train.ascii": "a87bd9e7eaf3b89f976df3f25d4aa5d5d871be8a00b58947df9e6e015559bd37",
    "test.ascii": "de6714ab105a5e0be8166be7fd313901b29f9a09c3bea307901457d1d71847e6",
    "user_item_features/item_features.ascii": None,
    "user_item_features/item_features_map.txt": None,
    "user_item_features/user_features.ascii": None,
    "user_item_features/user_features_map.txt": None,
}


# ---------------------------------------------------------------------------
# 1. Monty Hall as missing data
# ---------------------------------------------------------------------------

def monty_missing_data():
    """Complete data: which door hides the car. Observation: the host opens door 3 and shows a goat.

    Returns the ignorable answer (drop the mechanism), the full answer for a
    knowing host with tie-break q = 1/2, the full answer for an ignorant host,
    and the CAR ratios P(open 3 | car 1) / P(open 3 | car 2).
    """
    prior = {1: F(1, 3), 2: F(1, 3), 3: F(1, 3)}
    # Ignorable analysis: condition on the event "door 3 hides a goat" only.
    consistent = {c: p for c, p in prior.items() if c != 3}
    ignorable = consistent[2] / sum(consistent.values())
    # Full analysis, knowing host: P(open 3 | car) = 1/2, 1, 0 for car = 1, 2, 3.
    mech_know = {1: F(1, 2), 2: F(1), 3: F(0)}
    num = prior[2] * mech_know[2]
    den = sum(prior[c] * mech_know[c] for c in (1, 2))
    full_know = num / den
    # Ignorant host: opens 2 or 3 at random; door 3 shows a goat unless the car is there.
    mech_ign = {1: F(1, 2), 2: F(1, 2), 3: F(0)}  # P(open 3 and goat | car)
    full_ign = prior[2] * mech_ign[2] / sum(prior[c] * mech_ign[c] for c in (1, 2))
    car_know = mech_know[1] / mech_know[2]
    car_ign = mech_ign[1] / mech_ign[2]
    return ignorable, full_know, full_ign, car_know, car_ign


def switch_given_ratio(lam: F) -> F:
    """P(switch wins | door 3 opened) when P(open 3 | car 1) / P(open 3 | car 2) = lam."""
    return 1 / (1 + lam)


# ---------------------------------------------------------------------------
# 2. Clips selected on the outcome
# ---------------------------------------------------------------------------

def clip_estimate(p_win: F, s_win: F, s_loss: F) -> F:
    """Share of wins among clips, when a won game is shown w.p. s_win and a lost one w.p. s_loss."""
    return p_win * s_win / (p_win * s_win + (1 - p_win) * s_loss)


# ---------------------------------------------------------------------------
# 3. Coat
# ---------------------------------------------------------------------------

def fetch_coat() -> None:
    for name, digest in FILES.items():
        path = DATA / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(MIRROR + name, timeout=120) as r:
                path.write_bytes(r.read())
        if digest is not None:
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            if got != digest:
                raise RuntimeError(f"{name}: SHA-256 {got} does not match the pinned digest {digest}")


def load_coat():
    fetch_coat()
    train = np.loadtxt(DATA / "train.ascii").astype(int)  # self-selected ratings, 0 = not rated
    test = np.loadtxt(DATA / "test.ascii").astype(int)    # ratings of randomly chosen coats
    item_x = np.loadtxt(DATA / "user_item_features/item_features.ascii")
    user_x = np.loadtxt(DATA / "user_item_features/user_features.ascii")
    return train, test, item_x, user_x


def rating_shares(values: np.ndarray) -> np.ndarray:
    return np.array([(values == k).mean() for k in range(1, 6)])


def nb_selection(sel_values: np.ndarray, mar_values: np.ndarray, p_obs: float, smooth: float = 0.0) -> np.ndarray:
    """Naive-Bayes selection probabilities P(rated | rating = k), k = 1..5.

    P(O | r) = P(r | O) P(O) / P(r), with P(r | O) from the self-selected
    ratings and P(r) from ratings of randomly chosen items. `smooth` adds that
    many pseudo-counts to each rating level of the random sample (used when the
    random sample is small enough for a level to be empty).
    """
    counts = np.array([(mar_values == k).sum() for k in range(1, 6)], dtype=float) + smooth
    return rating_shares(sel_values) * p_obs / (counts / counts.sum())


def snips_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def tilt_mean(values: np.ndarray, gamma: float) -> float:
    """Mean rating if P(rated | r) is proportional to exp(gamma * r) (exponential tilting)."""
    w = np.exp(-gamma * values)
    return snips_mean(values, w)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """L2-penalised logistic regression (intercept unpenalised) by L-BFGS."""
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])

    def loss(beta):
        z = Xb @ beta
        ll = np.sum(np.logaddexp(0, z) - y * z)
        pen = 0.5 * l2 * np.sum(beta[1:] ** 2)
        grad = Xb.T @ (1 / (1 + np.exp(-z)) - y)
        grad[1:] += l2 * beta[1:]
        return ll + pen, grad

    res = optimize.minimize(loss, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B",
                            options={"maxiter": 5000, "gtol": 1e-8})
    # The loss is strictly convex, so a few Newton steps from the L-BFGS point reach the optimum to
    # near machine precision; without them the stopping point (and the last printed digit) depends
    # on the BLAS kernel of the machine.
    beta = res.x
    ridge = np.full(Xb.shape[1], l2)
    ridge[0] = 0.0
    for _ in range(10):
        p = 1 / (1 + np.exp(-(Xb @ beta)))
        grad = Xb.T @ (p - y) + ridge * beta
        hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(ridge)
        step = linalg.solve(hess, grad, assume_a="pos")
        beta = beta - step
        if np.max(np.abs(step)) < 1e-11:
            break
    return beta


def covariate_propensities(obs: np.ndarray, item_x: np.ndarray, user_x: np.ndarray,
                           spec: str = "full") -> np.ndarray:
    """P(rated | user features, item features): logistic regression on all user-item pairs (a MAR model).

    spec "main": main effects of user and coat features; "gender": plus the user-gender x coat-gender
    interaction; "full": plus every user-feature x coat-feature interaction; "coat": one effect per coat.
    """
    n_users, n_items = obs.shape
    U = np.repeat(user_x, n_items, axis=0)
    I = np.tile(item_x, (n_users, 1))
    if spec == "coat":
        X = np.tile(np.eye(n_items)[:, 1:], (n_users, 1))
    elif spec == "main":
        X = np.hstack([U, I])
    elif spec == "gender":
        X = np.hstack([U, I, U[:, :2] * I[:, :1], U[:, :2] * I[:, 1:2]])
    else:
        inter = (U[:, :, None] * I[:, None, :]).reshape(U.shape[0], -1)
        X = np.hstack([U, I, inter])
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    beta = fit_logistic(X, obs.reshape(-1).astype(float), l2=1.0)
    z = beta[0] + X @ beta[1:]
    return (1 / (1 + np.exp(-z))).reshape(n_users, n_items)


# --- rating models trained on self-selected ratings --------------------------

def fit_biases(R: np.ndarray, M: np.ndarray, W: np.ndarray, lam: float, iters: int = 30):
    """Weighted ridge model r_ui = mu + b_u + c_i, by alternating exact updates."""
    mu = np.sum(W * M * R) / np.sum(W * M)
    bu = np.zeros(R.shape[0])
    ci = np.zeros(R.shape[1])
    WM = W * M
    for _ in range(iters):
        bu = np.sum(WM * (R - mu - ci[None, :]), axis=1) / (np.sum(WM, axis=1) + lam)
        ci = np.sum(WM * (R - mu - bu[:, None]), axis=0) / (np.sum(WM, axis=0) + lam)
        mu = np.sum(WM * (R - bu[:, None] - ci[None, :])) / np.sum(WM)
    return mu, bu, ci


def fit_mf(R: np.ndarray, M: np.ndarray, W: np.ndarray, k: int, lam: float, rng, iters: int = 40):
    """Weighted ALS matrix factorisation with biases: r_ui = mu + b_u + c_i + p_u . q_i."""
    mu, bu, ci = fit_biases(R, M, W, lam)
    n_u, n_i = R.shape
    P = 0.1 * rng.standard_normal((n_u, k))
    Q = 0.1 * rng.standard_normal((n_i, k))
    WM = W * M
    eye = np.eye(k)
    for _ in range(iters):
        E = R - mu - bu[:, None] - ci[None, :]
        for u in range(n_u):
            w = WM[u]
            A = (Q * w[:, None]).T @ Q + lam * eye
            P[u] = np.linalg.solve(A, (Q * w[:, None]).T @ E[u])
        for i in range(n_i):
            w = WM[:, i]
            A = (P * w[:, None]).T @ P + lam * eye
            Q[i] = np.linalg.solve(A, (P * w[:, None]).T @ E[:, i])
        resid = R - P @ Q.T
        mu, bu, ci = fit_biases(resid, M, W, lam, iters=5)
    return mu, bu, ci, P, Q


def predict_biases(model):
    mu, bu, ci = model
    return mu + bu[:, None] + ci[None, :]


def predict_mf(model):
    mu, bu, ci, P, Q = model
    return mu + bu[:, None] + ci[None, :] + P @ Q.T


def evaluate_models(train: np.ndarray, test: np.ndarray, seed: int = SEED + 10, frac: float = 1.0) -> dict:
    """Hold out 4 of each user's 24 self-selected ratings, split the users of the random sample
    into halves (A estimates the selection probabilities, B is the truth), and fit six models
    on the remaining self-selected ratings, unweighted and inverse-propensity weighted."""
    rng = np.random.default_rng(seed)
    n_u, n_i = train.shape
    obs = train > 0
    hold = np.zeros_like(obs)
    for u in range(n_u):
        idx = np.flatnonzero(obs[u])
        hold[u, rng.choice(idx, 4, replace=False)] = True
    tr_mask = obs & ~hold
    users = rng.permutation(n_u)
    half_a, half_b = np.sort(users[: n_u // 2]), np.sort(users[n_u // 2:])
    if frac < 1.0:
        half_a = np.sort(rng.choice(half_a, max(1, int(round(frac * len(half_a)))), replace=False))
    mar_a = test[half_a][test[half_a] > 0]
    mask_b = np.zeros_like(obs)
    mask_b[half_b] = test[half_b] > 0
    pr_tr = nb_selection(train[tr_mask], mar_a, tr_mask.mean(), smooth=1.0 if frac < 1.0 else 0.0)
    R = train.astype(float)
    W1 = np.ones_like(R)
    Wips = np.where(tr_mask, 1 / pr_tr[np.clip(train, 1, 5) - 1], 0.0)
    Wips = Wips / Wips[tr_mask].mean()
    mu = R[tr_mask].mean()
    item_mean = ((R * tr_mask).sum(axis=0) + 5.0 * mu) / (tr_mask.sum(axis=0) + 5.0)
    preds = {
        "global mean": np.full_like(R, mu),
        "coat mean (shrunk)": np.tile(item_mean, (n_u, 1)),
        "user + coat biases": predict_biases(fit_biases(R, tr_mask, W1, lam=5.0)),
        "matrix factorisation (k = 4)": predict_mf(fit_mf(R, tr_mask, W1, 4, 5.0, np.random.default_rng(SEED))),
        "user + coat biases, IPS-weighted": predict_biases(fit_biases(R, tr_mask, Wips, lam=5.0)),
        "matrix factorisation, IPS-weighted": predict_mf(fit_mf(R, tr_mask, Wips, 4, 5.0,
                                                                np.random.default_rng(SEED))),
    }
    preds = {k: np.clip(v, 1, 5) for k, v in preds.items()}
    w_hold = 1 / pr_tr[np.clip(train[hold], 1, 5) - 1]
    rows = []
    for name, P in preds.items():
        err_h = (P[hold] - R[hold]) ** 2
        rows.append((name, err_h.mean(), np.sum(err_h * w_hold) / np.sum(w_hold),
                     np.mean((P[mask_b] - test[mask_b]) ** 2), P[mask_b].mean()))
    ess = np.sum(w_hold) ** 2 / np.sum(w_hold ** 2)
    return {"tr_mask": tr_mask, "hold": hold, "mask_b": mask_b, "half_a": half_a, "half_b": half_b,
            "mar_a": mar_a, "pr_tr": pr_tr, "preds": preds, "rows": rows, "ess": ess,
            "truth_mean": test[mask_b].mean()}


def repeated_evaluation(train: np.ndarray, test: np.ndarray, n_splits: int = 100):
    """Repeat evaluate_models over independent splits. Returns model names and an array of shape
    (n_splits, n_models, 3) holding naive, SNIPS and true MSE."""
    names = None
    out, extra = [], []
    for s in range(n_splits):
        ev = evaluate_models(train, test, seed=SEED + 100 + s)
        if names is None:
            names = [r[0] for r in ev["rows"]]
        out.append([[r[1], r[2], r[3]] for r in ev["rows"]])
        extra.append([ev["ess"], ev["truth_mean"]] + [r[4] for r in ev["rows"]])
    repeated_evaluation.extra = np.array(extra)
    return names, np.array(out)


# ---------------------------------------------------------------------------
# 4. A host who records his rule: the Open Bandit Dataset sample
# ---------------------------------------------------------------------------

OBD_MIRROR = "https://raw.githubusercontent.com/st-tech/zr-obp/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/obd/"
OBD_FILES = {
    "bts/all/all.csv": "0ad874e4dbf6902f0845dd478ad8dde5ef6903583d3ffaace78411bdad064106",
    "random/all/all.csv": "7168295b6e0a9eabcf3392320a5dd434e542b68e705d5cd9491499af589812f1",
}


def load_obd() -> dict:
    """Small sample (10,000 impressions per logging policy, 'All' campaign) shipped with the
    Open Bandit Pipeline (Apache-2.0). Returns item, position, click and logged propensity."""
    out = {}
    for name, digest in OBD_FILES.items():
        path = DATA / "obd" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(OBD_MIRROR + name, timeout=120) as r:
                path.write_bytes(r.read())
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != digest:
            raise RuntimeError(f"{name}: SHA-256 {got} does not match the pinned digest {digest}")
        import csv
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        out[name.split("/")[0]] = {
            "item": np.array([int(r["item_id"]) for r in rows]),
            "click": np.array([int(r["click"]) for r in rows]),
            "p": np.array([float(r["propensity_score"]) for r in rows]),
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(x) -> str:
    return f"{x} = {float(x):.4f}" if isinstance(x, F) else f"{x:.4f}"


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("=" * 76)
    print("1. MONTY HALL AS MISSING DATA (contestant holds door 1, door 3 opened, goat)")
    print("=" * 76)
    ign, full_k, full_i, car_k, car_i = monty_missing_data()
    print(f"  ignorable analysis (mechanism dropped):       P(switch wins) = {fmt(ign)}")
    print(f"  full likelihood, knowing host, fair tie-break: P(switch wins) = {fmt(full_k)}")
    print(f"  full likelihood, ignorant host:                P(switch wins) = {fmt(full_i)}")
    print(f"  CAR check P(open 3 | car 1) / P(open 3 | car 2): knowing host {car_k}, ignorant host {car_i}"
          " (CAR holds iff the ratio is 1)")
    print("  unknown mechanism: ratio lam in [0, 1] gives P(switch wins) = 1/(1 + lam) in [1/2, 1]")
    for lam in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
        print(f"    lam = {str(lam):>4}: {fmt(switch_given_ratio(lam))}")
    print("  uniform prior on the tie-break q (= lam for the knowing host):")
    print(f"    averaging the conditional answer 1/(1+q) over q: ln 2 = {math.log(2):.4f}  (not the predictive)")
    print("    predictive: P(car 2, open 3) = 1/3, P(car 1, open 3) = E[q]/3 = 1/6, so P(switch wins | open 3) = 2/3")
    print()

    print("=" * 76)
    print("2. CLIPS SELECTED ON THE OUTCOME (true switching win rate 2/3)")
    print("=" * 76)
    p = F(2, 3)
    for label, sw, sl in (("every game shown (MCAR)", F(1), F(1)),
                          ("one game in ten shown at random (MCAR)", F(1, 10), F(1, 10)),
                          ("wins shown always, losses one time in five", F(1), F(1, 5)),
                          ("losses shown always, wins one time in five", F(1, 5), F(1)),
                          ("only wins shown", F(1), F(0))):
        print(f"  {label:<46} estimated win rate {fmt(clip_estimate(p, sw, sl))}")
    print("  the estimate is p s_w / (p s_w + (1 - p) s_l): the selection ratio s_w / s_l plays the"
          " role of the host's likelihood ratio")
    print()

    print("=" * 76)
    print("3. COAT: SELF-SELECTED AND RANDOMLY ASSIGNED RATINGS (Schnabel et al. 2016)")
    print("=" * 76)
    train, test, item_x, user_x = load_coat()
    n_u, n_i = train.shape
    obs = train > 0
    sel = train[obs]
    mar = test[test > 0]
    print(f"  {n_u} users, {n_i} items; self-selected ratings {obs.sum()} ({obs.sum(axis=1).min()} per user),"
          f" random ratings {(test > 0).sum()} ({(test > 0).sum(axis=1).min()} per user)")
    print(f"  user-item pairs rated in both samples: {((train > 0) & (test > 0)).sum()}")
    both = (train > 0) & (test > 0)
    print(f"    on those pairs: same rating {np.mean(train[both] == test[both]):.3f},"
          f" mean |difference| {np.mean(np.abs(train[both] - test[both])):.3f}")
    print("  rating shares 1..5, self-selected: " + " ".join(f"{x:.3f}" for x in rating_shares(sel)))
    print("  rating shares 1..5, random:        " + " ".join(f"{x:.3f}" for x in rating_shares(mar)))
    print(f"  mean rating: self-selected {sel.mean():.3f}, random {mar.mean():.3f},"
          f" difference {sel.mean() - mar.mean():.3f}")
    # user bootstrap for the difference
    boot = []
    for _ in range(2000):
        us = rng.integers(0, n_u, n_u)
        s = train[us][train[us] > 0]
        m = test[us][test[us] > 0]
        boot.append(s.mean() - m.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"    95% user-bootstrap interval for the difference: [{lo:.3f}, {hi:.3f}]")
    share_45_sel = np.mean(sel >= 4)
    share_45_mar = np.mean(mar >= 4)
    print(f"  share of ratings 4 or 5: self-selected {share_45_sel:.3f}, random {share_45_mar:.3f}")
    print()

    p_obs = obs.mean()
    pr = nb_selection(sel, mar, p_obs)
    print(f"  P(rated) overall = {p_obs:.4f}")
    print("  naive-Bayes selection probability P(rated | rating = k), k = 1..5: "
          + " ".join(f"{x:.4f}" for x in pr))
    print(f"  ratio P(rated | 5) / P(rated | 1) = {pr[4] / pr[0]:.2f};"
          f" P(rated | 4) / P(rated | 1) = {pr[3] / pr[0]:.2f}")
    boot_pr = []
    for _ in range(2000):
        us = rng.integers(0, n_u, n_u)
        s = train[us][train[us] > 0]
        m = test[us][test[us] > 0]
        boot_pr.append(nb_selection(s, m, p_obs))
    boot_pr = np.array(boot_pr)
    lo_pr, hi_pr = np.percentile(boot_pr, [2.5, 97.5], axis=0)
    print("    95% user-bootstrap intervals: " + "; ".join(
        f"{k + 1}: [{lo_pr[k]:.4f}, {hi_pr[k]:.4f}]" for k in range(5)))
    slope = np.polyfit(np.arange(1, 6), np.log(pr), 1)[0]
    print(f"  log-linear fit of log P(rated | r) on r: slope {slope:.3f} per rating point")
    print()

    print("  Identification without and with the random sample:")
    lower = p_obs * sel.mean() + (1 - p_obs) * 1
    upper = p_obs * sel.mean() + (1 - p_obs) * 5
    print(f"    no assumption (Manski bounds on the mean over all user-item pairs): [{lower:.3f}, {upper:.3f}]")
    print(f"    ignorable (selection unrelated to the rating): {sel.mean():.3f}")
    gamma_star = optimize.brentq(lambda g: tilt_mean(sel.astype(float), g) - mar.mean(), 0, 5)
    print(f"    exponential tilting P(rated | r) ~ exp(gamma r): gamma = {gamma_star:.3f} reproduces"
          f" the random-sample mean {mar.mean():.3f}")
    for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        print(f"      gamma = {g:.1f}: implied mean rating {tilt_mean(sel.astype(float), g):.3f}")
    delta = (sel.mean() - mar.mean()) / (1 - p_obs)
    print(f"    pattern-mixture alternative: unrated pairs would have to average {delta:.3f} points below"
          f" the rated ones (mean {sel.mean() - delta:.3f}) to reproduce the random-sample mean")
    print(f"    SNIPS mean with the naive-Bayes propensities (depend only on the rating):"
          f" {snips_mean(sel.astype(float), 1 / pr[sel - 1]):.3f} = the random-sample mean by construction")
    print()

    print("  Which coats get rated, and who rates them:")
    item_sel_n = obs.sum(axis=0)
    item_mar_n = (test > 0).sum(axis=0)
    print(f"    self-selected ratings per coat: min {item_sel_n.min()}, median {int(np.median(item_sel_n))},"
          f" max {item_sel_n.max()}; coats never self-selected: {(item_sel_n == 0).sum()}")
    print(f"    random ratings per coat: min {item_mar_n.min()}, median {int(np.median(item_mar_n))},"
          f" max {item_mar_n.max()}")
    item_sel_mean = np.array([train[:, i][obs[:, i]].mean() if item_sel_n[i] else np.nan for i in range(n_i)])
    item_mar_mean = np.array([test[:, i][test[:, i] > 0].mean() if item_mar_n[i] else np.nan
                              for i in range(n_i)])
    ok = item_sel_n > 0
    print(f"    mean over coats of the random-rating mean (all coats): {np.nanmean(item_mar_mean):.3f}")
    print(f"    mean over coats of the random-rating mean, coats ever self-selected ({ok.sum()}):"
          f" {np.nanmean(item_mar_mean[ok]):.3f}")
    post = np.nanmean(item_sel_mean[ok])
    print(f"    post-stratified by coat (each self-selected coat weighted equally): {post:.3f}")
    print(f"    decomposition of the gap {sel.mean():.3f} - {mar.mean():.3f} = {sel.mean() - mar.mean():.3f}:")
    print(f"      which coats get rated (naive minus post-stratified): {sel.mean() - post:.3f}")
    print(f"      who rates each coat (post-stratified minus the random mean over the same coats):"
          f" {post - np.nanmean(item_mar_mean[ok]):.3f}")
    print(f"      remainder (random mean over those coats minus the random mean over all ratings):"
          f" {np.nanmean(item_mar_mean[ok]) - mar.mean():.3f}")
    both_ok = (item_sel_n >= 5) & (item_mar_n >= 5)
    d = item_sel_mean[both_ok] - item_mar_mean[both_ok]
    print(f"    coats with at least 5 ratings of each kind: {both_ok.sum()};"
          f" self-selected mean exceeds random mean for {np.mean(d > 0):.3f} of them;"
          f" mean within-coat difference {d.mean():.3f}")
    print()

    ug = user_x[:, :2].argmax(axis=1)  # user gender: 0 men, 1 women
    ig = item_x[:, :2].argmax(axis=1)  # coat gender: 0 men's, 1 women's
    match = ug[:, None] == ig[None, :]
    print("  An observable part of the selection: coats of the user's own gender")
    for label, M in (("self-selected", train), ("random", test)):
        o = M > 0
        print(f"    {label:<13}: share own-gender {match[o].mean():.3f}; mean rating own-gender"
              f" {M[o & match].mean():.3f}, other-gender {M[o & ~match].mean():.3f}")
    both_pairs = (train > 0) & (test > 0)
    s_ex = train[(train > 0) & ~both_pairs]
    r_ex = test[(test > 0) & ~both_pairs]
    print(f"  excluding the {both_pairs.sum()} pairs present in both files: self-selected {s_ex.mean():.3f},"
          f" random {r_ex.mean():.3f}, difference {s_ex.mean() - r_ex.mean():.3f}")
    print()

    print("  Propensities from observed covariates only (logistic regressions on all 87,000 pairs;")
    print("  missing-at-random models of increasing richness):")
    gap = sel.mean() - mar.mean()
    for spec, label in (("coat", "one effect per coat"),
                        ("main", "user and coat features, main effects"),
                        ("gender", "+ user gender x coat gender"),
                        ("full", "+ every user x coat feature interaction")):
        prop = covariate_propensities(obs, item_x, user_x, spec)
        m_ = snips_mean(sel.astype(float), 1 / prop[obs])
        extra = f"; propensity range {prop.min():.4f} to {prop.max():.4f}" if spec == "full" else ""
        print(f"    {label:<42} SNIPS mean {m_:.3f}; share of the gap removed {(sel.mean() - m_) / gap:.3f}{extra}")
    print()

    print("=" * 76)
    print("4. COAT: OFFLINE EVALUATION OF RATING MODELS UNDER SELECTION")
    print("=" * 76)
    ev = evaluate_models(train, test)
    tr_mask, hold, mask_b, half_a, half_b, mar_a, pr_tr, preds = (
        ev["tr_mask"], ev["hold"], ev["mask_b"], ev["half_a"], ev["half_b"], ev["mar_a"], ev["pr_tr"], ev["preds"])
    R = train.astype(float)
    print(f"  training: {tr_mask.sum()} self-selected ratings; held out: {hold.sum()} self-selected;"
          f" truth: {mask_b.sum()} random ratings from {len(half_b)} users;"
          f" propensities from {len(mar_a)} random ratings of the other {len(half_a)} users")
    print("  P(rated | r) from the training half: " + " ".join(f"{x:.4f}" for x in pr_tr))
    test_f = test.astype(float)
    print("  model                                   | naive MSE on held-out | SNIPS MSE on held-out |"
          " true MSE on random | mean prediction on random pairs")
    rows = [r[:4] for r in ev["rows"]]
    for name, naive, snips, true, meanp in ev["rows"]:
        print(f"  {name:<40} | {naive:>21.3f} | {snips:>21.3f} | {true:>18.3f} | {meanp:.3f}")
    print(f"  mean of the true ratings on those random pairs: {test_f[mask_b].mean():.3f}")
    order = lambda col: [r[0] for r in sorted(rows, key=lambda r: r[col])]
    print("  ranking by naive MSE: " + " < ".join(order(1)))
    print("  ranking by SNIPS MSE: " + " < ".join(order(2)))
    print("  ranking by true MSE:  " + " < ".join(order(3)))
    print()

    n_splits = 100
    names, res = repeated_evaluation(train, test, n_splits)
    print(f"  Over {n_splits} independent splits (hold-out draw and halves of the random sample):")
    print("  model                                   | naive MSE     | SNIPS MSE     | true MSE")
    for j, name in enumerate(names):
        m = res[:, j, :].mean(axis=0)
        s = res[:, j, :].std(axis=0, ddof=1)
        print(f"  {name:<40} | {m[0]:.3f} ({s[0]:.3f}) | {m[1]:.3f} ({s[1]:.3f}) | {m[2]:.3f} ({s[2]:.3f})")
    ex = repeated_evaluation.extra
    print(f"  effective sample size of the 1,160 reweighted held-out ratings: mean {ex[:, 0].mean():.0f}"
          f" (range {ex[:, 0].min():.0f} to {ex[:, 0].max():.0f})")
    print(f"  mean true rating on the random pairs of the truth half: {ex[:, 1].mean():.3f}")
    print("  mean prediction on those pairs: " + "; ".join(
        f"{names[j]} {ex[:, 2 + j].mean():.3f}" for j in range(len(names))))
    from scipy.stats import kendalltau
    tau_n = np.array([kendalltau(res[s, :, 0], res[s, :, 2]).statistic for s in range(n_splits)])
    tau_s = np.array([kendalltau(res[s, :, 1], res[s, :, 2]).statistic for s in range(n_splits)])
    print(f"  Kendall tau with the true ranking, mean over splits: naive {tau_n.mean():.3f},"
          f" SNIPS {tau_s.mean():.3f}")
    best_true = res[:, :, 2].argmin(axis=1)
    best_naive = res[:, :, 0].argmin(axis=1)
    best_snips = res[:, :, 1].argmin(axis=1)
    print(f"  share of splits in which the model chosen is the truly best one: naive {np.mean(best_naive == best_true):.2f},"
          f" SNIPS {np.mean(best_snips == best_true):.2f}")
    print("  truly best model, share of splits: " + "; ".join(
        f"{names[k]} {np.mean(best_true == k):.2f}" for k in range(len(names)) if np.mean(best_true == k) > 0))
    print("  model chosen by naive evaluation, share of splits: " + "; ".join(
        f"{names[k]} {np.mean(best_naive == k):.2f}" for k in range(len(names)) if np.mean(best_naive == k) > 0))
    print()
    print("  How large a randomised sample is needed? Propensities from a share of the 145 users of half A:")
    for frac in (0.05, 0.1, 0.25, 0.5):
        agree, ips_better, taus, sizes = [], [], [], []
        for s in range(25):
            ev = evaluate_models(train, test, seed=SEED + 500 + s, frac=frac)
            rr = np.array([[r[1], r[2], r[3]] for r in ev["rows"]])
            sizes.append(len(ev["mar_a"]))
            nm = [r[0] for r in ev["rows"]]
            pairs = ((nm.index("user + coat biases"), nm.index("user + coat biases, IPS-weighted")),
                     (nm.index("matrix factorisation (k = 4)"), nm.index("matrix factorisation, IPS-weighted")))
            for a_, b_ in pairs:
                ips_better.append(rr[b_, 2] < rr[a_, 2])
                agree.append((rr[b_, 1] < rr[a_, 1]) == (rr[b_, 2] < rr[a_, 2]))
            taus.append(kendalltau(rr[:, 1], rr[:, 2]).statistic)
        print(f"    {frac:>4.0%} of half A ({np.mean(sizes):.0f} random ratings, add-one smoothing):"
              f" weighted model better in truth"
              f" {np.mean(ips_better):.2f}; SNIPS agrees on weighted vs unweighted {np.mean(agree):.2f};"
              f" mean Kendall tau {np.mean(taus):.3f}  (25 splits, two model pairs)")
    i_b, i_bw = names.index("user + coat biases"), names.index("user + coat biases, IPS-weighted")
    i_m, i_mw = names.index("matrix factorisation (k = 4)"), names.index("matrix factorisation, IPS-weighted")
    for a, b, label in ((i_b, i_bw, "biases"), (i_m, i_mw, "factorisation")):
        true_better = np.mean(res[:, b, 2] < res[:, a, 2])
        naive_better = np.mean(res[:, b, 0] < res[:, a, 0])
        snips_better = np.mean(res[:, b, 1] < res[:, a, 1])
        print(f"  {label}: IPS-weighted beats unweighted in truth in {true_better:.2f} of splits;"
              f" naive evaluation says so in {naive_better:.2f}; SNIPS in {snips_better:.2f}")
    print()
    print("=" * 76)
    print("5. A HOST WHO RECORDS HIS RULE: OPEN BANDIT DATASET SAMPLE (Saito et al. 2021)")
    print("=" * 76)
    obd = load_obd()
    for pol, label in (("bts", "Bernoulli Thompson sampling"), ("random", "uniform random")):
        d = obd[pol]
        cnt = np.bincount(d["item"], minlength=80)
        top5 = np.sort(cnt)[::-1][:5].sum() / len(d["item"])
        print(f"  {label:<28}: {len(d['p'])} impressions, {d['click'].sum()} clicks (CTR {d['click'].mean():.4f});"
              f" logged propensity min {d['p'].min():.6f}, median {np.median(d['p']):.4f}, max {d['p'].max():.4f};"
              f" five most-shown items take {top5:.3f} of impressions")
    b = obd["bts"]
    w = (1 / 80) / b["p"]
    ess = w.sum() ** 2 / np.sum(w ** 2)
    print(f"  evaluating the uniform policy from the Thompson-sampling log: weights (1/80) / p,"
          f" max {w.max():.0f}; effective sample size {ess:.0f} of {len(w)} ({ess / len(w):.3f})")
    ips = np.mean(w * b["click"])
    snips = np.sum(w * b["click"]) / np.sum(w)
    r_ = obd["random"]["click"]
    rng_b = np.random.default_rng(SEED + 9)
    boot_ips, boot_on = [], []
    for _ in range(2000):
        i = rng_b.integers(0, len(w), len(w))
        boot_ips.append(np.sum(w[i] * b["click"][i]) / np.sum(w[i]))
        j = rng_b.integers(0, len(r_), len(r_))
        boot_on.append(r_[j].mean())
    lo1, hi1 = np.percentile(boot_ips, [2.5, 97.5])
    lo2, hi2 = np.percentile(boot_on, [2.5, 97.5])
    print(f"  CTR of the uniform policy: SNIPS from the Thompson log {snips:.4f} [{lo1:.4f}, {hi1:.4f}]"
          f" (IPS {ips:.4f}); on-policy from its own log {r_.mean():.4f} [{lo2:.4f}, {hi2:.4f}];"
          f" naive mean of the Thompson log {b['click'].mean():.4f}")


if __name__ == "__main__":
    main()
