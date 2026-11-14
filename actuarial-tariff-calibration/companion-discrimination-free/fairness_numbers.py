"""Fifth companion to the actuarial-tariff-calibration series: leaving a protected attribute out of a tariff, and
the discrimination-free price.

Every number quoted in the article is printed by this script. Run:

    python fairness_numbers.py > output.txt     # about six minutes, single-threaded; writes figure_data.json

The question. Since 21 December 2012 insurers in the EU may not charge men and women different premiums for the same
cover. The usual way to comply is to leave sex out of the tariff and refit it (the unawareness price). Rating factors
that are correlated with sex then carry part of its effect into the price. Lindholm, Richman, Tsanakas and Wuthrich
(2022) define a discrimination-free price: fit the best-estimate model with the protected attribute D, then average
its prices over the portfolio's distribution of D, not over the distribution of D given the rating factors.

A small-effect approximation tells how much the unawareness price puts back. If the true log frequency is
f(x) + gamma d, the unawareness price is the discrimination-free price times (1 + p(x) c) / (1 + pi c), with
p(x) = P(D = 1 | x), pi = P(D = 1) and c = exp(gamma) - 1. The gap between the average log premiums of the two groups is
then about gamma times E[p(X) | D = 1] - E[p(X) | D = 0], against gamma for the best-estimate price. So the share of the
direct effect that the unawareness price restores is about Tjur's coefficient of discrimination of the classifier
p(x): the difference between its mean prediction in the two groups.

Sections
1. Belgium (beMTPL97, 163,212 policies, sex recorded): who the women in the portfolio are, and how well the rating
   factors predict sex (AUC and Tjur's coefficient, logistic regression on the GLM design and gradient boosting).
2. Belgium: three prices for each of two model classes (banded Poisson GLM, Poisson gradient boosting), over R random
   80/20 splits into a training and a test sample: best estimate (with sex), unawareness (sex left out and the model
   refitted) and discrimination-free (fitted with sex, averaged over the training sample's share of women). Every
   price is rescaled to collect the training sample's claims. The ratio of women's to men's average premium per
   policy-year, the share of the direct effect restored by the unawareness price, and the Tjur prediction.
3. Belgium: what each price costs: deviance and Gini on the test sample, actual over expected claims by sex, and by
   decile of the predicted probability that the policyholder is a woman (pooled over splits).
4. A controlled experiment with known costs, on freMTPL2: a synthetic protected attribute whose probability depends
   on rating factors with a chosen strength and shape (additive in the factors, or through an interaction), and a
   chosen direct effect. True expected claims are known, so the shares, the calibration and the deviance against
   the truth are exact up to the models' estimation error.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "companion-winners-curse"))
import curse_numbers as c1  # noqa: E402

p3 = c1.p3
from data import belgian_mtpl97  # noqa: E402
from tariffcal import gini, poisson_deviance  # noqa: E402

header = c1.header
SEED = 20261114
R = 20
CLASSES = ("GLM", "GBM")
PRICES = ("best estimate", "unawareness", "implied unawareness", "discrimination-free")
SHORT = {"best estimate": "BE", "unawareness": "U", "implied unawareness": "U*", "discrimination-free": "DF"}
TRAIN_SYN = 200_000
# Synthetic settings: (shape of the proxy, strength kappa, direct effect gamma)
SETTINGS = [("additive", 0.0, 0.2), ("additive", 1.0, 0.2), ("additive", 2.0, 0.2), ("additive", 4.0, 0.2),
            ("interaction", 2.0, 0.2), ("interaction", 4.0, 0.2), ("additive", 2.0, 0.5)]


def summary(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.3f} [{x.min():.3f}, {x.max():.3f}]"


def wmean(x, w) -> float:
    return float(np.sum(x * w) / np.sum(w))


def tjur(p, d, w) -> float:
    """Tjur's coefficient of discrimination, exposure-weighted: mean prediction among D = 1 minus among D = 0."""
    return wmean(p[d == 1], w[d == 1]) - wmean(p[d == 0], w[d == 0])


def gbm_fit(X, y, w, cat):
    g = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=2000, max_leaf_nodes=31,
                                      min_samples_leaf=200, early_stopping=True, validation_fraction=0.15,
                                      n_iter_no_change=30, random_state=7, categorical_features=cat)
    return g.fit(X, y, sample_weight=w)


def prices(cls, Xg, Xb, D, e, N, tr, te, cat, p_tr, p_te, total=None):
    """Four rates on the test sample for one model class.

    best estimate        the model fitted with D, at each policy's own D (direct discrimination)
    unawareness          the model refitted without D (what a pricing team does when D is banned)
    implied unawareness  the best-estimate model averaged over P(D | x): p(x) mu(x, 1) + (1 - p(x)) mu(x, 0), the price
                         the unawareness refit estimates, without the noise of a second fit
    discrimination-free  the best-estimate model averaged over the training sample's share of D: pi mu(x, 1) +
                         (1 - pi) mu(x, 0) (Lindholm, Richman, Tsanakas and Wuthrich 2022)
    Xg(idx) is the GLM design, Xb(idx) the boosting features; p_tr and p_te are P(D = 1 | x) on the two samples. Each
    rate is rescaled to collect `total` on the training sample (default: its claims). Returns the rates, pi and the
    direct effect (GLM coefficient, or the exposure-weighted mean log ratio of the boosting prices for D = 1 and 0)."""
    pi = wmean(D[tr], e[tr])
    total = N[tr].sum() if total is None else total
    out = {}
    if cls == "GLM":
        withD = lambda idx, d: np.column_stack([Xg(idx), d])  # noqa: E731
        b = c1.irls_ridge(withD(tr, D[tr]), N[tr], e[tr])
        bU = c1.irls_ridge(Xg(tr), N[tr], e[tr])
        mu = lambda idx, d: np.exp(withD(idx, d) @ b)  # noqa: E731
        un = lambda idx: np.exp(Xg(idx) @ bU)  # noqa: E731
    else:
        withD = lambda idx, d: np.column_stack([Xb(idx), d])  # noqa: E731
        m = gbm_fit(withD(tr, D[tr]), N[tr] / e[tr], e[tr], cat)
        mU = gbm_fit(Xb(tr), N[tr] / e[tr], e[tr], cat)
        mu = lambda idx, d: m.predict(withD(idx, d))  # noqa: E731
        un = lambda idx: mU.predict(Xb(idx))  # noqa: E731
    for idx, key, p in ((tr, "tr", p_tr), (te, "te", p_te)):
        one, zero = mu(idx, np.ones(idx.size)), mu(idx, np.zeros(idx.size))
        out[key] = {"best estimate": mu(idx, D[idx]), "unawareness": un(idx),
                    "implied unawareness": p * one + (1 - p) * zero,
                    "discrimination-free": pi * one + (1 - pi) * zero}
        if key == "te":
            effect = float(b[-1]) if cls == "GLM" else wmean(np.log(one / zero), e[idx])
            out["eff"] = np.log(one / zero)
    for k in PRICES:
        c = total / (e[tr] * out["tr"][k]).sum()
        out["te"][k] = c * out["te"][k]
    out["te"]["eff"] = out["eff"]
    return out["te"], pi, effect


def ratio(rate, D, e) -> float:
    """Women's average premium per policy-year over men's."""
    return wmean(rate[D == 1], e[D == 1]) / wmean(rate[D == 0], e[D == 0])


def deciles(p, w) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    cum = np.cumsum(w[order]) - 0.5 * w[order]
    dec = np.empty(p.size, dtype=int)
    dec[order] = np.minimum((10 * cum / w.sum()).astype(int), 9)
    return dec


# ---------------------------------------------------------------------------------------------------
# Belgium
# ---------------------------------------------------------------------------------------------------
def belgium() -> dict:
    P = c1.belgian()
    df = belgian_mtpl97()
    e, N = P["e"], P["N"]
    n = e.size
    D = (df["sex"].astype(str) == "female").to_numpy(dtype=float)
    age = df["ageph"].to_numpy(dtype=float)
    pw = df["power"].to_numpy(dtype=float)
    bm = df["bm"].to_numpy(dtype=float)
    cov = df["coverage"].astype(str).to_numpy()
    use = df["use"].astype(str).to_numpy()
    fleet = df["fleet"].astype(str).to_numpy()
    fuel = df["fuel"].astype(str).to_numpy()
    region = df["postcode"].to_numpy(dtype=int) // 1000
    cat = P["cat"]

    header("Belgium (beMTPL97): who the women in the portfolio are")
    print(f"  Policies {n:,}; women {int(D.sum()):,} ({D.mean():.3f} of policies, {wmean(D, e):.3f} of exposure)")
    fw = N[D == 1].sum() / e[D == 1].sum()
    fm = N[D == 0].sum() / e[D == 0].sum()
    print(f"  Claim frequency per policy-year: women {fw:.4f}, men {fm:.4f}, ratio {fw / fm:.3f}")
    groups = [("driver age", np.digitize(age, [26, 31, 41, 51, 61, 71]), ["18-25", "26-30", "31-40", "41-50", "51-60",
                                                                       "61-70", "71+"]),
              ("engine power (kW)", np.digitize(pw, [50, 66, 92]), ["below 50", "50-65", "66-91", "92+"]),
              ("bonus-malus level", np.digitize(bm, [1, 4, 7, 10, 14]), ["0", "1-3", "4-6", "7-9", "10-13", "14+"]),
              ("coverage", np.searchsorted(["TPL", "TPL+", "TPL++"], cov), ["TPL", "TPL+", "TPL++"]),
              ("use", (use == "work").astype(int), ["private", "work"]),
              ("fuel", (fuel == "diesel").astype(int), ["petrol", "diesel"]),
              ("fleet", (fleet == "1").astype(int), ["no", "yes"]),
              ("region (first digit of postcode)", region - 1, [str(k) for k in range(1, 10)])]
    res = {"groups": {}}
    for name, g, labels in groups:
        parts = []
        for k, lab in enumerate(labels):
            m = g == k
            if m.any():
                parts.append(f"{lab} {wmean(D[m], e[m]):.3f} ({e[m].sum() / e.sum():.3f})")
        print(f"  share of women (share of exposure) by {name}: " + ", ".join(parts))
        res["groups"][name] = [[lab, wmean(D[g == k], e[g == k]), float(e[g == k].sum() / e.sum())]
                               for k, lab in enumerate(labels) if (g == k).any()]

    acc = {cls: {"ratio": {k: [] for k in PRICES}, "share": [], "tjur": [], "auc": [], "effect": [], "pi": [],
                 "dev": {k: [] for k in PRICES}, "gini": {k: [] for k in PRICES}, "ae_f": {k: [] for k in PRICES},
                 "ae_m": {k: [] for k in PRICES}, "sd": [], "corr": [], "share_i": [], "sd_proxy": [], "sd_refit": [],
                 "dec_n": {k: np.zeros(10) for k in PRICES}, "dec_x": {k: np.zeros(10) for k in PRICES},
                 "dec_p": np.zeros(10), "dec_e": np.zeros(10), "dec_eff": np.zeros(10)}
           for cls in CLASSES}
    for r in range(R):
        print(f"beMTPL97: split {r + 1} of {R}", file=sys.stderr, flush=True)
        rng = np.random.default_rng(SEED + r)
        tr, ca, te = p3.split(n, rng)
        tr = np.sort(np.concatenate([tr, ca]))
        et, Nt, Dt = e[te], N[te], D[te]
        per = et.sum() / 1000.0
        # How well the rating factors predict sex, by each model class's own representation of them.
        Xl = P["X_glm"](tr)[:, 1:]
        lr = LogisticRegression(C=1e4, max_iter=5000).fit(Xl, D[tr], sample_weight=e[tr])
        pl_tr = lr.predict_proba(Xl)[:, 1]
        pl = lr.predict_proba(P["X_glm"](te)[:, 1:])[:, 1]
        gc = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=2000, max_leaf_nodes=31, min_samples_leaf=200,
                                            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
                                            random_state=7, categorical_features=cat)
        gc.fit(P["X_gbm"](tr), D[tr], sample_weight=e[tr])
        pg_tr = gc.predict_proba(P["X_gbm"](tr))[:, 1]
        pg = gc.predict_proba(P["X_gbm"](te))[:, 1]
        for cls, pp_tr, pp in (("GLM", pl_tr, pl), ("GBM", pg_tr, pg)):
            a = acc[cls]
            rates, pi, effect = prices(cls, P["X_glm"], P["X_gbm"], D, e, N, tr, te, cat, pp_tr, pp)
            a["pi"].append(pi)
            a["effect"].append(effect)
            a["tjur"].append(tjur(pp, Dt, et))
            a["auc"].append(roc_auc_score(Dt, pp, sample_weight=et))
            for k in PRICES:
                a["ratio"][k].append(ratio(rates[k], Dt, et))
                a["dev"][k].append(poisson_deviance(Nt, et * rates[k]) / per)
                a["gini"][k].append(gini(rates[k], et, Nt))
                a["ae_f"][k].append(Nt[Dt == 1].sum() / (et * rates[k])[Dt == 1].sum())
                a["ae_m"][k].append(Nt[Dt == 0].sum() / (et * rates[k])[Dt == 0].sum())
            lb, lu, li, ld = (np.log(a["ratio"][k][-1]) for k in PRICES)
            a["share"].append((lu - ld) / (lb - ld))
            a["share_i"].append((li - ld) / (lb - ld))
            sd = lambda q: float(np.sqrt(wmean((q - wmean(q, et)) ** 2, et)))  # noqa: E731
            q = np.log(rates["unawareness"] / rates["discrimination-free"])
            a["sd"].append(sd(q))
            a["sd_proxy"].append(sd(np.log(rates["implied unawareness"] / rates["discrimination-free"])))
            a["sd_refit"].append(sd(np.log(rates["unawareness"] / rates["implied unawareness"])))
            a["corr"].append(float(np.corrcoef(q, pp)[0, 1]))
            dec = deciles(pp, et)
            for j in range(10):
                m = dec == j
                a["dec_p"][j] += (et[m] * pp[m]).sum()
                a["dec_e"][j] += et[m].sum()
                a["dec_eff"][j] += (et[m] * rates["eff"][m]).sum()
                for k in PRICES:
                    a["dec_n"][k][j] += Nt[m].sum()
                    a["dec_x"][k][j] += (et[m] * rates[k][m]).sum()

    header(f"Belgium: how well the rating factors predict sex (test samples of {R} random 80/20 splits; exposure-"
           f"weighted; mean [min, max])")
    for cls, lab in (("GLM", "logistic regression on the GLM design"), ("GBM", "gradient boosting on the raw features")):
        a = acc[cls]
        print(f"  {lab}: AUC {summary(a['auc'])}; Tjur's coefficient {summary(a['tjur'])}")
    print(f"  share of women in the training samples (pi) {summary(acc['GLM']['pi'])}")

    header(f"Belgium: three prices, women's average premium per policy-year over men's, on the test samples ({R} splits)")
    for cls in CLASSES:
        a = acc[cls]
        lab = "sex coefficient" if cls == "GLM" else "mean log ratio of prices for a woman and a man with the same factors"
        print(f"  {cls}: direct effect of sex ({lab}) {summary(a['effect'])}, positive in"
              f" {int(np.sum(np.array(a['effect']) > 0))} of {R} splits")
        for k in PRICES:
            print(f"      {k}: ratio {summary(a['ratio'][k])}")
        lb, lu, li, ld = (np.log(np.array(a["ratio"][k])) for k in PRICES)
        pooled = float((lu - ld).mean() / (lb - ld).mean())
        pooled_i = float((li - ld).mean() / (lb - ld).mean())
        print(f"      share of the direct effect restored, pooled over splits (ratio of the mean log gaps): unawareness"
              f" {pooled:.3f}, implied unawareness {pooled_i:.3f}; per split, unawareness {summary(a['share'])},"
              f" implied {summary(a['share_i'])}; Tjur prediction {np.mean(a['tjur']):.3f}")
        print(f"      per policy, sd of the log of the unawareness price over the discrimination-free price"
              f" {summary(a['sd'])}: of which proxy (implied unawareness over discrimination-free) {summary(a['sd_proxy'])},"
              f" refit (unawareness over implied unawareness) {summary(a['sd_refit'])}; correlation of the log ratio"
              f" with the predicted probability of a woman {summary(a['corr'])}")
        res[cls] = {"ratio": {k: [round(float(x), 6) for x in a["ratio"][k]] for k in PRICES},
                    "share": [round(float(x), 6) for x in a["share"]], "pooled": pooled, "pooled_i": pooled_i,
                    "share_i": [round(float(x), 6) for x in a["share_i"]],
                    "sd": [float(np.mean(a[k])) for k in ("sd", "sd_proxy", "sd_refit")],
                    "tjur": [round(float(x), 6) for x in a["tjur"]], "effect": [round(float(x), 6) for x in a["effect"]]}

    header(f"Belgium: what each price costs, on the test samples ({R} splits): deviance per 1,000 policy-years, Gini,"
           f" and actual over expected claims by sex")
    for cls in CLASSES:
        a = acc[cls]
        for k in PRICES:
            print(f"  {cls}, {k}: deviance {np.mean(a['dev'][k]):.2f}; Gini {summary(a['gini'][k])}; A/E women"
                  f" {summary(a['ae_f'][k])}, men {summary(a['ae_m'][k])}")
        du = np.array(a["dev"]["unawareness"]) - np.array(a["dev"]["discrimination-free"])
        di = np.array(a["dev"]["implied unawareness"]) - np.array(a["dev"]["discrimination-free"])
        db = np.array(a["dev"]["best estimate"]) - np.array(a["dev"]["discrimination-free"])
        gu = np.array(a["gini"]["unawareness"]) - np.array(a["gini"]["discrimination-free"])
        print(f"  {cls}: deviance, unawareness minus discrimination-free {summary(du)} (lower in"
              f" {int(np.sum(du < 0))} of {R}); implied unawareness minus discrimination-free {summary(di)} (lower in"
              f" {int(np.sum(di < 0))} of {R}); best estimate minus discrimination-free {summary(db)} (lower in"
              f" {int(np.sum(db < 0))} of {R}); Gini, unawareness minus discrimination-free {summary(gu)}")
    print("  pooled over splits, by decile of the predicted probability that the policyholder is a woman (lowest"
          " first): mean probability; A/E of each price")
    for cls in CLASSES:
        a = acc[cls]
        print(f"    {cls} classifier, probability: " + " ".join(f"{x:.3f}" for x in a["dec_p"] / a["dec_e"]))
        print(f"    {cls}, direct effect of sex (mean log ratio of a woman's to a man's price): "
              + " ".join(f"{x:+.3f}" for x in a["dec_eff"] / a["dec_e"]))
        for k in PRICES:
            ae = a["dec_n"][k] / a["dec_x"][k]
            print(f"    {cls}, {k}: " + " ".join(f"{x:.3f}" for x in ae))
        res[cls]["dec"] = {"p": (a["dec_p"] / a["dec_e"]).round(6).tolist(),
                           "eff": (a["dec_eff"] / a["dec_e"]).round(6).tolist(),
                           **{k: (a["dec_n"][k] / a["dec_x"][k]).round(6).tolist() for k in PRICES}}
    return res


# ---------------------------------------------------------------------------------------------------
# A controlled experiment with known costs
# ---------------------------------------------------------------------------------------------------
def synthetic() -> dict:
    P = c1.french()
    d = p3.load()
    e = P["e"]
    n = e.size
    tr0, te = P["tr"], P["te"]
    allidx = np.arange(n)
    b = c1.irls_ridge(P["X_glm"](tr0), P["N"][tr0], e[tr0])
    base = np.exp(P["X_glm"](allidx) @ b)                 # plays the true frequency without the protected attribute
    z = lambda v: (v - wmean(v, e)) / np.sqrt(wmean((v - wmean(v, e)) ** 2, e))  # noqa: E731
    young, power, dens = z(-d["dage"]), z(np.minimum(d["vpow"], 12.0)), z(d["logdens"])
    index = {"additive": z(young + power + dens), "interaction": z(young * power)}
    rng0 = np.random.default_rng(SEED + 99)
    tr = np.sort(rng0.permutation(tr0)[:TRAIN_SYN])
    et = e[te]
    per = et.sum() / 1000.0

    header(f"A controlled experiment with known costs, on freMTPL2: a synthetic protected attribute D (share 0.3 of"
           f" exposure). True frequency = the GLM rate fitted on the whole training pool times exp(gamma D), rescaled;"
           f" P(D = 1 | x) = logistic(a + kappa s(x)). Models fitted on {TRAIN_SYN:,} training policies with claims"
           f" simulated from the truth and every price rescaled to their true expected claims; outcomes on the"
           f" {te.size:,} test policies against the true expected claims; implied unawareness uses the true P(D = 1 | x)")
    print("  s(x), additive: the standardized sum of standardized (minus driver age), vehicle power (capped at 12) and log"
          " density; interaction: the standardized product of standardized (minus driver age) and vehicle power")
    res = []
    for si, (shape, kappa, gamma) in enumerate(SETTINGS):
        print(f"freMTPL2 synthetic: setting {si + 1} of {len(SETTINGS)}", file=sys.stderr, flush=True)
        s = index[shape]
        lo, hi = -10.0, 10.0
        for _ in range(100):                                   # intercept so that the exposure share of D is 0.3
            a = (lo + hi) / 2
            if wmean(1 / (1 + np.exp(-(a + kappa * s))), e) > 0.3:
                hi = a
            else:
                lo = a
        p = 1 / (1 + np.exp(-(a + kappa * s)))
        rng = np.random.default_rng(SEED + 1000 + si)
        D = (rng.random(n) < p).astype(float)
        mu = base * np.exp(gamma * D)
        mu *= (e * base).sum() / (e * mu).sum()
        N = rng.poisson(e * mu).astype(float)
        Dt = D[te]
        tj = tjur(p[te], Dt, et)
        row = {"shape": shape, "kappa": kappa, "gamma": gamma, "tjur": tj, "auc": roc_auc_score(Dt, p[te], sample_weight=et)}
        truth_ratio = ratio(mu[te], Dt, et)
        for cls in CLASSES:
            rates, pi, effect = prices(cls, P["X_glm"], P["X_gbm"], D, e, N, tr, te, P["cat"], p[tr], p[te],
                                       total=float((e[tr] * mu[tr]).sum()))
            lb, lu, li, ld = (np.log(ratio(rates[k], Dt, et)) for k in PRICES)
            out = {"effect": effect, "share": float((lu - ld) / (lb - ld)), "share_i": float((li - ld) / (lb - ld)),
                   "ratios": [float(np.exp(x)) for x in (lb, lu, li, ld)]}
            dec = deciles(p[te], et)
            for k in PRICES:
                x = et * rates[k]
                y = et * mu[te]
                out[SHORT[k]] = {
                    "excess": float(2 * np.sum(y * np.log(y / x) - (y - x)) / per),
                    "ae_f": float(y[Dt == 1].sum() / x[Dt == 1].sum()), "ae_m": float(y[Dt == 0].sum() / x[Dt == 0].sum()),
                    "dec": [float(y[dec == j].sum() / x[dec == j].sum()) for j in range(10)]}
            row[cls] = out
        res.append(row)
        print(f"  {shape}, kappa {kappa:.0f}, gamma {gamma:.1f}: AUC of the true P(D = 1 | x) {row['auc']:.3f}, Tjur's"
              f" coefficient {tj:.3f}; true ratio of expected claims per policy-year, D = 1 over D = 0 {truth_ratio:.3f}")
        for cls in CLASSES:
            o = row[cls]
            print(f"      {cls}: estimated direct effect {o['effect']:.3f}; D = 1 over D = 0 premium: best estimate"
                  f" {o['ratios'][0]:.3f}, unawareness {o['ratios'][1]:.3f}, implied unawareness {o['ratios'][2]:.3f},"
                  f" discrimination-free {o['ratios'][3]:.3f}; share restored: unawareness {o['share']:.3f}, implied"
                  f" unawareness {o['share_i']:.3f}")
            print(f"          true over charged claims, D = 1 / D = 0: " + "; ".join(
                f"{k} {o[SHORT[k]]['ae_f']:.3f} / {o[SHORT[k]]['ae_m']:.3f}" for k in PRICES))
            print(f"          deviance against the truth per 1,000 policy-years: " + "; ".join(
                f"{k} {o[SHORT[k]]['excess']:.2f}" for k in PRICES))
            print(f"          true over charged claims by decile of P(D = 1 | x), lowest first: unawareness "
                  + " ".join(f"{x:.3f}" for x in o["U"]["dec"]) + "; discrimination-free "
                  + " ".join(f"{x:.3f}" for x in o["DF"]["dec"]))
    return {"settings": res}


def main() -> None:
    print("Fifth companion: leaving a protected attribute out of a tariff, and the discrimination-free price.")
    quick = os.environ.get("FAIR_QUICK")
    global R, SETTINGS
    if quick:
        R = 2
        SETTINGS = SETTINGS[:2]
    out = {"beMTPL97": belgium(), "synthetic": synthetic()}
    if not quick:
        (HERE / "figure_data.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
