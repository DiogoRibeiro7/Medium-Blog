"""Survival Analysis for Data Scientists - Part 6.

Machine learning for survival data, and how to evaluate it, on real unemployment
spells: who is at risk of still being jobless after 26 weeks?

Everything the article reports is printed by ``python benchmark.py``.
The figures are produced by ``python figures.py``.

Data: UnempDur, from the R package Ecdat (Croissant and Graves, GPL >= 2), as
distributed by the Rdatasets project. Source: McCall (1996), Econometrica 64,
647-682, from the Displaced Worker Supplements of the US Current Population Survey.
The script downloads the file on first use and checks its SHA-256 checksum.

Requires numpy, pandas, scikit-learn and scikit-survival.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (brier_score, concordance_index_censored, concordance_index_ipcw,
                            cumulative_dynamic_auc, integrated_brier_score)
from sksurv.nonparametric import kaplan_meier_estimator

SEED = 20261017
DATA_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/Ecdat/UnempDur.csv"
DATA_SHA256 = "b2af5323840983b8d758f40a0c9367de04066d79351db0ee8ba53509aa39296e"
DATA_FILE = Path(__file__).parent / "data" / "UnempDur.csv"

HORIZON = 13          # two-week intervals: 26 weeks
TIMES = np.arange(1, 27)   # evaluation grid for the integrated Brier score, intervals 1 to 26
TARGET_SHARE = 0.20   # capacity: intensive support for one worker in five
FEATURES = ("age", "ui", "reprate", "disrate", "logwage", "tenure")
N_SPLITS, N_REPEATS = 5, 10


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def load_data():
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    digest = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise RuntimeError(f"UnempDur.csv has changed (sha256 {digest}); results would not match")
    d = pd.read_csv(DATA_FILE)
    flags = d[["censor1", "censor2", "censor3", "censor4"]].to_numpy()
    reemployed = flags[:, :3].sum(axis=1) > 0
    X = pd.DataFrame({"age": d.age, "ui": (d.ui == "yes").astype(float), "reprate": d.reprate,
                      "disrate": d.disrate, "logwage": d.logwage, "tenure": d.tenure.astype(float)})
    return {"X": X.to_numpy(float), "time": d.spell.to_numpy(float), "event": reemployed,
            "no_flag": flags.sum(axis=1) == 0, "frame": d}


def surv_array(event, time):
    return np.array(list(zip(event.astype(bool), time.astype(float))), dtype=[("event", bool), ("time", float)])


def km_at(event, time, t):
    if len(time) == 0:
        return np.nan
    tt, ss = kaplan_meier_estimator(event.astype(bool), time)
    inside = tt <= t
    return float(ss[inside][-1]) if inside.any() else 1.0


# ---------------------------------------------------------------------------
# 2. Models. Each returns S(t | x) at t = 1, ..., 26 intervals.
# ---------------------------------------------------------------------------

class SksurvModel:
    def __init__(self, model):
        self.model = model

    def fit(self, X, event, time):
        self.model.fit(X, surv_array(event, time))
        return self

    def survival(self, X):
        fns = self.model.predict_survival_function(X)
        return np.array([fn(TIMES.astype(float)) for fn in fns])


def person_period(X, event, time):
    """One row per person and two-week interval at risk; the outcome is 1 in the
    interval in which the person found a job."""
    k = time.astype(int)
    rows = np.repeat(np.arange(len(k)), k)
    interval = np.concatenate([np.arange(1, n + 1) for n in k])
    y = (interval == k[rows]) & event[rows]
    return X[rows], interval, y.astype(int)


class DiscreteTime:
    """Discrete-time survival model: a classifier for the probability of finding a
    job in each interval, given still jobless at its start; S(t) = prod (1 - h)."""

    def __init__(self, kind):
        self.kind = kind

    def _design(self, X, interval):
        if self.kind == "logit":
            dummies = np.eye(20)[np.minimum(interval, 20) - 1]             # intervals 20 and later pooled
            return np.column_stack([(X - self.mean) / self.sd, dummies])
        return np.column_stack([X, interval])

    def fit(self, X, event, time):
        self.mean, self.sd = X.mean(axis=0), X.std(axis=0)
        Xp, interval, y = person_period(X, event, time)
        if self.kind == "logit":
            self.clf = LogisticRegression(C=1e6, max_iter=5000, fit_intercept=False)
        else:
            self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                                      min_samples_leaf=50, early_stopping=False,
                                                      random_state=SEED)
        self.clf.fit(self._design(Xp, interval), y)
        return self

    def survival(self, X):
        n = len(X)
        Xr = np.repeat(X, len(TIMES), axis=0)
        interval = np.tile(TIMES, n)
        h = self.clf.predict_proba(self._design(Xr, interval))[:, 1].reshape(n, len(TIMES))
        return np.cumprod(1 - h, axis=1)


def make_models():
    return {
        "Cox": SksurvModel(CoxPHSurvivalAnalysis(ties="efron")),
        "random survival forest": SksurvModel(RandomSurvivalForest(n_estimators=300, min_samples_leaf=15,
                                                                   max_features="sqrt", n_jobs=-1,
                                                                   random_state=SEED)),
        "discrete-time logistic": DiscreteTime("logit"),
        "discrete-time boosting": DiscreteTime("boost"),
    }


MODELS = tuple(make_models())


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def top_share(score, share, seed):
    """Exactly round(share * n) people with the highest scores; ties broken at random
    (with a fixed seed), so that every method selects groups of the same size."""
    n_sel = int(round(share * len(score)))
    order = np.lexsort((np.random.default_rng(seed).random(len(score)), -np.asarray(score, float)))
    mask = np.zeros(len(score), bool)
    mask[order[:n_sel]] = True
    return mask


def evaluate(S, event_tr, time_tr, event_te, time_te, X_te, seed):
    """All metrics for one test fold. S: predicted survival at TIMES, one row per person."""
    y_tr, y_te = surv_array(event_tr, time_tr), surv_array(event_te, time_te)
    s13 = S[:, HORIZON - 1]
    risk = 1 - s13                                   # probability of a job within 26 weeks
    out = {"Harrell C": concordance_index_censored(event_te.astype(bool), time_te, risk)[0],
           "Uno C": concordance_index_ipcw(y_tr, y_te, risk, tau=26.0)[0],
           "AUC at 26 weeks": cumulative_dynamic_auc(y_tr, y_te, risk, [float(HORIZON)])[0][0],
           "Brier at 26 weeks": brier_score(y_tr, y_te, s13[:, None], [float(HORIZON)])[1][0],
           "integrated Brier": integrated_brier_score(y_tr, y_te, S, TIMES.astype(float))}
    observed = km_at(event_te, time_te, HORIZON)
    out["calibration in the large"] = float(s13.mean() - observed)
    top = top_share(s13, TARGET_SHARE, seed)
    out["still jobless, top 20%"] = km_at(event_te[top], time_te[top], HORIZON)
    out["UI share, top 20%"] = X_te[top, FEATURES.index("ui")].mean()
    out["still jobless, all"] = observed
    return out


def censoring_survival(event, time, x):
    """Kaplan-Meier estimate of the probability of still being observed just before x
    (censoring distribution; with tied times, censoring counts as after the event)."""
    tt, ss = kaplan_meier_estimator(event.astype(bool), time, reverse=True)
    idx = np.searchsorted(tt, x, side="left") - 1
    return np.where(idx >= 0, ss[np.maximum(idx, 0)], 1.0)


class HorizonClassifier:
    """Logistic regression for 'still jobless after 26 weeks', fitted to the people
    whose status at 26 weeks is known, each weighted by one over the probability of
    having stayed under observation that long (inverse probability of censoring
    weighting)."""

    def fit(self, X, event, time):
        self.mean, self.sd = X.mean(axis=0), X.std(axis=0)
        found = event & (time <= HORIZON)                       # re-employed within 26 weeks
        known = found | (time >= HORIZON)                       # status at 26 weeks observed
        weight = 1.0 / censoring_survival(event, time, np.minimum(time, HORIZON))
        self.clf = LogisticRegression(max_iter=5000).fit(((X - self.mean) / self.sd)[known],
                                                         (~found)[known].astype(int),
                                                         sample_weight=weight[known])
        return self

    def still_jobless(self, X):
        return self.clf.predict_proba((X - self.mean) / self.sd)[:, 1]


def rule_of_thumb(X):
    """UI claimants first, oldest first: a ranking anyone could apply without a model."""
    return X[:, FEATURES.index("ui")] * 100 + X[:, FEATURES.index("age")]


def cross_validate(data, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    X, time, event = data["X"], data["time"], data["event"]
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    rows, pooled = [], {m: np.zeros((len(X), len(TIMES))) for m in MODELS}
    pooled["26-week classifier"] = np.zeros(len(X))
    for f, (tr, te) in enumerate(cv.split(X)):
        rep = f // n_splits
        for name, model in make_models().items():
            S = model.fit(X[tr], event[tr], time[tr]).survival(X[te])
            row = evaluate(S, event[tr], time[tr], event[te], time[te], X[te], SEED + f)
            row.update(model=name, repeat=rep, fold=f)
            rows.append(row)
            if rep == 0:
                pooled[name][te] = S
        p = HorizonClassifier().fit(X[tr], event[tr], time[tr]).still_jobless(X[te])
        y_tr, y_te = surv_array(event[tr], time[tr]), surv_array(event[te], time[te])
        top = top_share(p, TARGET_SHARE, SEED + f)
        rows.append({"model": "26-week classifier", "repeat": rep, "fold": f,
                     "Harrell C": concordance_index_censored(event[te].astype(bool), time[te], 1 - p)[0],
                     "Uno C": concordance_index_ipcw(y_tr, y_te, 1 - p, tau=26.0)[0],
                     "AUC at 26 weeks": cumulative_dynamic_auc(y_tr, y_te, 1 - p, [float(HORIZON)])[0][0],
                     "Brier at 26 weeks": brier_score(y_tr, y_te, p[:, None], [float(HORIZON)])[1][0],
                     "calibration in the large": float(p.mean() - km_at(event[te], time[te], HORIZON)),
                     "still jobless, top 20%": km_at(event[te][top], time[te][top], HORIZON),
                     "UI share, top 20%": X[te][top, 1].mean()})
        if rep == 0:
            pooled["26-week classifier"][te] = p
        score = rule_of_thumb(X[te])
        top = top_share(score, TARGET_SHARE, SEED + f)
        rows.append({"model": "rule of thumb", "repeat": rep, "fold": f,
                     "Harrell C": concordance_index_censored(event[te].astype(bool), time[te], -score)[0],
                     "Uno C": concordance_index_ipcw(y_tr, y_te, -score, tau=26.0)[0],
                     "AUC at 26 weeks": cumulative_dynamic_auc(y_tr, y_te, -score, [float(HORIZON)])[0][0],
                     "still jobless, top 20%": km_at(event[te][top], time[te][top], HORIZON),
                     "UI share, top 20%": X[te][top, 1].mean(),
                     "still jobless, all": km_at(event[te], time[te], HORIZON)})
    return pd.DataFrame(rows), pooled


METRICS = ("Harrell C", "Uno C", "AUC at 26 weeks", "Brier at 26 weeks", "integrated Brier",
           "calibration in the large", "still jobless, top 20%", "UI share, top 20%")
EXTRA = ("26-week classifier", "rule of thumb")


def calibration_table(S13, event, time, groups=10):
    order = np.argsort(S13)
    bins = np.array_split(order, groups)
    pred = np.array([S13[b].mean() for b in bins])
    obs = np.array([km_at(event[b], time[b], HORIZON) for b in bins])
    return pred, obs


def gain_curve(S13, event, time, shares, seed=SEED):
    out = []
    for q in shares:
        top = top_share(S13, q, seed)
        out.append(km_at(event[top], time[top], HORIZON))
    return np.array(out)


def run_all():
    data = load_data()
    results, pooled = cross_validate(data)
    return data, results, pooled


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def main():
    data = load_data()
    X, time, event, d = data["X"], data["time"], data["event"], data["frame"]

    print("=" * 72)
    print("1. THE DATA")
    print("=" * 72)
    print(f"Spells {len(time):,}; re-employed {event.sum():,} (full-time {int(d.censor1.sum()):,}, part-time "
          f"{int(d.censor2.sum()):,}, status unknown {int(d.censor3.sum()):,}); still jobless when last observed "
          f"{int(d.censor4.sum()):,}; no outcome recorded {int(data['no_flag'].sum())} (treated as censored)")
    print(f"Spell length in two-week intervals: median {np.median(time):.0f}, longest {time.max():.0f} "
          f"({2 * time.max():.0f} weeks)")
    print(f"Still jobless after 26 weeks (Kaplan-Meier): all {pct(km_at(event, time, HORIZON))}; filed a UI claim "
          f"{pct(km_at(event[X[:, 1] == 1], time[X[:, 1] == 1], HORIZON))}, did not "
          f"{pct(km_at(event[X[:, 1] == 0], time[X[:, 1] == 0], HORIZON))}")
    print(f"UI claimants {int(X[:, 1].sum()):,} ({pct(X[:, 1].mean())}); age median {np.median(X[:, 0]):.0f}")
    for ui in (1, 0):
        m = X[:, 1] == ui
        hz = [((time[m] == k) & event[m]).sum() / (time[m] >= k).sum() for k in range(1, 17)]
        print(f"Re-employment per interval, {'UI claimants' if ui else 'no UI claim'}, intervals 1-16: "
              + " ".join(f"{h:.2f}" for h in hz))
    print(f"Re-employments within the first 10 weeks (5 intervals): {pct(np.mean(time[event] <= 5))} of all "
          f"re-employments; among non-claimants {pct(np.mean(time[event & (X[:, 1] == 0)] <= 5))}, among claimants "
          f"{pct(np.mean(time[event & (X[:, 1] == 1)] <= 5))}")
    # Pairs Harrell's C can compare: the earlier spell ended in re-employment, and the other lasted longer
    # (or was censored at the same interval).
    t_ev, ui_ev = time[event], X[event, 1]
    n_pairs = np.array([np.sum(time > t) + np.sum((time == t) & ~event) for t in t_ev])
    print(f"Pairs Harrell's C can compare: {n_pairs.sum():,}; with the earlier re-employment in the first 10 weeks "
          f"{pct(n_pairs[t_ev <= 5].sum() / n_pairs.sum())}; with the earlier re-employment by a non-claimant "
          f"{pct(n_pairs[ui_ev == 0].sum() / n_pairs.sum())}")
    print(f"Censored in the analysis: {int((~event).sum()):,} (still jobless {int(d.censor4.sum()):,} plus "
          f"{int(data['no_flag'].sum())} without an outcome)")
    Xp, interval, yp = person_period(X, event, time)
    boost = DiscreteTime("boost").fit(X, event, time)
    print(f"Person-period rows {len(Xp):,}; discrete-time boosting fitted to all spells: mean predicted share still "
          f"jobless at 26 weeks {boost.survival(X)[:, HORIZON - 1].mean():.4f}")
    found = event & (time <= HORIZON)
    known = found | (time >= HORIZON)
    weight = 1.0 / censoring_survival(event, time, np.minimum(time, HORIZON))
    print(f"Status at 26 weeks known for {int(known.sum()):,} spells; censoring-weighted share still jobless "
          f"{np.average(~found[known], weights=weight[known]):.4f} (Kaplan-Meier {km_at(event, time, HORIZON):.4f})")
    dropped = ~data["no_flag"]
    cox_all = CoxPHSurvivalAnalysis(ties="efron").fit(X, surv_array(event, time))
    cox_drop = CoxPHSurvivalAnalysis(ties="efron").fit(X[dropped], surv_array(event[dropped], time[dropped]))
    print("Cox hazard ratios of re-employment (all spells; dropping the 102 without an outcome): "
          + "; ".join(f"{f} {np.exp(a):.3f} / {np.exp(b):.3f}" for f, a, b in zip(FEATURES, cox_all.coef_, cox_drop.coef_)))

    print()
    print("=" * 72)
    print(f"2. {N_REPEATS} REPEATS OF {N_SPLITS}-FOLD CROSS-VALIDATION")
    print("=" * 72)
    results, pooled = cross_validate(data)
    by_rep = results.groupby(["model", "repeat"]).mean(numeric_only=True)
    for metric in METRICS:
        print(metric)
        cox = by_rep.loc["Cox"][metric]
        for name in list(MODELS) + list(EXTRA):
            v = by_rep.loc[name][metric]
            if v.isna().all():
                continue
            line = f"    {name:24s} mean {v.mean():.4f}, range over repeats {v.min():.4f} to {v.max():.4f}"
            if name != "Cox":
                diff = v - cox
                better = np.sum(diff < 0) if metric in ("Brier at 26 weeks", "integrated Brier") else np.sum(diff > 0)
                if metric == "calibration in the large":
                    better = np.sum(np.abs(v) < np.abs(cox))
                line += f"; difference from Cox {diff.mean():+.4f} (better in {better} of {N_REPEATS} repeats)"
            print(line)
    allv = by_rep.loc["Cox"]["still jobless, all"]
    print(f"Still jobless after 26 weeks among all workers in the test folds: {allv.mean():.4f}")
    r, c = by_rep.loc["rule of thumb"]["still jobless, top 20%"], by_rep.loc["26-week classifier"]["still jobless, top 20%"]
    print(f"Rule of thumb against 26-week classifier, top 20%: difference {(r - c).mean():+.4f} "
          f"(rule better in {int(np.sum(r > c))} of {N_REPEATS} repeats)")
    fold = results[results.model == "Cox"]["still jobless, top 20%"]
    print(f"Spread of the top-20% share across the 50 test folds, Cox: standard deviation {fold.std():.4f}")
    print("Share of all workers still jobless at 26 weeks who are in the selected 20% (capture): " + ", ".join(
        f"{name} {pct(TARGET_SHARE * by_rep.loc[name]['still jobless, top 20%'].mean() / allv.mean())}"
        for name in list(MODELS) + list(EXTRA)))
    # Kaplan-Meier with no covariates, as a reference for the Brier scores
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    ref = []
    for tr, te in cv.split(X):
        tt, ss = kaplan_meier_estimator(event[tr].astype(bool), time[tr])
        S = np.tile(np.array([km_at(event[tr], time[tr], k) for k in TIMES]), (len(te), 1))
        y_tr, y_te = surv_array(event[tr], time[tr]), surv_array(event[te], time[te])
        ref.append((brier_score(y_tr, y_te, S[:, [HORIZON - 1]], [float(HORIZON)])[1][0],
                    integrated_brier_score(y_tr, y_te, S, TIMES.astype(float))))
    ref = np.array(ref).mean(axis=0)
    print(f"Kaplan-Meier with no covariates: Brier at 26 weeks {ref[0]:.4f}, integrated Brier {ref[1]:.4f}")

    print()
    print("=" * 72)
    print("3. IN-SAMPLE AGAINST OUT-OF-SAMPLE")
    print("=" * 72)
    for name, model in make_models().items():
        S = model.fit(X, event, time).survival(X)
        c_in = concordance_index_censored(event, time, 1 - S[:, HORIZON - 1])[0]
        c_out = by_rep.loc[name]["Harrell C"].mean()
        print(f"{name:24s} Harrell C fitted and scored on all data {c_in:.4f}; cross-validated {c_out:.4f}")

    print()
    print("=" * 72)
    print("4. CALIBRATION AND TARGETING (FIRST REPEAT, OUT-OF-FOLD PREDICTIONS)")
    print("=" * 72)
    shares = np.array([0.1, 0.2, 0.3, 0.5])
    for name in MODELS:
        s13 = pooled[name][:, HORIZON - 1]
        pred, obs = calibration_table(s13, event, time)
        gains = gain_curve(s13, event, time, shares)
        print(f"{name:24s} deciles of predicted 26-week joblessness: predicted {pred[0]:.3f} to {pred[-1]:.3f}, "
              f"observed {obs[0]:.3f} to {obs[-1]:.3f}; mean absolute gap {np.mean(np.abs(pred - obs)):.3f}; "
              "still jobless among the top " + ", ".join(f"{pct(q, 0)}: {pct(g)}" for q, g in zip(shares, gains)))
    p = pooled["26-week classifier"]
    pred, obs = calibration_table(p, event, time)
    gains = gain_curve(p, event, time, shares)
    print(f"{'26-week classifier':24s} deciles of predicted 26-week joblessness: predicted {pred[0]:.3f} to {pred[-1]:.3f}, "
          f"observed {obs[0]:.3f} to {obs[-1]:.3f}; mean absolute gap {np.mean(np.abs(pred - obs)):.3f}; "
          "still jobless among the top " + ", ".join(f"{pct(q, 0)}: {pct(g)}" for q, g in zip(shares, gains)))
    score = rule_of_thumb(X)
    gains = gain_curve(score.astype(float), event, time, shares)
    print("rule of thumb (UI claimants, oldest first): still jobless among the top "
          + ", ".join(f"{pct(q, 0)}: {pct(g)}" for q, g in zip(shares, gains)))


if __name__ == "__main__":
    main()
