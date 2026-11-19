"""Base rates and flags: the arithmetic, the detector data and the card-fraud model."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- the arithmetic


def ppv(base, sens, fpr):
    """Share of flags that are right (positive predictive value), by Bayes' rule."""
    base, sens, fpr = np.asarray(base, float), np.asarray(sens, float), np.asarray(fpr, float)
    hit = base * sens
    return hit / (hit + (1 - base) * fpr)


def false_per_true(base, sens, fpr):
    """False flags for every true one."""
    return (1 - base) * fpr / (base * sens)


def frequencies(n, base, sens, fpr):
    """Natural frequencies: out of n cases, how many are positive, flagged correctly and flagged wrongly."""
    pos = n * base
    return {"positive": pos, "true_flags": pos * sens, "negative": n - pos, "false_flags": (n - pos) * fpr}


def required_fpr(base, sens, target):
    """Largest false-positive rate at which a flag is right with probability `target`."""
    base = np.asarray(base, float)
    return base * sens * (1 - target) / ((1 - base) * target)


def likelihood_ratio_needed(base, target):
    """Sensitivity / false-positive rate needed to turn prior odds into posterior odds target/(1-target)."""
    return (target / (1 - target)) / (base / (1 - base))


# ---------------------------------------------------------------- uncertainty from small samples


def jeffreys(k, n, level=0.95):
    """Jeffreys interval for a proportion k/n."""
    from scipy.stats import beta

    a = (1 - level) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k + 0.5, n - k + 0.5)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 0.5, n - k + 0.5)
    return float(lo), float(hi)


def ppv_interval(base, k_sens, n_sens, k_fpr, n_fpr, draws=20_000, seed=0, level=0.95):
    """Interval for the share of right flags, drawing sensitivity and false-positive rate from their Jeffreys
    posteriors."""
    rng = np.random.default_rng(seed)
    s = rng.beta(k_sens + 0.5, n_sens - k_sens + 0.5, draws)
    f = rng.beta(k_fpr + 0.5, n_fpr - k_fpr + 0.5, draws)
    v = ppv(base, s, f)
    a = (1 - level) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


# ---------------------------------------------------------------- AI-text detectors (Liang et al. 2023)

DETECTORS = ["Crossplag", "GPTZero", "HFOpenAI", "OriginalityAI", "Quil", "Sapling", "ZeroGPT"]
GROUPS = {
    "native": ["Human_Data/HewlettStudentEssay_real_88"],
    "non-native": ["Human_Data/TOEFL_real_91"],
    "college": ["Human_Data/CollegeEssay_real_70"],
    "ai": ["GPT_Data/CollegeEssay_gpt3_31", "GPT_Data/CS224N_gpt3_145"],
    "ai_edited": ["GPT_Data/CollegeEssay_gpt3PromptEng_31", "GPT_Data/CS224N_gpt3PromptEng_145"],
}
THRESHOLD = 0.5  # as in Liang et al.: a score above 0.5 is a flag


def detector_counts(root):
    """Flags (score above 0.5) and essays, for every detector and group of essays."""
    rows = []
    for group, folders in GROUPS.items():
        for det in DETECTORS:
            k = n = 0
            for folder in folders:
                recs = json.loads((Path(root) / "Data_and_Results" / folder / f"{det}.json").read_text())
                scores = np.array([float(r["score"]) for r in recs])
                k += int((scores > THRESHOLD).sum())
                n += len(scores)
            rows.append({"group": group, "detector": det, "flagged": k, "essays": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- card fraud (ULB / Worldline)


def load_cards(zip_path):
    """The transactions, with a day indicator (the data cover two days)."""
    with zipfile.ZipFile(zip_path) as z:
        d = pd.read_csv(io.BytesIO(z.read("creditcard.csv")))
    d["day"] = np.where(d.Time >= 86_400, 2, 1)
    return d


def fraud_model(d):
    """Logistic regression trained on day 1 and scored on day 2.

    Features: the 28 anonymised components and the log of the amount. Scores are rounded to eight decimals
    and ranked with a stable sort, so ties and the last bits of the arithmetic cannot reorder them.
    """
    cols = [f"V{i}" for i in range(1, 29)]
    x = d[cols].copy()
    x["log_amount"] = np.log1p(d.Amount)
    train, test = d.day == 1, d.day == 2
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(x[train], d.Class[train])
    p = np.round(model.predict_proba(x[test])[:, 1], 8)
    y = d.Class[test].to_numpy()
    return y, p, float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def ranked(y, p):
    """True and false alarms when the top k scores are flagged, for every k."""
    order = np.argsort(-p, kind="stable")
    tp = np.cumsum(y[order])
    fp = np.cumsum(1 - y[order])
    return tp, fp


def at_recall(y, p, recall):
    """Alerts, true and false alarms, and false-positive rate at the first cut that catches `recall` of frauds."""
    tp, fp = ranked(y, p)
    k = int(np.argmax(tp >= recall * y.sum()))
    return {"alerts": k + 1, "true": int(tp[k]), "false": int(fp[k]), "fpr": fp[k] / (len(y) - y.sum())}


def rebase(true, false, base_from, base_to):
    """Share of right alerts if frauds were `base_to` of transactions instead of `base_from`, with the same
    detector: every false alarm is reweighted by the change in the odds of a legitimate transaction."""
    w = (base_from / (1 - base_from)) / (base_to / (1 - base_to))
    return true / (true + false * w), false * w / true


def precision_recall_curve(y, p, base_to=None):
    """Precision against recall for every cut, optionally at another base rate."""
    tp, fp = ranked(y, p)
    recall = tp / y.sum()
    if base_to is None:
        precision = tp / (tp + fp)
    else:
        b = y.mean()
        w = (b / (1 - b)) / (base_to / (1 - base_to))
        precision = tp / (tp + fp * w)
    keep = np.r_[True, np.diff(recall) > 0]  # one point per recall level
    return recall[keep], precision[keep]
