"""Core of the analysis: how much numbers decide a battle, what else does, and what the codes hide.

One row per battle, from the attacker's point of view (CDB90's convention). The force ratio is the
attacker's initial strength over the defender's, used as a natural logarithm, so 0 means equal
numbers, +0.69 twice as many attackers and -0.69 half as many. A battle is decisive when CDB90
names a winner; draws are left out of the win models.

Lanchester's laws (1916) give two benchmarks for the role of numbers.

* Square law (aimed fire): each side's losses are proportional to the size of the other side, so
  fighting strength grows with effectiveness times numbers squared. The larger side should win
  unless the smaller one is much more effective, and the larger side's share of the losses should
  fall in proportion to the force ratio: slope -1 of log(attacker losses / defender losses) on the
  log force ratio.
* Linear law (area fire): losses are proportional to the product of the two sizes, so strength
  grows with numbers, not their square, and the loss ratio does not depend on the force ratio:
  slope 0.

Coded advantages (leadership, morale, surprise and so on) were assigned by historians at HERO who
knew how each battle ended. Prior records are built from earlier battles only, so they are known
before the battle starts.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 20261118

CODED = ["leada", "morala", "cea", "inita", "intela", "aeroa", "momnta", "trnga", "logsa", "techa"]
CODED_NAMES = {
    "leada": "leadership",
    "morala": "morale",
    "cea": "combat effectiveness",
    "inita": "initiative",
    "intela": "intelligence",
    "aeroa": "air superiority",
    "momnta": "momentum",
    "trnga": "training",
    "logsa": "logistics",
    "techa": "technology",
}
ERAS = [(1600, 1799, "1600-1799"), (1800, 1899, "1800-1899"), (1900, 1945, "1900-1945"), (1946, 1982, "1946-1982")]


def load():
    """Battles with force ratio, outcome, era, war, posture, terrain and both sides' strength errors."""
    b = pd.read_csv(DATA / "battles.csv")
    b["year"] = b.datetime.str[:4].astype(int)
    b["date"] = pd.to_datetime(b.datetime.str[:10])
    b["lr"] = np.log(b.attacker_str / b.defender_str)
    b["win"] = (b.wina == 1).astype(int)
    b["decisive"] = b.wina.isin([1, -1])
    b["era"] = ""
    for lo, hi, name in ERAS:
        b.loc[b.year.between(lo, hi), "era"] = name
    b["war_key"] = b.war4.fillna(b.war)
    b["att_surprise"] = (b.surpa > 0).astype(int)
    b["def_surprise"] = (b.surpa < 0).astype(int)
    t = pd.read_csv(DATA / "terrain.csv").drop_duplicates("isqno")[["isqno", "terra1", "terra2"]]
    b = b.merge(t, on="isqno", how="left")
    be = pd.read_csv(DATA / "belligerents.csv")
    err = be.assign(half=(be.strpl - be.strmi) / 2 / 100)[["isqno", "attacker", "half"]]
    a = err[err.attacker == 1].drop_duplicates("isqno").rename(columns={"half": "att_err"})
    d = err[err.attacker == 0].drop_duplicates("isqno").rename(columns={"half": "def_err"})
    b = b.merge(a[["isqno", "att_err"]], on="isqno", how="left").merge(d[["isqno", "def_err"]], on="isqno", how="left")
    return b


def decisive(b):
    """Battles with a winner and both strengths known."""
    return b[b.decisive & b.lr.notna()].reset_index(drop=True)


def larger_side_share(d):
    """Share of decisive battles with unequal numbers won by the larger side."""
    x = d[d.lr != 0]
    return ((x.lr > 0) == (x.win == 1)).mean(), len(x)


def logit(y, x):
    """Logistic regression with an intercept; returns the fitted statsmodels result."""
    import statsmodels.api as sm

    return sm.Logit(np.asarray(y, float), sm.add_constant(np.asarray(x, float), has_constant="add")).fit(
        disp=False, maxiter=200
    )


def implied_spread(slope, exponent):
    """Standard deviation of the log effectiveness ratio a Lanchester law needs to match a logit slope.

    Under a law with exponent p (2 for the square law, 1 for the linear law) the attacker wins when
    p*log(force ratio) + log(effectiveness ratio) > 0. If the log effectiveness ratio, independent
    of the force ratio, follows a logistic distribution with scale s, P(win) is a logistic curve in
    the log force ratio with slope p/s. Its standard deviation is s*pi/sqrt(3) = p*pi/(sqrt(3)*slope).
    """
    return exponent * np.pi / (np.sqrt(3) * slope)


def auc(y, score):
    """Area under the ROC curve; ties count one half."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    r = rankdata(np.asarray(score, float))
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def mz_r2(fit):
    """McKelvey-Zavoina R2 of a logit: share of the latent variance explained by the predictors."""
    xb = fit.fittedvalues
    v = np.var(xb)
    return v / (v + np.pi**2 / 3)


def cluster_bootstrap(d, stat, cluster="campgn", draws=2000, seed=SEED):
    """Percentile 95% interval of stat(frame) resampling whole campaigns with replacement.

    Battles of one campaign share armies, commanders and ground, so they are resampled together.
    Wars are too few and too unequal (World War II alone has 192 battles) to resample reliably;
    the within-war estimate in within_war_slope() is the check at that level.
    """
    rng = np.random.default_rng(seed)
    d = d.reset_index(drop=True)
    groups = [np.asarray(ix) for ix in d.groupby(cluster).indices.values()]
    vals = []
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(d.iloc[np.concatenate([groups[i] for i in pick])].reset_index(drop=True)))
    return np.nanpercentile(vals, [2.5, 97.5])


def within_war_slope(d):
    """Conditional-logit slope of winning on the log force ratio, comparing battles only within a war.

    Wars in which one side won every battle carry no information and drop out.
    Returns slope, standard error and number of battles used.
    """
    from statsmodels.discrete.conditional_models import ConditionalLogit

    x = d[d.groupby("war_key").win.transform("nunique") == 2]
    fit = ConditionalLogit(x.win.to_numpy(), x[["lr"]].to_numpy(), groups=pd.factorize(x.war_key)[0]).fit(disp=False)
    return fit.params[0], fit.bse[0], len(x)


def casualty_frame(b):
    """Battles with both strengths and positive casualties on both sides."""
    c = b[b.lr.notna() & (b.attacker_cas > 0) & (b.defender_cas > 0)].copy()
    c["llr"] = np.log(c.attacker_cas / c.defender_cas)
    return c.reset_index(drop=True)


def casualty_slope(c, control_outcome=False):
    """Slope of log(attacker losses / defender losses) on the log force ratio (least squares)."""
    x = c[["lr"]].assign(w=c.win) if control_outcome else c[["lr"]]
    X = np.column_stack([np.ones(len(c)), x.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, c.llr.to_numpy(float), rcond=None)
    return beta[1]


def attenuation(d):
    """Reliability of the log force ratio given CDB90's stated strength error ranges.

    Each side's half-range (in percent of strength) is read as one standard deviation of the log
    strength, which overstates the error if the ranges are meant as bounds. Reliability is the share
    of the variance of the log force ratio that is not error; a slope is shrunk by that factor.
    """
    x = d.dropna(subset=["att_err", "def_err"])
    err = np.log1p(x.att_err) ** 2 + np.log1p(x.def_err) ** 2
    return 1 - err.mean() / x.lr.var(), len(x)


def principal_commanders(links):
    """The first linked commander of each side of each battle (CDB90 lists the commander first)."""
    return links.groupby(["isqno", "attacker"], sort=False).first().reset_index()[["isqno", "attacker", "wikidata_id"]]


def prior_records(b, links):
    """Each side's principal commander's record in CDB90 battles fought strictly before this one.

    Returns one row per battle with the attacker's and defender's prior wins, losses and battles.
    Battles on the same date do not count for each other.
    """
    pc = principal_commanders(links).merge(b[["isqno", "date", "wina"]], on="isqno")
    pc["res"] = pc.wina * np.where(pc.attacker == 1, 1, -1)
    rows = []
    for _, g in pc.groupby("wikidata_id"):
        dates, res = g.date.to_numpy(), g.res.to_numpy()
        for i, r in enumerate(g.itertuples()):
            earlier = dates < dates[i]
            rows.append((r.isqno, r.attacker, earlier.sum(), (res[earlier] == 1).sum(), (res[earlier] == -1).sum()))
    rec = pd.DataFrame(rows, columns=["isqno", "attacker", "n", "w", "l"])
    out = b[["isqno"]].copy()
    for side, flag in (("a", 1), ("d", 0)):
        s = rec[rec.attacker == flag].drop(columns="attacker").set_index("isqno").add_prefix(f"{side}_prior_")
        out = out.join(s, on="isqno")
    return out.fillna(0)


def record_edge(rec):
    """+1 if the attacker's commander had the better prior net record (wins minus losses), -1 if worse, 0 if equal."""
    net = (rec.a_prior_w - rec.a_prior_l) - (rec.d_prior_w - rec.d_prior_l)
    return np.sign(net)


def side_rows(b, links):
    """One row per principal commander and decisive battle: his side's coded advantages and result."""
    pc = principal_commanders(links).merge(b[["isqno", "wina", "lr", "surpa", *CODED]], on="isqno")
    sgn = np.where(pc.attacker == 1, 1, -1)
    out = pd.DataFrame({"isqno": pc.isqno, "commander": pc.wikidata_id, "res": pc.wina * sgn})
    out["lr"] = pc.lr * sgn
    out["surpa"] = pc.surpa * sgn
    for col in CODED:
        out[col] = pc[col] * sgn
    out = out[out.res != 0]
    out["won"] = (out.res == 1).astype(int)
    return out.reset_index(drop=True)


def within_commander(rows, col):
    """Correlation of a side's advantage with its winning, within commanders who both won and lost.

    A trait of the commander would not vary with his own results; a code read off the result would.
    """
    x = rows.dropna(subset=[col])
    k = x.groupby("commander").won.agg(["size", "mean"])
    keep = k[(k["size"] >= 2) & (k["mean"] > 0) & (k["mean"] < 1)].index
    x = x[x.commander.isin(keep)]
    dv = x[col] - x.groupby("commander")[col].transform("mean")
    dw = x.won - x.groupby("commander").won.transform("mean")
    return np.corrcoef(dv, dw)[0, 1], x.commander.nunique(), len(x)


def pair_rows(b, links):
    """Decisive battles between principal commanders who met more than once with different results.

    Each battle is oriented to the pair's first commander (by Wikidata id): his result (+1/-1),
    his side's coded leadership advantage, force ratio and surprise.
    """
    pc = principal_commanders(links).merge(b[["isqno", "wina", "leada", "lr", "surpa", "name", "year"]], on="isqno")
    a = pc[pc.attacker == 1].rename(columns={"wikidata_id": "att"})
    d = pc[pc.attacker == 0][["isqno", "wikidata_id"]].rename(columns={"wikidata_id": "dfn"})
    m = a.merge(d, on="isqno")
    m = m[m.wina.isin([1, -1]) & m.leada.notna()].copy()
    first = np.where(m.att < m.dfn, m.att, m.dfn)
    m["c1"], m["c2"] = first, np.where(m.att < m.dfn, m.dfn, m.att)
    sgn = np.where(m.att == m.c1, 1, -1)
    m["res"], m["lead"], m["lr1"], m["surp"] = m.wina * sgn, m.leada * sgn, m.lr * sgn, m.surpa * sgn
    k = m.groupby(["c1", "c2"]).res.agg(["size", "nunique"])
    keep = k[(k["size"] >= 2) & (k["nunique"] == 2)].index
    return m.set_index(["c1", "c2"]).loc[keep].reset_index()


def within_pair(rows, col, permutations=0, seed=SEED):
    """Correlation of a coded advantage with the result within pairs of opposing commanders.

    With permutations > 0, also the share of within-pair shuffles of the results giving a
    correlation at least as large (a one-sided permutation p-value).
    """
    keys = rows.groupby(["c1", "c2"]).ngroup().to_numpy()
    counts = np.bincount(keys)

    def centred(v):
        return v - (np.bincount(keys, weights=v) / counts)[keys]

    dv = centred(rows[col].to_numpy(float))
    res = rows.res.to_numpy(float)
    r0 = np.corrcoef(dv, centred(res))[0, 1]
    if not permutations:
        return r0, None
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(keys == k) for k in range(counts.size)]
    hits = 0
    r = res.copy()
    for _ in range(permutations):
        for ix in idx:
            r[ix] = res[rng.permutation(ix)]
        hits += np.corrcoef(dv, centred(r))[0, 1] >= r0
    return r0, (hits + 1) / (permutations + 1)
