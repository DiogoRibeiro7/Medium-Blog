"""Screens for bid rigging, evaluated the way an agency would use them.

    python screens.py            # prints every number in the article, writes results/

Data: the coalition-level files of Imhof and Wallimann (2021), downloaded by fetch.py from
Mendeley Data (doi:10.17632/x8jwmbcpgn.1, CC BY 4.0). A row is a coalition of three firms
in one period (cartel or competitive); its 36 features are the minimum, maximum, mean and
median, across the tenders in which the three firms bid together, of nine screens computed
from the three firms' bids in each tender.

The script compares the published evaluation (balanced classes, random 75/25 split, 100
repetitions) with evaluations that hold out what a screen meets in practice: firms it has
not seen, another country, and the natural share of cartels.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
RES = HERE / "results"

SCREENS = ["cvbid", "skew1", "D", "RD1", "RDnorm", "RDalter", "Diffperc", "Spread", "Kolmo_stat"]
STATS = ["min3", "max3", "mean3", "median3"]
ALL = [s + c for c in SCREENS for s in STATS]
SCALE_FREE = [c for c in ALL if not c.endswith("3D")]      # D is in the currency of the bids
# Direction expected under collusion, from the screening literature: bids of a cartel are
# closer together (lower coefficient of variation and spread) and further from a uniform
# spread of bids (higher Kolmogorov-Smirnov statistic).
SIGNED = {"median3cvbid": -1, "median3Spread": -1, "median3Kolmo_stat": +1}
CORE = ["median3cvbid", "median3Spread", "median3Diffperc", "median3Kolmo_stat"]
# Screens whose direction under collusion is the same in all three data sets (section 3)
VARIANCE = [s + c for c in ("cvbid", "Spread", "Kolmo_stat") for s in STATS]
NAMES = {"CH": "Switzerland", "JP": "Okinawa", "IT": "Turin"}
PUBLISHED_RF = {"JP": (94.9, 96.9, 92.8), "CH": (89.7, 88.3, 91.1), "IT": (89.1, 87.6, 90.6)}

N_TREES = 500
SEED = 20261121


def forest(seed: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=N_TREES, max_features="sqrt", n_jobs=2,
                                  random_state=seed)


def load() -> dict[str, pd.DataFrame]:
    """Swiss rows from the file that keeps the firm identifiers; the screens are identical."""
    ch = pd.read_csv(RAW / "Swissdata_addpred.csv")
    jp = pd.read_csv(RAW / "Japanesedata.csv")
    it = pd.read_csv(RAW / "Italiandata.csv").dropna(subset=["cartelperiod"])
    out = {}
    for k, d in (("CH", ch), ("JP", jp), ("IT", it)):
        d = d.copy()
        d["y"] = d.cartelperiod.astype(int)
        d["n_tenders"] = d.number_project.astype(int)
        ids = d.idfirm_.astype(str)
        d["coalition"] = ids if k != "IT" else pd.Series(range(len(d)), index=d.index).astype(str)
        d["firms"] = [tuple(s.strip("()").split(",")) if s.startswith("(") else None for s in ids]
        out[k] = d[ALL + ["y", "n_tenders", "coalition", "firms"]].reset_index(drop=True)
    return out


def auc(y, score) -> float:
    return float(roc_auc_score(y, score))


def balanced(d: pd.DataFrame, rng: np.random.Generator, n: int | None = None) -> pd.DataFrame:
    a, b = d[d.y == 0], d[d.y == 1]
    k = min(len(a), len(b)) if n is None else n // 2
    ia = rng.choice(len(a), size=min(k, len(a)), replace=False)
    ib = rng.choice(len(b), size=min(k, len(b)), replace=False)
    return pd.concat([a.iloc[ia], b.iloc[ib]])


def shares(y, pred) -> tuple[float, float, float]:
    y, pred = np.asarray(y), np.asarray(pred)
    return (100 * (y == pred).mean(), 100 * (pred[y == 1] == 1).mean(),
            100 * (pred[y == 0] == 0).mean())


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def describe(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for k, d in data.items():
        known = d[d.firms.notna()]
        firms = {f for t in known.firms for f in t}
        rows.append(dict(country=k, rows=len(d), cartel=int(d.y.sum()), competitive=int((d.y == 0).sum()),
                         cartel_share=d.y.mean(), coalitions=d.coalition.nunique() if k != "IT" else np.nan,
                         rows_with_firm_ids=len(known), firms=len(firms),
                         tenders_median_cartel=d[d.y == 1].n_tenders.median(),
                         tenders_median_comp=d[d.y == 0].n_tenders.median(),
                         cv_median_cartel=d[d.y == 1].median3cvbid.median(),
                         cv_median_comp=d[d.y == 0].median3cvbid.median()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. The published protocol
# ---------------------------------------------------------------------------

def published_protocol(d: pd.DataFrame, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    acc, coll, comp, aucs = [], [], [], []
    for r in range(reps):
        b = balanced(d, rng)
        idx = rng.permutation(len(b))
        ntr = int(0.75 * len(b))
        tr, te = b.iloc[idx[:ntr]], b.iloc[idx[ntr:]]
        m = forest(r).fit(tr[ALL], tr.y)
        p = m.predict_proba(te[ALL])[:, 1]
        a, c1, c0 = shares(te.y, (p > 0.5).astype(int))
        acc.append(a); coll.append(c1); comp.append(c0); aucs.append(auc(te.y, p))
    return dict(ccr=np.mean(acc), ccr_cartel=np.mean(coll), ccr_comp=np.mean(comp),
                auc=np.mean(aucs), n_train=ntr)


# ---------------------------------------------------------------------------
# 3. Single screens, no training
# ---------------------------------------------------------------------------

def single_screens(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for k, d in data.items():
        for c in [f"median3{s}" for s in SCREENS if s != "D"]:
            a = auc(d.y, d[c])                       # AUC if higher values meant collusion
            rows.append(dict(country=k, screen=c, auc_high_means_cartel=a,
                             direction="higher in cartel" if a > 0.5 else "lower in cartel",
                             auc_oriented=max(a, 1 - a)))
    return pd.DataFrame(rows)


def signed_screen(d: pd.DataFrame, col: str = "median3cvbid") -> np.ndarray:
    return SIGNED[col] * d[col].to_numpy()


# ---------------------------------------------------------------------------
# 4. Firms the model has not seen
# ---------------------------------------------------------------------------

def split_by_firms(d: pd.DataFrame, held: set) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Training rows: coalitions with none of the held-out firms. Test rows: the others.
    Also returns, for every row, how many of its three firms are held out."""
    k = d.firms.map(lambda t: sum(f in held for f in t))
    return d[k == 0], d[k > 0], k


def unseen_firms(d: pd.DataFrame, reps: int, seed: int, test_share: float = 0.3) -> pd.DataFrame:
    """Hold out a random 30% of firms. Train on coalitions with none of them; test on
    coalitions with one, two or three of them. Compare with a random split of the same
    training size, and with the median coefficient of variation, which needs no training."""
    d = d[d.firms.notna()].reset_index(drop=True)
    firms = sorted({f for t in d.firms for f in t})
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(reps):
        held = set(rng.choice(firms, size=int(round(test_share * len(firms))), replace=False))
        tr, te, k = split_by_firms(d, held)
        if tr.y.nunique() < 2 or te.y.nunique() < 2:
            continue
        m = forest(r).fit(tr[ALL], tr.y)
        p = m.predict_proba(te[ALL])[:, 1]
        cv = signed_screen(te)
        # random split, same number of training rows, tested on the rest
        idx = rng.permutation(len(d))
        rtr, rte = d.iloc[idx[:len(tr)]], d.iloc[idx[len(tr):]]
        pr = forest(r).fit(rtr[ALL], rtr.y).predict_proba(rte[ALL])[:, 1]
        row = dict(rep=r, n_train=len(tr), n_test=len(te), rf_unseen=auc(te.y, p),
                   rf_random_same_size=auc(rte.y, pr), cv_unseen=auc(te.y, cv))
        for j in (1, 2, 3):
            s = (k[k > 0] == j).to_numpy()
            if s.sum() >= 20 and te.y[s].nunique() == 2:
                row[f"rf_k{j}"] = auc(te.y[s], p[s])
                row[f"cv_k{j}"] = auc(te.y[s], cv[s])
                row[f"n_k{j}"] = int(s.sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Number of tenders shared by a coalition
# ---------------------------------------------------------------------------

def within_strata_auc(d: pd.DataFrame, score: np.ndarray, strata: pd.Series) -> float:
    num = den = 0.0
    for _, idx in d.groupby(strata).groups.items():
        g = d.loc[idx]
        if g.y.nunique() == 2:
            w = (g.y == 1).sum() * (g.y == 0).sum()
            num += w * auc(g.y, score[g.index.to_numpy()])
            den += w
    return num / den


def tender_counts(data: dict[str, pd.DataFrame], seed: int) -> pd.DataFrame:
    rows = []
    for k, d in data.items():
        row = dict(country=k, auc_n_tenders=auc(d.y, d.n_tenders))
        for c in ("min3cvbid", "max3cvbid", "median3cvbid"):
            s = -d[c].to_numpy()
            row[f"{c}_pooled"] = auc(d.y, s)
            row[f"{c}_within_n"] = within_strata_auc(d, s, d.n_tenders)
            for y in (0, 1):
                g = d[d.y == y]
                row[f"{c}_spearman_n_y{y}"] = g[c].corr(np.log(g.n_tenders), method="spearman")
        rng = np.random.default_rng(seed)
        ext = [c for c in SCALE_FREE if c.startswith(("min3", "max3"))]
        cen = [c for c in SCALE_FREE if c.startswith(("mean3", "median3"))]
        res = {"extremes": [], "centre": []}
        for r in range(5):
            idx = rng.permutation(len(d)); ntr = int(0.75 * len(d))
            tr, te = d.iloc[idx[:ntr]], d.iloc[idx[ntr:]]
            for name, cols in (("extremes", ext), ("centre", cen)):
                res[name].append(auc(te.y, forest(r).fit(tr[cols], tr.y).predict_proba(te[cols])[:, 1]))
        row["rf_extremes_auc"] = np.mean(res["extremes"])
        row["rf_centre_auc"] = np.mean(res["centre"])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Another country
# ---------------------------------------------------------------------------

def ranks(d: pd.DataFrame) -> pd.DataFrame:
    """Each screen as a percentile rank within its own country."""
    r = d.copy()
    for c in SCALE_FREE:
        r[c] = d[c].rank(pct=True)
    return r


def transfer(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    R = {k: ranks(v) for k, v in data.items()}
    pair = []
    for s in data:
        m = forest(0).fit(data[s][SCALE_FREE], data[s].y)
        for t in data:
            if t != s:
                pair.append(dict(train=s, test=t, rf_raw=auc(data[t].y, m.predict_proba(data[t][SCALE_FREE])[:, 1])))
    loo = []
    for t in data:
        others = [s for s in data if s != t]
        tr = pd.concat([data[s] for s in others]); trR = pd.concat([R[s] for s in others])
        rf_raw = auc(data[t].y, forest(0).fit(tr[SCALE_FREE], tr.y).predict_proba(data[t][SCALE_FREE])[:, 1])
        rf_rank = auc(R[t].y, forest(0).fit(trR[SCALE_FREE], trR.y).predict_proba(R[t][SCALE_FREE])[:, 1])
        rf_var = auc(R[t].y, forest(0).fit(trR[VARIANCE], trR.y).predict_proba(R[t][VARIANCE])[:, 1])
        rf_core = auc(R[t].y, forest(0).fit(trR[CORE], trR.y).predict_proba(R[t][CORE])[:, 1])
        lr_all = LogisticRegression(max_iter=5000).fit(trR[SCALE_FREE], trR.y)
        lr = LogisticRegression(max_iter=5000).fit(trR[CORE], trR.y)
        loo.append(dict(test=t, train="+".join(others), rf_raw=rf_raw, rf_within_country_ranks=rf_rank,
                        rf_ranks_variance_screens=rf_var, rf_ranks_four_medians=rf_core,
                        logit_ranks_all_screens=auc(R[t].y, lr_all.predict_proba(R[t][SCALE_FREE])[:, 1]),
                        logit_ranks_four_medians=auc(R[t].y, lr.predict_proba(R[t][CORE])[:, 1]),
                        median_cv_no_training=auc(data[t].y, signed_screen(data[t])),
                        **{f"coef_{c.replace('median3', '')}": float(b) for c, b in zip(CORE, lr.coef_[0])}))
    return pd.DataFrame(pair), pd.DataFrame(loo)


# ---------------------------------------------------------------------------
# 7. Thresholds and lists for investigation, at the natural share of cartels
# ---------------------------------------------------------------------------

def thresholds(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    ch = data["CH"]
    # threshold on the median CV that flags 80% of the Swiss cartel coalitions
    thr = float(np.quantile(ch[ch.y == 1].median3cvbid, 0.80))
    rows = []
    for k, d in data.items():
        flag = d.median3cvbid <= thr
        rows.append(dict(country=k, rule=f"median CV <= {thr:.2f} (Swiss 80% rule)",
                         flagged_share=flag.mean(), sensitivity=flag[d.y == 1].mean(),
                         false_flag_share_of_competitive=flag[d.y == 0].mean(),
                         precision=d.y[flag].mean() if flag.any() else np.nan))
        q = d.median3cvbid.rank(pct=True) <= 0.10
        rows.append(dict(country=k, rule="lowest 10% of median CV within the country",
                         flagged_share=q.mean(), sensitivity=q[d.y == 1].mean(),
                         false_flag_share_of_competitive=q[d.y == 0].mean(), precision=d.y[q].mean()))
    return pd.DataFrame(rows), {"threshold": thr}


def investigation_lists(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Precision of the top 5% and 10% of coalitions, at each country's own cartel share.
    The forest is trained on the other two countries (within-country ranks), so that no
    coalition it scores was seen in training."""
    R = {k: ranks(v) for k, v in data.items()}
    rows = []
    for t in data:
        others = [s for s in data if s != t]
        trR = pd.concat([R[s] for s in others])
        p_rf = forest(0).fit(trR[SCALE_FREE], trR.y).predict_proba(R[t][SCALE_FREE])[:, 1]
        cv = signed_screen(data[t])
        y = data[t].y.to_numpy()
        for frac in (0.05, 0.10):
            n = int(round(frac * len(y)))
            for name, s in (("median CV", cv), ("forest, other countries", p_rf)):
                top = np.argsort(-s, kind="stable")[:n]
                rows.append(dict(country=t, top_share=frac, scorer=name, n_flagged=n,
                                 precision=y[top].mean(), recall=y[top].sum() / y.sum(),
                                 base_rate=y.mean()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def main() -> None:
    RES.mkdir(exist_ok=True)
    data = load()
    out = io.StringIO()
    with redirect_stdout(out):
        print("=" * 78); print("1. DATA"); print("=" * 78)
        desc = describe(data); desc.to_csv(RES / "data_summary.csv", index=False)
        for r in desc.itertuples():
            print(f"{NAMES[r.country]}: {r.rows:,} coalition-periods, {r.cartel:,} cartel and {r.competitive:,} competitive "
                  f"(cartel share {100 * r.cartel_share:.1f}%)")
            print(f"  median tenders per coalition: cartel {r.tenders_median_cartel:.0f}, competitive {r.tenders_median_comp:.0f}")
            print(f"  median of the median CV: cartel {r.cv_median_cartel:.2f}, competitive {r.cv_median_comp:.2f}")
            if r.rows_with_firm_ids:
                print(f"  rows with firm identifiers: {r.rows_with_firm_ids:,}, firms {r.firms}")
        ch = data["CH"]; chk = ch[ch.firms.notna()]
        both = chk.groupby("coalition").y.nunique().eq(2).sum()
        print(f"Swiss market with firm identifiers: {chk.coalition.nunique()} coalitions, {both} seen in both periods")
        jpc = data["JP"].groupby("coalition").y.nunique().eq(2).sum()
        print(f"Okinawa: {data['JP'].coalition.nunique():,} coalitions, {jpc} seen in both periods")

        print(); print("=" * 78); print("2. THE PUBLISHED PROTOCOL (balanced, random 75/25 split, 100 repetitions)"); print("=" * 78)
        pub = []
        for i, k in enumerate(("JP", "CH", "IT")):
            r = published_protocol(data[k], reps=100, seed=SEED + i)
            pub.append(dict(country=k, **r))
            pb = PUBLISHED_RF[k]
            print(f"{NAMES[k]}: correct classification {fmt(r['ccr'], 1)}% (cartel {fmt(r['ccr_cartel'], 1)}%, "
                  f"competitive {fmt(r['ccr_comp'], 1)}%), AUC {fmt(r['auc'])}; published random forest "
                  f"{pb[0]}% ({pb[1]}%, {pb[2]}%); training rows {r['n_train']:,}")
        pd.DataFrame(pub).to_csv(RES / "published_protocol.csv", index=False)

        print(); print("=" * 78); print("3. SINGLE SCREENS, NO TRAINING (AUC of the median across tenders)"); print("=" * 78)
        ss = single_screens(data); ss.to_csv(RES / "single_screens.csv", index=False)
        for k in data:
            g = ss[ss.country == k].sort_values("auc_oriented", ascending=False)
            print(f"{NAMES[k]}: " + "; ".join(f"{r.screen.replace('median3', '')} {fmt(r.auc_oriented)} ({r.direction})"
                                              for r in g.itertuples()))
        flips = ss.pivot(index="screen", columns="country", values="direction")
        flipped = flips[flips.nunique(axis=1) > 1]
        print("screens whose direction differs between countries: " + ", ".join(
            f"{s.replace('median3', '')} ({', '.join(f'{c} {v}' for c, v in row.items())})" for s, row in flipped.iterrows()))

        print(); print("=" * 78); print("4. FIRMS THE MODEL HAS NOT SEEN (30% of firms held out, 50 draws)"); print("=" * 78)
        uf_all = []
        for i, k in enumerate(("JP", "CH")):
            uf = unseen_firms(data[k], reps=50, seed=SEED + 10 + i); uf["country"] = k; uf_all.append(uf)
            q = lambda c: f"{fmt(uf[c].mean())} [{fmt(uf[c].quantile(0.1))}, {fmt(uf[c].quantile(0.9))}]"
            print(f"{NAMES[k]} ({'market with firm identifiers' if k == 'CH' else 'all coalitions'}): draws {len(uf)}, "
                  f"training rows {uf.n_train.mean():.0f}, test rows {uf.n_test.mean():.0f} (means)")
            print(f"  forest, random split of the same training size: AUC {q('rf_random_same_size')}")
            print(f"  forest, test coalitions with at least one unseen firm: AUC {q('rf_unseen')}")
            print(f"  median CV, same test coalitions, no training: AUC {q('cv_unseen')}")
            for j in (1, 2, 3):
                if f"rf_k{j}" in uf and uf[f"rf_k{j}"].notna().sum() >= 10:
                    v = uf.dropna(subset=[f"rf_k{j}"])
                    print(f"  {j} unseen firm(s): forest {fmt(v[f'rf_k{j}'].mean())}, median CV {fmt(v[f'cv_k{j}'].mean())} "
                          f"({len(v)} draws, {v[f'n_k{j}'].mean():.0f} test rows on average)")
            print(f"  draws in which the median CV beats the forest on unseen firms: {(uf.cv_unseen > uf.rf_unseen).mean() * 100:.0f}%")
        pd.concat(uf_all).to_csv(RES / "unseen_firms.csv", index=False)

        print(); print("=" * 78); print("5. NUMBER OF TENDERS A COALITION SHARES"); print("=" * 78)
        tc = tender_counts(data, SEED + 20); tc.to_csv(RES / "tender_counts.csv", index=False)
        for r in tc.itertuples():
            print(f"{NAMES[r.country]}: AUC of the number of tenders alone {fmt(r.auc_n_tenders)}")
            for c in ("min3cvbid", "max3cvbid", "median3cvbid"):
                print(f"  {c}: pooled AUC {fmt(getattr(r, c + '_pooled'))}, within tenders-count strata "
                      f"{fmt(getattr(r, c + '_within_n'))}; Spearman with log tenders: competitive "
                      f"{getattr(r, c + '_spearman_n_y0'):.2f}, cartel {getattr(r, c + '_spearman_n_y1'):.2f}")
            print(f"  forest on minima and maxima only {fmt(r.rf_extremes_auc)}, on means and medians only {fmt(r.rf_centre_auc)}")

        print(); print("=" * 78); print("6. ANOTHER COUNTRY (scale-free screens; D left out)"); print("=" * 78)
        pair, loo = transfer(data); pair.to_csv(RES / "transfer_pairs.csv", index=False); loo.to_csv(RES / "transfer_leave_one_out.csv", index=False)
        for r in pair.itertuples():
            print(f"forest trained on {NAMES[r.train]}, tested on {NAMES[r.test]}: AUC {fmt(r.rf_raw)}")
        for r in loo.itertuples():
            print(f"{NAMES[r.test]} held out (trained on {r.train}): forest {fmt(r.rf_raw)}, forest on within-country ranks "
                  f"{fmt(r.rf_within_country_ranks)}, forest on ranks of the variance and uniformity screens "
                  f"{fmt(r.rf_ranks_variance_screens)}, forest on ranks of four medians {fmt(r.rf_ranks_four_medians)}, "
                  f"logit on ranks of all screens {fmt(r.logit_ranks_all_screens)}, logit on ranks of four medians {fmt(r.logit_ranks_four_medians)}, "
                  f"median CV alone {fmt(r.median_cv_no_training)}")
            print(f"  logit coefficients on the four ranked medians: CV {r.coef_cvbid:+.2f}, spread {r.coef_Spread:+.2f}, "
                  f"percentage difference {r.coef_Diffperc:+.2f}, KS {r.coef_Kolmo_stat:+.2f}")
        ranked = pd.concat([ranks(v) for v in data.values()])
        print(f"rank correlation of median CV and median spread, all coalitions (within-country ranks): "
              f"{ranked.median3cvbid.corr(ranked.median3Spread, method='spearman'):.2f}")

        print(); print("=" * 78); print("7. THRESHOLDS AND INVESTIGATION LISTS"); print("=" * 78)
        th, info = thresholds(data); th.to_csv(RES / "thresholds.csv", index=False)
        print(f"Swiss threshold that flags 80% of Swiss cartel coalitions: median CV <= {info['threshold']:.2f}")
        for r in th.itertuples():
            print(f"{NAMES[r.country]}, {r.rule}: flags {100 * r.flagged_share:.1f}% of coalitions, "
                  f"{100 * r.sensitivity:.1f}% of cartel ones, {100 * r.false_flag_share_of_competitive:.1f}% of competitive ones; "
                  f"precision {100 * r.precision:.1f}%")
        il = investigation_lists(data); il.to_csv(RES / "investigation_lists.csv", index=False)
        for r in il.itertuples():
            print(f"{NAMES[r.country]} (cartel share {100 * r.base_rate:.1f}%), top {100 * r.top_share:.0f}% "
                  f"({r.n_flagged:,} coalitions) by {r.scorer}: precision {100 * r.precision:.1f}%, recall {100 * r.recall:.1f}%")

    text = out.getvalue()
    (RES / "summary.txt").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
