"""Part 4 of the Simpson's paradox series: reversals in data-science practice.

Every number quoted in the article is printed by this script. Run:

    python practice_numbers.py > output.txt

Data downloaded on first run from pinned commits on GitHub, checked against SHA-256 digests and cached
in data/:
  - Kevin Hillstrom's MineThatData e-mail experiment (2008): 64,000 customers randomised to a men's
    e-mail, a women's e-mail or no e-mail, from a public copy of the original file;
  - nycflights13 flights (Wickham): every flight that departed New York City in 2013, via Rdatasets;
  - ProPublica's COMPAS two-year recidivism file (Larson, Mattu, Kirchner and Angwin, 2016).

Sections
1. A/B tests: what randomisation guarantees, segment effects, and a ramp-up re-enacted on real outcomes.
2. Metric decomposition: a monthly on-time rate, and carrier comparisons split into mix and within-segment
   parts, with the mix part attributed to origin airport and departure hour by Shapley values.
3. Fairness metrics computed in aggregate and within strata.
4. Scanning a data set for reversals automatically, with an evidence filter and a split-sample check.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import itertools
import math
import urllib.request
from pathlib import Path

import numpy as np
from scipy import stats

SEED = 2026
HERE = Path(__file__).parent
DATA = HERE / "data"
RD = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/"
FILES = {
    "hillstrom.csv": ("https://raw.githubusercontent.com/haijinwootest/email_data_analysis/"
                      "5635e3d6ac33bc6f902aad8dd35b08af7160191c/"
                      "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
                      "27bab8c5d3669f26ec08ebb50a0a78317542f29501156f2e2af6781fab4cd7e2"),
    "flights.csv": (RD + "csv/nycflights13/flights.csv",
                    "b791a2590c9167aa0bd3cb45af3f64101180878cf088c183eae4011e3a654f42"),
    "compas-scores-two-years.csv": ("https://raw.githubusercontent.com/propublica/compas-analysis/"
                                    "bafff5da3f2e45eca6c2d5055faad269defd135a/compas-scores-two-years.csv",
                                    "c451db85908b2f7fef1d83203bedf6b71ecda0d5af468d82ae62178f91d0cc7d"),
}
HOUR_BANDS = ((0, 9, "before 9"), (9, 12, "9-12"), (12, 15, "12-15"), (15, 18, "15-18"), (18, 24, "18 on"))


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fetch(name: str) -> Path:
    url, digest = FILES[name]
    path = DATA / name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as r:
            path.write_bytes(r.read())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise RuntimeError(f"{name}: SHA-256 {got} does not match the pinned digest {digest}")
    return path


def pct(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}%"


def pts(x: float, d: int = 2) -> str:
    return f"{100 * x:+.{d}f}"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_hillstrom() -> dict[str, np.ndarray]:
    with open(fetch("hillstrom.csv")) as f:
        rows = list(csv.DictReader(f))
    out = {k: np.array([r[k] for r in rows]) for k in ("segment", "history_segment", "zip_code", "channel")}
    for k in ("newbie", "mens", "womens", "visit", "conversion", "recency"):
        out[k] = np.array([int(r[k]) for r in rows])
    for k in ("spend", "history"):
        out[k] = np.array([float(r[k]) for r in rows])
    return out


def hour_band(hour: np.ndarray) -> np.ndarray:
    out = np.empty(len(hour), dtype=object)
    for lo, hi, name in HOUR_BANDS:
        out[(hour >= lo) & (hour < hi)] = name
    return out.astype(str)


def load_flights() -> dict[str, np.ndarray]:
    """Flights with an arrival delay recorded (cancelled and diverted flights have none). On time, as the US
    Department of Transportation counts it, means arriving less than 15 minutes after schedule."""
    with open(fetch("flights.csv")) as f:
        rows = list(csv.DictReader(f))
    n_all = len(rows)
    rows = [r for r in rows if r["arr_delay"] not in ("NA", "")]
    out = {
        "month": np.array([int(r["month"]) for r in rows]),
        "day": np.array([int(r["day"]) for r in rows]),
        "carrier": np.array([r["carrier"] for r in rows]),
        "origin": np.array([r["origin"] for r in rows]),
        "dest": np.array([r["dest"] for r in rows]),
        "hour": np.array([int(r["hour"]) for r in rows]),
        "distance": np.array([float(r["distance"]) for r in rows]),
        "arr_delay": np.array([float(r["arr_delay"]) for r in rows]),
    }
    out["ontime"] = (out["arr_delay"] < 15).astype(int)
    out["hour_band"] = hour_band(out["hour"])
    out["quarter"] = np.array([f"Q{(m - 1) // 3 + 1}" for m in out["month"]])
    out["month_name"] = np.array([datetime.date(2013, m, 1).strftime("%b") for m in out["month"]])
    out["weekday"] = np.array([datetime.date(2013, m, d).strftime("%a") for m, d in zip(out["month"], out["day"])])
    bands = np.array(["under 500", "500-999", "1000-1499", "1500 and over"])
    out["distance_band"] = bands[np.digitize(out["distance"], [500, 1000, 1500])]
    out["n_all"] = n_all
    return out


def load_compas() -> dict[str, np.ndarray]:
    """ProPublica's filters: charge date within 30 days of the screening, a recidivism flag recorded, an
    ordinary traffic offence excluded, a score text present."""
    with open(fetch("compas-scores-two-years.csv")) as f:
        r = csv.reader(f)
        hdr = next(r)
        col = {h: hdr.index(h) for h in hdr}  # first occurrence for the duplicated columns
        rows = list(r)
    n_all = len(rows)
    keep = []
    for x in rows:
        days = x[col["days_b_screening_arrest"]]
        if days == "" or not -30 <= int(float(days)) <= 30:
            continue
        if int(x[col["is_recid"]]) == -1 or x[col["c_charge_degree"]] == "O" or x[col["score_text"]] == "N/A":
            continue
        keep.append(x)
    out = {
        "race": np.array([x[col["race"]] for x in keep]),
        "sex": np.array([x[col["sex"]] for x in keep]),
        "age_cat": np.array([x[col["age_cat"]] for x in keep]),
        "decile": np.array([int(x[col["decile_score"]]) for x in keep]),
        "priors": np.array([int(x[col["priors_count"]]) for x in keep]),
        "charge": np.array([x[col["c_charge_degree"]] for x in keep]),
        "recid": np.array([int(x[col["two_year_recid"]]) for x in keep]),
    }
    out["n_all"] = n_all
    return out


# ---------------------------------------------------------------------------
# Tools quoted in the article
# ---------------------------------------------------------------------------

def counts_by(y: np.ndarray, *keys: np.ndarray) -> dict[tuple, tuple[int, int]]:
    """{key tuple: (successes, trials)} for a binary outcome y grouped by one or more key arrays."""
    out: dict[tuple, list[int]] = {}
    for k, v in zip(zip(*keys), y):
        c = out.setdefault(k, [0, 0])
        c[0] += int(v)
        c[1] += 1
    return {k: (s, n) for k, (s, n) in out.items()}


def kitagawa(pa, wa, pb, wb):
    """Split the difference between two weighted averages, sum(wa * pa) - sum(wb * pb), into a mix part
    (different weights at average rates) and a within part (different rates at average weights)."""
    mix = np.sum((wa - wb) * (pa + pb) / 2)
    within = np.sum((pa - pb) * (wa + wb) / 2)
    return mix, within


def stratified_difference(sa, na, sb, nb):
    """Difference of rates a - b standardised to the pooled stratum sizes, with a normal-approximation
    standard error. sa, na, sb, nb: successes and trials by stratum."""
    pa, pb = sa / na, sb / nb
    w = (na + nb) / (na + nb).sum()
    d = np.sum(w * (pa - pb))
    se = math.sqrt(np.sum(w ** 2 * (pa * (1 - pa) / na + pb * (1 - pb) / nb)))
    return d, se


def mh_difference(sa, na, sb, nb):
    """Mantel-Haenszel risk difference a - b with the Sato variance; weights na * nb / (na + nb)."""
    pa, pb = sa / na, sb / nb
    w = na * nb / (na + nb)
    d = np.sum(w * (pa - pb)) / w.sum()
    p_num = np.sum((nb ** 2 * sa - na ** 2 * sb + na * nb * (na - nb) / 2) / (na + nb) ** 2)
    q_num = np.sum((sa * (nb - sb) + sb * (na - sa)) / (2 * (na + nb)))
    var = (d * p_num + q_num) / w.sum() ** 2
    return d, math.sqrt(var)


def difference(sa, na, sb, nb):
    pa, pb = sa / na, sb / nb
    return pa - pb, math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)


def shapley_mix(mix_of) -> dict[str, float]:
    """Shapley attribution of a mix effect to two factors. mix_of(set of factors) returns the mix part of a
    Kitagawa decomposition over the strata defined by that set (zero for the empty set)."""
    a, b = "origin", "hour_band"
    m_a, m_b, m_ab = mix_of({a}), mix_of({b}), mix_of({a, b})
    return {a: (m_a + (m_ab - m_b)) / 2, b: (m_b + (m_ab - m_a)) / 2, "joint": m_ab}


def scan(y, entity, strata: dict[str, np.ndarray], min_n: int = 30):
    """Every pair of levels of `entity`, against every stratifier: the aggregate difference in the rate of y
    over the strata both levels share (each with at least min_n records), the difference standardised to
    the pooled stratum sizes, their standard errors, and whether the stratum differences all point one way
    while the aggregate points the other."""
    found = []
    levels = sorted(set(entity))
    for name, s in strata.items():
        cnt = counts_by(y, entity, s)
        for a, b in itertools.combinations(levels, 2):
            common = sorted(k for (e, k) in cnt if e == a and (b, k) in cnt
                            and cnt[(a, k)][1] >= min_n and cnt[(b, k)][1] >= min_n)
            if len(common) < 2:
                continue
            sa, na = (np.array([cnt[(a, k)][i] for k in common], float) for i in (0, 1))
            sb, nb = (np.array([cnt[(b, k)][i] for k in common], float) for i in (0, 1))
            agg, agg_se = difference(sa.sum(), na.sum(), sb.sum(), nb.sum())
            std, std_se = stratified_difference(sa, na, sb, nb)
            d = sa / na - sb / nb
            full = (np.all(d > 0) and agg < 0) or (np.all(d < 0) and agg > 0)
            found.append(dict(stratifier=name, a=a, b=b, strata=len(common), n=int(na.sum() + nb.sum()),
                              agg=agg, agg_se=agg_se, std=std, std_se=std_se,
                              flip=bool(np.sign(agg) != np.sign(std)), full=bool(full),
                              counts=(sa, na, sb, nb)))
    return found


def posterior_flip(sa, na, sb, nb, rng, draws: int = 4000) -> float:
    """Posterior probability, under uniform priors on every stratum rate, that the aggregate and the
    standardised differences have opposite signs (stratum sizes held at their observed values)."""
    pa = rng.beta(1 + sa, 1 + na - sa, size=(draws, len(sa)))
    pb = rng.beta(1 + sb, 1 + nb - sb, size=(draws, len(sb)))
    agg = (pa * na).sum(axis=1) / na.sum() - (pb * nb).sum(axis=1) / nb.sum()
    w = (na + nb) / (na + nb).sum()
    std = (w * (pa - pb)).sum(axis=1)
    return float(np.mean(np.sign(agg) != np.sign(std)))


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def ramp_multiplier(a1: float, n1: float = 1.0, n2: float = 1.0, a2: float = 0.5) -> float:
    """Pooling a two-period test adds (baseline of period 1 - baseline of period 2) times this multiplier to
    the difference treatment - control: the share of treatment drawn from period 1 minus the share of
    control drawn from period 1."""
    t1, t2 = a1 * n1, a2 * n2
    c1, c2 = (1 - a1) * n1, (1 - a2) * n2
    return t1 / (t1 + t2) - c1 / (c1 + c2)


def ramp_reenactment(h, rng, per_day: int = 10_700, share_day1: float = 0.01):
    """Men's e-mail (treatment) against women's e-mail (control) on visits. Day 1 draws returning customers
    (newbie = 0) with share_day1 of them on the treatment; day 2 draws new customers (newbie = 1) split
    50/50. Customers are drawn without replacement from the real experiment arms."""
    out = {}
    pools = {(day, arm): h["visit"][(h["segment"] == arm) & (h["newbie"] == (day - 1))]
             for day in (1, 2) for arm in ("Mens E-Mail", "Womens E-Mail")}
    n_t1 = round(per_day * share_day1)
    sizes = {(1, "Mens E-Mail"): n_t1, (1, "Womens E-Mail"): per_day - n_t1,
             (2, "Mens E-Mail"): per_day // 2, (2, "Womens E-Mail"): per_day - per_day // 2}
    for key, n in sizes.items():
        out[key] = rng.choice(pools[key], size=n, replace=False)
    return out


def summarise_ramp(sample):
    s = {k: (v.sum(), len(v)) for k, v in sample.items()}
    t = [s[(d, "Mens E-Mail")] for d in (1, 2)]
    c = [s[(d, "Womens E-Mail")] for d in (1, 2)]
    naive = difference(sum(x[0] for x in t), sum(x[1] for x in t), sum(x[0] for x in c), sum(x[1] for x in c))
    sa, na = np.array([x[0] for x in t], float), np.array([x[1] for x in t], float)
    sb, nb = np.array([x[0] for x in c], float), np.array([x[1] for x in c], float)
    mh = mh_difference(sa, na, sb, nb)
    days = [difference(sa[i], na[i], sb[i], nb[i]) for i in range(2)]
    return naive, mh, days, (sa, na, sb, nb)


def ramp_draws(h, rng, reps: int = 2000):
    """Repeat the re-enactment: pooled and Mantel-Haenszel differences and their z statistics."""
    out = np.zeros((4, reps))
    for r in range(reps):
        naive, mh, _, _ = summarise_ramp(ramp_reenactment(h, rng))
        out[:, r] = naive[0], naive[0] / naive[1], mh[0], mh[0] / mh[1]
    return out


def flights_scan(fl, mask=None):
    """The scan of section 4 over carriers, origin airports and months, optionally on a subset of flights."""
    mask = np.ones(len(fl["ontime"]), bool) if mask is None else mask
    strata = {k: fl[k][mask] for k in ("origin", "hour_band", "month_name", "quarter", "dest", "distance_band",
                                         "weekday", "carrier")}
    found = []
    for ent in ("carrier", "origin", "month_name"):
        sub = {k: v for k, v in strata.items() if k != ent and not (ent == "month_name" and k == "quarter")}
        for f in scan(fl["ontime"][mask], fl[ent][mask], sub):
            f["entity"] = ent
            found.append(f)
    return found


BIG = ["UA", "B6", "EV", "DL", "AA", "MQ", "US", "9E", "WN"]
SHOW = {("UA", "9E"), ("EV", "DL"), ("EV", "MQ"), ("US", "9E"), ("AA", "9E"), ("MQ", "9E"), ("B6", "9E"), ("UA", "B6")}


def carrier_gap(fl, a: str, b: str) -> dict:
    """On-time rate of carrier a minus carrier b, split by Kitagawa over the cells of origin x departure-hour
    band both carriers fly, with the mix part attributed to origin and hour by Shapley values."""
    y = fl["ontime"]
    ma, mb = fl["carrier"] == a, fl["carrier"] == b

    def parts(factors):
        ca = counts_by(y[ma], *(fl[f][ma] for f in factors))
        cb = counts_by(y[mb], *(fl[f][mb] for f in factors))
        common = [k for k in ca if k in cb]
        sa = np.array([ca[k][0] for k in common], float); na = np.array([ca[k][1] for k in common], float)
        sb = np.array([cb[k][0] for k in common], float); nb = np.array([cb[k][1] for k in common], float)
        mix, within = kitagawa(sa / na, na / na.sum(), sb / nb, nb / nb.sum())
        return mix, within, (na.sum() + nb.sum()) / (ma.sum() + mb.sum()), (sa, na, sb, nb)

    mix, within, cover, cnt = parts(["origin", "hour_band"])
    sh = shapley_mix(lambda fs: parts([f for f in ("origin", "hour_band") if f in fs])[0])
    std, std_se = stratified_difference(*cnt)
    return dict(a=a, b=b, total=y[ma].mean() - y[mb].mean(), mix=mix, within=within, cover=cover,
                origin=sh["origin"], hour_band=sh["hour_band"], within3=parts(["origin", "hour_band", "month"])[1],
                std=std, std_se=std_se)


def fairness_metrics(recid, high):
    """base rate, selection rate, false positive rate, false negative rate, positive and negative predictive
    values, each as (successes, trials)."""
    return {
        "base rate": (recid.sum(), len(recid)),
        "selection rate": (high.sum(), len(high)),
        "FPR": (high[recid == 0].sum(), (recid == 0).sum()),
        "FNR": ((1 - high[recid == 1]).sum(), (recid == 1).sum()),
        "PPV": (recid[high == 1].sum(), (high == 1).sum()),
        "NPV": ((1 - recid[high == 0]).sum(), (high == 0).sum()),
    }


def stratified_metric(metric, recid, high, group_a, strata, min_n: int = 10):
    """Metric difference (group a minus the rest of the two-group sample) standardised to the pooled sizes of
    the metric's denominator in each stratum; strata with fewer than min_n in either group are left out."""
    sa, na, sb, nb, used = [], [], [], [], 0
    total = fairness_metrics(recid, high)[metric][1]
    for s in sorted(set(strata)):
        m = strata == s
        fa = fairness_metrics(recid[m & group_a], high[m & group_a])[metric]
        fb = fairness_metrics(recid[m & ~group_a], high[m & ~group_a])[metric]
        if fa[1] < min_n or fb[1] < min_n:
            continue
        used += 1
        sa.append(fa[0]); na.append(fa[1]); sb.append(fb[0]); nb.append(fb[1])
    d, se = stratified_difference(*(np.array(v, float) for v in (sa, na, sb, nb)))
    return d, se, used, (sum(na) + sum(nb)) / total


def priors_band(priors: np.ndarray, cuts) -> np.ndarray:
    labels = []
    lo = 0
    for c in cuts:
        labels.append(f"{lo}" if c - 1 == lo else f"{lo}-{c - 1}")
        lo = c
    labels.append(f"{lo}+")
    return np.array(labels)[np.digitize(priors, cuts)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)

    header("1. A/B TESTS: HILLSTROM'S E-MAIL EXPERIMENT (64,000 customers, two-week outcome window)")
    h = load_hillstrom()
    arms = ("Mens E-Mail", "Womens E-Mail", "No E-Mail")
    sizes = np.array([np.sum(h["segment"] == a) for a in arms])
    chi = stats.chisquare(sizes)
    print(f"  arm sizes: " + "; ".join(f"{a} {n:,}" for a, n in zip(arms, sizes))
          + f"; sample-ratio test against equal thirds: chi-square {chi.statistic:.3f}, p = {chi.pvalue:.3f}")
    for a in arms:
        m = h["segment"] == a
        print(f"    {a:<14} visit {pct(h['visit'][m].mean())}, conversion {pct(h['conversion'][m].mean())},"
              f" spend ${h['spend'][m].mean():.3f} per customer")
    for a in arms[:2]:
        for k in ("visit", "conversion"):
            ya, yb = h[k][h["segment"] == a], h[k][h["segment"] == "No E-Mail"]
            d, se = difference(ya.sum(), len(ya), yb.sum(), len(yb))
            print(f"    {a} against no e-mail, {k}: {pts(d)} points [{pts(d - 1.96 * se)}, {pts(d + 1.96 * se)}]")
    print("  balance of pre-treatment covariates across the three arms (chi-square test of independence):")
    covs = {"history_segment": h["history_segment"], "zip_code": h["zip_code"], "channel": h["channel"],
            "newbie": h["newbie"], "mens": h["mens"], "womens": h["womens"]}
    for name, v in covs.items():
        levels = sorted(set(v))
        tab = np.array([[np.sum((v == l) & (h["segment"] == a)) for a in arms] for l in levels])
        res = stats.chi2_contingency(tab)
        print(f"    {name:<16} {len(levels)} levels: chi-square {res.statistic:6.2f} on {res.dof:>2} df, p = {res.pvalue:.3f}")
    print("  men's e-mail against no e-mail, visits: aggregate difference against the stratum differences averaged")
    print("  with weights equal to each stratum's share of the two arms combined:")
    tm, tc = h["segment"] == "Mens E-Mail", h["segment"] == "No E-Mail"
    agg = h["visit"][tm].mean() - h["visit"][tc].mean()
    gaps = []
    for name, v in covs.items():
        levels = sorted(set(v))
        w = np.array([np.sum((v == l) & (tm | tc)) for l in levels], float)
        w /= w.sum()
        dl = np.array([h["visit"][tm & (v == l)].mean() - h["visit"][tc & (v == l)].mean() for l in levels])
        gaps.append(agg - np.sum(w * dl))
        print(f"    by {name:<16} stratum differences {pts(dl.min())} to {pts(dl.max())}; weighted average {pts(np.sum(w * dl), 3)};"
              f" aggregate {pts(agg, 3)}; gap {pts(agg - np.sum(w * dl), 3)}")
    print(f"    largest gap between the aggregate and a weighted average of stratum differences: {pts(max(np.abs(gaps)), 3)} points")
    print()
    print("  men's against women's e-mail (two creatives), visits:")
    tw = h["segment"] == "Womens E-Mail"
    d, se = difference(h["visit"][tm].sum(), tm.sum(), h["visit"][tw].sum(), tw.sum())
    print(f"    aggregate {pts(d)} points [{pts(d - 1.96 * se)}, {pts(d + 1.96 * se)}]")
    subgroup = {}
    for name in ("mens", "womens", "newbie"):
        for l in (0, 1):
            m = h[name] == l
            dd, ss = difference(h["visit"][tm & m].sum(), (tm & m).sum(), h["visit"][tw & m].sum(), (tw & m).sum())
            subgroup[(name, l)] = (dd, ss)
            print(f"    {name} = {l}: {pts(dd)} points [{pts(dd - 1.96 * ss)}, {pts(dd + 1.96 * ss)}],"
                  f" {(tm & m).sum():,} and {(tw & m).sum():,} customers")
        (d0, s0), (d1, s1) = subgroup[(name, 0)], subgroup[(name, 1)]
        z = (d1 - d0) / math.sqrt(s0 ** 2 + s1 ** 2)
        print(f"      difference between the two subgroup effects {pts(d1 - d0)} points, z = {z:.2f}")
    print()
    print("  Ramp-up, two days of equal traffic, treatment share a on day 1 and 1/2 on day 2: pooling the days adds")
    print("  (baseline on day 1 - baseline on day 2) x multiplier to the treatment-control difference")
    for a1 in (0.01, 0.05, 0.10, 0.20, 0.50):
        print(f"    a = {a1:.2f}: multiplier {ramp_multiplier(a1):+.3f}")
    # real rates for the re-enactment
    rates = {}
    for day, nb in ((1, 0), (2, 1)):
        for arm in ("Mens E-Mail", "Womens E-Mail"):
            m = (h["segment"] == arm) & (h["newbie"] == nb)
            rates[(day, arm)] = (h["visit"][m].mean(), m.sum())
    print("  Re-enactment on the real outcomes: day 1 draws returning customers (newbie = 0), 1% to the men's e-mail;")
    print("  day 2 draws new customers (newbie = 1), split evenly; 10,700 customers a day, drawn without replacement:")
    for (day, arm), (p, n) in rates.items():
        print(f"    population rate, day {day}, {arm:<14} {pct(p)} (from {n:,} customers)")
    base_gap = (rates[(1, "Mens E-Mail")][0] + rates[(1, "Womens E-Mail")][0]) / 2 - \
               (rates[(2, "Mens E-Mail")][0] + rates[(2, "Womens E-Mail")][0]) / 2
    n_t1 = round(10_700 * 0.01)
    exp_t = (n_t1 * rates[(1, "Mens E-Mail")][0] + 5350 * rates[(2, "Mens E-Mail")][0]) / (n_t1 + 5350)
    exp_c = ((10_700 - n_t1) * rates[(1, "Womens E-Mail")][0] + 5350 * rates[(2, "Womens E-Mail")][0]) / (10_700 - n_t1 + 5350)
    print(f"    average baseline, day 1 minus day 2: {pts(base_gap)} points; multiplier at 1% {ramp_multiplier(n_t1 / 10_700):+.3f}")
    print(f"    expected pooled rates: men's {pct(exp_t)}, women's {pct(exp_c)}, difference {pts(exp_t - exp_c)} points")
    sample = ramp_reenactment(h, rng)
    naive, mh, days, _ = summarise_ramp(sample)
    n_t = sum(len(sample[(d, "Mens E-Mail")]) for d in (1, 2))
    n_c = sum(len(sample[(d, "Womens E-Mail")]) for d in (1, 2))
    srm = stats.chisquare([n_t, n_c])
    print(f"    one draw: {n_t:,} treated and {n_c:,} controls; sample-ratio test against 50/50: chi-square {srm.statistic:,.0f}")
    for i, (dd, ss) in enumerate(days, 1):
        print(f"      day {i}: {pts(dd)} points [{pts(dd - 1.96 * ss)}, {pts(dd + 1.96 * ss)}]")
    print(f"      pooled: {pts(naive[0])} points [{pts(naive[0] - 1.96 * naive[1])}, {pts(naive[0] + 1.96 * naive[1])}],"
          f" z = {naive[0] / naive[1]:.2f}")
    print(f"      Mantel-Haenszel by day: {pts(mh[0])} points [{pts(mh[0] - 1.96 * mh[1])}, {pts(mh[0] + 1.96 * mh[1])}],"
          f" z = {mh[0] / mh[1]:.2f}")
    reps = 2000
    nv, nz, mv, mz = ramp_draws(h, rng, reps)
    print(f"    {reps:,} draws: pooled difference mean {pts(nv.mean())} points (2.5th to 97.5th percentile {pts(np.percentile(nv, 2.5))}"
          f" to {pts(np.percentile(nv, 97.5))}); negative in {np.mean(nv < 0):.3f} of draws, significantly positive"
          f" (z > 1.96) in {np.mean(nz > 1.96):.3f}, significantly negative in {np.mean(nz < -1.96):.3f}")
    print(f"      Mantel-Haenszel by day: mean {pts(mv.mean())} points ({pts(np.percentile(mv, 2.5))} to {pts(np.percentile(mv, 97.5))});"
          f" significantly positive in {np.mean(mz > 1.96):.3f} of draws")
    full_t = h["visit"][tm].mean() - h["visit"][tw].mean()
    print(f"    for reference, the full experiment: {pts(full_t)} points; new customers only {pts(rates[(2, 'Mens E-Mail')][0] - rates[(2, 'Womens E-Mail')][0])};"
          f" returning only {pts(rates[(1, 'Mens E-Mail')][0] - rates[(1, 'Womens E-Mail')][0])}")

    header("2. METRIC DECOMPOSITION: ON-TIME ARRIVAL OF FLIGHTS FROM NEW YORK, 2013 (nycflights13)")
    fl = load_flights()
    y = fl["ontime"]
    print(f"  {fl['n_all']:,} scheduled departures; {len(y):,} with an arrival delay recorded"
          f" ({fl['n_all'] - len(y):,} cancelled or diverted, left out); on time (arriving less than 15 minutes late): {pct(y.mean())}")
    for var in ("origin", "hour_band"):
        c = counts_by(y, fl[var])
        order = [k for k in (("EWR",), ("JFK",), ("LGA",))] if var == "origin" else [(n,) for _, _, n in HOUR_BANDS]
        print(f"  on-time rate by {var}: " + "; ".join(f"{k[0]} {pct(c[k][0] / c[k][1])} of {c[k][1]:,}" for k in order))
    print("  the nine largest carriers: flights, on-time rate, share of flights from EWR / JFK / LGA, share departing at 15:00 or later")
    for cr in BIG:
        m = fl["carrier"] == cr
        shares = " / ".join(f"{np.mean(fl['origin'][m] == o):.3f}" for o in ("EWR", "JFK", "LGA"))
        print(f"    {cr:<3} {m.sum():>6,} flights; on time {pct(y[m].mean())}; origin shares {shares}; after 15:00 {np.mean(fl['hour'][m] >= 15):.3f}")
    cells = (fl["carrier"], fl["origin"], fl["hour_band"])
    year = counts_by(y, *cells)
    w_year = {k: n / len(y) for k, (_, n) in year.items()}
    monthly = []
    for m in range(1, 13):
        mm = fl["month"] == m
        cm = counts_by(y[mm], *(c[mm] for c in cells))
        common = [k for k in cm if k in w_year]
        wsum = sum(w_year[k] for k in common)
        adj = sum(w_year[k] * cm[k][0] / cm[k][1] for k in common) / wsum
        monthly.append((m, y[mm].mean(), adj, wsum, cm))
    print("  month: on-time rate; standardised to the year's mix of carrier x origin x departure-hour band")
    for m, raw, adj, wsum, _ in monthly:
        print(f"    {datetime.date(2013, m, 1).strftime('%b')}: {pct(raw)}; {pct(adj)} (difference {pts(adj - raw)};"
              f" cells covering {wsum:.3f} of the year's flights)")
    diffs = np.array([adj - raw for _, raw, adj, _, _ in monthly])
    print(f"    largest absolute adjustment {pts(np.max(np.abs(diffs)))} points; range of the raw monthly rate"
          f" {pct(min(r for _, r, _, _, _ in monthly))} to {pct(max(r for _, r, _, _, _ in monthly))}")
    print("  month-to-month changes split into mix and within (Kitagawa, cells present in both months):")
    mix_abs, within_abs = 0.0, 0.0
    for (m1, r1, _, _, c1), (m2, r2, _, _, c2) in zip(monthly[:-1], monthly[1:]):
        common = [k for k in c1 if k in c2]
        n1 = sum(c1[k][1] for k in common); n2 = sum(c2[k][1] for k in common)
        p1 = np.array([c1[k][0] / c1[k][1] for k in common]); p2 = np.array([c2[k][0] / c2[k][1] for k in common])
        w1 = np.array([c1[k][1] / n1 for k in common]); w2 = np.array([c2[k][1] / n2 for k in common])
        mix, within = kitagawa(p2, w2, p1, w1)
        mix_abs += abs(mix); within_abs += abs(within)
        print(f"    {datetime.date(2013, m1, 1).strftime('%b')} to {datetime.date(2013, m2, 1).strftime('%b')}:"
              f" change {pts(r2 - r1)} points = mix {pts(mix)} + within {pts(within)}"
              f" (+ {pts((r2 - r1) - mix - within)} from cells not in both months)")
    print(f"    sum of absolute mix parts {pts(mix_abs)} points; sum of absolute within parts {pts(within_abs)} points")
    print()
    print("  Carrier comparisons among the nine largest carriers: difference in on-time rate, split into the mix")
    print("  of origin airport x departure-hour band and the within-cell part; mix attributed by Shapley values:")
    flips = []
    rows = [carrier_gap(fl, a, b) for a, b in itertools.combinations(BIG, 2)]
    for r in rows:
        if np.sign(r["within"]) != np.sign(r["total"]):
            flips.append(f"{r['a']}-{r['b']}")
    print(f"    {len(rows)} pairs; within-cell part of the opposite sign to the total in {len(flips)}: {', '.join(flips)}")
    for var, levels in (("origin", ("EWR", "JFK", "LGA")), ("hour_band", [n for _, _, n in HOUR_BANDS])):
        parts_ = []
        for lv in levels:
            mu, m9 = (fl["carrier"] == "UA") & (fl[var] == lv), (fl["carrier"] == "9E") & (fl[var] == lv)
            parts_.append(f"{lv} UA {pct(y[mu].mean(), 1)} of {mu.sum():,}, 9E {pct(y[m9].mean(), 1)} of {m9.sum():,}")
        print(f"    UA and 9E by {var}: " + "; ".join(parts_))
    for r in rows:
        if (r["a"], r["b"]) in SHOW:
            print(f"    {r['a']} minus {r['b']}: total {pts(r['total'])}; mix {pts(r['origin'] + r['hour_band'])}"
                  f" (origin {pts(r['origin'])}, hour {pts(r['hour_band'])}); within {pts(r['within'])};"
                  f" remainder from cells only one carrier flies {pts(r['total'] - r['mix'] - r['within'])};"
                  f" within with month added {pts(r['within3'])}; cells cover {r['cover']:.3f} of the flights;"
                  f" standardised difference {pts(r['std'])} (SE {pts(r['std_se'])})")

    header("3. FAIRNESS METRICS IN AGGREGATE AND WITHIN STRATA (ProPublica's COMPAS two-year file)")
    cp = load_compas()
    print(f"  {cp['n_all']:,} defendants in the file; {len(cp['race']):,} after ProPublica's filters")
    two = np.isin(cp["race"], ["African-American", "Caucasian"])
    for k in ("race", "sex", "age_cat", "decile", "priors", "charge", "recid"):
        cp[k] = cp[k][two]
    black = cp["race"] == "African-American"
    high = (cp["decile"] >= 5).astype(int)
    recid = cp["recid"]
    print(f"  Black (ProPublica's 'African-American') {black.sum():,}; white ('Caucasian') {(~black).sum():,};"
          f" 'higher risk' means a decile score of 5 or more (ProPublica's medium or high)")
    mets = {}
    for label, m in (("Black", black), ("white", ~black)):
        mets[label] = fairness_metrics(recid[m], high[m])
    for k in mets["Black"]:
        ab, aw = mets["Black"][k], mets["white"][k]
        cib = stats.binomtest(int(ab[0]), int(ab[1])).proportion_ci()
        ciw = stats.binomtest(int(aw[0]), int(aw[1])).proportion_ci()
        d, se = difference(ab[0], ab[1], aw[0], aw[1])
        print(f"    {k:<15} Black {pct(ab[0] / ab[1], 1)} [{pct(cib.low, 1)}, {pct(cib.high, 1)}] of {ab[1]:,};"
              f" white {pct(aw[0] / aw[1], 1)} [{pct(ciw.low, 1)}, {pct(ciw.high, 1)}] of {aw[1]:,}; difference {pts(d, 1)} points")
    print("  check of Chouldechova's identity FPR = p / (1 - p) x (1 - PPV) / PPV x (1 - FNR):")
    for label in ("Black", "white"):
        mm = mets[label]
        p = mm["base rate"][0] / mm["base rate"][1]
        ppv = mm["PPV"][0] / mm["PPV"][1]
        fnr = mm["FNR"][0] / mm["FNR"][1]
        fpr = mm["FPR"][0] / mm["FPR"][1]
        print(f"    {label}: FPR {fpr:.4f}; right-hand side {p / (1 - p) * (1 - ppv) / ppv * (1 - fnr):.4f}")
    print("  two-year recidivism by decile score (calibration):")
    sa, na, sb, nb = [], [], [], []
    for dcl in range(1, 11):
        mb_, mw_ = black & (cp["decile"] == dcl), ~black & (cp["decile"] == dcl)
        sa.append(recid[mb_].sum()); na.append(mb_.sum()); sb.append(recid[mw_].sum()); nb.append(mw_.sum())
        print(f"    decile {dcl:>2}: Black {pct(recid[mb_].mean(), 1)} of {mb_.sum():,}; white {pct(recid[mw_].mean(), 1)} of {mw_.sum():,}")
    d_cal, se_cal = stratified_difference(*(np.array(v, float) for v in (sa, na, sb, nb)))
    print(f"    recidivism difference within deciles, standardised to the pooled decile sizes: {pts(d_cal)} points"
          f" [{pts(d_cal - 1.96 * se_cal)}, {pts(d_cal + 1.96 * se_cal)}], z = {d_cal / se_cal:.2f}")
    coarse = priors_band(cp["priors"], [1, 4])
    fine = priors_band(cp["priors"], [1, 2, 3, 4, 6, 10])
    strat = {
        "prior-count band (0, 1-3, 4+)": coarse,
        "age category": cp["age_cat"],
        "prior band x age": np.char.add(np.char.add(coarse, " | "), cp["age_cat"]),
        "finer prior band x age": np.char.add(np.char.add(fine, " | "), cp["age_cat"]),
        "finer prior band x age x sex x charge degree": np.char.add(np.char.add(np.char.add(
            np.char.add(fine, " | "), cp["age_cat"]), np.char.add(" | ", cp["sex"])), np.char.add(" | ", cp["charge"])),
    }
    print("  Black minus white, aggregate and standardised within strata (strata with at least 10 in each group"
          " in the metric's denominator), points:")
    strat_res = {}
    for k in ("base rate", "selection rate", "FPR", "FNR", "PPV"):
        d0, s0 = difference(*mets["Black"][k], *mets["white"][k])
        line = [f"aggregate {pts(d0, 1)} [{pts(d0 - 1.96 * s0, 1)}, {pts(d0 + 1.96 * s0, 1)}]"]
        strat_res[(k, "aggregate")] = (d0, s0)
        for name, s in strat.items():
            d, se, used, cover = stratified_metric(k, recid, high, black, s)
            strat_res[(k, name)] = (d, se)
            line.append(f"{name} {pts(d, 1)} [{pts(d - 1.96 * se, 1)}, {pts(d + 1.96 * se, 1)}] ({used} strata, {cover:.3f} of the denominator)")
        print(f"    {k}: " + "; ".join(line))
    print("  mean prior count within the coarse bands:")
    for band in ("0", "1-3", "4+"):
        mb_, mw_ = black & (coarse == band), ~black & (coarse == band)
        print(f"    band {band:<4} Black {cp['priors'][mb_].mean():.2f} (n = {mb_.sum():,}); white {cp['priors'][mw_].mean():.2f} (n = {mw_.sum():,})")
    print(f"    share with 10 or more priors among those with 4 or more: Black"
          f" {np.mean(cp['priors'][black & (coarse == '4+')] >= 10):.3f}, white {np.mean(cp['priors'][~black & (coarse == '4+')] >= 10):.3f}")

    header("4. SCANNING FOR REVERSALS")
    found = flights_scan(fl)
    flips = [f for f in found if f["flip"]]
    full = [f for f in found if f["full"]]
    print(f"  flights: entities carrier (16), origin (3), month (12); {len(found):,} comparisons of a pair of entity"
          f" levels within a stratifier (strata shared by both, at least 30 flights each)")
    for ent in ("carrier", "origin", "month_name"):
        fe = [f for f in found if f["entity"] == ent]
        print(f"    {ent:<10} {len(fe):>4} comparisons; aggregate and standardised differences of opposite sign {sum(f['flip'] for f in fe)};"
              f" full reversals {sum(f['full'] for f in fe)}")
    print(f"    total: {len(flips)} with opposite signs ({len(flips) / len(found):.4f}); {len(full)} full reversals")
    both = [f for f in flips if abs(f["agg"]) > 1.96 * f["agg_se"] and abs(f["std"]) > 1.96 * f["std_se"]]
    print(f"    with both differences beyond 1.96 standard errors: {len(both)}")
    rng_post = np.random.default_rng(SEED + 4)
    for f in sorted(flips, key=lambda f: -f["n"]):
        f["post"] = posterior_flip(*f["counts"], rng_post)
    for f in sorted(flips, key=lambda f: -f["post"]):
        print(f"      {f['entity']} {f['a']} minus {f['b']} by {f['stratifier']} ({f['strata']} strata, {int(f['counts'][1].sum()):,}"
              f" and {int(f['counts'][3].sum()):,} flights):"
              f" aggregate {pts(f['agg'])} (z {f['agg'] / f['agg_se']:+.2f}), standardised {pts(f['std'])}"
              f" (z {f['std'] / f['std_se']:+.2f}); full {f['full']}; posterior P(opposite signs) {f['post']:.3f}")
    print(f"    posterior probability of opposite signs at least 0.95: {sum(f['post'] >= 0.95 for f in flips)}; at least 0.99:"
          f" {sum(f['post'] >= 0.99 for f in flips)}")
    print("  split-sample check: scan flights on odd-numbered days, confirm on even-numbered days")
    halves = {}
    for label, mask in (("odd", fl["day"] % 2 == 1), ("even", fl["day"] % 2 == 0)):
        halves[label] = {(f["entity"], f["stratifier"], f["a"], f["b"]): f for f in flights_scan(fl, mask)}
    disc = {k: f for k, f in halves["odd"].items() if f["flip"]}
    rep = [k for k, f in disc.items() if k in halves["even"] and halves["even"][k]["flip"]
           and np.sign(halves["even"][k]["agg"]) == np.sign(f["agg"])]
    disc_ev = {k: f for k, f in disc.items() if abs(f["agg"]) > 1.96 * f["agg_se"] and abs(f["std"]) > 1.96 * f["std_se"]}
    rep_ev = [k for k in disc_ev if k in rep]
    print(f"    odd days: {len(halves['odd']):,} comparisons, {len(disc)} with opposite signs; replicated on even days"
          f" (same signs again): {len(rep)}")
    print(f"    of the {len(disc_ev)} odd-day flips with both differences beyond 1.96 standard errors, replicated: {len(rep_ev)}")
    for k in sorted(rep):
        f, g = disc[k], halves["even"][k]
        print(f"      {k[0]} {k[2]} minus {k[3]} by {k[1]}: aggregate / standardised, odd days {pts(f['agg'])} / {pts(f['std'])};"
              f" even days {pts(g['agg'])} / {pts(g['std'])}{'; both beyond 1.96 SE on odd days' if k in disc_ev else ''}")
    for k in sorted(disc_ev):
        if k not in rep:
            print(f"      not replicated despite the odd-day evidence: {k}")
    print()
    print("  Hillstrom's experiment: arms as entities, pre-treatment covariates as stratifiers")
    strata_h = {k: np.asarray(v).astype(str) for k, v in covs.items()}
    for outcome in ("visit", "conversion"):
        fh = scan(h[outcome], h["segment"], strata_h)
        print(f"    {outcome}: {len(fh)} comparisons; opposite signs {sum(f['flip'] for f in fh)}; full reversals"
              f" {sum(f['full'] for f in fh)}; largest |aggregate - standardised| {pts(max(abs(f['agg'] - f['std']) for f in fh), 3)} points")


if __name__ == "__main__":
    main()
