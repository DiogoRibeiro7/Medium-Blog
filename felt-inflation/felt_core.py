"""Methods for the felt-inflation article: rebuilding Eurostat's frequent out-of-pocket purchases index for
every EU country, and weighing inflation the way survey answers weigh it.

Every sum that reaches a printed number is either a fixed-order Python sum, math.fsum or np.bincount (a
sequential loop), so results do not depend on the processor's vector kernels.
"""

import csv
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

EU27 = "AT BE BG CY CZ DE DK EE EL ES FI FR HR HU IE IT LT LU LV MT NL PL PT RO SE SI SK".split()
EURO_2025 = "AT BE CY DE EE EL ES FI FR HR IE IT LT LU LV MT NL PT SI SK".split()
EURO_2002 = "AT BE DE EL ES FI FR IE IT LU NL PT".split()

# Frequent out-of-pocket purchases (FROOPP), ECOICOP ver.1 classes as published by Eurostat: food and
# non-alcoholic beverages, alcohol and tobacco, 03.1.4, 05.6.1, 05.6.2, 06.1.1, 07.2.2, 07.2.4, 07.3.1,
# 07.3.2, 07.3.5, 08.1, 09.1.4, 09.3.4/5, 09.4.1, 09.4.2, 09.5.1, 09.5.2, 09.5.3/4, 11.1.1, 11.1.2, 12.1.1,
# 12.1.2/3. The food and drink divisions are taken at group level, which covers the same classes.
FRO = [
    "CP011",
    "CP012",
    "CP021",
    "CP022",
    "CP0314",
    "CP0561",
    "CP0562",
    "CP0611",
    "CP0722",
    "CP0724",
    "CP0731",
    "CP0732",
    "CP0735",
    "CP081",
    "CP0914",
    "CP0934",
    "CP0935",
    "CP0941",
    "CP0942",
    "CP0951",
    "CP0952",
    "CP0953",
    "CP0954",
    "CP1111",
    "CP1112",
    "CP1211",
    "CP1212",
    "CP1213",
]
# Classes that some countries (and every country before 2017) publish only as a pair.
COMB = {"CP0934_0935": ["CP0934", "CP0935"], "CP0953_0954": ["CP0953", "CP0954"], "CP1212_1213": ["CP1212", "CP1213"]}
GROCERIES = ["CP011", "CP012", "CP021", "CP022"]
FUEL = ["CP0722"]
ENERGY = ["CP045"]
# Rent, water and refuse, phone and internet services, insurance. Before 2017 many countries publish phone
# services only together with phones (CP082_083), which is then used instead.
BILLS = ["CP041", "CP044", "CP0830", "CP125"]
PANEL_CODES = ["CP00", "FROOPP", "TOT_X_FROOPP"] + FRO + list(COMB) + ENERGY + BILLS + ["CP082_083"]
COMPONENTS = ["groceries", "fuel", "other_frequent", "energy", "bills", "rest"]

# Items shown one by one for the euro area, with the basket each belongs to.
SHOWCASE = {
    "CP0111": ("Bread and cereals", "frequent"),
    "CP0112": ("Meat", "frequent"),
    "CP0114": ("Milk, cheese and eggs", "frequent"),
    "CP0115": ("Oils and fats", "frequent"),
    "CP0121": ("Coffee, tea and cocoa", "frequent"),
    "CP022": ("Tobacco", "frequent"),
    "CP0722": ("Fuel for cars", "frequent"),
    "CP1111": ("Restaurants and cafés", "frequent"),
    "CP1211": ("Hairdressing", "frequent"),
    "CP0952": ("Newspapers and magazines", "frequent"),
    "CP041": ("Rent", "bill"),
    "CP0451": ("Electricity", "bill"),
    "CP0452": ("Gas", "bill"),
    "CP08": ("Post, phone and internet", "bill"),
    "CP125": ("Insurance", "bill"),
    "CP031": ("Clothing", "rare"),
    "CP0511": ("Furniture", "rare"),
    "CP0531_0532": ("Household appliances", "rare"),
    "CP0711": ("New cars", "rare"),
    "CP032": ("Shoes", "rare"),
    "CP0911": ("TVs and audio equipment", "rare"),
    "CP0913": ("Computers", "rare"),
    "CP096": ("Package holidays", "rare"),
}
SHOWCASE_EXTRA = [c for c in SHOWCASE if c not in PANEL_CODES]
V2_CODES = ["TOTAL", "FROOPP", "TOT_X_FROOPP", "FOOD", "NRG", "FUEL", "ELC_GAS", "SERV", "IGD_NNRG"]


# ---------------------------------------------------------------------------------------------- data files
def jsonstat_rows(d):
    """Rows (geo, item, period, value) from a Eurostat JSON-stat 2.0 response.

    The item is the classification code (coicop or coicop18) or, for the survey, the indicator."""
    dims, sizes = d["id"], d["size"]
    keys = [list(d["dimension"][k]["category"]["index"]) for k in dims]
    item_dim = next(k for k in ("coicop", "coicop18", "indic") if k in dims)
    rows = []
    for flat, value in d["value"].items():
        i, pos = int(flat), []
        for s in reversed(sizes):
            pos.append(i % s)
            i //= s
        pos.reverse()
        label = {k: keys[j][p] for j, (k, p) in enumerate(zip(dims, pos, strict=True))}
        rows.append((label["geo"], label[item_dim], label["time"], float(value)))
    return rows


def wide_csv(rows):
    """A deterministic wide CSV: one line per (geo, item), one column per period, values as repr(float)."""
    table, periods = {}, set()
    for geo, item, period, value in rows:
        key = (geo, item)
        if period in table.get(key, {}) and table[key][period] != value:
            raise ValueError(f"conflicting values for {geo} {item} {period}")
        table.setdefault(key, {})[period] = value
        periods.add(period)
    periods = sorted(periods)
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["geo", "item"] + periods)
    for key in sorted(table):
        w.writerow(list(key) + [repr(table[key][p]) if p in table[key] else "" for p in periods])
    return out.getvalue()


def cells(text):
    if not text:
        return {}
    r = list(csv.reader(io.StringIO(text)))
    head = r[0][2:]
    return {(row[0], row[1], p): v for row in r[1:] for p, v in zip(head, row[2:], strict=True) if v}


def read_wide(name):
    """{geo: DataFrame(index=period, columns=item)} from a snapshot file in data/."""
    df = pd.read_csv(DATA / f"{name}.csv", dtype={"geo": str, "item": str})
    out = {}
    for geo, g in df.groupby("geo", sort=True):
        t = g.drop(columns="geo").set_index("item").T
        t.index = t.index.astype(str)
        t.columns.name = None
        out[geo] = t.astype(float)
    return out


# ----------------------------------------------------------------------------------------- chain-linking
def _ok(P, W, code, year):
    y, dec = str(year), f"{year - 1}-12"
    return (
        code in W.columns
        and code in P.columns
        and y in W.index
        and dec in P.index
        and pd.notna(W.at[y, code])
        and W.at[y, code] > 0
        and pd.notna(P.at[dec, code])
    )


def frequent_items(year, P, W):
    """{code: weight} for the frequent basket in one year. A pair of classes is used separately when both
    are published, and as the combined index otherwise."""
    paired = {c for parts in COMB.values() for c in parts}
    out = {c: W.at[str(year), c] for c in FRO if c not in paired and _ok(P, W, c, year)}
    for comb, parts in COMB.items():
        if all(_ok(P, W, p, year) for p in parts):
            out.update({p: W.at[str(year), p] for p in parts})
        elif _ok(P, W, comb, year):
            out[comb] = W.at[str(year), comb]
    return out


def fixed_items(codes):
    def items(year, P, W):
        return {c: W.at[str(year), c] for c in codes if _ok(P, W, c, year)}

    return items


def bill_items(year, P, W):
    out = fixed_items(["CP041", "CP044", "CP125"])(year, P, W)
    if _ok(P, W, "CP0830", year):
        out["CP0830"] = W.at[str(year), "CP0830"]
    elif _ok(P, W, "CP082_083", year):
        out["CP082_083"] = W.at[str(year), "CP082_083"]
    return out


def chain(P, W, items, y0, y1):
    """Annually chain-linked index with December overlap, as in the HICP.

    Within year t, the index relative to December t-1 is the weighted mean of item relatives
    I_i(t, m) / I_i(t-1, Dec), with the year-t weights; the chain multiplies these at each December.
    Returns DataFrame(index=month, columns=[index, weight]) with December y0-1 = 100."""
    rows, level = [], 100.0
    for year in range(y0, y1 + 1):
        dec = f"{year - 1}-12"
        it = items(year, P, W)
        for m in range(1, 13):
            mo = f"{year}-{m:02d}"
            if mo not in P.index:
                continue
            num, den = [], []
            for code in sorted(it):
                v = P.at[mo, code]
                if pd.notna(v):
                    num.append(it[code] * v / P.at[dec, code])
                    den.append(it[code])
            rows.append((mo, level * math.fsum(num) / math.fsum(den) if den else np.nan, math.fsum(den)))
        level = rows[-1][1] if rows and rows[-1][0] == f"{year}-12" else np.nan
    return pd.DataFrame(rows, columns=["month", "index", "weight"]).set_index("month")


def complement(total, parts):
    """Chain-linked index of everything in `total` not in `parts`, from link relatives and weights.

    total and each part are DataFrames from chain(); December y0-1 = 100 in all."""
    rows, level = [], 100.0
    months = list(total.index)
    for mo in months:
        year = int(mo[:4])
        dec = f"{year - 1}-12"

        def rel(df, mo=mo, dec=dec):
            return df.at[mo, "index"] / (df.at[dec, "index"] if dec in df.index else 100.0)

        w_rest = total.at[mo, "weight"] - math.fsum(p.at[mo, "weight"] for p in parts)
        r = (total.at[mo, "weight"] * rel(total) - math.fsum(p.at[mo, "weight"] * rel(p) for p in parts)) / w_rest
        rows.append((mo, level * r, w_rest))
        if mo.endswith("-12"):
            level = level * r
    return pd.DataFrame(rows, columns=["month", "index", "weight"]).set_index("month")


def first_year(P, W):
    """First year with an all-items weight, a December index before it, and most frequent classes."""
    for year in range(1997, 2026):
        if _ok(P, W, "CP00", year) and len(frequent_items(year, P, W)) >= 20:
            return year
    raise ValueError("no usable year")


def rebuild(P, W, y0=None, y1=2025):
    """Frequent (F), all items (T) and everything else (X) for one country, December y0-1 = 100."""
    y0 = y0 or first_year(P, W)
    total = chain(P, W, fixed_items(["CP00"]), y0, y1)
    freq = chain(P, W, frequent_items, y0, y1)
    rest = complement(total, [freq])
    return pd.DataFrame(
        {"F": freq["index"], "T": total["index"], "X": rest["index"], "share_F": freq["weight"] / total["weight"]}
    )


def split(P, W, y0=None, y1=2025):
    """Six baskets that add up to all items: groceries, fuel, other frequent purchases, household energy,
    other bills (rent, water and refuse, phone and internet, insurance), and the rest."""
    y0 = y0 or first_year(P, W)
    total = chain(P, W, fixed_items(["CP00"]), y0, y1)

    def other_frequent(year, P, W):
        return {c: w for c, w in frequent_items(year, P, W).items() if c not in GROCERIES + FUEL}

    parts = {
        "groceries": chain(P, W, fixed_items(GROCERIES), y0, y1),
        "fuel": chain(P, W, fixed_items(FUEL), y0, y1),
        "other_frequent": chain(P, W, other_frequent, y0, y1),
        "energy": chain(P, W, fixed_items(ENERGY), y0, y1),
        "bills": chain(P, W, bill_items, y0, y1),
    }
    parts["rest"] = complement(total, list(parts.values()))
    out = pd.DataFrame({k: v["index"] for k, v in parts.items()})
    for k, v in parts.items():
        out["w_" + k] = v["weight"] / total["weight"]
    return out


def mix(rF, rX, w):
    """Chain a two-basket index from monthly levels of F and X with weight w (scalar or {year: w}) on F."""
    rows, level = [], 100.0
    for mo in rF.index:
        year = int(mo[:4])
        dec = f"{year - 1}-12"
        if dec not in rF.index:
            continue
        ww = w[year] if isinstance(w, dict) else w
        r = ww * rF[mo] / rF[dec] + (1 - ww) * rX[mo] / rX[dec]
        rows.append((mo, level * r))
        if mo.endswith("-12"):
            level *= r
    return pd.Series(dict(rows))


def rate(s, months=12):
    """Change over `months`, annualised, in per cent."""
    return ((s / s.shift(months)) ** (12 / months) - 1) * 100


def cumulative(s, a, b):
    return (s[b] / s[a] - 1) * 100


def annualised(s, a, b):
    years = (int(b[:4]) - int(a[:4])) + (int(b[5:]) - int(a[5:])) / 12
    return ((s[b] / s[a]) ** (1 / years) - 1) * 100


# -------------------------------------------------------------------------------------------- regression
def _codes(labels):
    uniq = sorted(set(labels))
    pos = {u: i for i, u in enumerate(uniq)}
    return np.array([pos[x] for x in labels]), len(uniq)


def _demean(M, g, ng):
    counts = np.bincount(g, minlength=ng).astype(float)
    means = np.column_stack([np.bincount(g, weights=M[:, j], minlength=ng) / counts for j in range(M.shape[1])])
    return M - means[g]


def _solve(A, b):
    """Gauss-Jordan elimination with partial pivoting, in Python floats."""
    n = len(b)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c], strict=True)]
    return np.array([M[i][n] / M[i][i] for i in range(n)])


def within(df, y, xs, two_way=False):
    """Fixed-effects (within) transform: by country, and also by month when two_way."""
    M = df[[y] + xs].to_numpy(float)
    g, ng = _codes(df["geo"].tolist())
    if not two_way:
        return _demean(M, g, ng)
    t, nt = _codes(df["month"].tolist())
    for _ in range(1000):
        before = M
        M = _demean(_demean(M, g, ng), t, nt)
        if np.max(np.abs(M - before)) < 1e-11:
            break
    return M


def fe_fit(df, y, xs, two_way=False):
    """Slopes of y on xs with country (and month) fixed effects; returns (beta, within R^2)."""
    M = within(df, y, xs, two_way)
    Y, X = M[:, 0], M[:, 1:]
    k = X.shape[1]
    A = [[math.fsum(X[:, i] * X[:, j]) for j in range(k)] for i in range(k)]
    b = [math.fsum(X[:, i] * Y) for i in range(k)]
    beta = _solve(A, b)
    e = Y - sum(beta[i] * X[:, i] for i in range(k))
    return beta, 1 - math.fsum(e * e) / math.fsum(Y * Y)


def implied_weights(beta):
    """Weights the slopes put on each basket: beta_k / sum(beta)."""
    return beta / math.fsum(beta)


def bootstrap(df, stat, n=999, seed=20261119):
    """Resample whole countries with replacement; returns an array of stat(resampled panel)."""
    rng = np.random.default_rng(seed)
    geos = sorted(df["geo"].unique())
    by = {g: d for g, d in df.groupby("geo", sort=True)}
    out = []
    for _ in range(n):
        pick = rng.choice(len(geos), len(geos), replace=True)
        q = pd.concat([by[geos[j]].assign(geo=f"{geos[j]}#{i}") for i, j in enumerate(pick)], ignore_index=True)
        out.append(stat(q))
    return np.array(out)


def interval(draws, level=95):
    lo = (100 - level) / 2
    return np.percentile(draws, [lo, 100 - lo], axis=0)


def fe_predict(train, test, y, xs):
    """Fit country fixed effects and slopes on train; predict y on test (countries must appear in train)."""
    beta, _ = fe_fit(train, y, xs)
    means = train.groupby("geo")[[y] + xs].mean()
    a = means[y] - sum(beta[i] * means[x] for i, x in enumerate(xs))
    return test["geo"].map(a).to_numpy() + sum(beta[i] * test[x].to_numpy() for i, x in enumerate(xs)), beta


def ols(df, y, xs):
    """Ordinary least squares with an intercept; returns coefficients [intercept, *slopes]."""
    X = np.column_stack([np.ones(len(df))] + [df[x].to_numpy(float) for x in xs])
    Y = df[y].to_numpy(float)
    k = X.shape[1]
    A = [[math.fsum(X[:, i] * X[:, j]) for j in range(k)] for i in range(k)]
    b = [math.fsum(X[:, i] * Y) for i in range(k)]
    return _solve(A, b)


def rmse(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return math.sqrt(math.fsum(d * d) / len(d))


# ------------------------------------------------------------------------------------------------- panel
def country_panel(P, W, balance, t0="2002-01", t1="2025-12"):
    """One row per country and month: survey balance, inflation of each basket over 12, 24, 36 and 48
    months, and basket weights. Rows without a balance or a 12-month rate are dropped."""
    frames = []
    for geo in EU27:
        r = rebuild(P[geo], W[geo])
        s = split(P[geo], W[geo])
        d = pd.DataFrame(index=r.index)
        for h in (12, 24, 36, 48):
            for c in ("F", "X", "T"):
                d[f"{c}{h}"] = rate(r[c], h)
        for c in COMPONENTS:
            d[c] = rate(s[c])
            d["w_" + c] = s["w_" + c]
        d["share_F"] = r["share_F"]
        d["B"] = balance[geo]["BS-PT-LY"].reindex(d.index)
        d["geo"] = geo
        frames.append(d.loc[t0:t1])
    p = pd.concat(frames).rename_axis("month").reset_index()
    p = p.dropna(subset=["B", "F12", "X12", "T12"] + COMPONENTS).reset_index(drop=True)
    p["year"] = p["month"].str[:4].astype(int)
    w_freq = p[["w_groceries", "w_fuel", "w_other_frequent"]].sum(axis=1)
    w_bill = p[["w_energy", "w_bills"]].sum(axis=1)
    p["frequent"] = (p.groceries * p.w_groceries + p.fuel * p.w_fuel + p.other_frequent * p.w_other_frequent) / w_freq
    p["billed"] = (p.energy * p.w_energy + p.bills * p.w_bills) / w_bill
    p["w_frequent"], p["w_billed"] = w_freq, w_bill
    return p


def mean(x):
    x = np.asarray(x, float)
    return math.fsum(x) / len(x)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - mean(a), b - mean(b)
    return math.fsum(a * b) / math.sqrt(math.fsum(a * a) * math.fsum(b * b))
