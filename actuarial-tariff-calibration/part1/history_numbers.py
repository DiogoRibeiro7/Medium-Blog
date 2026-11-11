"""Part 1 of the actuarial-tariff-calibration series: how motor tariffs were judged, from minimum bias to the Gini.

Every number quoted in the article is printed by this script. Run:

    python history_numbers.py > output.txt

Two data sets, both historical. The Canadian private-passenger liability experience of policy years 1957
and 1958 that Bailey and Simon (1960) used to set out their four criteria for a set of rates: five classes
by four merit-rating groups, 4.15 million car-years, transcribed below from Tables A and D of their paper.
And the Swedish third-party motor experience of 1977 compiled by the Swedish Committee on the Analysis of
Risk Premium in Motor Insurance (2,182 tariff cells, 2.38 million policy-years), downloaded once from a
pinned commit of CASdatasets.

Sections
1. Bailey and Simon (1960): the data, and the effectiveness of class and merit rating (their Section A).
2. Their four methods (customary, minimum chi-square multiplicative, additive, and 3xy - 2), recomputed,
   with the tests of balance, average departure and chi-square set against their Table E.
3. Bailey (1963): the minimum-bias (balance) iteration, its first calculations against Berquist's table,
   and its fixed point, which is the Poisson maximum-likelihood fit (Jung 1968).
4. The same Canadian tariffs scored by an exposure-weighted Gini index: what the ranking can and cannot see.
5. Sweden 1977: the marginal-totals tariff, which is the Poisson GLM; nested tariffs by Gini and deviance;
   the customary one-way tariff.
6. Sweden 1977: tariffs with the same ranking and a different price. Power distortions and level errors
   leave the Gini unchanged and move deviance, balance by rating factor and balance by premium level.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import swedish_1977  # noqa: E402

np.set_printoptions(linewidth=120)


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def row(label: str, values, fmt: str = "{:8.3f}") -> str:
    return f"{label:<26}" + "".join(fmt.format(v) for v in values)


# ---------------------------------------------------------------------------------------------------
# Bailey and Simon (1960), "Two Studies in Automobile Insurance Ratemaking", PCAS XLVII 1-19
# (also ASTIN Bulletin 1(4), 192-217). Canada excluding Saskatchewan; private passenger automobile
# liability, non-farmers; policy years 1957 and 1958 as of 30 June 1959. Rows in the paper's order.
# ---------------------------------------------------------------------------------------------------
CLASSES = ("1", "5", "3", "2", "4")
MERITS = ("A", "X", "Y", "B")

# Table A: earned car-years, in thousands.
CAR_YEARS_THOUSANDS = np.array([
    [2758, 131, 164, 274],
    [64, 4, 5, 9],
    [247, 16, 20, 38],
    [131, 7, 10, 22],
    [157, 18, 21, 57],
], dtype=float)
N = 1000.0 * CAR_YEARS_THOUSANDS

# Table D, "actual": loss ratio at 1B rates divided by the overall loss ratio of 0.505.
R = np.array([
    [0.786, 1.016, 1.115, 1.358],
    [1.071, 1.079, 1.410, 1.642],
    [1.212, 1.285, 1.450, 1.885],
    [1.269, 1.747, 1.519, 1.784],
    [2.050, 2.192, 2.412, 2.853],
])
LOSS = N * R  # relative losses: car-years times relative loss ratio

# Tables 1, 2 and B: loss ratios at 1B rates.
LR_OVERALL = 0.505
LR_CLASS_TOTAL = {"1": 0.436, "5": 0.583, "3": 0.663, "2": 0.693, "4": 1.146}
LR_MERIT_TOTAL = {"A": 0.452, "X": 0.593, "Y": 0.645, "B": 0.813}
LR_1A = 0.397

# Table C: relativities as printed (Method 4 partly illegible in the scan; read as below).
PUBLISHED = {
    "Method 1": (np.array([0.863, 1.154, 1.313, 1.372, 2.269]), np.array([0.895, 1.174, 1.277, 1.610])),
    "Method 2": (np.array([0.881, 1.161, 1.309, 1.367, 2.125]), np.array([0.906, 1.113, 1.215, 1.462])),
    "Method 3": (np.array([0.869, 1.145, 1.291, 1.352, 2.172]), np.array([-0.083, 0.135, 0.237, 0.512])),
    "Method 4": (np.array([0.958, 1.049, 1.099, 1.118, 1.384]), np.array([0.971, 1.040, 1.076, 1.167])),
}
# Table E as printed: total balance, average departure, chi-square (K = 1/200).
PUBLISHED_E = {
    "Method 1": (1.0103, 0.0401, 98),
    "Method 2": (1.0011, 0.0317, 34),
    "Method 3": (1.0006, 0.0098, 10),
    "Method 4": (0.9983, 0.0111, 8),
}
PUBLISHED_BALANCE_M1 = (np.array([0.9886, 1.0099, 1.0195, 1.0230, 1.1067]),
                        np.array([0.9806, 1.0589, 1.0536, 1.1122]))
K = 1.0 / 200.0  # Bailey and Simon's constant for the chi-square of the Canadian data

# Berquist's discussion of Bailey (1963), Table 1: minimum-bias calculations started from the customary y.
BERQUIST_ITERATIONS = [
    (np.array([0.872, 1.143, 1.288, 1.341, 2.050]), np.array([0.918, 1.129, 1.232, 1.486])),
    (np.array([0.868, 1.144, 1.290, 1.345, 2.089]), np.array([0.919, 1.128, 1.232, 1.481])),
    (np.array([0.868, 1.143, 1.290, 1.345, 2.090]), np.array([0.919, 1.127, 1.232, 1.481])),
]
# Berquist's Table 2: fitted relative loss ratios after the third calculation (2Y read as 1.657).
BERQUIST_FITTED = np.array([
    [0.798, 0.979, 1.069, 1.286],
    [1.050, 1.288, 1.408, 1.693],
    [1.186, 1.454, 1.589, 1.910],
    [1.236, 1.516, 1.657, 1.992],
    [1.921, 2.355, 2.575, 3.095],
])


def criteria(F: np.ndarray) -> dict:
    """Bailey and Simon's tests: balance (criterion 1), average departure (3), chi-square (4)."""
    fitted = N * F
    return {
        "class": fitted.sum(1) / LOSS.sum(1),
        "merit": fitted.sum(0) / LOSS.sum(0),
        "total": fitted.sum() / LOSS.sum(),
        "departure": (N * np.abs(R - F)).sum() / LOSS.sum(),
        "chi2": K * (N * (R - F) ** 2 / F).sum(),
    }


def chi2_objective(model):
    def f(p):
        F = model(p)
        if np.any(F <= 0):
            return 1e12
        return float((N * (R - F) ** 2 / F).sum())
    return f


def mult(p):
    return np.outer(np.exp(p[:5]), np.exp(np.r_[0.0, p[5:]]))


def add(p):
    return p[:5, None] + np.r_[0.0, p[5:]][None, :]


def mixed(p):
    return 3.0 * mult(p) - 2.0


def minimise(model, start) -> np.ndarray:
    obj = chi2_objective(model)
    p = np.asarray(start, dtype=float)
    for method in ("Nelder-Mead", "BFGS", "Nelder-Mead", "BFGS"):
        opts = {"xatol": 1e-12, "fatol": 1e-12, "maxiter": 100000, "maxfev": 100000} if method == "Nelder-Mead" \
            else {"gtol": 1e-10, "maxiter": 10000}
        p = optimize.minimize(obj, p, method=method, options=opts).x
    return p


def minimum_bias(y0: np.ndarray, iterations: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bailey's balance iteration: x_i = sum_j n r / sum_j n y_j, then y_j = sum_i n r / sum_i n x_i."""
    out, y = [], y0.copy()
    for _ in range(iterations):
        x = LOSS.sum(1) / (N * y[None, :]).sum(1)
        y = LOSS.sum(0) / (N * x[:, None]).sum(0)
        out.append((x, y))
    return out


def poisson_irls(y: np.ndarray, exposure: np.ndarray, X: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """Poisson (quasi-)likelihood with log link and offset log(exposure), by iteratively reweighted least squares."""
    b = np.zeros(X.shape[1])
    b[0] = np.log(y.sum() / exposure.sum())
    for _ in range(200):
        mu = exposure * np.exp(X @ b)
        z = X @ b + (y - mu) / mu
        b_new = np.linalg.solve(X.T @ (mu[:, None] * X), X.T @ (mu * z))
        done = np.max(np.abs(b_new - b)) < tol
        b = b_new
        if done:
            break
    return exposure * np.exp(X @ b)


def dummies(codes: list[np.ndarray]) -> np.ndarray:
    cols = [np.ones(len(codes[0]))]
    for c in codes:
        for level in np.unique(c)[1:]:
            cols.append((c == level).astype(float))
    return np.column_stack(cols)


def lorenz(score: np.ndarray, exposure: np.ndarray, loss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ordered Lorenz curve: cells sorted by score from lowest to highest risk.

    With a flat base premium this is the ordered Lorenz curve of Frees, Meyers and Cummings (2011): the
    horizontal axis is the cumulative share of exposure, the vertical axis the cumulative share of losses.
    Cells with equal scores form one block, joined by a straight segment, so the curve does not depend on
    the order in which tied cells happen to be stored.
    """
    order = np.argsort(score, kind="stable")
    s, w, l = score[order], exposure[order], loss[order]
    last = np.r_[s[1:] != s[:-1], True]
    cw = np.r_[0.0, np.cumsum(w)[last]] / w.sum()
    cl = np.r_[0.0, np.cumsum(l)[last]] / l.sum()
    return cw, cl


def gini(score: np.ndarray, exposure: np.ndarray, loss: np.ndarray) -> float:
    """Exposure-weighted Gini index: twice the area between the ordered Lorenz curve and the diagonal.

    Only the order of the scores enters: any strictly increasing transformation of the scores leaves it unchanged.
    """
    cw, cl = lorenz(score, exposure, loss)
    return float(1.0 - 2.0 * np.trapezoid(cl, cw))


def kaggle_gini(actual: np.ndarray, pred: np.ndarray) -> float:
    """The Gini of the Kaggle insurance competitions: rows sorted by prediction, largest first, each row
    counting once whatever its exposure; normalised by the Gini of a perfect ordering of the rows."""
    def raw(a, p):
        order = np.lexsort((np.arange(len(p)), -p))
        c = np.cumsum(a[order])
        return (c.sum() / c[-1] - (len(a) + 1) / 2.0) / len(a)
    return float(raw(actual, pred) / raw(actual, actual))


def discordant_pairs(a: np.ndarray, b: np.ndarray) -> int:
    a, b = a.ravel(), b.ravel()
    return sum(1 for i, j in combinations(range(a.size), 2) if np.sign(a[i] - a[j]) != np.sign(b[i] - b[j]))


def rescale_mult(x, y, x_ref):
    """Relativities are identified only up to x*c, y/c; choose c to match the published x in logs."""
    c = np.exp(np.mean(np.log(x_ref) - np.log(x)))
    return x * c, y / c


# ---------------------------------------------------------------------------------------------------
# The Canadian computations
# ---------------------------------------------------------------------------------------------------
def canadian() -> dict:
    res: dict = {}
    # Method 1, as printed and as recomputed from car-years
    x1p, y1p = PUBLISHED["Method 1"]
    res["F1"] = np.outer(x1p, y1p)
    res["x1_caryears"] = LOSS.sum(1) / N.sum(1)
    res["y1_caryears"] = LOSS.sum(0) / N.sum(0)
    # Method 2: minimum chi-square, multiplicative
    p2 = minimise(mult, np.r_[np.log(x1p), np.log(y1p[1:] / y1p[0])])
    res["F2"] = mult(p2)
    res["x2"], res["y2"] = rescale_mult(np.exp(p2[:5]), np.exp(np.r_[0.0, p2[5:]]), PUBLISHED["Method 2"][0])
    # Method 3: minimum chi-square, additive
    x3p, y3p = PUBLISHED["Method 3"]
    p3 = minimise(add, np.r_[x3p + y3p[0], y3p[1:] - y3p[0]])
    res["F3"] = add(p3)
    shift = np.mean(x3p - p3[:5])
    res["x3"], res["y3"] = p3[:5] + shift, np.r_[0.0, p3[5:]] - shift
    # Method 4: minimum chi-square on 3 x y - 2
    x4p, y4p = PUBLISHED["Method 4"]
    p4 = minimise(mixed, np.r_[np.log(x4p * y4p[0]), np.log(y4p[1:] / y4p[0])])
    res["F4"] = mixed(p4)
    res["x4"], res["y4"] = rescale_mult(np.exp(p4[:5]), np.exp(np.r_[0.0, p4[5:]]), x4p)
    # Minimum bias, started from the customary y as in Berquist's table, then run to its fixed point
    its = minimum_bias(y1p, 2000)
    res["mb_iterations"] = its[:3]
    xm, ym = its[-1]
    res["Fmb"] = np.outer(xm, ym)
    res["xmb"], res["ymb"] = xm, ym
    # the Poisson fit of the 20 cells, relative losses as response, log car-years as offset
    cls = np.repeat(np.arange(5), 4)
    mer = np.tile(np.arange(4), 5)
    res["Fglm"] = poisson_irls(LOSS.ravel(), N.ravel(), dummies([cls, mer])).reshape(5, 4) / N
    return res


def report_canadian(res: dict) -> None:
    header("1. Bailey and Simon (1960): the Canadian experience, policy years 1957 and 1958")
    print(f"Cells: {N.size} (5 classes x 4 merit groups). Car-years: {N.sum():,.0f}"
          f" (Table A total 4,153 thousand; Table 1 total 4,150,075).")
    print(f"Share of car-years in class 1: {N[0].sum() / N.sum():.3f}; in merit group A: {N[:, 0].sum() / N.sum():.3f}.")
    print("Cell with most car-years: class 1, merit A, "
          f"{N[0, 0] / N.sum():.3f} of the total.")
    print()
    print("Effectiveness as Bailey and Simon measured it (their Section A), from the published loss ratios:")
    merit_red = 1 - LR_MERIT_TOTAL["A"] / LR_OVERALL
    class_red = 1 - LR_CLASS_TOTAL["1"] / LR_OVERALL
    both_red = 1 - LR_1A / LR_OVERALL
    print(f"  lowest merit group A below average:      {100 * merit_red:5.1f}%")
    print(f"  lowest class 1 below average:             {100 * class_red:5.1f}%")
    print(f"  cell 1A (both plans) below average:       {100 * both_red:5.1f}%")
    print(f"  merit rating as a fraction of class rating: {merit_red / class_red:.3f} (their 77%)")
    print()
    print(f"Share of each class's car-years in merit group B (1, 5, 3, 2, 4): {np.round(N[:, 3] / N.sum(1), 3)}")
    print(f"Merit B over merit A within each class (1, 5, 3, 2, 4): {np.round(R[:, 3] / R[:, 0], 3)}")
    print(f"Customary rate of cell 1A: {PUBLISHED['Method 1'][0][0]:.3f} x {PUBLISHED['Method 1'][1][0]:.3f}"
          f" = {PUBLISHED['Method 1'][0][0] * PUBLISHED['Method 1'][1][0]:.3f}")
    print("Spread of the relative loss ratios: lowest cell 1A = "
          f"{R.min():.3f}, highest cell 4B = {R.max():.3f}, ratio {R.max() / R.min():.2f}.")

    header("2. The four methods of Bailey and Simon, recomputed, against their Tables C and E")
    print("Relativities (classes 1, 5, 3, 2, 4 | merit A, X, Y, B). 'pub' = as printed; 'rec' = recomputed.")
    print("Multiplicative relativities are identified only up to x*c, y/c (additive: x+c, y-c); recomputed")
    print("relativities are put on the published scale by the c that matches the published class relativities.")
    print(row("Method 1 pub", np.r_[PUBLISHED["Method 1"][0], PUBLISHED["Method 1"][1]]))
    print(row("Method 1 from car-years", np.r_[res["x1_caryears"], res["y1_caryears"]]))
    print("  (Bailey and Simon weighted the customary relativities by premium at 1B rates; car-years are")
    print("   the only weights that can be transcribed, so the article uses their printed Method 1.)")
    for m, xk, yk in (("Method 2", "x2", "y2"), ("Method 3", "x3", "y3"), ("Method 4", "x4", "y4")):
        print(row(f"{m} pub", np.r_[PUBLISHED[m][0], PUBLISHED[m][1]]))
        print(row(f"{m} rec", np.r_[res[xk], res[yk]]))
    print()
    for m, F in (("Method 2", res["F2"]), ("Method 3", res["F3"]), ("Method 4", res["F4"])):
        xp, yp = PUBLISHED[m]
        Fp = np.outer(xp, yp) if m == "Method 2" else (xp[:, None] + yp[None, :] if m == "Method 3"
                                                         else 3 * np.outer(xp, yp) - 2)
        print(f"{m}: largest gap between recomputed and printed fitted cell values: {np.max(np.abs(F - Fp)):.4f}")
    print()
    print("Tests (balance = fitted / actual relative losses; departure = sum n|r - F| / sum n r;")
    print("chi-square = K sum n (r - F)^2 / F with n in car-years and K = 1/200; 12 df, 11 for Method 4):")
    methods = (("Method 1 (customary)", res["F1"], 12, "Method 1"), ("Method 2 (min chi2, x*y)", res["F2"], 12, "Method 2"),
               ("Method 3 (min chi2, x+y)", res["F3"], 12, "Method 3"), ("Method 4 (min chi2, 3xy-2)", res["F4"], 11, "Method 4"),
               ("Minimum bias (Bailey 1963)", res["Fmb"], 12, None))
    print(f"{'':<28}{'total bal':>10}{'worst bal':>11}{'departure':>11}{'chi2':>8}{'p':>9}   printed (bal, dep, chi2)")
    for name, F, df, key in methods:
        c = criteria(F)
        worst = np.r_[c["class"], c["merit"]]
        worst = worst[np.argmax(np.abs(worst - 1))]
        pub = PUBLISHED_E.get(key)
        pub_s = f"   {pub[0]:.4f}, {pub[1]:.4f}, {pub[2]}" if pub else ""
        print(f"{name:<28}{c['total']:10.4f}{worst:11.4f}{c['departure']:11.4f}{c['chi2']:8.1f}"
              f"{stats.chi2.sf(c['chi2'], df):9.4f}{pub_s}")
    c1 = criteria(res["F1"])
    print()
    print("Method 1 balance by class (1, 5, 3, 2, 4):  rec", np.round(c1["class"], 4), " pub", PUBLISHED_BALANCE_M1[0])
    print("Method 1 balance by merit (A, X, Y, B):     rec", np.round(c1["merit"], 4), " pub", PUBLISHED_BALANCE_M1[1])
    c2 = criteria(res["F2"])
    print("Method 2 balance by class:", np.round(c2["class"], 4), " by merit:", np.round(c2["merit"], 4))
    c3 = criteria(res["F3"])
    print("Method 3 balance by class:", np.round(c3["class"], 4), " by merit:", np.round(c3["merit"], 4))

    header("3. Bailey (1963): minimum bias, and why its answer is the Poisson fit")
    print("Iterations started from the customary merit relativities (as in Berquist's discussion, Table 1):")
    for k, ((x, y), (bx, by)) in enumerate(zip(res["mb_iterations"], BERQUIST_ITERATIONS), start=1):
        print(row(f"calculation {k} rec", np.r_[x, y]))
        print(row(f"calculation {k} Berquist", np.r_[bx, by]))
    c = criteria(res["Fmb"])
    print()
    print("Fixed point:", row("", np.r_[res["xmb"], res["ymb"]]))
    print("Largest gap to Berquist's fitted values (Table 2):", f"{np.max(np.abs(res['Fmb'] - BERQUIST_FITTED)):.4f}")
    print("Balance at the fixed point, by class:", np.round(c["class"], 6), " by merit:", np.round(c["merit"], 6))
    print(f"Largest relative gap between the minimum-bias fit and the Poisson log-link fit: "
          f"{np.max(np.abs(res['Fmb'] / res['Fglm'] - 1)):.2e}")
    print(f"Minimum bias against Method 2: chi-square {criteria(res['Fmb'])['chi2']:.2f} vs {criteria(res['F2'])['chi2']:.2f};"
          f" largest cell gap {np.max(np.abs(res['Fmb'] - res['F2'])):.4f}")

    header("4. The same Canadian tariffs scored by an exposure-weighted Gini index")
    flat = np.ones_like(R)
    cases = [("Method 1 (customary)", res["F1"]), ("Method 2 (min chi2, x*y)", res["F2"]),
             ("Method 3 (min chi2, x+y)", res["F3"]), ("Method 4 (min chi2, 3xy-2)", res["F4"]),
             ("Minimum bias", res["Fmb"]),
             ("Class only (customary x)", np.outer(PUBLISHED["Method 1"][0], np.ones(4))),
             ("Merit only (customary y)", np.outer(np.ones(5), PUBLISHED["Method 1"][1])),
             ("Flat rate", flat), ("Cells' own experience", R)]
    g = {}
    for name, F in cases:
        g[name] = gini(F.ravel(), N.ravel(), LOSS.ravel())
        print(f"  {name:<28} Gini {g[name]:.4f}")
    share_n, share_l = N / N.sum(), LOSS / LOSS.sum()
    print(f"Cell 1A: {share_n[0, 0]:.3f} of the car-years and {share_l[0, 0]:.3f} of the relative losses "
          f"(first point of every tariff's Lorenz curve); cell 4B: {share_n[4, 3]:.4f} of the car-years and "
          f"{share_l[4, 3]:.4f} of the losses; the other 18 cells: {1 - share_n[0, 0] - share_n[4, 3]:.3f} of the car-years.")
    ceiling = g["Cells' own experience"]
    print(f"Ceiling for a tariff on these 20 cells (cells ranked by their own experience): {ceiling:.4f}")
    for name in ("Method 1 (customary)", "Method 3 (min chi2, x+y)", "Minimum bias"):
        print(f"  {name} as a share of the ceiling: {g[name] / ceiling:.4f}")
    print(f"Merit-only Gini as a fraction of class-only Gini: {g['Merit only (customary y)'] / g['Class only (customary x)']:.3f}")
    print()
    print("Cell orderings: number of the 190 pairs of cells ranked in opposite order:")
    for (a, Fa), (b, Fb) in combinations([("M1", res["F1"]), ("M2", res["F2"]), ("M3", res["F3"]),
                                           ("M4", res["F4"]), ("MB", res["Fmb"])], 2):
        print(f"  {a} vs {b}: {discordant_pairs(Fa, Fb)}")
    print()
    print("The minimum-bias tariff raised to a power and rescaled to balance in total:")
    for gam in (0.5, 2.0, 3.0):
        F = res["Fmb"] ** gam
        F = F * LOSS.sum() / (N * F).sum()
        c = criteria(F)
        print(f"  power {gam:3.1f}: Gini {gini(F.ravel(), N.ravel(), LOSS.ravel()):.4f}, total balance {c['total']:.4f},"
              f" class 1 {c['class'][0]:.3f}, class 4 {c['class'][4]:.3f}, merit B {c['merit'][3]:.3f},"
              f" chi-square {c['chi2']:.0f}")
    res["gini"] = g


# ---------------------------------------------------------------------------------------------------
# Sweden 1977
# ---------------------------------------------------------------------------------------------------
FACTORS = ("Kilometres", "Zone", "Bonus", "Make")


def marginal_totals(y, exposure, codes, factors, tol=1e-13, max_iter=10000):
    """Jung's (1968) marginal-totals equations for a multiplicative tariff, solved by Bailey's iteration."""
    rel = {f: np.ones(codes[f].max() + 1) for f in factors}
    base = y.sum() / exposure.sum()
    for it in range(1, max_iter + 1):
        old = {f: rel[f].copy() for f in factors}
        for f in factors:
            other = base * np.prod([rel[h][codes[h]] for h in factors if h != f], axis=0) \
                if len(factors) > 1 else np.full(exposure.shape, base)
            rel[f] = (np.bincount(codes[f], weights=y, minlength=rel[f].size)
                      / np.bincount(codes[f], weights=exposure * other, minlength=rel[f].size))
        for f in factors:
            base *= rel[f][0]
            rel[f] = rel[f] / rel[f][0]
        if max(np.max(np.abs(rel[f] - old[f])) for f in factors) < tol:
            break
    mu = exposure * base * np.prod([rel[f][codes[f]] for f in factors], axis=0)
    return mu, rel, base, it


def poisson_deviance(y, mu) -> float:
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * np.sum(t - (y - mu)))


def balance_by(codes, premium, claims):
    return np.bincount(codes, weights=premium) / np.bincount(codes, weights=claims)


def decile_balance(rate, exposure, premium, claims, groups=10):
    """Premium over claims within groups of equal exposure, cells sorted by premium rate (whole cells)."""
    order = np.argsort(rate, kind="stable")
    cum = np.cumsum(exposure[order]) - 0.5 * exposure[order]
    g = np.minimum((groups * cum / exposure.sum()).astype(int), groups - 1)
    lab = np.empty_like(g)
    lab[order] = g
    return np.bincount(lab, weights=premium, minlength=groups) / np.bincount(lab, weights=claims, minlength=groups)


GAMMAS_REPORTED = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
GAMMA_GRID = np.round(np.linspace(0.2, 3.0, 57), 4)


def swedish() -> dict:
    d = swedish_1977()
    E, C, P = d["Insured"], d["Claims"], d["Payment"]
    codes = {f: d[f] - 1 for f in FACTORS}
    res = {"E": E, "C": C, "P": P, "codes": codes}
    mu, rel, base, it = marginal_totals(C, E, codes, FACTORS)
    res.update(mu=mu, rel=rel, base=base, iterations=it)
    res["mu_irls"] = poisson_irls(C, E, dummies([codes[f] for f in FACTORS]))
    res["nested"] = []
    for fs in (("Bonus",), ("Bonus", "Zone"), ("Bonus", "Zone", "Kilometres"), ("Bonus", "Zone", "Kilometres", "Make")):
        m = marginal_totals(C, E, codes, fs)[0]
        res["nested"].append((fs, gini(m / E, E, C), poisson_deviance(C, m)))
    res["flat_dev"] = poisson_deviance(C, E * C.sum() / E.sum())
    res["ceiling"] = gini(C / E, E, C)
    # customary one-way tariff: product of one-way relativities, no rebalancing
    freq = C.sum() / E.sum()
    one_way = {f: balance_by(codes[f], C, E) / freq for f in FACTORS}  # claims / exposure by level, relative
    cust = E * freq * np.prod([one_way[f][codes[f]] for f in FACTORS], axis=0)
    res["customary"] = cust
    # distortions of the GLM: same order, different price
    rate = mu / E
    dist = []
    for gam in GAMMA_GRID:
        s = rate ** gam
        prem = E * s * C.sum() / (E * s).sum()
        dist.append((float(gam), gini(s, E, C), poisson_deviance(C, prem)))
    res["distortions"] = dist
    res["rate"] = rate
    return res


def distorted_premium(res, gam, level=1.0):
    E, C, rate = res["E"], res["C"], res["rate"]
    s = rate ** gam
    return level * E * s * C.sum() / (E * s).sum()


def under_bound(value: float, bound: float) -> str:
    """State a machine-precision check as a bound.

    The last digits of a gap near 1e-14 change with the CPU's BLAS code path, so
    printing them makes the committed output irreproducible across runners. A gap
    at or above the bound is printed in full, so a real regression still shows.
    """
    return f"below {bound:.0e}" if value < bound else f"{value:.1e} (NOT below {bound:.0e})"


def report_swedish(res: dict) -> None:
    E, C, P, codes = res["E"], res["C"], res["P"], res["codes"]
    header("5. Sweden 1977: the marginal-totals tariff is the Poisson GLM")
    print(f"Possible cells 5 x 7 x 7 x 9 = {5 * 7 * 7 * 9:,}; cells present: {E.size:,}")
    print(f"Cells: {E.size:,}; policy-years: {E.sum():,.0f}; claims: {C.sum():,.0f}; payments: {P.sum():,.0f} kronor")
    print(f"Claim frequency: {C.sum() / E.sum():.4f} per policy-year; mean payment per claim: {P.sum() / C.sum():,.0f} kronor")
    print(f"Cells with no claims: {(C == 0).sum()}; median exposure per cell: {np.median(E):.1f} policy-years")
    print(f"Marginal-totals iteration (Bailey 1963, Jung 1968), four factors: converged in {res['iterations']} sweeps.")
    print(f"Largest relative gap to the Poisson GLM fitted by IRLS: {under_bound(np.max(np.abs(res['mu'] / res['mu_irls'] - 1)), 1e-13)}")
    for f in FACTORS:
        b = balance_by(codes[f], res["mu"], C)
        print(f"  {f:<10} relativities {np.round(res['rel'][f], 3)}  balance max |1 - ratio| {under_bound(np.max(np.abs(b - 1)), 1e-14)}")
    print(f"  base frequency (Kilometres 1, Zone 1, Bonus 1, Make 1): {res['base']:.4f}")
    print(f"Range of the four-factor frequency tariff: {res['rate'].min():.4f} to {res['rate'].max():.4f} "
          f"(ratio {res['rate'].max() / res['rate'].min():.1f})")
    print()
    print("Nested tariffs (each fitted by marginal totals), Gini of claims and Poisson deviance:")
    print(f"  {'flat rate':<40} Gini 0.0000   deviance {res['flat_dev']:10.1f}")
    for fs, g, dv in res["nested"]:
        print(f"  {' + '.join(fs):<40} Gini {g:.4f}   deviance {dv:10.1f}")
    print(f"  Ceiling: cells ranked by their own observed frequency, Gini {res['ceiling']:.4f}")
    print(f"  Four-factor tariff as a share of the ceiling (a 'normalised' Gini): {res['nested'][-1][1] / res['ceiling']:.3f}")
    print()
    print("Kaggle-style normalised Gini over the 2,182 cells as rows (each cell counts once, exposure ignored):")
    print(f"  claims ranked by the GLM frequency rate:       {kaggle_gini(C, res['rate']):.4f}")
    print(f"  claims ranked by the GLM expected claim count: {kaggle_gini(C, res['mu']):.4f}")
    print(f"  claims ranked by exposure alone (no tariff):   {kaggle_gini(C, E):.4f}")
    rate, top = res["rate"], res["rate"] >= np.quantile(res["rate"], 0.9)
    rho = stats.spearmanr(rate, E).statistic
    print(f"  Spearman correlation between a cell's GLM rate and its exposure: {rho:.3f}")
    print(f"  median exposure of the 10% of cells with the highest rates: {np.median(E[top]):.1f} policy-years;"
          f" of the other cells: {np.median(E[~top]):.1f}")
    print(f"  share of the highest-rate 10% of cells in Bonus 1: {np.mean(codes['Bonus'][top] == 0):.3f};"
          f" in Zones 1-2: {np.mean(codes['Zone'][top] <= 1):.3f}")
    print()
    cust = res["customary"]
    print("Customary tariff (product of the four one-way relativities, as in Method 1):")
    print(f"  total balance {cust.sum() / C.sum():.4f}; Gini {gini(cust / E, E, C):.4f}; deviance {poisson_deviance(C, cust):.1f}")
    for f in FACTORS:
        b = balance_by(codes[f], cust, C)
        print(f"  {f:<10} balance by level {np.round(b, 3)}")

    header("6. Sweden 1977: the same ranking at a different price")
    print("Premium rate proportional to (GLM frequency)^gamma, rescaled so that total premium = total claims.")
    print("Gini is computed to 9 decimals; deviance is Poisson; balance = premium / claims.")
    print(f"{'gamma':>6}{'Gini':>13}{'deviance':>11}   {'Bonus 1':>8}{'Bonus 7':>8}{'Zone 1':>8}{'Zone 7':>8}"
          f"{'low dec':>9}{'top dec':>9}")
    for gam in GAMMAS_REPORTED:
        prem = distorted_premium(res, gam)
        s = res["rate"] ** gam
        bb = balance_by(codes["Bonus"], prem, C)
        zz = balance_by(codes["Zone"], prem, C)
        dec = decile_balance(s, E, prem, C)
        print(f"{gam:6.2f}{gini(s, E, C):13.9f}{poisson_deviance(C, prem):11.1f}   {bb[0]:8.3f}{bb[6]:8.3f}"
              f"{zz[0]:8.3f}{zz[6]:8.3f}{dec[0]:9.3f}{dec[-1]:9.3f}")
    print()
    for gam in (1.0, 2.0):
        prem = distorted_premium(res, gam)
        print(f"gamma {gam}: balance by premium decile {np.round(decile_balance(res['rate'] ** gam, E, prem, C), 3)}")
        for f in FACTORS:
            print(f"  {f:<10} {np.round(balance_by(codes[f], prem, C), 3)}")
    best = min(res["distortions"], key=lambda t: t[2])
    worse_than_flat = [g for g, _, dv in res["distortions"] if dv > res["flat_dev"]]
    print()
    print(f"Grid of {len(GAMMA_GRID)} exponents from {GAMMA_GRID[0]} to {GAMMA_GRID[-1]}: deviance smallest at gamma = {best[0]}")
    print(f"Distinct Gini values over the grid: {len({round(g, 12) for _, g, _ in res['distortions']})}")
    print(f"Smallest exponent on the grid above 1 whose deviance exceeds the flat rate's: {min(g for g in worse_than_flat if g > 1)}")
    def excess(gam):
        return poisson_deviance(res["C"], distorted_premium(res, gam)) - res["flat_dev"]
    root = optimize.brentq(excess, 1.0, 3.0, xtol=1e-10)
    print(f"Exact exponent above 1 at which the deviance equals the flat rate's: {root:.3f}")
    below = [g for g in worse_than_flat if g < 1]
    print(f"Exponents below 1 whose deviance exceeds the flat rate's: {below if below else 'none on the grid'}")
    print()
    print("Level errors: the GLM tariff multiplied by a constant.")
    for lev in (0.8, 0.9, 1.1):
        prem = distorted_premium(res, 1.0, lev)
        print(f"  level {lev:.1f}: Gini {gini(prem / E, E, C):.9f}; deviance {poisson_deviance(C, prem):.1f};"
              f" total premium / claims {prem.sum() / C.sum():.3f}")
    print()
    mup, relp, _, _ = marginal_totals(P, E, codes, FACTORS)
    print("Pure premium (payments) tariff by marginal totals, four factors:")
    print(f"  Gini of payments {gini(mup / E, E, P):.4f}; the frequency tariff ranks payments with Gini"
          f" {gini(res['rate'], E, P):.4f}; ceiling {gini(P / E, E, P):.4f}")
    for f in FACTORS:
        print(f"  {f:<10} pure-premium relativities {np.round(relp[f], 3)}")


def main() -> None:
    print("Part 1: how motor tariffs were judged, from minimum bias to the Gini index.")
    print(f"NumPy {np.__version__}")
    can = canadian()
    report_canadian(can)
    sw = swedish()
    report_swedish(sw)


if __name__ == "__main__":
    main()
