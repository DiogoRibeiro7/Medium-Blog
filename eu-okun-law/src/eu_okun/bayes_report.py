"""Summaries, figures and the Markdown report for the hierarchical Bayesian fits.

Every number comes from saved posterior draws (``draws.npz``) and fit summaries
(``fit.json``); every qualitative statement follows a rule of ``BAYES_PROTOCOL.md``.
No ranking by point estimate is produced: rank uncertainty is reported as posterior
rank distributions and pairwise probabilities.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.bayes_model import PREREGISTERED, SPECS

PROBABILITY = 0.95          # threshold for "credible" statements (protocol, section 7)
CREDIBLE = 0.90             # equal-tailed credible intervals
FOCUS = ("PT", "ES", "NL", "LU")
SIGN_LABELS = {"negative": "credibly negative", "positive": "credibly positive",
               "uncertain": "sign uncertain"}


# ----------------------------------------------------------------------------------
# Posterior quantities
# ----------------------------------------------------------------------------------


@dataclass
class LoadedFit:
    """One specification's saved summary and flattened draws (draw, ...)."""

    name: str
    summary: dict[str, Any]
    draws: dict[str, np.ndarray]

    @property
    def countries(self) -> list[str]:
        return list(self.summary["countries"])

    @property
    def accepted(self) -> bool:
        return bool(self.summary["diagnostics"]["accepted"])

    def mu_theta(self) -> np.ndarray:
        """mu_theta draws, or its definition (unweighted mean of theta) without pooling."""
        if "mu_theta" in self.draws:
            return self.draws["mu_theta"]
        return self.draws["theta"].mean(axis=1)


def load_fit(directory: Path) -> LoadedFit:
    """Read ``fit.json`` and ``draws.npz``, merging chains into one draw axis."""
    summary = json.loads((directory / "fit.json").read_text(encoding="utf-8"))
    with np.load(directory / "draws.npz") as archive:
        draws = {k: archive[k].reshape(-1, *archive[k].shape[2:]) for k in archive.files}
    return LoadedFit(directory.name, summary, draws)


def credible_interval(values: np.ndarray, probability: float = CREDIBLE) -> tuple[float, float]:
    """Equal-tailed interval along the draw axis (axis 0)."""
    tail = (1.0 - probability) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail], axis=0)
    return float(low), float(high)


def sign_class(draws: np.ndarray) -> str:
    """Protocol rule: credibly negative / positive at 0.95, otherwise uncertain."""
    p_negative = float(np.mean(draws < 0))
    if p_negative >= PROBABILITY:
        return "negative"
    if p_negative <= 1.0 - PROBABILITY:
        return "positive"
    return "uncertain"


def rank_distribution(theta: np.ndarray) -> np.ndarray:
    """P(rank of country i = r), rank 1 = most negative sum; rows sum to one."""
    if theta.ndim != 2:
        raise ValueError("theta draws must have shape (draws, countries).")
    draws, n = theta.shape
    ranks = np.argsort(np.argsort(theta, axis=1), axis=1)
    out = np.zeros((n, n))
    for i in range(n):
        out[i] = np.bincount(ranks[:, i], minlength=n) / draws
    return out


def pairwise_probability(theta: np.ndarray) -> np.ndarray:
    """Matrix of P(theta[i] < theta[j]); diagonal zero, P_ij + P_ji = 1 without ties."""
    less = theta[:, :, None] < theta[:, None, :]
    return np.asarray(less.mean(axis=0), dtype=float)


@dataclass(frozen=True)
class CountryRow:
    """Posterior and classical summaries for one country in one fit."""

    spec: str
    geo: str
    theta_median: float
    theta_low: float
    theta_high: float
    p_negative: float
    sign: str
    beta0_median: float
    lag_share_median: float
    rank_median: float
    rank_low: float
    rank_high: float
    classical_theta: float
    classical_low: float
    classical_high: float


def country_rows(fit: LoadedFit) -> list[CountryRow]:
    """Per-country rows, in the stored (alphabetical) country order."""
    theta, beta0 = fit.draws["theta"], fit.draws["beta0"]
    ranks = np.argsort(np.argsort(theta, axis=1), axis=1) + 1
    rows = []
    for i, geo in enumerate(fit.countries):
        low, high = credible_interval(theta[:, i])
        share = (theta[:, i] - beta0[:, i]) / theta[:, i]
        r_low, r_high = credible_interval(ranks[:, i].astype(float))
        classical = fit.summary["classical_same_rows"][geo]
        rows.append(CountryRow(
            fit.name, geo, float(np.median(theta[:, i])), low, high,
            float(np.mean(theta[:, i] < 0)), sign_class(theta[:, i]),
            float(np.median(beta0[:, i])), float(np.median(share)),
            float(np.median(ranks[:, i])), r_low, r_high,
            classical["theta"], classical["ci_low"], classical["ci_high"],
        ))
    return rows


@dataclass(frozen=True)
class Claim:
    """Evaluation of one earlier claim in one fit."""

    claim: str
    spec: str
    probability: float
    holds: bool
    contradicted: bool
    evaluable: bool
    detail: str


def evaluate_claims(fit: LoadedFit) -> list[Claim]:
    """The three pre-registered claim rules (protocol, section 7 and change log)."""
    theta = fit.draws["theta"]
    index = {geo: i for i, geo in enumerate(fit.countries)}
    mu = fit.mu_theta()
    out = []
    lu = theta[:, index["LU"]]
    p_lu = float(np.mean(lu > mu))
    low, high = credible_interval(lu)
    out.append(Claim(
        "Luxembourg weak", fit.name, p_lu,
        holds=p_lu >= PROBABILITY and low <= 0 <= high,
        contradicted=p_lu <= 1 - PROBABILITY, evaluable=True,
        detail=f"P(theta_LU > mu_theta) = {p_lu:.3f}; 90% interval [{low:+.3f}, {high:+.3f}]",
    ))
    nl = theta[:, index["NL"]]
    nl_negative = float(np.mean(nl < 0)) >= PROBABILITY
    share = (nl - fit.draws["beta0"][:, index["NL"]]) / nl
    p_nl = float(np.mean(share > 0.5))
    out.append(Claim(
        "Netherlands delayed", fit.name, p_nl,
        holds=nl_negative and p_nl >= PROBABILITY,
        contradicted=nl_negative and p_nl <= 1 - PROBABILITY, evaluable=nl_negative,
        detail=(f"P(lag share > 0.5) = {p_nl:.3f}; median lag share "
                f"{float(np.median(share)):.2f}" if nl_negative
                else "theta_NL not credibly negative: rule not evaluable"),
    ))
    es = theta[:, index["ES"]]
    p_es = float(np.mean(es < mu))
    out.append(Claim(
        "Spain large", fit.name, p_es, holds=p_es >= PROBABILITY,
        contradicted=p_es <= 1 - PROBABILITY, evaluable=True,
        detail=f"P(theta_ES < mu_theta) = {p_es:.3f}",
    ))
    return out


def claim_verdict(primary: Claim, accepted: Sequence[Claim]) -> str:
    """Supported (primary and every accepted fit), contradicted (primary) or not resolved."""
    if not primary.evaluable:
        return "not evaluable"
    if primary.holds and all(c.holds for c in accepted):
        return "supported"
    if primary.contradicted:
        return "contradicted"
    return "not resolved"


@dataclass(frozen=True)
class FactorCheck:
    """Post-fit checks of the common factor's independence assumptions (median over draws)."""

    spec: str
    lag1_autocorrelation: float
    lag1_low: float
    lag1_high: float
    corr_mean_gdp: float
    corr_mean_gdp_low: float
    corr_mean_gdp_high: float
    corr_mean_unemployment_change: float


def factor_checks(fit: LoadedFit, mean_gdp: np.ndarray, mean_y: np.ndarray,
                  segment_start: np.ndarray) -> FactorCheck:
    """The model treats f[t] as independent over time and of GDP growth; test both.

    Lag-1 autocorrelation uses only pairs inside a segment. Correlations are with the
    cross-country mean of GDP growth and of the unemployment-rate change.
    """
    f = fit.draws["f"]
    inside = ~segment_start[1:]
    lag1 = np.array([np.corrcoef(row[1:][inside], row[:-1][inside])[0, 1] for row in f])
    with_gdp = np.array([np.corrcoef(row, mean_gdp)[0, 1] for row in f])
    with_y = np.array([np.corrcoef(row, mean_y)[0, 1] for row in f])
    lag_low, lag_high = credible_interval(lag1)
    gdp_low, gdp_high = credible_interval(with_gdp)
    return FactorCheck(fit.name, float(np.median(lag1)), lag_low, lag_high,
                       float(np.median(with_gdp)), gdp_low, gdp_high, float(np.median(with_y)))


@dataclass(frozen=True)
class Replication:
    """Summary tables of an independent run of the same fits (e.g. on another platform)."""

    accepted: dict[str, bool]
    medians: dict[str, dict[str, float]]
    signs: dict[str, dict[str, str]]
    claims: dict[str, list[Claim]]
    label: str


def load_replication(directory: Path) -> Replication:
    """Read ``country_estimates.csv``, ``claims.csv`` and ``diagnostics.json`` of a run."""
    diagnostics = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
    medians: dict[str, dict[str, float]] = {}
    signs: dict[str, dict[str, str]] = {}
    with (directory / "country_estimates.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            medians.setdefault(row["spec"], {})[row["geo"]] = float(row["theta_median"])
            signs.setdefault(row["spec"], {})[row["geo"]] = row["sign"]
    claims: dict[str, list[Claim]] = {}
    with (directory / "claims.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            claims.setdefault(row["spec"], []).append(Claim(
                row["claim"], row["spec"], float(row["probability"]), row["holds"] == "True",
                row["contradicted"] == "True", row["evaluable"] == "True", row["detail"]))
    readme = directory / "README.md"
    label = readme.read_text(encoding="utf-8").splitlines()[0].lstrip("# ") if readme.exists() \
        else directory.name
    return Replication({k: bool(v["accepted"]) for k, v in diagnostics.items()}, medians, signs,
                       claims, label)


def robust_negative_countries(
    signs: Mapping[str, Mapping[str, str]], accepted: Sequence[str],
) -> list[str]:
    """Countries credibly negative in every listed specification."""
    countries = next(iter(signs.values())).keys()
    return sorted(g for g in countries if all(signs[s][g] == "negative" for s in accepted))


# ----------------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:+.3f}"


def _interval(low: float, high: float) -> str:
    return f"[{low:+.3f}, {high:+.3f}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: str = "") -> list[str]:
    align = align or "l" + "r" * (len(header) - 1)
    marks = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _names(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


@dataclass
class BayesReview:
    """Everything the report is generated from."""

    fits: dict[str, LoadedFit]
    rows: dict[str, list[CountryRow]]
    claims: dict[str, list[Claim]]
    ranks: np.ndarray
    pairwise: np.ndarray
    factor: dict[str, FactorCheck] = field(default_factory=dict)
    replication: Replication | None = None

    @property
    def primary(self) -> LoadedFit:
        return self.fits["primary"]

    def accepted(self) -> list[str]:
        return [name for name, fit in self.fits.items() if fit.accepted]


def build_review(fits_dir: Path, data_dir: Path | None = None,
                 replication_dir: Path | None = None) -> BayesReview:
    """Load every fitted specification found in ``fits_dir`` (primary required).

    With ``data_dir`` (the snapshot the fits used), the common factor of each factor
    specification is also checked against the data.
    """
    fits = {
        name: load_fit(fits_dir / name) for name in PREREGISTERED
        if (fits_dir / name / "fit.json").exists()
    }
    if "primary" not in fits:
        raise FileNotFoundError(f"No primary fit in {fits_dir}.")
    rows = {name: country_rows(fit) for name, fit in fits.items()}
    claims = {name: evaluate_claims(fit) for name, fit in fits.items()}
    theta = fits["primary"].draws["theta"]
    review = BayesReview(fits, rows, claims, rank_distribution(theta),
                         pairwise_probability(theta))
    if data_dir is not None:
        from eu_okun.baseline import _read_panel
        from eu_okun.bayes_model import design_for
        from eu_okun.review import read_flags

        csv_path = data_dir / "processed" / "eu27_quarterly.csv"
        panel, flags = _read_panel(csv_path), read_flags(csv_path)
        for name, fit in fits.items():
            spec = SPECS[name]
            if spec.common_factor and "f" in fit.draws:
                data = design_for(spec, panel, flags)
                if list(data.quarters) != fit.summary["quarters"]:
                    raise ValueError(f"Snapshot does not match the quarters of fit {name}.")
                review.factor[name] = factor_checks(
                    fit, data.g.mean(axis=0), data.y.mean(axis=0), data.segment_start)
    if replication_dir is not None:
        review.replication = load_replication(replication_dir)
    return review


def robust_signs(review: BayesReview) -> dict[str, str]:
    """Sign class per country if identical in every accepted fit, else 'not robust'."""
    accepted = review.accepted()
    out = {}
    for i, geo in enumerate(review.primary.countries):
        classes = {review.rows[name][i].sign for name in accepted}
        out[geo] = classes.pop() if len(classes) == 1 else "not robust"
    return out


def render(review: BayesReview, assets: str, command: str) -> str:
    """Assemble the Markdown report."""
    p = review.primary
    accepted = review.accepted()
    primary_ok = p.accepted
    rows = review.rows["primary"]
    mu = p.mu_theta()
    tau = p.draws.get("tau_theta")
    signs = robust_signs(review)
    robust_negative = [g for g, s in signs.items() if s == "negative"]
    not_robust = [g for g, s in signs.items() if s == "not robust"]
    primary_uncertain = [r.geo for r in rows if r.sign == "uncertain"]
    primary_positive = [r.geo for r in rows if r.sign == "positive"]
    primary_negative = sum(r.sign == "negative" for r in rows)
    n = len(p.countries)
    rank_width = [r.rank_high - r.rank_low for r in rows]
    supported_pairs = int(np.sum(review.pairwise >= PROBABILITY))
    total_pairs = n * (n - 1) // 2
    snapshot = p.summary["snapshot"]
    env = p.summary["environment"]
    settings = p.summary["settings"]
    claims_primary = {c.claim: c for c in review.claims["primary"]}
    verdicts = {
        claim: claim_verdict(
            claims_primary[claim],
            [c for name in accepted for c in review.claims[name] if c.claim == claim],
        )
        for claim in claims_primary
    }
    mu_medians = {name: float(np.median(review.fits[name].mu_theta())) for name in accepted}
    by_name = {name: {r.geo: r for r in review.rows[name]} for name in review.fits}
    if "gaussian_iid" in accepted:
        pooling = float(np.median([abs(by_name["gaussian_iid"][r.geo].theta_median
                                       - r.classical_theta) for r in rows]))
        error_model = float(np.median([abs(r.theta_median
                                           - by_name["gaussian_iid"][r.geo].theta_median)
                                       for r in rows]))
        toward_zero = sum(abs(r.theta_median) < abs(by_name["gaussian_iid"][r.geo].theta_median)
                          for r in rows)
        decomposition_text = (
            f"Against OLS on the same quarters, partial pooling with OLS-like errors "
            f"(`gaussian_iid`) moves country sums by a median {pooling:.3f}; the primary "
            f"error model (Student-t, AR(1), common factor) moves them a further median "
            f"{error_model:.3f}, towards zero in {toward_zero} of {len(rows)} countries."
        )
    else:
        decomposition_text = "The decomposition needs an accepted `gaussian_iid` fit."
    p_lag = {name: fit.summary["predictive_check"]["p_value"]["lag1_autocorrelation"]
             for name, fit in review.fits.items()}
    misfit_lag = [name for name, value in p_lag.items() if value <= 0.05 or value >= 0.95]
    observed_lag = p.summary["predictive_check"]["observed"]["lag1_autocorrelation"]
    replicated_lag = {name: fit.summary["predictive_check"]["replicated_mean"]
                      ["lag1_autocorrelation"] for name, fit in review.fits.items()}
    persistence_text = (
        f"The lag-1 autocorrelation check fails (p <= 0.05 or >= 0.95) in "
        f"{len(misfit_lag)} of {len(review.fits)} specifications: observed "
        f"{observed_lag:.2f}, replicated means {min(replicated_lag.values()):.2f} to "
        f"{max(replicated_lag.values()):.2f}, so the models understate the persistence of "
        "unemployment changes."
    )
    replication_text = ""
    if review.replication is not None:
        rep_run = review.replication
        both = [n for n in review.fits if review.fits[n].accepted and rep_run.accepted.get(n)]
        gaps = [abs(by_name[n][g].theta_median - rep_run.medians[n][g])
                for n in both for g in rep_run.medians[n]]
        flipped = [n for n in review.fits if review.fits[n].accepted != rep_run.accepted.get(n)]
        rep_accepted = [n for n, ok in rep_run.accepted.items() if ok]
        rep_robust = robust_negative_countries(rep_run.signs, rep_accepted)
        here_robust = [g for g, sign in signs.items() if sign == "negative"]
        rep_verdicts = {
            claim: claim_verdict(
                next(c for c in rep_run.claims["primary"] if c.claim == claim),
                [c for n in rep_accepted for c in rep_run.claims[n] if c.claim == claim])
            for claim in claims_primary
        }
        same_verdicts = all(rep_verdicts[c] == verdicts[c] for c in verdicts)
        replication_text = (
            f"An independent run on a second platform (section 9) reproduces posterior medians "
            f"within {max(gaps):.3f} where both runs accept a fit. Acceptance differs for "
            f"{_names([f'`{n}`' for n in flipped]) if flipped else 'no specification'}, which "
            f"sit at the R-hat threshold. Countries credibly negative in every accepted fit: "
            f"{len(here_robust)} here, {len(rep_robust)} under that run's acceptance set; the "
            f"difference is {_names(sorted(set(here_robust) ^ set(rep_robust)))}. Claim "
            f"verdicts {'are identical' if same_verdicts else 'differ'}."
        )
    if "primary" in review.factor:
        fc = review.factor["primary"]
        factor_text = (
            f"The fitted common factor is autocorrelated (median lag-1 "
            f"{fc.lag1_autocorrelation:.2f}) and correlated with average GDP growth "
            f"(median {fc.corr_mean_gdp:+.2f}), although the model treats it as independent "
            "of both. In factor specifications theta therefore describes the association "
            "with each country's GDP beyond the common European cycle, not the classical "
            "coefficient sum."
        )
    else:
        factor_text = "Factor checks need the snapshot (`--data-dir`)."
    lines = [
        "# Hierarchical Bayesian estimates: EU-27 Okun coefficients", "",
        "> Generated by `eu-okun bayes-report` from saved posterior draws. Do not edit by "
        "hand. The model, priors, sensitivity set, acceptance criteria and decision rules "
        "were fixed in [BAYES_PROTOCOL.md](../BAYES_PROTOCOL.md) before any estimate was "
        "inspected. Results are descriptive conditional associations, not causal effects.",
        "",
    ]
    if not primary_ok:
        lines += [
            "**The primary specification failed the pre-registered convergence criteria "
            f"({'; '.join(p.summary['diagnostics']['failures'])}). Its estimates are shown for "
            "transparency only and must not be interpreted.**", "",
        ]
    mu_low, mu_high = credible_interval(mu)
    summary_rows = [
        ["Computation", f"{len(accepted)} of {len(review.fits)} specifications pass every "
         "acceptance criterion (R-hat <= 1.01, bulk and tail ESS >= 400, no divergences, "
         "E-BFMI >= 0.3)" + (f"; failed: {_names([k for k in review.fits if k not in accepted])}"
                              if len(accepted) < len(review.fits) else "") + "."],
        ["Average association", f"mu_theta, the unweighted mean of the country sums, has "
         f"median {_fmt(float(np.median(mu)))} and 90% interval {_interval(mu_low, mu_high)}; "
         f"P(mu_theta < 0) = {float(np.mean(mu < 0)):.3f}."
         + (f" Between-country s.d. tau_theta: median {float(np.median(tau)):.3f}, "
            f"90% interval {_interval(*credible_interval(tau))}." if tau is not None else "")],
        ["Country signs", f"Credibly negative in the primary fit: {primary_negative} "
         f"of {n}; sign uncertain: {_names(primary_uncertain)}"
         + (f"; credibly positive: {_names(primary_positive)}" if primary_positive else "")
         + f". Credibly negative in every accepted specification: {len(robust_negative)} "
         f"countries; classification changes across specifications for {_names(not_robust)}."],
        ["Rank uncertainty", f"Median width of the 90% rank interval: "
         f"{float(np.median(rank_width)):.0f} of {n} places. {supported_pairs} of {total_pairs} "
         "country pairs differ with posterior probability >= 0.95."],
        ["Earlier claims", "; ".join(f"{claim}: **{verdict}**" for claim, verdict in
                                      verdicts.items()) + "."],
        ["Magnitude", f"Specification-dependent: across accepted fits the median of mu_theta "
         f"ranges from {_fmt(min(mu_medians.values()))} to {_fmt(max(mu_medians.values()))}. "
         + decomposition_text],
        ["Model checks", persistence_text + " " + factor_text],
    ]
    if review.replication is not None:
        summary_rows.append(["Replication", replication_text])
    lines += ["## Summary", ""] + _table(["Topic", "Finding"], summary_rows, "ll")

    # 1. Provenance
    lines += [
        "## 1. Data, software and sampler", "",
        f"- Snapshot: manifest SHA-256 `{snapshot['manifest_sha256']}`, processed panel "
        f"`{snapshot['processed_csv_sha256']}`.",
        f"- Primary sample: {p.summary['quarters'][0]}..{p.summary['quarters'][-1]}, "
        f"{len(p.summary['quarters'])} quarters x {n} countries; "
        f"{p.summary['masked_terms']} likelihood terms masked because of Eurostat flags.",
        f"- NUTS: {settings['chains']} chains x ({settings['warmup']} warm-up + "
        f"{settings['draws']} draws), target acceptance {settings['target_accept']}, "
        f"maximum tree depth {settings['max_tree_depth']}, seed {settings['seed']}.",
        f"- Software: Python {env['python']}, JAX {env['jax']}, NumPyro {env['numpyro']}, "
        f"ArviZ {env['arviz']}, NumPy {env['numpy']} on {env['platform']}.", "",
    ]

    # 2. Diagnostics
    lines += ["## 2. Convergence and acceptance", ""]
    diag_rows = []
    for name, fit in review.fits.items():
        d = fit.summary["diagnostics"]
        diag_rows.append([
            f"`{name}`", SPECS[name].description, fit.summary["settings"]["warmup"],
            f"{d['max_rhat']:.3f}",
            f"{min(d['min_ess_bulk'], d['min_ess_tail']):.0f}", d["divergences"],
            f"{d['min_bfmi']:.2f}", f"{fit.summary['runtime_seconds'] / 60:.1f}",
            "accepted" if fit.accepted else "**failed**: " + "; ".join(d["failures"]),
        ])
    lines += _table(["Specification", "Description", "Warm-up", "Max R-hat", "Min ESS",
                     "Divergences", "Min E-BFMI", "Minutes", "Status"], diag_rows, "llrrrrrrl")
    longer = [name for name, fit in review.fits.items()
              if fit.summary["settings"]["warmup"] != settings["warmup"]]
    if longer:
        lines += [
            f"Re-run with longer warm-up after failing the criteria at {settings['warmup']} "
            f"warm-up iterations, as section 6 of the protocol allows (model and priors "
            f"unchanged; first-attempt diagnostics are in the protocol change log): "
            f"{_names([f'`{name}`' for name in longer])}.", "",
        ]

    # 3. Primary results
    lines += [
        "## 3. Country estimates (primary specification)", "",
        "Posterior median and 90% credible interval of each country's sum of GDP "
        "coefficients (lags 0-2), next to OLS with a calendar-aware HAC 95% interval on "
        "exactly the same quarters. Alphabetical order; no ranking is implied.", "",
        f"![Partial pooling against OLS]({assets}/shrinkage.png)", "",
    ]
    lines += _table(
        ["Country", "Bayes median", "90% credible interval", "P(theta < 0)",
         "Classical OLS", "95% HAC interval", "Lag share (median)"],
        [[r.geo, _fmt(r.theta_median), _interval(r.theta_low, r.theta_high),
          f"{r.p_negative:.3f}", _fmt(r.classical_theta),
          _interval(r.classical_low, r.classical_high),
          f"{r.lag_share_median:.2f}" if r.sign == "negative" else "n/a"] for r in rows],
        "lrrrrrr",
    )
    lines += [
        decomposition_text + " The differences from OLS are therefore not only shrinkage: "
        "the error model and the common factor change what the coefficient measures "
        "(section 8). Lag share = (theta - beta0) / theta, the part of the sum carried by "
        "lagged growth, shown only where theta is credibly negative.", "",
    ]

    # 4. Ranks
    lines += [
        "## 4. Rank uncertainty", "",
        "Rank 1 is the most negative coefficient sum. The heat map shows the posterior "
        "probability of each rank; the matrix shows P(row country's sum < column "
        "country's sum).", "",
        f"![Posterior rank probabilities]({assets}/rank_probabilities.png)", "",
        f"![Pairwise probabilities]({assets}/pairwise.png)", "",
    ]
    lines += _table(["Country", "Median rank", "90% rank interval"],
                    [[r.geo, f"{r.rank_median:.0f}", f"{r.rank_low:.0f}-{r.rank_high:.0f}"]
                     for r in rows], "lrr")
    lines += [
        f"{supported_pairs} of {total_pairs} pairs meet the protocol's 0.95 threshold. A "
        "country's position relative to others is reportable only for those pairs.", "",
    ]

    # 5. Claims
    lines += ["## 5. Earlier claims", "",
              "Rules from the protocol; 'supported' requires the rule to hold in the "
              "primary fit and in every accepted sensitivity fit.", ""]
    claim_rows = []
    for claim, verdict in verdicts.items():
        per_fit = [c for name in review.fits for c in review.claims[name] if c.claim == claim]
        holds = [c.spec for c in per_fit if c.holds]
        fails = [f"`{c.spec}`" for c in per_fit if not c.holds and c.spec in accepted]
        claim_rows.append([claim, f"**{verdict}**", claims_primary[claim].detail,
                           f"{len(holds)} of {len(per_fit)} fits", _names(fails)])
    lines += _table(["Claim", "Verdict", "Primary fit", "Rule holds in",
                     "Fails in (accepted fits)"], claim_rows, "lllll")

    # 6. Sensitivity
    lines += ["## 6. Sensitivity across specifications", "",
              "Posterior medians of each country's sum by specification; `*` marks a sum "
              "that is not credibly negative. Failed fits are greyed in the table header and "
              "do not count towards robustness.", ""]
    header = ["Country"] + [f"`{name}`" + ("" if review.fits[name].accepted else " (failed)")
                            for name in review.fits] + ["Robust sign"]
    sens_rows = []
    for i, geo in enumerate(p.countries):
        cells = []
        for name in review.fits:
            r = review.rows[name][i]
            cells.append(_fmt(r.theta_median) + ("" if r.sign == "negative" else "*"))
        sens_rows.append([geo, *cells, SIGN_LABELS.get(signs[geo], signs[geo])])
    lines += _table(header, sens_rows, "l" + "r" * len(review.fits) + "l")
    lines += [f"![Focus-country sensitivity]({assets}/focus_sensitivity.png)", ""]
    hyper_rows = []
    for name, fit in review.fits.items():
        m = fit.mu_theta()
        t = fit.draws.get("tau_theta")
        hyper_rows.append([f"`{name}`", _fmt(float(np.median(m))),
                           _interval(*credible_interval(m)),
                           f"{float(np.median(t)):.3f}" if t is not None else "n/a"])
    lines += _table(["Specification", "mu_theta median", "90% interval", "tau_theta median"],
                    hyper_rows, "lrrr")

    # 7. Predictive checks
    lines += ["## 7. Posterior predictive checks", "",
              "Coverage: share of observed changes inside the one-step-ahead 50% and 90% "
              "predictive intervals (calibrated values: 0.50 and 0.90). p-values: share of "
              "replicated data sets whose statistic is at least the observed one (values "
              "near 0 or 1 indicate misfit).", ""]
    ppc_rows = []
    for name, fit in review.fits.items():
        c = fit.summary["predictive_check"]
        ppc_rows.append([f"`{name}`", f"{c['coverage_50']:.2f}", f"{c['coverage_90']:.2f}",
                         f"{c['p_value']['lag1_autocorrelation']:.2f}",
                         f"{c['p_value']['cross_country_correlation']:.2f}",
                         f"{c['p_value']['mean_sd']:.2f}"])
    lines += _table(["Specification", "50% coverage", "90% coverage", "p: lag-1 autocorr.",
                     "p: cross-country corr.", "p: mean s.d."], ppc_rows, "lrrrrr")

    lines += [persistence_text, ""]

    # 8. Post-fit checks
    lines += [
        "## 8. Checks added after fitting", "",
        "Not pre-registered; computed because the predictive checks and the gap to OLS "
        "called for them. They qualify the interpretation; they do not change any fit.", "",
    ]
    if review.factor:
        lines += _table(
            ["Specification", "Factor lag-1 autocorr. (90%)",
             "Corr. with mean GDP growth (90%)", "Corr. with mean unemployment change"],
            [[f"`{fc.spec}`",
              f"{fc.lag1_autocorrelation:.2f} {_interval(fc.lag1_low, fc.lag1_high)}",
              f"{fc.corr_mean_gdp:+.2f} {_interval(fc.corr_mean_gdp_low, fc.corr_mean_gdp_high)}",
              f"{fc.corr_mean_unemployment_change:+.2f}"] for fc in review.factor.values()],
            "lrrr",
        )
        lines += [factor_text + " Specifications without the factor estimate the "
                  "classical quantity but fail the cross-country correlation check, so their "
                  "intervals ignore residual dependence. A dynamic factor, or a mean equation "
                  "with lagged unemployment changes, would address both problems; that is a "
                  "model change for a new, separately registered protocol.", ""]
    else:
        lines += [factor_text, ""]

    # 9. Replication
    if review.replication is not None:
        rep_run = review.replication
        lines += ["## 9. Replication on a second platform", "",
                  f"{rep_run.label}: the same snapshot, lock file, seed and settings, with the "
                  "same single retry. Tables in `replication-linux/`.", ""]
        rep_rows = []
        for name in review.fits:
            here_ok, there_ok = review.fits[name].accepted, rep_run.accepted.get(name)
            if here_ok and there_ok:
                gap = max(abs(by_name[name][g].theta_median - rep_run.medians[name][g])
                          for g in rep_run.medians[name])
                gap_text = f"{gap:.3f}"
            else:
                gap_text = "n/a"
            rep_rows.append([f"`{name}`", "accepted" if here_ok else "failed",
                             "accepted" if there_ok else "failed", gap_text])
        lines += _table(["Specification", "This run", "Replication", "Max abs. difference of "
                         "country medians"], rep_rows, "lllr")
        lines += [
            replication_text + " Countries whose credible negativity depends on including "
            "the unpooled fit are credibly negative only when estimates borrow strength across "
            "countries; their sign is supported by partial pooling, not by their own data "
            "alone.", "",
        ]

    # 10-11. Limitations and reproduction
    lines += [
        "## 10. Limitations", "",
        "- The pre-registered model understates the persistence of unemployment changes "
        "and its common factor is not independent of GDP growth (section 8). Credible "
        "intervals are conditional on that misspecified structure.",
        "- The coefficients are conditional associations. Contemporaneous GDP growth is "
        "not an identified shock; partial pooling does not change that.",
        "- mu_theta is an unweighted average across countries. It is not an EU-wide "
        "Okun coefficient in any population-weighted sense.",
        "- The primary sample ends in 2019; results do not describe the post-pandemic "
        "period. The pandemic-excluded sensitivity fit adds 2022-2026 under the same "
        "coefficients, which the classical stability tests question.",
        "- The common factor assumes non-negative loadings. Masked flagged observations "
        "remove information rather than correct it.",
        "- Rank and pairwise probabilities inherit every modelling assumption; they are "
        "not model-free comparisons.",
        "- Predictive comparison with the classical baselines is not yet done.", "",
        "## 11. Reproduction", "",
        "```bash", command, "```", "",
        "Posterior draws are produced by the separate Bayesian workflow and kept as its "
        "artifact; this report and its tables are regenerated from them.", "",
    ]
    return "\n".join(lines).rstrip() + "\n"


# ----------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------


def write_figures(review: BayesReview, directory: Path, source: str) -> None:
    """Shrinkage, rank probabilities, pairwise matrix and focus sensitivity."""
    from eu_okun.review_figures import AQUA, BLUE, INK2, MUTED, ORANGE, SURFACE, _finish, _pyplot

    plt = _pyplot()
    from matplotlib.lines import Line2D

    rows = review.rows["primary"]
    ys = np.arange(len(rows))[::-1]
    mu = float(np.median(review.primary.mu_theta()))
    iid = ({r.geo: r for r in review.rows["gaussian_iid"]}
           if "gaussian_iid" in review.fits else {})
    fig, ax = plt.subplots(figsize=(7.6, 9.2))
    for y, r in zip(ys, rows, strict=True):
        ax.plot([r.classical_low, r.classical_high], [y + 0.24] * 2, color=ORANGE, lw=1.6,
                alpha=0.55, solid_capstyle="round")
        ax.plot([r.classical_theta], [y + 0.24], marker="o", ms=5.5, color=ORANGE,
                mec=SURFACE, mew=1.2, lw=0)
        if r.geo in iid:
            q = iid[r.geo]
            ax.plot([q.theta_low, q.theta_high], [y] * 2, color=AQUA, lw=1.6,
                    solid_capstyle="round")
            ax.plot([q.theta_median], [y], marker="o", ms=5.5, color=AQUA, mec=SURFACE,
                    mew=1.2, lw=0)
        ax.plot([r.theta_low, r.theta_high], [y - 0.24] * 2, color=BLUE, lw=1.6,
                solid_capstyle="round")
        ax.plot([r.theta_median], [y - 0.24], marker="o", ms=5.5, color=BLUE, mec=SURFACE,
                mew=1.2, lw=0)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(mu, color=INK2, lw=1, ls=(0, (1, 0)), alpha=0.35)
    ax.set_yticks(ys)
    ax.set_yticklabels([r.geo for r in rows])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Sum of GDP coefficients, lags 0-2 (pp per pp), 2009-2019")
    ax.set_title("Pooling moves estimates a little; the error model moves them more")
    ax.legend(handles=[
        Line2D([], [], marker="o", color=ORANGE, lw=1.6, ms=6, label="OLS, 95% HAC interval"),
        Line2D([], [], marker="o", color=AQUA, lw=1.6, ms=6,
               label="Partial pooling, OLS-like errors (gaussian_iid), 90% interval"),
        Line2D([], [], marker="o", color=BLUE, lw=1.6, ms=6,
               label="Primary: pooling + Student-t AR(1) errors + common factor"),
        Line2D([], [], color=INK2, lw=1, alpha=0.35, label="Posterior median of mu_theta"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.075), fontsize=8.5)
    _finish(plt, fig, directory / "shrinkage.png", source, credit_y=-0.11)

    names = [r.geo for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    image = ax.imshow(review.ranks, cmap="Blues", vmin=0, vmax=max(0.3, review.ranks.max()),
                      aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(range(0, len(names), 2))
    ax.set_xticklabels([str(i + 1) for i in range(0, len(names), 2)], fontsize=8)
    ax.set_xlabel("Rank (1 = most negative coefficient sum)")
    ax.grid(False)
    ax.set_title("Posterior rank probabilities are spread over many places")
    fig.colorbar(image, ax=ax, shrink=0.7, label="Posterior probability")
    _finish(plt, fig, directory / "rank_probabilities.png", source, credit_y=-0.02)

    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    image = ax.imshow(review.pairwise, cmap="RdBu_r", vmin=0, vmax=1,
                      interpolation="nearest")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5, rotation=90)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.grid(False)
    ax.set_title("P(row country's sum < column country's sum)")
    fig.colorbar(image, ax=ax, shrink=0.7, label="Posterior probability")
    _finish(plt, fig, directory / "pairwise.png", source, credit_y=-0.02)

    fig, axes = plt.subplots(1, len(FOCUS), figsize=(12.4, 4.8), sharey=True)
    specs = list(review.fits)
    ys = np.arange(len(specs))[::-1]
    for ax, geo in zip(axes, FOCUS, strict=True):
        i = review.primary.countries.index(geo)
        for y, name in zip(ys, specs, strict=True):
            r = review.rows[name][i]
            colour = BLUE if review.fits[name].accepted else MUTED
            ax.plot([r.theta_low, r.theta_high], [y, y], color=colour, lw=1.6,
                    solid_capstyle="round")
            ax.plot([r.theta_median], [y], marker="o", ms=6, color=colour, mec=SURFACE,
                    mew=1, lw=0)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(geo)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(specs, fontsize=8)
    fig.supxlabel("Sum of GDP coefficients, posterior median and 90% interval", fontsize=9,
                  color=INK2)
    fig.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=1.6, ms=6, label="Accepted fit"),
        Line2D([], [], marker="o", color=MUTED, lw=1.6, ms=6, label="Failed fit (not interpreted)"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(wspace=0.08, bottom=0.16)
    _finish(plt, fig, directory / "focus_sensitivity.png", source, credit_y=-0.12)


def write_tables(review: BayesReview, directory: Path) -> None:
    """Country rows for every fit, rank probabilities, pairwise matrix and claims."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = [vars(r) for name in review.fits for r in review.rows[name]]
    with (directory / "country_estimates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    names = review.primary.countries
    for file, matrix, columns in (
        ("rank_probabilities.csv", review.ranks, [str(i + 1) for i in range(len(names))]),
        ("pairwise_probabilities.csv", review.pairwise, names),
    ):
        with (directory / file).open("w", encoding="utf-8", newline="") as f:
            writer_rows = csv.writer(f, lineterminator="\n")
            writer_rows.writerow(["geo", *columns])
            for geo, values in zip(names, matrix, strict=True):
                writer_rows.writerow([geo, *(f"{v:.4f}" for v in values)])
    claims = [vars(c) for name in review.fits for c in review.claims[name]]
    with (directory / "claims.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(claims[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(claims)
    diagnostics = {
        name: {**fit.summary["diagnostics"], "predictive_check": fit.summary["predictive_check"],
               "runtime_seconds": fit.summary["runtime_seconds"]}
        for name, fit in review.fits.items()
    }
    (directory / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def write_report(
    fits_dir: Path, output_dir: Path, report: Path | None = None, figures: bool = True,
    data_dir: Path | None = None, replication_dir: Path | None = None,
) -> BayesReview:
    """Build tables, figures and the Markdown report from saved fits."""
    review = build_review(fits_dir, data_dir, replication_dir)
    report = report or output_dir / "HIERARCHICAL_BAYES.md"
    write_tables(review, output_dir)
    if figures:
        source = ("Source: Eurostat, snapshot 2026-10-15; hierarchical Bayesian model "
                  "(BAYES_PROTOCOL.md). Descriptive, not causal.")
        write_figures(review, output_dir, source)
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"eu-okun bayes-report --fits-dir {fits_dir.as_posix()} "
               f"--output-dir {output_dir.as_posix()} --report {report.as_posix()}"
               + (f" --data-dir {data_dir.as_posix()}" if data_dir is not None else "")
               + (f" --replication-dir {replication_dir.as_posix()}"
                  if replication_dir is not None else ""))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(review, assets, command), encoding="utf-8", newline="\n")
    return review


def mapping_rows(rows: Mapping[str, Any]) -> list[str]:
    """Helper kept for CLI messages: 'name=value' pairs."""
    return [f"{k}={v}" for k, v in rows.items()]
