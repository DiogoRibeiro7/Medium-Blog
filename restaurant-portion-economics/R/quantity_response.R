# Break-even sensitivity of demand to the quantity served.
# Source R/economics.R first.
#
# The menu price is unchanged, so cutting the default portion by a fraction r
# raises the price per unit of food by r / (1 - r). If retained meals follow a
# constant-elasticity response to that effective unit price,
#
#   meals_after / meals_before = (1 - r)^eps,   eps >= 0,
#
# then eps is also the elasticity of demand with respect to portion size.
# The functions below return the largest eps compatible with a nonnegative
# change in monthly contribution. They do NOT estimate eps: no data in this
# project identify how restaurant customers respond to smaller portions.

#' Customer loss implied by a constant quantity elasticity.
#'
#' @param portion_reduction Fractions in [0, 1).
#' @param elasticity Nonnegative elasticities (recycled with the reductions).
#' @return Fraction of meals lost, 1 - (1 - r)^eps.
implied_customer_loss <- function(portion_reduction, elasticity) {
  if (!is.numeric(portion_reduction) || !is.numeric(elasticity) ||
      anyNA(c(portion_reduction, elasticity)) ||
      any(!is.finite(c(portion_reduction, elasticity))) ||
      any(portion_reduction < 0 | portion_reduction >= 1) ||
      any(elasticity < 0)) {
    stop("Use reductions in [0, 1) and nonnegative elasticities.",
         call. = FALSE)
  }
  1 - (1 - portion_reduction)^elasticity
}

#' Largest quantity elasticity at which a same-price portion cut breaks even.
#'
#' @param assumptions Baseline list validated by validate_assumptions().
#' @param portion_reductions Fractions in (0, 1) at which to evaluate.
#' @param include_implementation Whether the monthly implementation cost
#'   enters the break-even condition.
#' @return Data.frame with the ingredient saving, the break-even customer
#'   loss and the break-even elasticity for each reduction.
#' @details With m = p - c - v, S = c*s*r*a and implementation cost I over
#'   N meals, the change in contribution is
#'   N * ((1 - r)^eps * (m + S) - m) - I, which is zero at
#'   eps* = log((m + I/N) / (m + S)) / log(1 - r).
#'   A negative eps* means the cut loses money even if no customer reacts.
#'   For small r and I = 0, eps* tends to c*s*a/m: the avoidable ingredient
#'   cost per unit of portion cut, divided by the contribution margin.
quantity_elasticity_threshold <- function(
    assumptions,
    portion_reductions = seq(0.025, 0.30, by = 0.005),
    include_implementation = TRUE) {
  validate_assumptions(assumptions)
  if (!is.numeric(portion_reductions) || length(portion_reductions) < 1L ||
      anyNA(portion_reductions) || any(!is.finite(portion_reductions)) ||
      any(portion_reductions <= 0 | portion_reductions >= 1)) {
    stop("portion_reductions must be fractions strictly between 0 and 1.",
         call. = FALSE)
  }
  if (!is.logical(include_implementation) ||
      length(include_implementation) != 1L || is.na(include_implementation)) {
    stop("include_implementation must be TRUE or FALSE.", call. = FALSE)
  }
  a <- assumptions
  m <- a$price_eur - a$ingredient_cost_eur - a$other_variable_cost_eur
  if (m <= 0) {
    stop("The baseline contribution per meal must be positive.",
         call. = FALSE)
  }
  per_meal_cost <- if (include_implementation) {
    a$monthly_implementation_eur / a$meals_per_month
  } else {
    0
  }
  r <- portion_reductions
  saving <- a$ingredient_cost_eur * a$affected_ingredient_share * r *
    a$purchase_savings_realisation
  loss_star <- (saving - per_meal_cost) / (m + saving)
  eps_star <- log((m + per_meal_cost) / (m + saving)) / log(1 - r)
  data.frame(
    portion_reduction = r,
    unit_price_increase = r / (1 - r),
    saving_per_retained_meal_eur = saving,
    break_even_customer_loss = loss_star,
    break_even_quantity_elasticity = eps_star,
    includes_implementation_cost = include_implementation,
    evidence_status = "break_even_condition_not_an_estimate",
    stringsAsFactors = FALSE
  )
}

#' Small-cut limit of the break-even elasticity, ignoring implementation.
#'
#' @param assumptions Baseline list validated by validate_assumptions().
#' @return Scalar c*s*a / m.
quantity_elasticity_limit <- function(assumptions) {
  validate_assumptions(assumptions)
  a <- assumptions
  m <- a$price_eur - a$ingredient_cost_eur - a$other_variable_cost_eur
  if (m <= 0) {
    stop("The baseline contribution per meal must be positive.",
         call. = FALSE)
  }
  a$ingredient_cost_eur * a$affected_ingredient_share *
    a$purchase_savings_realisation / m
}

#' Published response estimates from adjacent markets, used as calibration.
#'
#' @return Data.frame of external estimates. None of them measures restaurant
#'   diners' response to a smaller plate at an unchanged price.
quantity_response_benchmarks <- function() {
  data.frame(
    benchmark_id = c(
      "cakir_balagtas_2014_package_size",
      "cakir_balagtas_2014_price",
      "andreyeva_2010_food_away_from_home_price"
    ),
    label = c(
      "Ice cream, package size (Chicago scanner data)",
      "Ice cream, price (same study)",
      "Food away from home, price (systematic review)"
    ),
    elasticity_low = c(0.12, 0.50, 0.70),
    elasticity_high = c(0.12, 0.50, 0.80),
    measures = c("quantity", "price", "price"),
    source_url = c(
      "https://doi.org/10.1016/j.jretai.2013.06.002",
      "https://doi.org/10.1016/j.jretai.2013.06.002",
      "https://doi.org/10.2105/AJPH.2008.151415"
    ),
    stringsAsFactors = FALSE
  )
}

#' Weeks of sales per arm needed to detect a given proportional loss.
#'
#' @param customer_loss Proportional drop in weekly meals to detect, (0, 1).
#' @param weekly_cv Week-to-week coefficient of variation of meals sold that
#'   the comparison cannot explain (an assumption, not an estimate).
#' @param alpha Two-sided test size.
#' @param power Probability of detecting the loss if it is real.
#' @return Weeks per arm under a two-sample normal approximation on log
#'   weekly meals with independent weeks. Serial correlation, seasonality
#'   and a single treated restaurant all make the real requirement larger;
#'   matched control restaurants can make it smaller.
pilot_weeks_needed <- function(customer_loss, weekly_cv, alpha = 0.05,
                               power = 0.80) {
  values <- c(customer_loss, weekly_cv, alpha, power)
  if (!is.numeric(values) || anyNA(values) || any(!is.finite(values)) ||
      any(customer_loss <= 0 | customer_loss >= 1) || any(weekly_cv <= 0) ||
      length(alpha) != 1L || length(power) != 1L ||
      alpha <= 0 || alpha >= 1 || power <= 0 || power >= 1) {
    stop("Use a loss in (0, 1), a positive CV and alpha, power in (0, 1).",
         call. = FALSE)
  }
  z <- stats::qnorm(1 - alpha / 2) + stats::qnorm(power)
  effect <- -log(1 - customer_loss)
  2 * z^2 * log1p(weekly_cv^2) / effect^2
}
