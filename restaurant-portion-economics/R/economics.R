# Economic scenarios for restaurant portion-size decisions.
# All monetary quantities are EUR; volumes refer to meals per month.
# These functions use assumptions, not fitted restaurant demand or cost data.

#' Validate named economic assumptions.
#'
#' @param x Named list of scalar numeric assumptions.
#' @return The validated list, invisibly.
validate_assumptions <- function(x) {
  required <- c(
    "price_eur", "ingredient_cost_eur", "other_variable_cost_eur",
    "meals_per_month", "affected_ingredient_share", "portion_reduction",
    "purchase_savings_realisation", "demand_change",
    "monthly_implementation_eur"
  )
  if (!is.list(x) || !all(required %in% names(x))) {
    stop("Assumptions must be a named list containing: ",
         paste(required, collapse = ", "), call. = FALSE)
  }
  for (name in required) {
    value <- x[[name]]
    if (!is.numeric(value) || length(value) != 1L ||
        is.na(value) || !is.finite(value)) {
      stop(name, " must be a finite numeric scalar.", call. = FALSE)
    }
  }
  nonnegative <- c(
    "price_eur", "ingredient_cost_eur", "other_variable_cost_eur",
    "meals_per_month", "monthly_implementation_eur"
  )
  for (name in nonnegative) {
    if (x[[name]] < 0) {
      stop(name, " must be nonnegative.", call. = FALSE)
    }
  }
  if (x$meals_per_month <= 0) {
    stop("meals_per_month must be positive.", call. = FALSE)
  }
  shares <- c(
    "affected_ingredient_share", "portion_reduction",
    "purchase_savings_realisation"
  )
  for (name in shares) {
    if (x[[name]] < 0 || x[[name]] > 1) {
      stop(name, " must be in [0, 1].", call. = FALSE)
    }
  }
  if (x$demand_change <= -1) {
    stop("demand_change must be greater than -1.", call. = FALSE)
  }
  invisible(x)
}

#' Create an explicitly hypothetical baseline.
#'
#' @return Named list validated by validate_assumptions().
example_assumptions <- function() {
  values <- list(
    price_eur = 15,
    ingredient_cost_eur = 5,
    other_variable_cost_eur = 2,
    meals_per_month = 2500,
    affected_ingredient_share = 0.40,
    portion_reduction = 0.15,
    purchase_savings_realisation = 0.85,
    demand_change = 0,
    monthly_implementation_eur = 150
  )
  validate_assumptions(values)
  values
}

#' Calculate the before/after monthly contribution scenario.
#'
#' @details Fixed overhead is omitted because it cancels from the difference.
#'   The procurement saving applies only to the affected share of ingredient
#'   cost and only to the fraction actually translated into lower purchases.
#'   A reduction in food left on plates does not automatically imply this saving.
#'   No unmeasured storage or waste-disposal saving is included.
#' @param assumptions Named list of numeric assumptions.
#' @return Single-row data.frame; EUR values are monthly unless stated.
evaluate_scenario <- function(assumptions) {
  validate_assumptions(assumptions)
  a <- assumptions
  savings_per_meal <- a$ingredient_cost_eur *
    a$affected_ingredient_share *
    a$portion_reduction * a$purchase_savings_realisation

  baseline_margin <- a$price_eur - a$ingredient_cost_eur -
    a$other_variable_cost_eur
  post_margin <- baseline_margin + savings_per_meal
  meals_after <- a$meals_per_month * (1 + a$demand_change)
  before <- a$meals_per_month * baseline_margin
  after <- meals_after * post_margin - a$monthly_implementation_eur

  data.frame(
    portion_reduction = a$portion_reduction,
    demand_change = a$demand_change,
    savings_per_meal_eur = savings_per_meal,
    meals_before = a$meals_per_month,
    meals_after = meals_after,
    contribution_before_eur = before,
    contribution_after_eur = after,
    incremental_contribution_eur = after - before,
    stringsAsFactors = FALSE
  )
}

#' Calculate the break-even demand change for the portion intervention.
#'
#' @details A negative result means the largest proportional decline in meals
#'   the scenario could tolerate before losing contribution; it is not an
#'   estimate of actual demand elasticity or customer behaviour.
#' @param assumptions Named list of numeric assumptions.
#' @return Numeric scalar break-even demand change, or NA if undefined.
break_even_demand_change <- function(assumptions) {
  validate_assumptions(assumptions)
  a <- assumptions
  savings <- a$ingredient_cost_eur * a$affected_ingredient_share *
    a$portion_reduction * a$purchase_savings_realisation
  baseline_margin <- a$price_eur - a$ingredient_cost_eur -
    a$other_variable_cost_eur
  new_margin <- baseline_margin + savings
  if (new_margin <= 0) {
    return(NA_real_)
  }
  (a$monthly_implementation_eur / a$meals_per_month - savings) / new_margin
}

#' Evaluate a deterministic sensitivity grid.
#'
#' @param assumptions Validated baseline list.
#' @param reductions Numeric vector of portion-size reduction fractions.
#' @param customer_losses Numeric vector of nonnegative demand loss fractions.
#' @return Data.frame with the economic results for each combination.
scenario_grid <- function(
    assumptions,
    reductions = seq(0, 0.30, by = 0.05),
    customer_losses = seq(0, 0.15, by = 0.025)) {
  validate_assumptions(assumptions)
  if (!is.numeric(reductions) || !is.numeric(customer_losses) ||
      anyNA(reductions) || anyNA(customer_losses) ||
      !all(is.finite(c(reductions, customer_losses))) ||
      length(reductions) == 0L || length(customer_losses) == 0L ||
      any(reductions < 0 | reductions > 1) ||
      any(customer_losses < 0 | customer_losses >= 1)) {
    stop("Grid inputs must be nonempty fractions in their valid ranges.",
         call. = FALSE)
  }
  design <- expand.grid(
    portion_reduction = reductions,
    customer_loss = customer_losses,
    KEEP.OUT.ATTRS = FALSE
  )
  results <- lapply(seq_len(nrow(design)), function(i) {
    a <- assumptions
    a$portion_reduction <- design$portion_reduction[i]
    a$demand_change <- -design$customer_loss[i]
    evaluate_scenario(a)
  })
  out <- do.call(rbind, results)
  rownames(out) <- NULL
  out
}
