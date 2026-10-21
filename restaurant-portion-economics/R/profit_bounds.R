# Deterministic profit bounds for a default restaurant portion-size change.
# Hypothetical business scenarios: NOT fitted demand, causal estimates,
# statistical confidence intervals or identified empirical effects.
# Source R/economics.R first to reuse the validated baseline model.

#' Define a transparent example uncertainty set (assumptions, not evidence).
#'
#' @return Named list of closed intervals, each c(lower, upper).
example_profit_uncertainty <- function() {
  list(
    portion_reduction = c(0.10, 0.20),
    affected_ingredient_share = c(0.25, 0.55),
    purchase_savings_realisation = c(0.50, 0.95),
    customer_loss = c(0, 0.05),
    monthly_implementation_eur = c(75, 250)
  )
}

#' Validate the rectangular uncertainty set.
#'
#' @param bounds Named list of five numeric lower/upper intervals.
#' @return Validated bounds invisibly.
#' @details A rectangle is a scenario assumption. It does not assert
#'   statistical independence of the variables or any empirical support.
validate_profit_uncertainty <- function(bounds) {
  required <- c(
    "portion_reduction", "affected_ingredient_share",
    "purchase_savings_realisation", "customer_loss",
    "monthly_implementation_eur"
  )
  if (!is.list(bounds) || !setequal(names(bounds), required) ||
      anyDuplicated(names(bounds))) {
    stop("Provide exactly these interval names: ",
         paste(required, collapse = ", "), call. = FALSE)
  }
  for (name in required) {
    interval <- bounds[[name]]
    if (!is.numeric(interval) || length(interval) != 2L ||
        anyNA(interval) || any(!is.finite(interval)) ||
        interval[[1L]] > interval[[2L]]) {
      stop(name, " must be a finite c(lower, upper) numeric interval.",
           call. = FALSE)
    }
    if (name == "monthly_implementation_eur") {
      if (interval[[1L]] < 0) {
        stop("Implementation costs must be nonnegative.", call. = FALSE)
      }
    } else {
      if (interval[[1L]] < 0 || interval[[2L]] > 1) {
        stop(name, " must be within [0, 1].", call. = FALSE)
      }
      if (name == "customer_loss" && interval[[2L]] >= 1) {
        stop("Customer-loss upper bound must be below 1.", call. = FALSE)
      }
    }
  }
  invisible(bounds)
}

#' Confirm that the existing model has positive baseline contribution.
#'
#' @param baseline Baseline list validated by validate_assumptions().
#' @return Positive contribution per original meal, in EUR.
#' @details Positive baseline margin ensures the change in contribution
#'   is decreasing in customer losses over the entire uncertainty rectangle.
positive_baseline_margin <- function(baseline) {
  if (!exists("validate_assumptions", mode = "function")) {
    stop("Source R/economics.R before using R/profit_bounds.R.",
         call. = FALSE)
  }
  validate_assumptions(baseline)
  margin <- baseline$price_eur - baseline$ingredient_cost_eur -
    baseline$other_variable_cost_eur
  if (margin <= 0) {
    stop("Profit bounds require a positive baseline contribution per meal.",
         call. = FALSE)
  }
  margin
}

#' Exact deterministic bounds over a rectangular set of uncertain inputs.
#'
#' @param baseline Baseline from example_assumptions() or equivalent.
#' @param bounds Named list from example_profit_uncertainty() or equivalent.
#' @return One-row data.frame with sharp lower and upper monthly EUR changes,
#'   break-even customer-loss thresholds, and a conservative classification.
#'
#' @details Let m = p - c - v (>0), N be monthly meals, S = c*s*r*a
#'   (ingredient saving/retained meal), l be fraction of meals lost,
#'   and I the monthly implementation cost. Then
#'   delta = N * ((1-l)*(m+S) - m) - I.
#'   For m>0, c,s,r,a,I>=0 and l in [0,1), delta increases with s,r,a,
#'   decreases with l and I. Therefore interval endpoints are attained
#'   at corners. Bound endpoints are hypothetical, not confidence limits.
profit_delta_bounds <- function(baseline, bounds) {
  margin <- positive_baseline_margin(baseline)
  validate_profit_uncertainty(bounds)

  low <- function(name) bounds[[name]][[1L]]
  high <- function(name) bounds[[name]][[2L]]
  cost <- baseline$ingredient_cost_eur
  meals <- baseline$meals_per_month

  savings_min <- cost * low("affected_ingredient_share") *
    low("portion_reduction") * low("purchase_savings_realisation")
  savings_max <- cost * high("affected_ingredient_share") *
    high("portion_reduction") * high("purchase_savings_realisation")
  delta_min <- meals * (
    (1 - high("customer_loss")) * (margin + savings_min) - margin
  ) - high("monthly_implementation_eur")
  delta_max <- meals * (
    (1 - low("customer_loss")) * (margin + savings_max) - margin
  ) - low("monthly_implementation_eur")

  # At zero incremental contribution:
  # customer_loss* = (savings_per_meal - I/N) / (margin + savings_per_meal).
  # This expression increases in savings and decreases in implementation
  # costs when margin>0; negative thresholds imply demand growth is needed.
  threshold_min <- (
    savings_min - high("monthly_implementation_eur") / meals
  ) / (margin + savings_min)
  threshold_max <- (
    savings_max - low("monthly_implementation_eur") / meals
  ) / (margin + savings_max)

  status <- if (delta_min > 0) {
    "gain_under_all_assumptions"
  } else if (delta_max < 0) {
    "loss_under_all_assumptions"
  } else {
    "sign_depends_on_assumptions"
  }

  result <- data.frame(
    price_eur = baseline$price_eur,
    meals_per_month = meals,
    baseline_contribution_per_meal_eur = margin,
    portion_reduction_min = low("portion_reduction"),
    portion_reduction_max = high("portion_reduction"),
    customer_loss_min = low("customer_loss"),
    customer_loss_max = high("customer_loss"),
    procurement_saving_per_retained_meal_min_eur = savings_min,
    procurement_saving_per_retained_meal_max_eur = savings_max,
    contribution_change_min_eur = delta_min,
    contribution_change_max_eur = delta_max,
    break_even_loss_threshold_min = threshold_min,
    break_even_loss_threshold_max = threshold_max,
    classification = status,
    stringsAsFactors = FALSE
  )
  if (delta_min > delta_max + 1e-7 ||
      threshold_min > threshold_max + 1e-12) {
    stop("Unexpected ordering of interval endpoints.", call. = FALSE)
  }
  result
}

#' Compute guaranteed/possible profit conditions over a grid.
#'
#' @param baseline Validated baseline with positive initial margin.
#' @param bounds Named uncertainty-set intervals; the r and l intervals
#'   are overridden by each grid pair.
#' @param portion_reductions Numeric vector of assumed reductions [0,1].
#' @param customer_losses Numeric vector of assumed lost meal shares [0,1).
#' @return Data.frame with exact EUR bounds and a classification for each cell.
#' @details This is a feasibility map, NOT an optimized portion policy:
#'   the data do not identify demand loss as a function of portion size.
profit_frontier_grid <- function(
    baseline,
    bounds,
    portion_reductions = seq(0, 0.30, by = 0.025),
    customer_losses = seq(0, 0.10, by = 0.005)) {
  positive_baseline_margin(baseline)
  validate_profit_uncertainty(bounds)
  if (!is.numeric(portion_reductions) ||
      !is.numeric(customer_losses) ||
      length(portion_reductions) < 1L ||
      length(customer_losses) < 1L ||
      anyNA(portion_reductions) || anyNA(customer_losses) ||
      any(!is.finite(portion_reductions)) ||
      any(!is.finite(customer_losses)) ||
      any(portion_reductions < 0 | portion_reductions > 1) ||
      any(customer_losses < 0 | customer_losses >= 1)) {
    stop("Grid entries must be finite valid portions and customer losses.",
         call. = FALSE)
  }
  parameters <- expand.grid(
    portion_reduction = portion_reductions,
    customer_loss = customer_losses,
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )
  rows <- lapply(seq_len(nrow(parameters)), function(i) {
    local <- bounds
    reduction <- parameters$portion_reduction[[i]]
    loss <- parameters$customer_loss[[i]]
    local$portion_reduction <- c(reduction, reduction)
    local$customer_loss <- c(loss, loss)
    result <- profit_delta_bounds(baseline, local)
    data.frame(
      portion_reduction = reduction,
      customer_loss = loss,
      contribution_change_min_eur = result$contribution_change_min_eur,
      contribution_change_max_eur = result$contribution_change_max_eur,
      classification = result$classification,
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  result
}
