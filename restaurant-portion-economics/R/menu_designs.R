# Counterfactual menu designs: same-price smaller portions, discounted portions,
# and optional free refills. Source R/economics.R before using this file.
#
# All numeric inputs are analyst assumptions. These equations do not identify
# willingness to pay, actual demand, food-waste reductions or repeat visits.

#' Validate a single menu intervention assumption set.
#'
#' @param option Named list of scalar, explicitly hypothetical parameters.
#' @return The list invisibly.
validate_menu_option <- function(option) {
  required <- c(
    "option_id", "portion_reduction", "discount_fraction",
    "refill_allowed", "refill_uptake", "refill_restoration_fraction",
    "refill_service_cost_eur", "customer_loss",
    "monthly_implementation_eur"
  )
  if (!is.list(option) || !setequal(names(option), required) ||
      anyDuplicated(names(option))) {
    stop("Menu option must contain exactly: ",
         paste(required, collapse = ", "), call. = FALSE)
  }
  id <- option$option_id
  if (!is.character(id) || length(id) != 1L || is.na(id) ||
      !grepl("^[a-z][a-z0-9_]*$", id)) {
    stop("option_id must be a nonempty lowercase identifier.",
         call. = FALSE)
  }
  allowed <- option$refill_allowed
  if (!is.logical(allowed) || length(allowed) != 1L || is.na(allowed)) {
    stop("refill_allowed must be a non-missing logical scalar.",
         call. = FALSE)
  }

  fields <- setdiff(required, c("option_id", "refill_allowed"))
  for (name in fields) {
    x <- option[[name]]
    if (!is.numeric(x) || length(x) != 1L ||
        is.na(x) || !is.finite(x)) {
      stop(name, " must be a finite numeric scalar.", call. = FALSE)
    }
  }
  shares <- c(
    "portion_reduction", "discount_fraction", "refill_uptake",
    "refill_restoration_fraction", "customer_loss"
  )
  for (name in shares) {
    x <- option[[name]]
    if (x < 0 || x > 1) {
      stop(name, " must be in [0,1].", call. = FALSE)
    }
  }
  if (option$portion_reduction >= 1 || option$discount_fraction >= 1 ||
      option$customer_loss >= 1) {
    stop("Portion, discount, and demand-loss fractions must be below 1.",
         call. = FALSE)
  }
  if (option$refill_service_cost_eur < 0 ||
      option$monthly_implementation_eur < 0) {
    stop("Implementation and refill handling costs must be nonnegative.",
         call. = FALSE)
  }
  if (!allowed && (option$refill_uptake != 0 ||
                   option$refill_restoration_fraction != 0 ||
                   option$refill_service_cost_eur != 0)) {
    stop("A no-refill option cannot charge or supply refills.",
         call. = FALSE)
  }
  if (allowed && option$portion_reduction == 0 &&
      option$refill_uptake > 0) {
    stop("A refill cannot restore a zero reduction.", call. = FALSE)
  }
  invisible(option)
}

#' Return three transparent hypothetical menu design assumptions.
#'
#' @return Named list of three valid menu options.
example_menu_options <- function() {
  options <- list(
    smaller_same_price = list(
      option_id = "smaller_same_price",
      portion_reduction = 0.15, discount_fraction = 0,
      refill_allowed = FALSE, refill_uptake = 0,
      refill_restoration_fraction = 0, refill_service_cost_eur = 0,
      customer_loss = 0.01, monthly_implementation_eur = 150
    ),
    smaller_discount = list(
      option_id = "smaller_discount",
      portion_reduction = 0.15, discount_fraction = 0.05,
      refill_allowed = FALSE, refill_uptake = 0,
      refill_restoration_fraction = 0, refill_service_cost_eur = 0,
      customer_loss = 0, monthly_implementation_eur = 150
    ),
    smaller_free_refill = list(
      option_id = "smaller_free_refill",
      portion_reduction = 0.15, discount_fraction = 0,
      refill_allowed = TRUE, refill_uptake = 0.25,
      refill_restoration_fraction = 1, refill_service_cost_eur = 0.20,
      customer_loss = 0, monthly_implementation_eur = 200
    )
  )
  for (option in options) validate_menu_option(option)
  options
}

#' Compare monthly contribution for one menu design against the baseline.
#'
#' @param baseline Numeric list validated by validate_assumptions().
#' @param option Menu assumption list validated by validate_menu_option().
#' @return Single-row data.frame with EUR changes, thresholds and units.
#' @details Savings per retained meal are:
#'   S = c*s*r*a, where c is ingredient cost, s affected-cost share,
#'   r default whole-portion reduction, and a procurement pass-through.
#'   A fraction u of buyers requests one free refill, restoring fraction f
#'   of S's underlying ingredient reduction; each costs h extra to serve.
#'   The new per-meal contribution is:
#'   m_new = m + S*(1-u*f) - p*discount_fraction - u*h.
#'   Thus monthly contribution change is:
#'   Delta = N*(1-loss)*m_new - N*m - I.
#'   This is an accounting counterfactual with constant variable costs
#'   besides explicitly modelled interventions, NOT an estimated demand curve.
evaluate_menu_design <- function(baseline, option) {
  if (!exists("validate_assumptions", mode = "function")) {
    stop("Source R/economics.R before R/menu_designs.R.", call. = FALSE)
  }
  validate_assumptions(baseline)
  validate_menu_option(option)
  p <- baseline$price_eur
  c <- baseline$ingredient_cost_eur
  v <- baseline$other_variable_cost_eur
  n <- baseline$meals_per_month
  m <- p - c - v
  if (p <= 0 || m <= 0) {
    stop("Baseline price and contribution per meal must be positive.",
         call. = FALSE)
  }

  r <- option$portion_reduction
  uptake <- option$refill_uptake
  restore <- option$refill_restoration_fraction
  loss <- option$customer_loss
  implementation <- option$monthly_implementation_eur
  gross_ingredient_saving <- c * baseline$affected_ingredient_share * r *
    baseline$purchase_savings_realisation
  savings_after_refills <- gross_ingredient_saving * (1 - uptake * restore)
  discount_eur <- p * option$discount_fraction
  refill_cost_per_sale <- uptake * option$refill_service_cost_eur
  new_contribution <- m + savings_after_refills -
    discount_eur - refill_cost_per_sale
  expected_meals <- n * (1 - loss)
  delta <- expected_meals * new_contribution - n * m - implementation

  # Negative threshold means additional sales would be necessary;
  # no empirical elasticity or repeat-purchase rate is implied.
  break_even_loss <- if (new_contribution > 0) {
    1 - (m + implementation / n) / new_contribution
  } else {
    NA_real_
  }
  # A negative maximum feasible discount means no nonnegative discount
  # is profitable given the other assumptions and expected volume.
  maximum_discount_eur <- (
    m + savings_after_refills - refill_cost_per_sale -
      (m + implementation / n) / (1 - loss)
  )
  data.frame(
    option_id = option$option_id,
    portion_reduction = r,
    assumed_customer_loss = loss,
    assumed_meals_after = expected_meals,
    menu_price_after_eur = p - discount_eur,
    discount_per_meal_eur = discount_eur,
    # Ratio assumes default portion mass decreases in exact proportion r.
    # It is a mass-price arithmetic indicator, NOT subjective perceived value.
    relative_price_per_default_portion_mass_change =
      (1 - option$discount_fraction) / (1 - r) - 1,
    assumed_refill_uptake = uptake,
    assumed_refill_restoration = restore,
    ingredient_saving_before_refills_per_meal_eur = gross_ingredient_saving,
    ingredient_saving_after_refills_per_meal_eur = savings_after_refills,
    expected_refill_handling_per_meal_eur = refill_cost_per_sale,
    contribution_per_meal_after_eur = new_contribution,
    implementation_cost_per_month_eur = implementation,
    monthly_contribution_change_eur = delta,
    maximum_customer_loss_for_break_even = break_even_loss,
    largest_discount_eur_for_break_even = maximum_discount_eur,
    evidence_status = "hypothetical_scenario_not_demand_estimate",
    stringsAsFactors = FALSE
  )
}

#' Generate a transparent customer-loss/refill uptake sensitivity grid.
#'
#' @param baseline Valid baseline list, with positive contribution.
#' @param options Nonempty list of menu options. Names need not match IDs.
#' @param customer_losses Hypothetical loss fractions in [0,1).
#' @param refill_uptakes Hypothetical probabilities in [0,1].
#' @return data.frame with deterministic counterfactual calculations.
#' @details No probability weights, WTP or consumer behaviour are inferred.
#'   No-refill designs are evaluated only at uptake 0; free-refill designs
#'   are evaluated at each supplied uptake. Refill variable costs and savings
#'   are counted per retained customer, not per baseline expected meal.
menu_design_sensitivity <- function(
    baseline, options,
    customer_losses = c(0, 0.01, 0.02, 0.05),
    refill_uptakes = c(0, 0.25, 0.50, 0.75)) {
  validate_assumptions(baseline)
  if (!is.list(options) || length(options) == 0L ||
      any(!vapply(options, is.list, logical(1)))) {
    stop("Options must be a nonempty list of menu option lists.",
         call. = FALSE)
  }
  for (o in options) validate_menu_option(o)
  ids <- vapply(options, function(o) o$option_id, character(1))
  if (anyDuplicated(ids)) {
    stop("Each menu design must have a distinct option_id.",
         call. = FALSE)
  }

  validate_range <- function(x, name, can_equal_one) {
    if (!is.numeric(x) || length(x) < 1L || anyNA(x) ||
        any(!is.finite(x)) || any(x < 0) ||
        any(x > 1) || (!can_equal_one && any(x >= 1))) {
      stop(name, " must be a nonempty vector of valid fractions.",
           call. = FALSE)
    }
  }
  validate_range(customer_losses, "customer_losses", FALSE)
  validate_range(refill_uptakes, "refill_uptakes", TRUE)

  rows <- list()
  for (option in options) {
    uptakes <- if (option$refill_allowed) refill_uptakes else 0
    for (loss in customer_losses) {
      for (uptake in uptakes) {
        trial <- option
        trial$customer_loss <- loss
        trial$refill_uptake <- uptake
        rows[[length(rows) + 1L]] <- evaluate_menu_design(baseline, trial)
      }
    }
  }
  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  result
}
