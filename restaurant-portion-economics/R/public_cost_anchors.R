# Public US industry cost medians as proxy scenarios, not restaurant estimates.
# Source R/economics.R and R/profit_bounds.R first.
# All monetary quantities in an example remain illustrative EUR.

#' Load externally documented restaurant food-cost benchmark references.
#' @param path CSV with the published provenance and values.
#' @return A validated data.frame of source medians and citations.
read_food_cost_benchmarks <- function(
    path = "data/reference/restaurant_cost_benchmarks.csv") {
  if (!is.character(path) || length(path) != 1L ||
      is.na(path) || !file.exists(path)) stop("Reference CSV missing.")
  x <- utils::read.csv(path, stringsAsFactors = FALSE)
  required <- c("reference_id", "geography", "observation_year",
                "segment", "metric", "statistic", "share_of_sales",
                "source_url")
  if (!all(required %in% names(x)) || nrow(x) == 0L ||
      anyDuplicated(x$reference_id) || anyNA(x[required])) {
    stop("Incomplete or duplicated source reference records.")
  }
  if (!is.numeric(x$share_of_sales) ||
      any(!is.finite(x$share_of_sales)) ||
      any(x$share_of_sales < 0 | x$share_of_sales >= 1) ||
      any(x$metric != "food_and_non_alcohol_beverage_costs") ||
      any(x$statistic != "respondent_median") ||
      any(x$geography != "United States") ||
      any(!grepl("^https://", x$source_url))) {
    stop("Unsupported restaurant benchmark data or units.")
  }
  x
}

#' Translate published sector sales-share medians into scenario costs.
#'
#' @param baseline Existing validated economics baseline.
#' @param benchmarks Validated source table.
#' @param demand_losses Scenario fractions, not empirical demand estimates.
#' @return Data.frame of hypothetical business results with source metadata.
#' @details A respondent median of a *business cost/sales ratio* is not
#'   the median cost per meal and includes non-alcoholic beverages. The
#'   calculation is an explicit cross-market proxy, not calibration of
#'   Portugal ingredient prices. Base-R sensitivity reuses the existing model.
cost_benchmark_scenarios <- function(
    baseline, benchmarks, demand_losses = c(0, 0.01, 0.02, 0.05)) {
  validate_assumptions(baseline)
  read_only <- benchmarks
  if (!is.data.frame(read_only) ||
      !all(c("reference_id", "share_of_sales", "source_url") %in%
           names(read_only)) ||
      anyNA(read_only$share_of_sales) ||
      any(read_only$share_of_sales < 0 | read_only$share_of_sales >= 1)) {
    stop("Invalid cost benchmark table.", call. = FALSE)
  }
  if (!is.numeric(demand_losses) || length(demand_losses) == 0L ||
      anyNA(demand_losses) || any(!is.finite(demand_losses)) ||
      any(demand_losses < 0 | demand_losses >= 1)) {
    stop("Demand losses must be valid fractions.", call. = FALSE)
  }
  index <- expand.grid(
    benchmark = seq_len(nrow(read_only)),
    demand_loss = demand_losses, KEEP.OUT.ATTRS = FALSE
  )
  rows <- lapply(seq_len(nrow(index)), function(i) {
    b <- read_only[index$benchmark[[i]], , drop = FALSE]
    local <- baseline
    local$ingredient_cost_eur <- local$price_eur * b$share_of_sales
    local$demand_change <- -index$demand_loss[[i]]
    output <- evaluate_scenario(local)
    data.frame(
      reference_id = b$reference_id,
      source_url = b$source_url,
      industry_food_beverage_sales_ratio = b$share_of_sales,
      assumed_menu_price_eur = local$price_eur,
      proxy_food_beverage_cost_per_meal_eur = local$ingredient_cost_eur,
      assumed_customer_loss = index$demand_loss[[i]],
      assumed_portion_reduction = local$portion_reduction,
      assumed_cost_share_affected = local$affected_ingredient_share,
      assumed_purchase_realisation = local$purchase_savings_realisation,
      assumed_implementation_cost_eur = local$monthly_implementation_eur,
      scenario_contribution_change_eur = output$incremental_contribution_eur,
      scenario_break_even_customer_loss =
        -break_even_demand_change(local),
      evidence_role = "cross_market_industry_proxy_scenario_not_estimate",
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}
