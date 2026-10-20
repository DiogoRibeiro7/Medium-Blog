# Small, deterministic tests; no external packages or network required.
source("R/economics.R")

a <- example_assumptions()
stopifnot(is.list(a), is.numeric(a$price_eur))
base <- a
base$portion_reduction <- 0
base$monthly_implementation_eur <- 0
stopifnot(abs(evaluate_scenario(base)$incremental_contribution_eur) < 1e-8)

# Only 40% of ingredients are affected: 5 * .40 * .15 * .85 = EUR 0.255.
a$monthly_implementation_eur <- 0
result <- evaluate_scenario(a)
stopifnot(abs(result$savings_per_meal_eur - 0.255) < 1e-10)
stopifnot(abs(result$incremental_contribution_eur - 637.5) < 1e-8)

a$monthly_implementation_eur <- 150
result <- evaluate_scenario(a)
stopifnot(abs(result$incremental_contribution_eur - 487.5) < 1e-8)

break_even <- break_even_demand_change(a)
stopifnot(is.finite(break_even), break_even < 0)
a$demand_change <- break_even
stopifnot(abs(evaluate_scenario(a)$incremental_contribution_eur) < 1e-7)
a$demand_change <- break_even - 0.01
stopifnot(evaluate_scenario(a)$incremental_contribution_eur < 0)

expect_error <- function(expr) {
  inherited_error <- inherits(tryCatch(force(expr), error = identity), "error")
  stopifnot(isTRUE(inherited_error))
}
bad <- a
bad$portion_reduction <- -0.01
expect_error(evaluate_scenario(bad))
bad <- a
bad$meals_per_month <- "100"
expect_error(evaluate_scenario(bad))
bad <- a
bad$demand_change <- -1
expect_error(evaluate_scenario(bad))

grid <- scenario_grid(
  example_assumptions(),
  reductions = c(0, 0.10, 0.20),
  customer_losses = c(0, 0.05)
)
stopifnot(is.data.frame(grid), nrow(grid) == 6L)
stopifnot(all(is.finite(grid$incremental_contribution_eur)))
stopifnot(all(grid$meals_after > 0))
message("All economic scenario tests passed.")
