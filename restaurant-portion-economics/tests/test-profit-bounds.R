# Deterministic, offline tests; figures are hypothetical.
source("R/economics.R")
source("R/profit_bounds.R")
a <- example_assumptions()
b <- example_profit_uncertainty()
z <- profit_delta_bounds(a, b)
stopifnot(z$contribution_change_min_eur < z$contribution_change_max_eur)
stopifnot(z$classification == "sign_depends_on_assumptions")

# Monotone corner solutions must exactly agree with the existing model.
for (corner in c("min", "max")) {
  at <- a
  idx <- if (corner == "min") 1L else 2L
  at$portion_reduction <- b$portion_reduction[[idx]]
  at$affected_ingredient_share <- b$affected_ingredient_share[[idx]]
  at$purchase_savings_realisation <- b$purchase_savings_realisation[[idx]]
  at$demand_change <- -b$customer_loss[[3L - idx]]
  at$monthly_implementation_eur <- b$monthly_implementation_eur[[3L - idx]]
  reference <- evaluate_scenario(at)$incremental_contribution_eur
  bound <- if (corner == "min") z$contribution_change_min_eur else z$contribution_change_max_eur
  stopifnot(isTRUE(all.equal(reference, bound, tolerance = 1e-10)))
}

# Exhaustively inspect every uncertainty-set corner (2^5 = 32).
corners <- expand.grid(
  r = b$portion_reduction, s = b$affected_ingredient_share,
  pass = b$purchase_savings_realisation,
  loss = b$customer_loss, implementation = b$monthly_implementation_eur
)
values <- vapply(seq_len(nrow(corners)), function(i) {
  at <- a
  at$portion_reduction <- corners$r[[i]]
  at$affected_ingredient_share <- corners$s[[i]]
  at$purchase_savings_realisation <- corners$pass[[i]]
  at$demand_change <- -corners$loss[[i]]
  at$monthly_implementation_eur <- corners$implementation[[i]]
  evaluate_scenario(at)$incremental_contribution_eur
}, numeric(1))
stopifnot(isTRUE(all.equal(min(values), z$contribution_change_min_eur)),
          isTRUE(all.equal(max(values), z$contribution_change_max_eur)))

# At the break-even customer loss, original model must give zero change.
at <- a
at$portion_reduction <- b$portion_reduction[[1L]]
at$affected_ingredient_share <- b$affected_ingredient_share[[1L]]
at$purchase_savings_realisation <- b$purchase_savings_realisation[[1L]]
at$monthly_implementation_eur <- b$monthly_implementation_eur[[2L]]
at$demand_change <- -z$break_even_loss_threshold_min
stopifnot(abs(evaluate_scenario(at)$incremental_contribution_eur) < 1e-8)

grid <- profit_frontier_grid(
  a, b, portion_reductions = c(0, 0.10, 0.20),
  customer_losses = c(0, 0.02, 0.05)
)
stopifnot(nrow(grid) == 9L,
          all(grid$contribution_change_min_eur <= grid$contribution_change_max_eur),
          all(grid$classification %in% c(
            "gain_under_all_assumptions", "loss_under_all_assumptions",
            "sign_depends_on_assumptions"
          )))
expect_error <- function(expr) {
  stopifnot(inherits(tryCatch(force(expr), error = identity), "error"))
}
bad <- b
bad$customer_loss <- c(0.1, 1)
expect_error(profit_delta_bounds(a, bad))
bad <- b
bad$affected_ingredient_share <- c(0.8, 0.4)
expect_error(profit_delta_bounds(a, bad))
bad <- a
bad$price_eur <- 5
expect_error(profit_delta_bounds(bad, b))
message("All deterministic partial-identification bounds tests passed.")
