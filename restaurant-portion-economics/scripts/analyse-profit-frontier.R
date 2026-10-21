# Run offline from restaurant-portion-economics/: hypothetical scenarios only.
source("R/economics.R")
source("R/profit_bounds.R")
dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
baseline <- example_assumptions()
uncertainty <- example_profit_uncertainty()
bounds <- profit_delta_bounds(baseline, uncertainty)
grid <- profit_frontier_grid(baseline, uncertainty)
utils::write.csv(bounds, "outputs/profit_bounds.csv", row.names = FALSE)
utils::write.csv(grid, "outputs/profit_frontier.csv", row.names = FALSE)
inputs <- data.frame(
  parameter = names(uncertainty),
  minimum = vapply(uncertainty, function(z) z[[1L]], numeric(1)),
  maximum = vapply(uncertainty, function(z) z[[2L]], numeric(1)),
  source = "hypothetical business assumption",
  stringsAsFactors = FALSE
)
utils::write.csv(inputs, "outputs/profit_assumptions.csv", row.names = FALSE)
print(bounds[, c("contribution_change_min_eur",
                 "contribution_change_max_eur", "classification")])
message("All profit frontier values are hypothetical, not observed effects.")
