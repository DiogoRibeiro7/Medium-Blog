# Checks reproducibility of the article and its worked economic examples.
# Run from restaurant-portion-economics/ as:
# Rscript publication/tests/test-article.R

source("R/economics.R")
source("R/profit_bounds.R")
source("R/menu_designs.R")
source("R/quantity_response.R")

# Research figures remain public even when the editorial manuscript moves.
figure_dir <- file.path("publication", "medium", "figures")
figure_names <- sprintf("fig%02d_", seq_len(10L))
archived <- list.files(figure_dir, pattern = "^fig[0-9]{2}_.*\\.png$", full.names = TRUE)
if (length(archived) != 10L || anyDuplicated(basename(archived))) {
  stop("Expected ten distinct public research PNGs.")
}
for (prefix in figure_names) {
  if (sum(startsWith(basename(archived), prefix)) != 1L) {
    stop("Missing or duplicated research figure prefix: ", prefix)
  }
}
expected_png_header <- as.raw(c(137, 80, 78, 71, 13, 10, 26, 10))
for (figure in archived) {
  connection <- file(figure, open = "rb")
  header <- readBin(connection, what = "raw", n = 8L)
  close(connection)
  if (!identical(header, expected_png_header)) {
    stop("Invalid PNG signature: ", figure)
  }
}

a <- example_assumptions()
values <- vapply(c(0, .01, .02, .05), function(loss) {
  x <- a
  x$demand_change <- -loss
  evaluate_scenario(x)$incremental_contribution_eur
}, numeric(1))
stopifnot(isTRUE(all.equal(
  unname(values), c(487.5, 281.125, 74.75, -544.375),
  tolerance = 1e-10
)))
stopifnot(abs(-break_even_demand_change(a) -
              (.255 - 150 / 2500) / (8 + .255)) < 1e-12)

bounds <- profit_delta_bounds(a, example_profit_uncertainty())
stopifnot(abs(bounds$contribution_change_min_eur + 1101.5625) < 1e-8)
stopifnot(abs(bounds$contribution_change_max_eur - 1231.25) < 1e-8)
stopifnot(bounds$classification == "sign_depends_on_assumptions")

options <- example_menu_options()
refill <- evaluate_menu_design(a, options$smaller_free_refill)
stopifnot(abs(refill$monthly_contribution_change_eur - 153.125) < 1e-8)
discount <- evaluate_menu_design(a, options$smaller_discount)
stopifnot(discount$monthly_contribution_change_eur < 0)

# Break-even elasticity, external calibration and pilot length.
eps <- quantity_elasticity_threshold(a, 0.15)$break_even_quantity_elasticity
stopifnot(sprintf("%.3f", eps) == "0.147")
stopifnot(abs(quantity_elasticity_limit(a) - 0.2125) < 1e-12)
grid <- quantity_elasticity_threshold(a, seq(0.10, 0.30, by = 0.001))
stopifnot(sprintf("%.3f", max(grid$break_even_quantity_elasticity)) == "0.154")
calibrate <- function(elasticity) {
  x <- a
  x$demand_change <- -implied_customer_loss(0.15, elasticity)
  c(loss = implied_customer_loss(0.15, elasticity),
    delta = evaluate_scenario(x)$incremental_contribution_eur)
}
stopifnot(sprintf("%.2f", calibrate(0.12)[["delta"]]) == "88.92",
          sprintf("%.2f", 100 * calibrate(0.12)[["loss"]]) == "1.93",
          sprintf("%.2f", 100 * calibrate(0.5)[["loss"]]) == "7.80",
          round(calibrate(0.5)[["delta"]]) == -1123)
weeks <- pilot_weeks_needed(-break_even_demand_change(a), c(0.05, 0.10))
stopifnot(identical(round(weeks), c(69, 273)))
stopifnot(abs(discount$monthly_contribution_change_eur + 1387.5) < 1e-8)
refill_half <- options$smaller_free_refill
refill_half$refill_uptake <- 0.5
stopifnot(round(evaluate_menu_design(a, refill_half)$
                  monthly_contribution_change_eur) == -131)

message("Economic regression checks and ", length(archived),
        " archived research figures pass base-R validation.")
