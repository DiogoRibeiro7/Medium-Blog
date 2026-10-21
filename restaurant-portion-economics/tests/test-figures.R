# Offline tests for the article figures and the quantity-response threshold.
# Empirical figures are drawn from SYNTHETIC fixtures shaped like the real
# outputs; they are smoke tests of the drawing code, not empirical results.
# Run from restaurant-portion-economics/: Rscript tests/test-figures.R
source("R/economics.R")
source("R/profit_bounds.R")
source("R/menu_designs.R")
source("R/quantity_response.R")
source("R/finland.R")
source("R/empirical_audit.R")
source("R/commercial_selection.R")
source("R/commercial_heterogeneity.R")
source("R/figures.R")

# --- Quantity-response threshold -------------------------------------------
a <- example_assumptions()
x <- quantity_elasticity_threshold(a, 0.15)
stopifnot(abs(x$saving_per_retained_meal_eur - 0.255) < 1e-12)
stopifnot(abs(x$break_even_customer_loss + break_even_demand_change(a)) <
            1e-12)
eps <- x$break_even_quantity_elasticity
stopifnot(abs(eps - log(8.06 / 8.255) / log(0.85)) < 1e-12)
# At the threshold elasticity the contribution change is exactly zero.
at_threshold <- a
at_threshold$demand_change <- -implied_customer_loss(0.15, eps)
stopifnot(abs(evaluate_scenario(at_threshold)$incremental_contribution_eur) <
            1e-8)
stopifnot(abs(quantity_elasticity_limit(a) - 5 * 0.40 * 0.85 / 8) < 1e-12)
no_impl <- quantity_elasticity_threshold(a, c(1e-6, 0.15), FALSE)
stopifnot(abs(no_impl$break_even_quantity_elasticity[1] -
                quantity_elasticity_limit(a)) < 1e-5)
stopifnot(no_impl$break_even_quantity_elasticity[2] > eps)
stopifnot(abs(implied_customer_loss(0.15, 0.12) - (1 - 0.85^0.12)) < 1e-15)
stopifnot(identical(implied_customer_loss(0.2, 0), 0))
error_expected <- function(expr) {
  stopifnot(inherits(tryCatch(force(expr), error = identity), "error"))
}
error_expected(implied_customer_loss(1, 0.5))
error_expected(implied_customer_loss(0.1, -0.5))
error_expected(quantity_elasticity_threshold(a, 0))
error_expected(quantity_elasticity_threshold(a, 0.1, NA))
b <- quantity_response_benchmarks()
stopifnot(nrow(b) == 3L, all(b$elasticity_low <= b$elasticity_high),
          all(grepl("^https://doi.org/", b$source_url)))

# --- Tornado ranges --------------------------------------------------------
t <- profit_tornado(a, example_profit_uncertainty())
stopifnot(nrow(t) == 5L, all(t$delta_min <= t$delta_max))
stopifnot(t$parameter[nrow(t)] == "customer_loss")
stopifnot(abs(attr(t, "baseline_delta") - 487.5) < 1e-9)
joint <- attr(t, "joint")
stopifnot(all(joint$contribution_change_min_eur <= t$delta_min),
          all(joint$contribution_change_max_eur >= t$delta_max))

# --- Synthetic empirical fixtures -----------------------------------------
build_figure_fixtures <- function(outputs_dir) {
  dir.create(outputs_dir, recursive = TRUE, showWarnings = FALSE)
  set.seed(2026L)
  eligible <- c(121, 70, 45, 30, 26, rep(c(10, 8, 6, 5, 4, 3, 2, 1),
                                         length.out = 20))
  eligible <- eligible[seq_len(25)]
  kitchens <- sprintf("K%02d", seq_len(34))
  rows <- lapply(seq_along(kitchens), function(i) {
    n_eligible <- if (i <= 25) eligible[i] else 0
    n_other <- 60
    n <- n_eligible + n_other
    portions <- c(sample(40:200, n_eligible, replace = TRUE),
                  rep(NA, n_other))
    weight <- c(portions[seq_len(n_eligible)] *
                  stats::rlnorm(n_eligible, log(0.2), 0.6),
                stats::runif(n_other, 1, 50))
    data.frame(
      kitchen_id = kitchens[i],
      timestamp_source = sprintf("2018-01-%02d", (seq_len(n) %% 28) + 1),
      waste_weight_source_unit = weight,
      "portions, Whole day" = portions,
      check.names = FALSE, stringsAsFactors = FALSE
    )
  })
  typed <- do.call(rbind, rows)
  audit <- commercial_selection_audit(typed)
  for (name in names(audit)) {
    utils::write.csv(audit[[name]], file.path(outputs_dir,
                                              paste0("commercial_", name, ".csv")),
                     row.names = FALSE)
  }
  het <- commercial_kitchen_heterogeneity(
    utils::read.csv(file.path(outputs_dir, "commercial_kitchen_coverage.csv")),
    utils::read.csv(file.path(outputs_dir, "commercial_source_overview.csv"))
  )
  utils::write.csv(het$leave_one_kitchen_out,
                   file.path(outputs_dir,
                             "commercial_heterogeneity_leave_one_kitchen_out.csv"),
                   row.names = FALSE)

  n <- 155L
  diners <- sample(300:380, n, replace = TRUE)
  menu <- rep(seq_len(25L), length.out = n)
  plate_rate <- c(stats::rnorm(135, 23, 4), stats::rnorm(20, 18, 3))
  kitchen_rate <- c(stats::rnorm(135, 22, 6), stats::rnorm(20, 10, 3))
  fixture <- data.frame(
    "PW" = plate_rate * diners / 1000,
    "PW (g/diner)" = plate_rate,
    "KSW" = kitchen_rate * diners / 1000,
    "KSW (g/diner)" = kitchen_rate,
    "DINERS" = diners, "MENU" = menu, check.names = FALSE
  )
  fixture[136, "PW (g/diner)"] <- 8.25
  fixture[136, "PW"] <- 11.633
  fixture[136, "KSW (g/diner)"] <- 21.77
  fixture[136, "KSW"] <- 2.41
  fixture[136, "DINERS"] <- 340
  daily <- analyse_finland(standardise_finland(fixture))$daily
  utils::write.csv(daily, file.path(outputs_dir, "finland_daily.csv"),
                   row.names = FALSE)
  flags <- find_finland_discrepancies(daily)
  utils::write.csv(finland_quality_sensitivity(daily, flags),
                   file.path(outputs_dir, "finland_sensitivity.csv"),
                   row.names = FALSE)
  invisible(outputs_dir)
}

if (!exists("figure_fixture_only")) {
  root <- tempfile("figures-")
  outputs <- file.path(root, "outputs")
  build_figure_fixtures(outputs)
  model_only <- make_article_figures(file.path(root, "model"),
                                     file.path(root, "missing"))
  stopifnot(sum(model_only) == 6L, !model_only[["finland_daily"]])
  written <- make_article_figures(file.path(root, "all"), outputs)
  stopifnot(all(written))
  files <- file.path(root, "all", article_figure_files())
  stopifnot(all(file.exists(files)), all(file.size(files) > 20000))
  unlink(root, recursive = TRUE)
  message("Figure and quantity-response tests passed (synthetic fixtures).")
}
