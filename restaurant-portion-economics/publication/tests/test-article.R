# Checks reproducibility of the article and its worked economic examples.
# Run from restaurant-portion-economics/ as:
# Rscript publication/tests/test-article.R

source("R/economics.R")
source("R/profit_bounds.R")
source("R/menu_designs.R")
source("R/quantity_response.R")

root <- "publication"
sections <- c(
  "00-question.md", "01-food-system.md", "02-evidence.md",
  "03-mathematics.md", "04-uncertainty.md", "05-menu-design.md",
  "06-business-and-conclusion.md"
)
paths <- file.path(root, "sections", sections)
if (!all(file.exists(paths))) stop("A source manuscript section is missing.")
read_text <- function(path) {
  paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
}
expected <- paste(
  vapply(paths, function(path) trimws(read_text(path)), character(1)),
  collapse = "\n\n"
)
actual <- read_text(file.path(root, "medium", "article.md"))
if (!identical(trimws(actual), trimws(expected))) {
  stop("Medium article differs from manuscript sections. Reassemble it.")
}
word_count <- length(strsplit(trimws(actual), "\\s+")[[1L]])
if (word_count < 5000L) {
  stop("Long-form Medium article has fewer than 5000 words.")
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

# Every figure the article references must exist next to it.
figures <- regmatches(actual, gregexpr("\\(figures/[^)]+\\.png\\)", actual))[[1L]]
figures <- gsub("^\\(|\\)$", "", figures)
if (length(figures) != 10L || anyDuplicated(figures)) {
  stop("The article should reference ten distinct figures.")
}
missing <- figures[!file.exists(file.path(root, "medium", figures))]
if (length(missing) > 0L) {
  stop("Article references missing figures: ", paste(missing, collapse = ", "))
}

for (required in c(
  "3,321", "380", "21.73%", "16.34%", "16.39%", "2.36%",
  "€487.50", "€0.255", "€153.13", "0.147", "€88.92", "€1,387.50",
  "about 69 weeks", "about 273 weeks"
)) {
  if (!grepl(required, actual, fixed = TRUE)) {
    stop("Article omitted a required audited or illustrative value: ",
         required)
  }
}
message("Article matches its sections, ", word_count,
        " words; economic figures and ", length(figures),
        " referenced figures pass base-R regression checks.")
