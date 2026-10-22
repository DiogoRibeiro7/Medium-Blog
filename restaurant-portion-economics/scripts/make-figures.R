# Draw the article figures into publication/medium/figures/.
# Usage (from restaurant-portion-economics/):
#   Rscript scripts/make-figures.R [outputs_dir] [figures_dir]
# Model figures always render. Empirical figures render only when the
# real-data outputs exist, i.e. after scripts/download-data.R and the
# analysis scripts have run (see .github/workflows/restaurant-portion-empirical.yml).
source("R/economics.R")
source("R/profit_bounds.R")
source("R/menu_designs.R")
source("R/quantity_response.R")
source("R/finland.R")
source("R/empirical_audit.R")
source("R/figures.R")

args <- commandArgs(trailingOnly = TRUE)
outputs_dir <- if (length(args) >= 1L) args[[1L]] else "outputs"
figures_dir <- if (length(args) >= 2L) {
  args[[2L]]
} else {
  file.path("publication", "medium", "figures")
}
written <- make_article_figures(figures_dir, outputs_dir)
for (name in names(written)) {
  message(sprintf("%-22s %s", name,
                  if (written[[name]]) "written" else
                    "skipped (real-data outputs not found)"))
}
