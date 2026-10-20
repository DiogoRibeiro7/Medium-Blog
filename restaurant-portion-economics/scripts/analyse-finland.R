# Run source-specific Finnish analysis after downloading the publisher workbook.
# Usage (from project root): Rscript scripts/analyse-finland.R
source("R/finland.R")
dir.create("outputs", recursive = TRUE, showWarnings = FALSE)
files <- list.files("data/raw/finland", pattern = "\\.xlsx?$",
                    full.names = TRUE, recursive = TRUE, ignore.case = TRUE)
if (length(files) != 1L) {
  stop("Expected exactly one Finnish Excel workbook; found ", length(files),
       ". Download source record 20825233 and inspect file list.", call. = FALSE)
}
daily <- read_finland_workbook(files[[1L]])
analysis <- analyse_finland(daily)
utils::write.csv(analysis$daily, "outputs/finland_daily.csv", row.names = FALSE)
utils::write.csv(analysis$period, "outputs/finland_period.csv", row.names = FALSE)
utils::write.csv(analysis$menu_matched, "outputs/finland_menu_matched.csv",
                 row.names = FALSE)
utils::write.csv(analysis$diagnostics, "outputs/finland_mass_diagnostics.csv",
                 row.names = FALSE)
if (any(analysis$diagnostics$n_discrepant > 0L)) {
  warning("Mass/per-diner values are not fully consistent. Review diagnostics.")
}
message("Finnish descriptive observations written to outputs/. No causal effect estimated.")
