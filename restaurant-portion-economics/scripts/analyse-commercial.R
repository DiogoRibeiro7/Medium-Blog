# Run from project root, after scripts/download-data.R.
source("R/finland.R")
source("R/empirical_audit.R")
source("R/commercial.R")
path <- "data/raw/urbanwaste/Food Tracking Device dataset.csv"
raw <- inspect_urbanwaste_csv(path)$data
typed <- normalise_commercial(raw)
summary <- audit_commercial(typed)
dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
write.csv(summary$fields, "outputs/commercial_fields.csv", row.names = FALSE)
write.csv(summary$kitchens, "outputs/commercial_kitchen_coverage.csv", row.names = FALSE)
write.csv(summary$denominators, "outputs/commercial_denominator_coverage.csv", row.names = FALSE)
cat("Rows:", nrow(typed), "kitchens:", length(unique(typed$kitchen_id)), "\n")
print(summary$denominators)
cat("Weight units are NOT yet established; no kg-per-portion claim.\n")
