# Run from restaurant-portion-economics after original publisher download.
source("R/empirical_audit.R")
source("R/commercial.R")
source("R/commercial_selection.R")
original <- inspect_urbanwaste_csv(
  "data/raw/urbanwaste/Food Tracking Device dataset.csv"
)
typed <- normalise_commercial(original$data)
audited <- commercial_selection_audit(typed)
dir.create("outputs", recursive = TRUE, showWarnings = FALSE)
for (name in names(audited)) {
  utils::write.csv(
    audited[[name]], file.path("outputs", paste0("commercial_", name, ".csv")),
    row.names = FALSE
  )
}
print(audited$source_overview)
print(audited$denominator_availability)
message("Provisional ratios use UNVERIFIED source units and time windows.")
