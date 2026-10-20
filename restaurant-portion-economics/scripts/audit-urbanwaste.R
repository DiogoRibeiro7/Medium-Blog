# Publisher-data audit: qualitative structure and denominator coverage only.
source("R/urbanwaste.R")
path <- "data/raw/urbanwaste/Food Tracking Device dataset.csv"
raw <- read_urbanwaste_source(path)
audit <- audit_urbanwaste(raw$data)
dir.create("outputs/urbanwaste", recursive = TRUE, showWarnings = FALSE)
for (name in names(audit)) {
  utils::write.csv(audit[[name]],
    file.path("outputs/urbanwaste", paste0(name, ".csv")),
    row.names = FALSE)
}
cat("Read encoding:", raw$encoding, " delimiter:", raw$delimiter, "\n")
print(audit$totals)
print(audit$portion_coverage)
print(audit$weight_patterns)
cat("No waste-per-portion ratio: source mass units and denominator comparability remain unverified.\n")
