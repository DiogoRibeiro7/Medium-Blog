# Reproduce kitchen heterogeneity from real-data workflow CSV summaries.
source("R/commercial_heterogeneity.R")
root <- "outputs"
k <- utils::read.csv(file.path(root,"commercial_kitchen_coverage.csv"))
o <- utils::read.csv(file.path(root,"commercial_source_overview.csv"))
results <- commercial_kitchen_heterogeneity(k,o)
for (name in names(results)) {
  utils::write.csv(results[[name]],
    file.path(root,paste0("commercial_heterogeneity_",name,".csv")),
    row.names=FALSE)
}
print(results$concentration)
print(results$minimum_records_sensitivity)
cat("Diagnostic ratios only: waste units and time windows unverified.\n")
