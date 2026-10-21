# Explicit cross-market cost-proxy scenarios, not fitted effects.
source("R/economics.R")
source("R/public_cost_anchors.R")
sources <- read_food_cost_benchmarks()
scenarios <- cost_benchmark_scenarios(example_assumptions(),sources)
dir.create("outputs",showWarnings=FALSE)
write.csv(scenarios,"outputs/public_cost_proxy_scenarios.csv",row.names=FALSE)
write.csv(sources,"outputs/public_cost_source_register.csv",row.names=FALSE)
cat("All derived EUR figures are hypothetical, not restaurant estimates.\n")
