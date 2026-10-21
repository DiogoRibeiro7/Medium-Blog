source("R/economics.R")
source("R/public_cost_anchors.R")
x <- read_food_cost_benchmarks()
stopifnot(nrow(x)==4L,all(x$share_of_sales>0.30),
          all(x$share_of_sales<0.35))
base <- example_assumptions()
r <- cost_benchmark_scenarios(base,x,demand_losses=c(0,0.02))
stopifnot(nrow(r)==8L,all(is.finite(r$scenario_contribution_change_eur)))
for(i in seq_len(nrow(r))) {
  a<-base
  a$ingredient_cost_eur<-a$price_eur*r$industry_food_beverage_sales_ratio[i]
  a$demand_change<--r$assumed_customer_loss[i]
  stopifnot(abs(evaluate_scenario(a)$incremental_contribution_eur -
                 r$scenario_contribution_change_eur[i])<1e-8)
}
stopifnot(all(r$evidence_role ==
 "cross_market_industry_proxy_scenario_not_estimate"))
message("Public cost proxy tests passed.")
