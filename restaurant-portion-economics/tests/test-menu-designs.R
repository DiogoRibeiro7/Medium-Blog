source("R/economics.R")
source("R/menu_designs.R")
a <- example_assumptions()
opts <- example_menu_options()
for (o in opts) validate_menu_option(o)
base <- opts$smaller_same_price
base$customer_loss <- 0
base$monthly_implementation_eur <- 150
actual <- evaluate_menu_design(a,base)
reference <- a
reference$portion_reduction <- base$portion_reduction
reference$demand_change <- 0
reference$monthly_implementation_eur <- 150
stopifnot(abs(actual$monthly_contribution_change_eur -
 evaluate_scenario(reference)$incremental_contribution_eur)<1e-9)
# Algebraic identity: zero change at threshold when positive contribution.
for(o in opts){
 result<-evaluate_menu_design(a,o)
 t<-result$maximum_customer_loss_for_break_even
 if(is.finite(t) && t>=0 && t<1) {
   o$customer_loss<-t
   stopifnot(abs(evaluate_menu_design(a,o)$monthly_contribution_change_eur)<1e-7)
 }
}
grid<-menu_design_sensitivity(a,opts,
 customer_losses=c(0,0.02),refill_uptakes=c(0,0.5))
stopifnot(nrow(grid)==8L,all(is.finite(grid$monthly_contribution_change_eur)))
stopifnot(all(grid$evidence_status=="hypothetical_scenario_not_demand_estimate"))
refill<-opts$smaller_free_refill
refill$refill_uptake<-1
refill$refill_restoration_fraction<-1
refill$refill_service_cost_eur<-0
stopifnot(abs(evaluate_menu_design(a,refill)$ingredient_saving_after_refills_per_meal_eur)<1e-10)
fail<-function(expr) stopifnot(inherits(tryCatch(force(expr),error=identity),"error"))
invalid<-opts$smaller_discount
invalid$discount_fraction<-1
fail(evaluate_menu_design(a,invalid))
invalid<-opts$smaller_same_price
invalid$refill_uptake<-0.4
fail(evaluate_menu_design(a,invalid))
message("Menu design tests passed.")
