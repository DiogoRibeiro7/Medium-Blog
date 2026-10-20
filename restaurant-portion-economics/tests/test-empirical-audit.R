# Synthetic observations, not real food-waste findings.
source("R/finland.R")
source("R/empirical_audit.R")
n<-155L
x<-data.frame(
plate_kg=rep(10,n),plate_g_diner=rep(100,n),
kitchen_serving_kg=rep(5,n),kitchen_serving_g_diner=rep(50,n),
diners=rep(100L,n),menu=rep(1:25,length.out=n),
day_index=seq_len(n),
phase=ifelse(seq_len(n)<=135,"baseline","intervention")
)
x$plate_kg[136]<-12
flags<-find_finland_discrepancies(x)
stopifnot(nrow(flags)==1L,flags$day_index==136L,flags$stage=="plate")
s<-finland_quality_sensitivity(x,flags)
stopifnot(nrow(s)==6L,all(s$comparable_intervention_days>0L))
stopifnot(abs(s$difference_g_diner[s$stage=="plate" &
 s$interpretation=="reported_per_diner"])<1e-10)
stopifnot(s$difference_g_diner[s$stage=="plate" &
 s$interpretation=="derive_all_rates_from_mass"]>0)
stopifnot(s$comparable_intervention_days[s$stage=="plate" &
 s$interpretation=="exclude_discrepant_days"]==19L)
message("Empirical audit fixture tests passed.")
