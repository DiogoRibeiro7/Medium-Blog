# Synthetic aggregated counts, not observed restaurant data.
source("R/commercial_heterogeneity.R")
k <- data.frame(
  kitchen=c("kitchen_001","kitchen_002","kitchen_003"),
  source_records=c(8L,12L,10L), eligible_records=c(4L,2L,0L),
  eligible_weight_source_units=c(20,4,0),
  eligible_reported_whole_day_portions=c(100,20,0),
  provisional_source_units_per_portion=c(0.2,0.2,NA_real_)
)
o <- data.frame(n_records=30L,n_eligible=6L,
  reported_weight_sum_eligible=24,
  reported_portions_sum_eligible=120)
res <- commercial_kitchen_heterogeneity(k,o,thresholds=c(1L,3L))
stopifnot(res$concentration$n_kitchens_without_eligible == 1L)
stopifnot(abs(res$concentration$top_one_kitchen_share_of_eligible_rows-4/6)<1e-10)
stopifnot(abs(res$concentration$provisional_pooled_source_units_per_portion-0.2)<1e-10)
stopifnot(nrow(res$leave_one_kitchen_out)==2L)
stopifnot(res$minimum_records_sensitivity$eligible_rows_retained[2]==4L)
bad <- o; bad$n_eligible <- 7L
stopifnot(inherits(tryCatch(commercial_kitchen_heterogeneity(k,bad),
 error=identity),"error"))
message("Commercial heterogeneity fixtures passed.")
