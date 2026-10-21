# Synthetic fixture tests of selection rules, not empirical results.
source("R/commercial_selection.R")
x <- data.frame(
  kitchen_id = c("B", "A", "B", "A", NA_character_, "B"),
  timestamp_source = c("1", "1", "2", "2", "3", "2"),
  waste_weight_source_unit = c(10, 6, 20, 4, 8, NA_real_),
  check.names = FALSE, stringsAsFactors = FALSE
)
x[["portions, Whole day"]] <- c(10, 3, 0, 2, 4, 10)
x[["portions, Breakfast"]] <- c(NA, 2, NA, 1, NA, 1)
out <- commercial_selection_audit(x)
stopifnot(out$source_overview$n_records == 6L)
stopifnot(out$source_overview$n_eligible == 3L)
stopifnot(out$source_overview$n_kitchens_with_eligible == 2L)
stopifnot(out$source_overview$n_repeated_kitchen_timestamp_pairs_among_eligible == 0L)
stopifnot(abs(out$source_overview$provisional_source_units_per_portion - (20 / 15)) < 1e-10)
stopifnot(all(!is.na(out$kitchen_coverage$kitchen)))
stopifnot(all(grepl("^kitchen_[0-9]{3}$", out$kitchen_coverage$kitchen)))
stopifnot(nrow(out$ratio_sensitivity) == 3L)
stopifnot(out$denominator_availability$n_with_weight_and_identifiers[
  out$denominator_availability$denominator == "portions, Breakfast"
] == 2L)
expect_error <- function(expr) {
  stopifnot(inherits(tryCatch(force(expr), error = identity), "error"))
}
bad <- x
bad[["portions, Whole day"]][1] <- -1
expect_error(commercial_selection_audit(bad))
bad <- x
bad$waste_weight_source_unit[1] <- -1
expect_error(commercial_selection_audit(bad))
stopifnot(is.na(safe_commercial_ratio(c(NA_real_), c(0))))
message("Commercial selection tests passed.")
