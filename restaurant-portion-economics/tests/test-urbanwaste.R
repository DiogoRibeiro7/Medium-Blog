# Offline artificial fixtures. No real commercial findings asserted.
source("R/urbanwaste.R")
headers <- urbanwaste_fields()
fixture <- as.data.frame(matrix(NA_character_, nrow = 4L, ncol = 13L))
names(fixture) <- c(headers$kitchen, headers$timestamp, headers$weight,
                    as.vector(rbind(headers$portions, headers$orders)))
fixture[[headers$kitchen]] <- c("A", "A", "B", "B")
fixture[[headers$timestamp]] <- c("t1", "t1", "t2", "t3")
fixture[[headers$weight]] <- c("1,25", "2.5", "10", NA_character_)
fixture[[headers$portions[1]]] <- c("12", NA, "0", NA)
fixture[[headers$portions[2]]] <- c(NA, "10", NA, NA)
patterns <- classify_weight_format(fixture[[headers$weight]])
stopifnot(identical(patterns,
  c("comma_decimal_or_grouping", "dot_decimal_or_grouping", "integer", "missing")))
audit <- audit_urbanwaste(fixture)
stopifnot(nrow(audit$schema) == 13L, audit$totals$records == 4L)
stopifnot(audit$totals$distinct_kitchen_labels == 2L)
stopifnot(audit$totals$any_portion_field == 3L)
stopifnot(audit$totals$any_positive_portion_field == 2L)
stopifnot(audit$totals$duplicate_kitchen_timestamp_keys == 1L)
stopifnot(sum(audit$row_coverage$records) == 4L)
stopifnot(sum(audit$weight_patterns$rows) == 4L)
error_expected <- function(x) stopifnot(inherits(
  tryCatch(force(x), error = identity), "error"))
error_expected(parse_portion_counts(c("1,5"), "test"))
error_expected(parse_portion_counts(c("-1"), "test"))
bad <- fixture
bad[[headers$portions[1]]][1] <- "garbage"
error_expected(audit_urbanwaste(bad))
message("All commercial-waste coverage fixture tests passed.")
