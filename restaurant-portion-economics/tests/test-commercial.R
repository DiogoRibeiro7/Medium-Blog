# Synthetic fixtures; no observed commercial result is fabricated.
source("R/commercial.R")
columns <- c("City / kitchen","Timestamp","Waste weight sum",
             "portions, Whole day","ordered portions, Whole day",
             "portions, Midday","ordered portions, Midday",
             "portions, Breakfast","ordered portions, Breakfast",
             "portions, Lunch","ordered portions, Lunch",
             "portions, Brunch","ordered portions, Brunch")
x <- as.data.frame(setNames(replicate(length(columns),rep(NA_character_,3L),
                                     simplify=FALSE),columns),
                   check.names=FALSE)
x[["City / kitchen"]] <- c("A","A","B")
x[["Timestamp"]] <- c("day1","day2","day3")
x[["Waste weight sum"]] <- c("1.5","2,5","3")
x[["portions, Whole day"]] <- c("20","0",NA)
typed <- normalise_commercial(x)
stopifnot(nrow(typed)==3L,
          identical(typed$waste_weight_source_unit,c(1.5,2.5,3)))
audit <- audit_commercial(typed)
whole <- audit$denominators[audit$denominators$denominator_field ==
                          "portions, Whole day",]
stopifnot(whole$records_eligible_for_ratio==1L,
          nrow(audit$kitchens)==2L, !audit$raw_weight_unit_confirmed)
expect_error <- function(expr) stopifnot(inherits(
  tryCatch(force(expr),error=identity),"error"))
bad <- x
bad[["Waste weight sum"]][1] <- "1 kg"
expect_error(normalise_commercial(bad))
bad <- x
bad[["portions, Whole day"]][1] <- "1.5"
expect_error(normalise_commercial(bad))
bad <- x
names(bad)[1] <- "unknown"
expect_error(normalise_commercial(bad))
message("Commercial source tests passed.")
