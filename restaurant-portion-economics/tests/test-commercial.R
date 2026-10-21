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

# Missing labels are retained for provenance but excluded from ratios.
incomplete <- x
incomplete[["City / kitchen"]][1] <- NA_character_
incomplete[["Timestamp"]][2] <- ""
cleaned <- normalise_commercial(incomplete)
stopifnot(is.na(cleaned$kitchen_id[1]),
          is.na(cleaned$timestamp_source[2]))
limited <- audit_commercial(cleaned)
stopifnot(limited$missing_kitchen_labels == 1L,
          limited$missing_timestamps == 1L,
          limited$denominators$records_eligible_for_ratio[
            limited$denominators$denominator_field ==
              "portions, Whole day"] == 0L)

# An independent semantically valid CSV with non-UTF-8 text must be
# decoded without dropping accented kitchen names or source rows.
source("R/empirical_audit.R")
latin <- x
latin[["City / kitchen"]][1] <- "Caf\u00e9"
input_file <- tempfile(fileext = ".csv")
write.csv(latin, input_file, row.names = FALSE, na = "", fileEncoding = "latin1")
decoded <- inspect_urbanwaste_csv(input_file)
stopifnot(nrow(decoded$data) == 3L, ncol(decoded$data) == 13L,
          decoded$data[["City / kitchen"]][1] == "Caf\u00e9")
unlink(input_file)
message("Commercial source encoding and incomplete-row tests passed.")


# Semicolon-separated publisher-style files keep commas inside column names.
# The parser must select the actual delimiter, not merely the widest parse.
semicolon_path <- tempfile(fileext = ".csv")
utils::write.table(x, semicolon_path, sep = ";", quote = TRUE,
                   row.names = FALSE, col.names = TRUE, na = "",
                   fileEncoding = "CP1252")
semicolon <- inspect_urbanwaste_csv(semicolon_path)
stopifnot(identical(semicolon$separator, ";"),
          identical(names(semicolon$data), columns),
          nrow(semicolon$data) == 3L)
unlink(semicolon_path)
