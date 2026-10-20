# Strict descriptive analysis of URBAN-WASTE commercial food tracking data.
# Source: Eriksson & Lindgren (2019), Zenodo 3355300.
# The waste-weight unit is not established from the CSV schema alone.
# Results remain in 'source weight units' until a published codebook confirms units.

#' Validate the published commercial file headers.
#' @param x An original CSV data.frame.
#' @return Validated input, invisibly.
validate_commercial_schema <- function(x) {
  required <- c(
    "City / kitchen", "Timestamp", "Waste weight sum",
    "portions, Whole day", "ordered portions, Whole day",
    "portions, Midday", "ordered portions, Midday",
    "portions, Breakfast", "ordered portions, Breakfast",
    "portions, Lunch", "ordered portions, Lunch",
    "portions, Brunch", "ordered portions, Brunch"
  )
  if (!is.data.frame(x) || nrow(x) < 1L ||
      !setequal(names(x), required) || anyDuplicated(names(x))) {
    stop("Commercial source schema changed; inspect before interpreting.",
         call. = FALSE)
  }
  invisible(x)
}

#' Parse a nonnegative decimal without silently dropping unit suffixes.
#' @param x Numeric or character measurement vector.
#' @param label Field name for error messages.
#' @return Numeric vector with missing values retained.
parse_commercial_number <- function(x, label) {
  if (!is.character(label) || length(label) != 1L) {
    stop("label must be one string.", call. = FALSE)
  }
  if (is.factor(x)) x <- as.character(x)
  if (is.numeric(x)) {
    z <- as.numeric(x)
  } else if (is.character(x)) {
    raw <- trimws(x)
    raw[raw == ""] <- NA_character_
    # Accept either a dot decimal or a comma decimal, not thousands separators.
    decimal_comma <- !is.na(raw) & grepl("^[0-9]+,[0-9]+$", raw)
    raw[decimal_comma] <- sub(",", ".", raw[decimal_comma], fixed = TRUE)
    valid <- is.na(raw) | grepl("^[0-9]+([.][0-9]+)?$", raw)
    if (!all(valid)) {
      stop(label, " includes unsupported numeric or unit formats: ",
           paste(unique(raw[!valid])[seq_len(min(5L, sum(!valid)))],
                 collapse = "; "), call. = FALSE)
    }
    z <- as.numeric(raw)
  } else {
    stop(label, " has an unsupported R type.", call. = FALSE)
  }
  if (any(!is.finite(z) & !is.na(z)) || any(z < 0, na.rm = TRUE)) {
    stop(label, " must be nonnegative and finite.", call. = FALSE)
  }
  z
}

#' Convert the original source to typed observations.
#' @param x The publisher's 13-column data.frame.
#' @return Data.frame with original denominators plus parsed waste weight.
normalise_commercial <- function(x) {
  validate_commercial_schema(x)
  kitchen <- trimws(as.character(x[["City / kitchen"]]))
  stamp <- trimws(as.character(x[["Timestamp"]]))
  # Keep incomplete source rows in the audit without assigning them to a
  # fabricated kitchen or date. Downstream coverage excludes missing labels.
  kitchen[is.na(kitchen) | kitchen == ""] <- NA_character_
  stamp[is.na(stamp) | stamp == ""] <- NA_character_
  denominator_names <- grep("^(ordered )?portions, ", names(x), value = TRUE)
  counts <- lapply(denominator_names, function(name) {
    v <- parse_commercial_number(x[[name]], name)
    if (any(v != trunc(v), na.rm = TRUE)) {
      stop("Portion counts must be whole numbers: ", name, call. = FALSE)
    }
    v
  })
  names(counts) <- denominator_names
  result <- data.frame(
    kitchen_id = kitchen, timestamp_source = stamp,
    waste_weight_source_unit = parse_commercial_number(
      x[["Waste weight sum"]], "Waste weight sum"),
    stringsAsFactors = FALSE
  )
  cbind(result, as.data.frame(counts, check.names = FALSE))
}

#' Calculate measurement coverage and safe denominator candidates.
#' @param x Normalised commercial observations.
#' @return List of field audit, kitchen coverage, and eligible denominator rows.
#' @details Multiple meal portions are NOT summed because the scopes may
#' overlap. Ordered quantities are not substitutes for portions served.
audit_commercial <- function(x) {
  required <- c("kitchen_id", "timestamp_source", "waste_weight_source_unit")
  if (!is.data.frame(x) || !all(required %in% names(x))) {
    stop("Expected typed commercial observations.", call. = FALSE)
  }
  measured <- !is.na(x$waste_weight_source_unit)
  candidates <- grep("^portions, ", names(x), value = TRUE)
  portions <- lapply(candidates, function(name) x[[name]])
  names(portions) <- candidates

  # These are individual *candidate* ratios, not automatically comparable
  # across the whole-day and meal-specific denominators.
  ratios <- do.call(rbind, lapply(names(portions), function(name) {
    p <- portions[[name]]
    valid <- measured & is.finite(p) & p > 0 & !is.na(x$kitchen_id) &
      !is.na(x$timestamp_source)
    data.frame(
      denominator_field = name,
      records_with_denominator = sum(is.finite(p)),
      records_eligible_for_ratio = sum(valid),
      distinct_kitchens_with_ratio = length(unique(x$kitchen_id[valid])),
      ratio_unit = "unverified weight units per reported portion",
      stringsAsFactors = FALSE
    )
  }))
  coverage <- do.call(rbind, lapply(split(
    which(!is.na(x$kitchen_id)), x$kitchen_id[
      !is.na(x$kitchen_id)]),
                                    function(rows) {
    w <- x$waste_weight_source_unit[rows]
    data.frame(
      kitchen_id = x$kitchen_id[rows[[1L]]],
      n_records = length(rows),
      n_measured_weight = sum(is.finite(w)),
      n_missing_weight = sum(is.na(w)),
      sum_weight_source_units = if (any(is.finite(w))) {
        sum(w, na.rm = TRUE)
      } else NA_real_,
      stringsAsFactors = FALSE
    )
  }))
  coverage <- coverage[order(-coverage$n_records, coverage$kitchen_id), ]
  rownames(coverage) <- NULL

  field_audit <- data.frame(
    field = names(x),
    n = nrow(x),
    missing = vapply(x, function(v) sum(is.na(v)), integer(1)),
    nonmissing = vapply(x, function(v) sum(!is.na(v)), integer(1)),
    stringsAsFactors = FALSE
  )
  list(
    fields = field_audit,
    kitchens = coverage,
    denominators = ratios,
    missing_kitchen_labels = sum(is.na(x$kitchen_id)),
    missing_timestamps = sum(is.na(x$timestamp_source)),
    raw_weight_unit_confirmed = FALSE
  )
}
