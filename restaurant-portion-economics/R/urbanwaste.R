# Auditable parsing of the URBAN-WASTE hotel and restaurant CSV (2019).
# Source DOI: 10.5281/zenodo.3355300.
# IMPORTANT: "Waste weight sum" is NOT assigned a mass unit here. Neither
# prices nor ingredient costs nor matching meal-level consumption are observed.

#' Validate the original published commercial-tracking field names.
#'
#' @return Named list of the mandatory raw columns.
urbanwaste_fields <- function() {
  list(
    kitchen = "City / kitchen",
    timestamp = "Timestamp",
    weight = "Waste weight sum",
    portions = c(
      "portions, Whole day", "portions, Midday",
      "portions, Breakfast", "portions, Lunch", "portions, Brunch"
    ),
    orders = c(
      "ordered portions, Whole day", "ordered portions, Midday",
      "ordered portions, Breakfast", "ordered portions, Lunch",
      "ordered portions, Brunch"
    )
  )
}

#' Read publisher file as UTF-8, falling back explicitly to Windows-1252.
#'
#' @param path Path to the checksum-verified URBAN-WASTE original CSV.
#' @return List(data=data.frame, encoding=character, delimiter=character).
#' @details Decoding happens before CSV parsing to avoid truncated rows and
#'   invalid-input scan warnings on non-UTF8 source characters. All columns
#'   remain character until their individual meanings are verified.
read_urbanwaste_source <- function(path) {
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
      !file.exists(path) || file.info(path)$size <= 0 ||
      file.info(path)$size > 10^8) {
    stop("Expected an existing source CSV smaller than 100 MB.",
         call. = FALSE)
  }
  bytes <- readBin(path, what = "raw", n = file.info(path)$size)
  if (any(bytes == as.raw(0))) {
    stop("Source CSV contains NUL bytes.", call. = FALSE)
  }
  raw_text <- rawToChar(bytes)
  decoded <- iconv(raw_text, from = "UTF-8", to = "UTF-8", sub = NA)
  encoding <- "UTF-8"
  if (is.na(decoded)) {
    decoded <- iconv(raw_text, from = "CP1252", to = "UTF-8", sub = NA)
    encoding <- "CP1252"
  }
  if (is.na(decoded)) {
    stop("Neither UTF-8 nor Windows-1252 decodes source losslessly.",
         call. = FALSE)
  }
  delimiters <- c(comma = ",", semicolon = ";", tab = "\t")
  fields <- urbanwaste_fields()
  required <- unique(unlist(fields, use.names = FALSE))
  candidates <- list()
  for (delimiter in names(delimiters)) {
    warnings <- character()
    parsed <- tryCatch(
      withCallingHandlers(
        utils::read.table(
          text = decoded, header = TRUE, sep = delimiters[[delimiter]],
          quote = "\"", comment.char = "", check.names = FALSE,
          stringsAsFactors = FALSE, colClasses = "character",
          na.strings = c("", "NA"), fill = TRUE,
          blank.lines.skip = TRUE, fileEncoding = "UTF-8"
        ),
        warning = function(w) {
          warnings <<- c(warnings, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ), error = function(e) NULL
    )
    if (!is.null(parsed) && !length(warnings) && nrow(parsed) > 0L &&
        all(required %in% names(parsed)) &&
        !anyNA(names(parsed)) && !anyDuplicated(names(parsed))) {
      candidates[[delimiter]] <- parsed
    }
  }
  if (length(candidates) != 1L) {
    stop("Could not determine an unambiguous CSV delimiter and schema; ",
         "inspect original source bytes, headers and warnings.",
         call. = FALSE)
  }
  list(
    data = candidates[[1L]],
    encoding = encoding,
    delimiter = names(candidates)[1L]
  )
}

#' Classify waste-weight strings without assuming their physical units.
#'
#' @param values Character waste-weight strings, NA permitted.
#' @return A character vector identifying lexical number patterns.
#' @details Numeric parsing is intentionally deferred until the publisher's
#'   notation and mass unit are documented. Commas might be decimal
#'   separators or thousands separators; treating them blindly as numeric
#'   could change estimates by orders of magnitude.
classify_weight_format <- function(values) {
  if (!is.character(values)) {
    stop("Waste weights must be character source tokens.", call. = FALSE)
  }
  token <- trimws(values)
  out <- rep("other_or_ambiguous", length(token))
  out[is.na(token) | token == ""] <- "missing"
  valid <- !(is.na(token) | token == "")
  out[valid & grepl("^[0-9]+$", token)] <- "integer"
  out[valid & grepl("^[0-9]+\\.[0-9]+$", token)] <- "dot_decimal_or_grouping"
  out[valid & grepl("^[0-9]+,[0-9]+$", token)] <- "comma_decimal_or_grouping"
  out[valid & grepl(
    "^[0-9]{1,3}(\\.[0-9]{3})+,[0-9]+$", token
  )] <- "dot_grouping_comma_decimal"
  out[valid & grepl(
    "^[0-9]{1,3}(,[0-9]{3})+\\.[0-9]+$", token
  )] <- "comma_grouping_dot_decimal"
  out
}

#' Strictly parse documented integer counts, treating blanks as unknown.
#'
#' @param tokens Raw character vectors for a portions field.
#' @param field Full publisher field label used for errors.
#' @return Numeric vector with NA for unreported counts.
parse_portion_counts <- function(tokens, field) {
  if (!is.character(tokens) || !is.character(field) ||
      length(field) != 1L || is.na(field)) {
    stop("Invalid portion-count arguments.", call. = FALSE)
  }
  tokens <- trimws(tokens)
  missing <- is.na(tokens) | tokens == ""
  valid <- missing | grepl("^[0-9]+$", tokens)
  if (any(!valid)) {
    stop("Unexpected non-integer portion counts in ", field,
         "; review source before interpreting.", call. = FALSE)
  }
  out <- rep(NA_real_, length(tokens))
  out[!missing] <- as.numeric(tokens[!missing])
  if (any(!is.finite(out[!missing]))) {
    stop("Invalid or oversized count in ", field, call. = FALSE)
  }
  out
}

#' Build a descriptive coverage audit for original restaurant/hotel data.
#'
#' @param x Original parsed CSV data.frame (all columns character).
#' @return List with schema, portion_coverage, kitchen_coverage,
#'   weight_patterns, row_coverage, totals, and observation quality flags.
#' @details Unknown weight units and non-equivalent denominator definitions
#'   prevent defensible mass-per-meal calculations at this stage.
audit_urbanwaste <- function(x) {
  fields <- urbanwaste_fields()
  required <- unique(unlist(fields, use.names = FALSE))
  if (!is.data.frame(x) || nrow(x) == 0L ||
      !all(required %in% names(x)) || anyDuplicated(names(x)) ||
      !all(vapply(x[, required, drop = FALSE], is.character, logical(1)))) {
    stop("The source audit expects original, unmodified character columns.",
         call. = FALSE)
  }
  n <- nrow(x)
  weight <- trimws(x[[fields$weight]])
  weight_present <- !is.na(weight) & nzchar(weight)
  kitchen <- trimws(x[[fields$kitchen]])
  kitchen_present <- !is.na(kitchen) & nzchar(kitchen)
  time <- trimws(x[[fields$timestamp]])
  time_present <- !is.na(time) & nzchar(time)
  portion_fields <- c(fields$portions, fields$orders)
  counts <- lapply(portion_fields, function(name) {
    parse_portion_counts(x[[name]], name)
  })
  names(counts) <- portion_fields
  count_matrix <- do.call(cbind, counts)
  colnames(count_matrix) <- portion_fields
  reports_per_row <- rowSums(!is.na(count_matrix))
  positive_per_row <- rowSums(!is.na(count_matrix) & count_matrix > 0)
  raw_patterns <- classify_weight_format(x[[fields$weight]])
  weight_patterns <- as.data.frame(
    table(factor(raw_patterns, levels = c(
      "missing", "integer", "dot_decimal_or_grouping",
      "comma_decimal_or_grouping", "dot_grouping_comma_decimal",
      "comma_grouping_dot_decimal", "other_or_ambiguous"
    ))), stringsAsFactors = FALSE
  )
  names(weight_patterns) <- c("token_format", "rows")

  # These fields are neither interchangeable nor necessarily additive.
  coverage <- do.call(rbind, lapply(portion_fields, function(name) {
    values <- counts[[name]]
    data.frame(
      field = name,
      records = n,
      reported = sum(!is.na(values)),
      strictly_positive = sum(!is.na(values) & values > 0),
      zero = sum(!is.na(values) & values == 0),
      missing = sum(is.na(values)),
      coverage_fraction = mean(!is.na(values)),
      weight_also_reported = sum(!is.na(values) & weight_present),
      stringsAsFactors = FALSE
    )
  }))
  rownames(coverage) <- NULL

  kitchens <- sort(unique(kitchen[kitchen_present]))
  group <- match(kitchen, kitchens)
  kitchen_summary <- do.call(rbind, lapply(seq_along(kitchens), function(i) {
    at <- which(group == i)
    data.frame(
      kitchen_code = sprintf("kitchen_%03d", i),
      records = length(at),
      weight_reported = sum(weight_present[at]),
      timestamp_reported = sum(time_present[at]),
      records_with_any_portion_field = sum(reports_per_row[at] > 0),
      records_with_positive_portion_field = sum(positive_per_row[at] > 0),
      stringsAsFactors = FALSE
    )
  }))
  if (is.null(kitchen_summary)) {
    kitchen_summary <- data.frame(
      kitchen_code = character(), records = integer(),
      weight_reported = integer(), timestamp_reported = integer(),
      records_with_any_portion_field = integer(),
      records_with_positive_portion_field = integer()
    )
  }
  schema <- data.frame(
    field = names(x),
    records = rep(n, ncol(x)),
    missing = vapply(x, function(v) {
      sum(is.na(v) | trimws(v) == "")
    }, integer(1)),
    distinct_nonmissing = vapply(x, function(v) {
      unique_tokens <- unique(trimws(v[!is.na(v) & nzchar(trimws(v))]))
      length(unique_tokens)
    }, integer(1)),
    stringsAsFactors = FALSE
  )

  # Count duplicate kitchen+timestamp keys, but do not collapse: the source
  # might legitimately have multiple waste measurements at one time.
  keys <- paste(kitchen[kitchen_present & time_present],
                time[kitchen_present & time_present], sep = "\034")
  duplicates <- sum(duplicated(keys))
  row_coverage <- as.data.frame(table(
    factor(
      ifelse(reports_per_row == 0, "none",
             ifelse(reports_per_row == 1, "one", "multiple")),
      levels = c("none", "one", "multiple")
    )
  ), stringsAsFactors = FALSE)
  names(row_coverage) <- c("portion_fields_per_row", "records")
  totals <- data.frame(
    records = n,
    distinct_kitchen_labels = length(kitchens),
    distinct_nonmissing_timestamps = length(unique(time[time_present])),
    weight_present = sum(weight_present),
    weight_nonstandard_format = sum(raw_patterns == "other_or_ambiguous"),
    any_portion_field = sum(reports_per_row > 0),
    any_positive_portion_field = sum(positive_per_row > 0),
    weight_and_any_positive_portion = sum(weight_present & positive_per_row > 0),
    duplicate_kitchen_timestamp_keys = duplicates,
    missing_kitchen = sum(!kitchen_present),
    missing_timestamp = sum(!time_present)
  )
  list(
    schema = schema,
    portion_coverage = coverage,
    kitchen_coverage = kitchen_summary,
    weight_patterns = weight_patterns,
    row_coverage = row_coverage,
    totals = totals
  )
}
