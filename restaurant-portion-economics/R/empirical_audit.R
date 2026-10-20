# Source-quality diagnostics and descriptive sensitivity analysis.
# This file assumes R/finland.R was sourced first; it uses base R only.
# Figures are observations, not causal estimates or price/portion elasticities.

#' Find mismatches between recorded kg/day and reported grams per diner.
#'
#' @param x Standardised 155-day source data with phase and day_index.
#' @param minimum_tolerance_g Absolute minimum discrepancy tolerance in g/diner.
#' @param relative_tolerance Allowed fractional discrepancy in derived rate.
#' @return Data.frame with one row for each observed, discrepant measurement.
#' @details Source rows are never repaired automatically; both quantities
#'   remain available for sensitivity analysis.
find_finland_discrepancies <- function(
    x, minimum_tolerance_g = 2, relative_tolerance = 0.03) {
  required <- c("day_index", "phase", "menu", "diners",
                "plate_kg", "plate_g_diner",
                "kitchen_serving_kg", "kitchen_serving_g_diner")
  if (!is.data.frame(x) || !all(required %in% names(x))) {
    stop("Expected standardised Finnish observations with phase.", call. = FALSE)
  }
  limits <- c(minimum_tolerance_g, relative_tolerance)
  if (!is.numeric(limits) || length(limits) != 2L ||
      anyNA(limits) || any(!is.finite(limits)) || any(limits < 0)) {
    stop("Tolerances must be finite nonnegative numeric scalars.",
         call. = FALSE)
  }
  if (!is.numeric(x$diners) || anyNA(x$diners) ||
      any(x$diners <= 0)) {
    stop("DINERS must be positive and non-missing.", call. = FALSE)
  }

  stages <- list(
    plate = c("plate_kg", "plate_g_diner"),
    kitchen_serving = c("kitchen_serving_kg", "kitchen_serving_g_diner")
  )
  rows <- lapply(names(stages), function(stage) {
    pair <- stages[[stage]]
    derived <- 1000 * x[[pair[1L]]] / x$diners
    reported <- x[[pair[2L]]]
    observed <- is.finite(derived) & is.finite(reported)
    delta <- abs(derived - reported)
    tolerance <- pmax(minimum_tolerance_g, relative_tolerance * derived)
    flagged <- which(observed & delta > tolerance)
    data.frame(
      stage = rep(stage, length(flagged)),
      day_index = x$day_index[flagged],
      phase = x$phase[flagged],
      menu = x$menu[flagged],
      diners = x$diners[flagged],
      recorded_kg = x[[pair[1L]]][flagged],
      published_g_diner = reported[flagged],
      mass_derived_g_diner = derived[flagged],
      discrepancy_g_diner = delta[flagged],
      threshold_g_diner = tolerance[flagged],
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  result
}

#' Calculate one menu-matched mean contrast for a specified waste measure.
#'
#' @param x Source data.frame with phase, menu, and numeric measure field.
#' @param field Name of a per-diner waste measure.
#' @param omit_days Integer day indices to omit (source data quality review).
#' @return One-row data.frame containing matched descriptive difference.
menu_matched_contrast <- function(x, field, omit_days = integer()) {
  if (!is.data.frame(x) || !all(c("phase", "menu", "day_index") %in%
                                 names(x)) ||
      !is.character(field) || length(field) != 1L ||
      !field %in% names(x) || !is.numeric(x[[field]]) ||
      !is.numeric(omit_days) || anyNA(omit_days)) {
    stop("Invalid menu-match inputs.", call. = FALSE)
  }
  z <- x[!x$day_index %in% omit_days, , drop = FALSE]
  before <- z[z$phase == "baseline", , drop = FALSE]
  after <- z[z$phase == "intervention", , drop = FALSE]
  if (nrow(before) == 0L || nrow(after) == 0L) {
    stop("Both source periods must contain observations.", call. = FALSE)
  }
  prior <- tapply(before[[field]], before$menu, function(v) {
    if (any(is.finite(v))) mean(v[is.finite(v)]) else NA_real_
  })
  comparable <- unname(prior[as.character(after$menu)])
  observed <- after[[field]]
  matched <- is.finite(comparable) & is.finite(observed)
  n <- sum(matched)
  before_mean <- if (n > 0L) mean(comparable[matched]) else NA_real_
  after_mean <- if (n > 0L) mean(observed[matched]) else NA_real_
  change <- after_mean - before_mean
  data.frame(
    measure = field,
    baseline_days_available = nrow(before),
    intervention_days_available = nrow(after),
    comparable_intervention_days = n,
    reference_baseline_g_diner = before_mean,
    intervention_g_diner = after_mean,
    difference_g_diner = change,
    percent_difference = if (is.finite(before_mean) && before_mean > 0) {
      100 * change / before_mean
    } else {
      NA_real_
    },
    stringsAsFactors = FALSE
  )
}

#' Compare interpretations of inconsistent published waste measurements.
#'
#' @param x Standardised daily source observations with phase.
#' @param discrepancies Data.frame returned by find_finland_discrepancies().
#' @return Data.frame containing three scenarios for each waste stream.
#' @details The derived-rate scenario assumes recorded mass and diner counts
#'   are correct; the exclusion scenario leaves out flagged days. None is a
#'   validated correction of publisher data.
finland_quality_sensitivity <- function(x, discrepancies) {
  if (!is.data.frame(discrepancies) ||
      !all(c("stage", "day_index") %in% names(discrepancies))) {
    stop("Provide validated discrepancy output.", call. = FALSE)
  }
  stages <- list(
    plate = c("plate_kg", "plate_g_diner"),
    kitchen_serving = c("kitchen_serving_kg", "kitchen_serving_g_diner")
  )
  results <- list()
  for (stage in names(stages)) {
    pair <- stages[[stage]]
    flagged <- unique(discrepancies$day_index[
      discrepancies$stage == stage
    ])
    reported <- menu_matched_contrast(x, pair[2L])
    reported$interpretation <- "reported_per_diner"
    excluded <- menu_matched_contrast(x, pair[2L], omit_days = flagged)
    excluded$interpretation <- "exclude_discrepant_days"
    mass_based <- x
    mass_based[[pair[2L]]] <- 1000 * x[[pair[1L]]] / x$diners
    derived <- menu_matched_contrast(mass_based, pair[2L])
    derived$interpretation <- "derive_all_rates_from_mass"
    combined <- rbind(reported, excluded, derived)
    combined$stage <- stage
    combined$flagged_day_count <- length(flagged)
    results[[stage]] <- combined
  }
  out <- do.call(rbind, results)
  rownames(out) <- NULL
  out[, c("stage", "interpretation", "measure",
          "flagged_day_count", "baseline_days_available",
          "intervention_days_available", "comparable_intervention_days",
          "reference_baseline_g_diner", "intervention_g_diner",
          "difference_g_diner", "percent_difference")]
}

#' Inspect a commercial food-waste CSV without presuming its meaning or units.
#'
#' @param path Existing, publisher-verified CSV filename.
#' @return List containing a parsed data.frame and field-quality audit.
#' @details Try explicit CSV delimiters, rejecting ambiguous structures.
#'   Does not treat any numerical column as kg or EUR without a codebook.
inspect_urbanwaste_csv <- function(path) {
  if (!is.character(path) || length(path) != 1L ||
      is.na(path) || !file.exists(path)) {
    stop("Provide a downloaded URBAN-WASTE CSV path.", call. = FALSE)
  }
  separators <- c(",", ";", "\t")
  candidates <- lapply(separators, function(sep) {
    tryCatch(
      utils::read.table(
        path, header = TRUE, sep = sep, quote = "\"",
        check.names = FALSE, stringsAsFactors = FALSE,
        fileEncoding = "UTF-8-BOM", blank.lines.skip = TRUE,
        comment.char = "", fill = FALSE
      ),
      error = function(e) NULL
    )
  })
  ncols <- vapply(candidates, function(x) {
    if (is.data.frame(x) && nrow(x) > 0L) ncol(x) else 0L
  }, integer(1))
  if (max(ncols) <= 1L || sum(ncols == max(ncols)) != 1L) {
    stop("Could not resolve a unique tabular CSV structure.", call. = FALSE)
  }
  data <- candidates[[which.max(ncols)]]
  if (anyNA(names(data)) || anyDuplicated(names(data))) {
    stop("Missing or duplicated source field labels.", call. = FALSE)
  }
  schema <- data.frame(
    column = names(data),
    r_type = vapply(data, function(v) paste(class(v), collapse = "/"),
                    character(1)),
    rows = rep(nrow(data), ncol(data)),
    missing = vapply(data, function(v) sum(is.na(v)), integer(1)),
    distinct_nonmissing = vapply(
      data, function(v) length(unique(v[!is.na(v)])), integer(1)
    ),
    stringsAsFactors = FALSE
  )
  list(data = data, schema = schema, separator = separators[[which.max(ncols)]])
}
