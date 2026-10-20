# Finnish school food-waste observations: a descriptive, not causal, analysis.
# Published units: PW and KSW in kg/day; per-diner columns in g/diner/day.
# Source: Kuusisto, Latva & Wilen (2026), DOI 10.5281/zenodo.20825233.

#' Normalise column names for comparison with documented source variables.
#'
#' @param x Character vector of field labels.
#' @return Character vector containing lowercase alphanumeric names.
normalise_finland_header <- function(x) {
  if (!is.character(x)) stop("Header labels must be characters.", call. = FALSE)
  gsub("[^a-z0-9]", "", tolower(trimws(x)))
}

#' Return the allowed aliases for explicitly documented measurements.
#'
#' @return Named list of character vectors of accepted header labels.
finland_source_fields <- function() {
  list(
    plate_kg = c("PW", "PW (kg)", "PW/kg", "PW kg"),
    plate_g_diner = c("PW/g/diner", "PW (g/diner)", "PW g diner"),
    kitchen_serving_kg = c("KSW", "KSW (kg)", "KSW/kg", "KSW kg"),
    kitchen_serving_g_diner = c("KSW/g/diner", "KSW (g/diner)",
                                "KSW g diner"),
    diners = c("DINERS"),
    menu = c("MENU")
  )
}

#' Resolve documented fields without guessing which measurements they mean.
#'
#' @param headers Character column names from the source.
#' @return Named integer vector of exactly one matching column per field.
resolve_finland_fields <- function(headers) {
  if (!is.character(headers) || anyNA(headers)) {
    stop("Column names must be a character vector without NA.", call. = FALSE)
  }
  keys <- normalise_finland_header(headers)
  aliases <- finland_source_fields()
  found <- vapply(names(aliases), function(field) {
    matches <- which(keys %in% normalise_finland_header(aliases[[field]]))
    if (length(matches) != 1L) {
      stop(
        "Expected exactly one source column for ", field,
        "; found ", length(matches), ". Actual fields: ",
        paste(headers, collapse = " | "),
        call. = FALSE
      )
    }
    matches
  }, integer(1))
  if (anyDuplicated(unname(found))) {
    stop("Two source variables resolved to one column.", call. = FALSE)
  }
  found
}

#' Parse strictly nonnegative measurements, allowing missing measurements.
#'
#' @param x Numeric or character vector of observations.
#' @param field Source-field name for diagnostics.
#' @return Numeric vector (NA retained).
parse_nonnegative <- function(x, field) {
  if (is.factor(x)) x <- as.character(x)
  if (is.numeric(x)) {
    result <- as.numeric(x)
  } else if (is.character(x)) {
    raw <- trimws(x)
    raw[raw == ""] <- NA_character_
    # Excel occasionally stores comma-decimal numbers as text.
    comma_decimal <- !is.na(raw) & grepl("^[0-9]+,[0-9]+$", raw)
    raw[comma_decimal] <- sub(",", ".", raw[comma_decimal], fixed = TRUE)
    pattern <- "^([0-9]+([.][0-9]+)?|[.][0-9]+)$"
    invalid <- !is.na(raw) & !grepl(pattern, raw)
    if (any(invalid)) {
      stop("Non-numeric or ambiguous values in ", field, ".", call. = FALSE)
    }
    result <- as.numeric(raw)
  } else {
    stop("Unsupported source type in ", field, ".", call. = FALSE)
  }
  if (any(!is.finite(result) & !is.na(result)) ||
      any(result < 0, na.rm = TRUE)) {
    stop("Expected finite nonnegative values in ", field, ".", call. = FALSE)
  }
  result
}

#' Standardise a Finnish observation table with explicit mass units.
#'
#' @param x Source data.frame with documented PW/KSW/DINERS/MENU columns.
#' @return Standardised data.frame; row order remains source order.
standardise_finland <- function(x) {
  if (!is.data.frame(x) || nrow(x) == 0L) {
    stop("Expected a nonempty source data.frame.", call. = FALSE)
  }
  indices <- resolve_finland_fields(names(x))
  numbers <- lapply(names(indices), function(field) {
    parse_nonnegative(x[[indices[[field]]]], field)
  })
  names(numbers) <- names(indices)
  result <- as.data.frame(numbers, stringsAsFactors = FALSE)
  if (anyNA(result$diners) || any(result$diners <= 0) ||
      any(result$diners != floor(result$diners))) {
    stop("DINERS must be a positive integer on every day.", call. = FALSE)
  }
  if (anyNA(result$menu) || any(result$menu < 1 | result$menu > 25) ||
      any(result$menu != floor(result$menu))) {
    stop("MENU must contain integers from 1 to 25 on every day.",
         call. = FALSE)
  }
  result$diners <- as.integer(result$diners)
  result$menu <- as.integer(result$menu)
  result$day_index <- seq_len(nrow(result))
  result
}

#' Read the Excel sheet whose header contains all six documented measurements.
#'
#' @param path Existing original publisher Excel workbook.
#' @return Standardised data.frame.
#' @details The header may follow introductory rows. Search at most the
#'   first 30 rows of each sheet; reject ambiguous or undocumented structures.
read_finland_workbook <- function(path) {
  if (!is.character(path) || length(path) != 1L ||
      is.na(path) || !file.exists(path)) {
    stop("Provide an existing Finnish source workbook.", call. = FALSE)
  }
  if (!requireNamespace("readxl", quietly = TRUE)) {
    stop("Install readxl before reading the Finnish workbook.", call. = FALSE)
  }
  candidates <- list()
  for (sheet in readxl::excel_sheets(path)) {
    preview <- suppressMessages(readxl::read_excel(
      path, sheet = sheet, col_names = FALSE, col_types = "text",
      n_max = 30, .name_repair = "minimal"
    ))
    if (nrow(preview) == 0L) next
    for (i in seq_len(nrow(preview))) {
      labels <- as.character(unlist(preview[i, , drop = TRUE],
                                    use.names = FALSE))
      matched <- tryCatch(resolve_finland_fields(labels),
                          error = function(e) NULL)
      if (!is.null(matched)) {
        candidates[[length(candidates) + 1L]] <- list(
          sheet = sheet, header_row = i
        )
      }
    }
  }
  if (length(candidates) != 1L) {
    stop("Expected one unambiguous header across workbook sheets; found ",
         length(candidates), ". Inspect the workbook manually.",
         call. = FALSE)
  }
  selected <- candidates[[1L]]
  raw <- as.data.frame(readxl::read_excel(
    path, sheet = selected$sheet, skip = selected$header_row - 1L,
    .name_repair = "minimal"
  ))
  standardise_finland(raw)
}

#' Quantify consistency between total mass and reported per-diner values.
#'
#' @param x Standardised observation table.
#' @return Per-measure count and discrepancy diagnostics.
#' @details Publisher rounds per-diner values. The tolerance is a diagnostic,
#'   not permission to overwrite source values.
finland_mass_diagnostics <- function(x) {
  measures <- list(
    plate = c("plate_kg", "plate_g_diner"),
    kitchen_serving = c("kitchen_serving_kg", "kitchen_serving_g_diner")
  )
  do.call(rbind, lapply(names(measures), function(name) {
    pair <- measures[[name]]
    predicted <- 1000 * x[[pair[[1L]]]] / x$diners
    reported <- x[[pair[[2L]]]]
    valid <- is.finite(predicted) & is.finite(reported)
    deviation <- abs(predicted[valid] - reported[valid])
    tolerance <- pmax(2, 0.03 * predicted[valid])
    data.frame(
      measure = name,
      n_checked = sum(valid),
      n_discrepant = sum(deviation > tolerance),
      max_absolute_difference_g_diner = if (any(valid)) {
        max(deviation)
      } else {
        NA_real_
      }
    )
  }))
}

#' Describe observed waste by period, retaining mass and exposure units.
#'
#' @param x Standardised data with a "phase" column.
#' @return Period-by-measure descriptive data.frame.
summarise_finland_periods <- function(x) {
  measures <- list(
    plate = c("plate_kg", "plate_g_diner"),
    kitchen_serving = c("kitchen_serving_kg", "kitchen_serving_g_diner")
  )
  result <- list()
  for (phase in c("baseline", "intervention")) {
    subset <- x[x$phase == phase, , drop = FALSE]
    for (measure in names(measures)) {
      pair <- measures[[measure]]
      mass <- subset[[pair[[1L]]]]
      rate <- subset[[pair[[2L]]]]
      valid_mass <- is.finite(mass) & !is.na(subset$diners)
      valid_rate <- is.finite(rate)
      total_kg <- if (any(valid_mass)) sum(mass[valid_mass]) else NA_real_
      pooled_g_diner <- if (any(valid_mass)) {
        1000 * total_kg / sum(subset$diners[valid_mass])
      } else {
        NA_real_
      }
      result[[length(result) + 1L]] <- data.frame(
        phase = phase,
        measure = measure,
        days_total = nrow(subset),
        days_with_mass = sum(valid_mass),
        days_with_reported_rate = sum(valid_rate),
        diners_with_mass = sum(subset$diners[valid_mass]),
        total_waste_kg = total_kg,
        pooled_g_diner = pooled_g_diner,
        mean_daily_reported_g_diner = if (any(valid_rate)) {
          mean(rate[valid_rate])
        } else {
          NA_real_
        }
      )
    }
  }
  do.call(rbind, result)
}

#' Compute descriptive contrasts within the observed intervention menu mix.
#'
#' @param x Standardised table with phase (baseline, intervention).
#' @return Two-row data.frame for plate and kitchen/serving waste.
#' @details This adjusts for menu ID only. It is NOT a treatment effect:
#'   time, attendance, awareness and other interventions remain confounded.
summarise_finland_menu_matched <- function(x) {
  measures <- c("plate_g_diner", "kitchen_serving_g_diner")
  before <- x[x$phase == "baseline", , drop = FALSE]
  after <- x[x$phase == "intervention", , drop = FALSE]
  rows <- lapply(measures, function(field) {
    # Baseline menu-specific means are compared against the same menu IDs
    # appearing on actual intervention days, without modelling assumptions.
    baseline_means <- tapply(before[[field]], before$menu, function(z) {
      valid <- is.finite(z)
      if (any(valid)) mean(z[valid]) else NA_real_
    })
    baseline_reference <- unname(
      baseline_means[as.character(after$menu)]
    )
    observed <- after[[field]]
    valid <- is.finite(baseline_reference) & is.finite(observed)
    baseline_mean <- if (any(valid)) {
      mean(baseline_reference[valid])
    } else {
      NA_real_
    }
    observed_mean <- if (any(valid)) mean(observed[valid]) else NA_real_
    difference <- observed_mean - baseline_mean
    data.frame(
      measure = field,
      baseline_days = nrow(before),
      intervention_days = nrow(after),
      intervention_days_menu_matched = sum(valid),
      reference_baseline_g_diner = baseline_mean,
      observed_intervention_g_diner = observed_mean,
      difference_g_diner = difference,
      percent_difference = if (is.finite(baseline_mean) &&
                               baseline_mean > 0) {
        100 * difference / baseline_mean
      } else {
        NA_real_
      },
      interpretation = "descriptive, not causal",
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

#' Run the source-defined 135-day/20-day descriptive phase comparison.
#'
#' @param x Standardised daily table.
#' @return List of daily data, period summaries, menu-matched contrasts and
#'   consistency diagnostics.
analyse_finland <- function(x) {
  if (!is.data.frame(x) || nrow(x) != 155L ||
      !all(c("plate_kg", "plate_g_diner", "kitchen_serving_kg",
             "kitchen_serving_g_diner", "diners", "menu", "day_index") %in%
           names(x))) {
    stop("Expected 155 standardised daily rows with documented fields.",
         call. = FALSE)
  }
  if (!identical(x$day_index, seq_len(155L))) {
    stop("Source rows must preserve their original chronological order.",
         call. = FALSE)
  }
  x$phase <- ifelse(x$day_index <= 135L, "baseline", "intervention")
  list(
    daily = x,
    period = summarise_finland_periods(x),
    menu_matched = summarise_finland_menu_matched(x),
    diagnostics = finland_mass_diagnostics(x)
  )
}
