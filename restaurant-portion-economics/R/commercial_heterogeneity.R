# Descriptive kitchen composition diagnostics for URBAN-WASTE, 2019.
# Requires outputs/commercial_kitchen_coverage.csv and
# outputs/commercial_source_overview.csv from the publisher-sourced R workflow.
# Units and measurement-window alignment remain unverified.
# Avoid making population, waste-causality, or restaurant-profitability claims.

#' Calculate diagnostic ratios with an explicit zero-denominator rule.
#'
#' @param numerator Nonnegative finite scalar.
#' @param denominator Nonnegative finite scalar.
#' @return Numeric scalar or NA_real_ when denominator equals zero.
safe_summary_ratio <- function(numerator, denominator) {
  if (!is.numeric(numerator) || !is.numeric(denominator) ||
      length(numerator) != 1L || length(denominator) != 1L ||
      anyNA(c(numerator, denominator)) ||
      !all(is.finite(c(numerator, denominator))) ||
      any(c(numerator, denominator) < 0)) {
    stop("Expected finite nonnegative numeric scalars.", call. = FALSE)
  }
  if (denominator == 0) return(NA_real_)
  numerator / denominator
}

#' Describe kitchen concentration and sensitivity to kitchen inclusion.
#'
#' @param kitchens Project-generated, pseudonymised commercial kitchen coverage
#'   data.frame (one row for each kitchen).
#' @param overview Project-generated one-row commercial source overview.
#' @param thresholds Positive integer vector of minimum eligible records needed
#'   to retain a kitchen in a sensitivity scenario.
#' @return A named list with aggregate diagnostics, leave-one-kitchen-out
#'   scenarios and minimum-observation thresholds.
#' @details Kitchen pseudonyms identify the source's grouping structure only.
#'   All ratios use unverified source-weight units per reported whole-day
#'   portion. None is an estimate of restaurant food-waste kg per meal.
commercial_kitchen_heterogeneity <- function(
    kitchens, overview, thresholds = c(1L, 5L, 10L, 20L)) {
  expected <- c(
    "kitchen", "source_records", "eligible_records",
    "eligible_weight_source_units", "eligible_reported_whole_day_portions",
    "provisional_source_units_per_portion"
  )
  if (!is.data.frame(kitchens) || nrow(kitchens) == 0L ||
      !all(expected %in% names(kitchens)) ||
      anyNA(kitchens$kitchen) ||
      anyDuplicated(kitchens$kitchen) ||
      !is.data.frame(overview) || nrow(overview) != 1L ||
      !all(c("n_records", "n_eligible", "reported_weight_sum_eligible",
             "reported_portions_sum_eligible") %in% names(overview))) {
    stop("Missing or malformed upstream commercial summary fields.",
         call. = FALSE)
  }

  counts <- c(
    kitchens$source_records, kitchens$eligible_records,
    kitchens$eligible_weight_source_units,
    kitchens$eligible_reported_whole_day_portions,
    overview$n_records, overview$n_eligible,
    overview$reported_weight_sum_eligible,
    overview$reported_portions_sum_eligible
  )
  if (!is.numeric(counts) || anyNA(counts) ||
      !all(is.finite(counts)) || any(counts < 0) ||
      any(kitchens$eligible_records > kitchens$source_records) ||
      any(kitchens$eligible_records != floor(kitchens$eligible_records)) ||
      any(kitchens$source_records != floor(kitchens$source_records))) {
    stop("Invalid nonnegative source counts and weights.", call. = FALSE)
  }
  if (!is.numeric(thresholds) || length(thresholds) < 1L ||
      anyNA(thresholds) || !all(is.finite(thresholds)) ||
      any(thresholds < 1 | thresholds != floor(thresholds)) ||
      anyDuplicated(thresholds)) {
    stop("Thresholds must be distinct positive integers.", call. = FALSE)
  }
  if (sum(kitchens$source_records) > overview$n_records ||
      sum(kitchens$eligible_records) != overview$n_eligible ||
      !isTRUE(all.equal(
        sum(kitchens$eligible_weight_source_units),
        overview$reported_weight_sum_eligible, tolerance = 1e-7
      )) ||
      !isTRUE(all.equal(
        sum(kitchens$eligible_reported_whole_day_portions),
        overview$reported_portions_sum_eligible, tolerance = 1e-7
      ))) {
    stop("Kitchen totals do not reconcile with source overview.",
         call. = FALSE)
  }

  n <- nrow(kitchens)
  eligible_count <- kitchens$eligible_records
  present <- eligible_count > 0
  total_eligible <- sum(eligible_count)
  weights <- kitchens$eligible_weight_source_units
  denominators <- kitchens$eligible_reported_whole_day_portions
  eligible_kitchens <- kitchens[present, , drop = FALSE]

  if (any(eligible_count[!present] == 0 &
          (weights[!present] > 0 | denominators[!present] > 0))) {
    stop("An ineligible kitchen has eligible weight or portions.",
         call. = FALSE)
  }
  if (any(present & denominators <= 0)) {
    stop("Eligible kitchen needs a positive portion total.",
         call. = FALSE)
  }

  top <- sort(eligible_count, decreasing = TRUE)
  distribution <- if (total_eligible > 0) {
    eligible_count / total_eligible
  } else {
    rep(0, n)
  }
  g <- data.frame(
    n_kitchens = n,
    n_kitchens_with_eligible = sum(present),
    n_kitchens_without_eligible = sum(!present),
    n_eligible_rows = total_eligible,
    eligible_share_of_source_rows = safe_summary_ratio(
      total_eligible, overview$n_records
    ),
    top_one_kitchen_share_of_eligible_rows =
      safe_summary_ratio(sum(head(top, 1L)), total_eligible),
    top_five_kitchen_share_of_eligible_rows =
      safe_summary_ratio(sum(head(top, 5L)), total_eligible),
    selected_row_kitchen_hhi = if (total_eligible > 0) {
      sum(distribution ^ 2)
    } else {
      NA_real_
    },
    provisional_pooled_source_units_per_portion = safe_summary_ratio(
      sum(weights), sum(denominators)
    ),
    provisional_equal_kitchen_mean_of_ratios =
      if (any(present)) {
        mean(eligible_kitchens$provisional_source_units_per_portion)
      } else {
        NA_real_
      },
    measurement_unit_validated = FALSE,
    observation_windows_validated = FALSE,
    stringsAsFactors = FALSE
  )

  W <- sum(weights)
  P <- sum(denominators)
  loo <- data.frame(
    kitchen = kitchens$kitchen[present],
    eligible_records_removed = eligible_count[present],
    remaining_eligible_records = total_eligible - eligible_count[present],
    remaining_kitchens = sum(present) - 1L,
    remaining_source_weight = W - weights[present],
    remaining_portions = P - denominators[present],
    stringsAsFactors = FALSE
  )
  loo$provisional_ratio_without_kitchen <- vapply(
    seq_len(nrow(loo)), function(i) {
      safe_summary_ratio(
        loo$remaining_source_weight[[i]], loo$remaining_portions[[i]]
      )
    }, numeric(1)
  )
  loo <- loo[order(
    loo$provisional_ratio_without_kitchen,
    na.last = TRUE
  ), , drop = FALSE]
  rownames(loo) <- NULL

  threshold_scenarios <- do.call(
    rbind,
    lapply(thresholds, function(k) {
      chosen <- eligible_count >= k
      data.frame(
        minimum_eligible_records_per_kitchen = k,
        eligible_kitchens_retained = sum(chosen),
        eligible_rows_retained = sum(eligible_count[chosen]),
        share_eligible_rows_retained = safe_summary_ratio(
          sum(eligible_count[chosen]), total_eligible
        ),
        total_source_weight_units_retained = sum(weights[chosen]),
        reported_whole_day_portions_retained = sum(denominators[chosen]),
        provisional_source_units_per_portion = safe_summary_ratio(
          sum(weights[chosen]), sum(denominators[chosen])
        ),
        stringsAsFactors = FALSE
      )
    })
  )
  rownames(threshold_scenarios) <- NULL

  list(
    concentration = g,
    leave_one_kitchen_out = loo,
    minimum_records_sensitivity = threshold_scenarios
  )
}
