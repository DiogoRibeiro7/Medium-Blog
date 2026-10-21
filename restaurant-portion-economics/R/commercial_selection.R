# Observational selection and provisional ratio diagnostics for URBAN-WASTE.
# Source: Eriksson and Lindgren (2019), Zenodo 3355300.
#
# Do not interpret the source-weight column as kg or as a particular waste
# stream; the published CSV metadata do not independently identify either.
# A "Whole day" count and a row's waste sum may still refer to mismatched
# measurement windows. Ratios below are diagnostic, not validated KPIs.

#' Validate typed commercial observations from normalise_commercial().
#'
#' @param x A typed data.frame with kitchen, timestamp, waste and whole-day
#'   portion columns.
#' @return The input invisibly; errors on invalid types and values.
validate_commercial_selection_input <- function(x) {
  required <- c(
    "kitchen_id", "timestamp_source", "waste_weight_source_unit",
    "portions, Whole day"
  )
  if (!is.data.frame(x) || nrow(x) < 1L ||
      !all(required %in% names(x))) {
    stop("Expected nonempty typed commercial observations.", call. = FALSE)
  }
  if (!is.character(x$kitchen_id) ||
      !is.character(x$timestamp_source)) {
    stop("Kitchen and time identifiers must be character columns.",
         call. = FALSE)
  }
  weight <- x$waste_weight_source_unit
  portions <- x[["portions, Whole day"]]
  if (!is.numeric(weight) || !is.numeric(portions)) {
    stop("Waste and portion counts must be numeric.", call. = FALSE)
  }
  if (any(!is.finite(weight) & !is.na(weight)) ||
      any(weight < 0, na.rm = TRUE) ||
      any(!is.finite(portions) & !is.na(portions)) ||
      any(portions < 0, na.rm = TRUE) ||
      any(portions != floor(portions), na.rm = TRUE)) {
    stop("Invalid waste weights or nonnegative integer portion counts.",
         call. = FALSE)
  }
  invisible(x)
}

#' Calculate a source-unit ratio on exactly the selected records.
#'
#' @param weight Numeric vector of source-reported waste values.
#' @param portions Positive, numeric portion counts of equal length.
#' @return Numeric scalar or NA_real_ if no positive denominator is present.
#' @details The ratio of sums differs from the unweighted mean of row ratios.
safe_commercial_ratio <- function(weight, portions) {
  if (!is.numeric(weight) || !is.numeric(portions) ||
      length(weight) != length(portions)) {
    stop("weight and portions must be equally sized numeric vectors.",
         call. = FALSE)
  }
  good <- is.finite(weight) & is.finite(portions) & portions > 0
  if (!any(good)) return(NA_real_)
  sum(weight[good]) / sum(portions[good])
}

#' Produce a robust, non-inferential selection and ratio sensitivity audit.
#'
#' @param x Typed commercial data from normalise_commercial().
#' @return A list: source_overview, kitchen_coverage, ratio_sensitivity,
#'   and denominator_availability.
#' @details A kitchen identifier remains in memory but is replaced by a
#'   stable within-source pseudonym in outputs. No input rows or IDs are
#'   modified. Only whole-day denominator candidates appear in the ratio
#'   calculation; meal-specific denominators are audited but never summed.
commercial_selection_audit <- function(x) {
  validate_commercial_selection_input(x)
  w <- x$waste_weight_source_unit
  p <- x[["portions, Whole day"]]
  kitchen <- x$kitchen_id
  timestamp <- x$timestamp_source

  labelled <- !is.na(kitchen) & nzchar(trimws(kitchen)) &
    !is.na(timestamp) & nzchar(trimws(timestamp))
  observed_w <- is.finite(w)
  observed_p <- is.finite(p)
  eligible <- labelled & observed_w & observed_p & p > 0
  ratio <- w[eligible] / p[eligible]

  keys <- sort(unique(kitchen[!is.na(kitchen) & nzchar(trimws(kitchen))]))
  anon <- setNames(sprintf("kitchen_%03d", seq_along(keys)), keys)
  kitchen_coverage <- do.call(
    rbind,
    lapply(keys, function(id) {
      ix <- which(!is.na(kitchen) & kitchen == id)
      subset_eligible <- ix[eligible[ix]]
      weight_total <- if (any(observed_w[ix])) {
        sum(w[ix][observed_w[ix]])
      } else {
        NA_real_
      }
      data.frame(
        kitchen = unname(anon[[id]]),
        source_records = length(ix),
        complete_time_and_kitchen = sum(labelled[ix]),
        measured_weight_records = sum(observed_w[ix]),
        whole_day_count_records = sum(observed_p[ix]),
        eligible_records = length(subset_eligible),
        eligible_share_of_kitchen_records = length(subset_eligible) / length(ix),
        all_observed_weight_source_units = weight_total,
        eligible_weight_source_units = sum(w[subset_eligible]),
        eligible_reported_whole_day_portions = sum(p[subset_eligible]),
        provisional_source_units_per_portion = safe_commercial_ratio(
          w[subset_eligible], p[subset_eligible]
        ),
        stringsAsFactors = FALSE
      )
    })
  )
  if (is.null(kitchen_coverage)) {
    kitchen_coverage <- data.frame(
      kitchen = character(), source_records = integer(),
      complete_time_and_kitchen = integer(),
      measured_weight_records = integer(),
      whole_day_count_records = integer(), eligible_records = integer(),
      eligible_share_of_kitchen_records = numeric(),
      all_observed_weight_source_units = numeric(),
      eligible_weight_source_units = numeric(),
      eligible_reported_whole_day_portions = numeric(),
      provisional_source_units_per_portion = numeric()
    )
  }
  rownames(kitchen_coverage) <- NULL

  eligible_ids <- which(eligible)
  duplicate_pairs <- if (length(eligible_ids) > 0L) {
    sum(duplicated(
      data.frame(
        kitchen = kitchen[eligible_ids],
        timestamp = timestamp[eligible_ids]
      )
    ))
  } else {
    0L
  }

  source_overview <- data.frame(
    n_records = nrow(x),
    n_kitchens_with_label = length(keys),
    n_records_with_weight = sum(observed_w),
    n_records_with_whole_day_count = sum(observed_p),
    n_records_with_positive_whole_day_count = sum(observed_p & p > 0),
    n_complete_identifier_rows = sum(labelled),
    n_eligible = sum(eligible),
    eligible_fraction = mean(eligible),
    n_kitchens_with_eligible = length(unique(kitchen[eligible])),
    n_repeated_kitchen_timestamp_pairs_among_eligible = duplicate_pairs,
    reported_weight_sum_all_observed = sum(w[observed_w]),
    reported_weight_sum_eligible = sum(w[eligible]),
    share_reported_weight_in_eligible = if (sum(w[observed_w]) > 0) {
      sum(w[eligible]) / sum(w[observed_w])
    } else {
      NA_real_
    },
    reported_portions_sum_eligible = sum(p[eligible]),
    provisional_source_units_per_portion = safe_commercial_ratio(
      w[eligible], p[eligible]
    ),
    median_source_units_per_row_portion = if (length(ratio) > 0L) {
      stats::median(ratio)
    } else {
      NA_real_
    },
    is_waste_unit_confirmed = FALSE,
    is_window_alignment_confirmed = FALSE,
    stringsAsFactors = FALSE
  )

  # Descriptive outlier sensitivity only: these cutoffs are not confidence
  # intervals, nor do they correct selection bias or identify source units.
  cutoffs <- c(all = 1, below_99pct = 0.99, below_95pct = 0.95)
  ratio_sensitivity <- do.call(
    rbind,
    lapply(names(cutoffs), function(label) {
      threshold <- if (length(ratio) > 0L) {
        unname(stats::quantile(
          ratio, probs = cutoffs[[label]], names = FALSE, type = 1
        ))
      } else {
        NA_real_
      }
      keep <- if (is.finite(threshold)) {
        ratio <= threshold
      } else {
        logical()
      }
      weight_selected <- w[eligible][keep]
      portion_selected <- p[eligible][keep]
      data.frame(
        scenario = label,
        cutoff_row_ratio = threshold,
        n_rows = sum(keep),
        n_kitchens = length(unique(kitchen[eligible][keep])),
        source_weight_sum = sum(weight_selected),
        whole_day_portions_sum = sum(portion_selected),
        provisional_ratio_of_sums = safe_commercial_ratio(
          weight_selected, portion_selected
        ),
        median_row_ratio = if (any(keep)) {
          stats::median(ratio[keep])
        } else {
          NA_real_
        },
        stringsAsFactors = FALSE
      )
    })
  )
  rownames(ratio_sensitivity) <- NULL

  # Missingness is reported separately for each candidate denominator.
  # Meal-specific numbers are NOT combined with the whole-day waste sum.
  candidates <- grep("^portions, ", names(x), value = TRUE)
  denominator_availability <- do.call(
    rbind,
    lapply(candidates, function(field) {
      value <- x[[field]]
      if (!is.numeric(value)) {
        stop("Non-numeric portion field: ", field, call. = FALSE)
      }
      pos <- is.finite(value) & value > 0
      usable <- pos & observed_w & labelled
      data.frame(
        denominator = field,
        n_nonmissing = sum(is.finite(value)),
        n_positive = sum(pos),
        n_with_weight_and_identifiers = sum(usable),
        n_kitchens = length(unique(kitchen[usable])),
        eligible_fraction = mean(usable),
        ratio_interpretation = if (field == "portions, Whole day") {
          "provisional, measurement window unverified"
        } else {
          "no ratio: daily waste sum and meal count may misalign"
        },
        stringsAsFactors = FALSE
      )
    })
  )
  rownames(denominator_availability) <- NULL

  list(
    source_overview = source_overview,
    kitchen_coverage = kitchen_coverage,
    ratio_sensitivity = ratio_sensitivity,
    denominator_availability = denominator_availability
  )
}
