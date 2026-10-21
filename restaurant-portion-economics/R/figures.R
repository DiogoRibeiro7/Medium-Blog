# Article figures for "The Economics of the Uneaten Meal". Base R graphics only.
# Source R/economics.R, R/profit_bounds.R, R/menu_designs.R,
# R/quantity_response.R and R/empirical_audit.R before this file.
#
# Model figures need no external data. Empirical figures read the CSV
# outputs written by the real-data scripts and are only drawn when those
# outputs exist (the empirical GitHub Actions workflow downloads the sources).

fig_colours <- function() {
  list(
    surface = "#ffffff", ink = "#0b0b0b", secondary = "#52514e",
    muted = "#898781", grid = "#e1e0d9", axis = "#c3c2b7",
    blue = "#2a78d6", orange = "#eb6834", aqua = "#1baf7a",
    yellow = "#eda100", red = "#e34948", violet = "#6250d6",
    blue_light = "#b7d3f6", red_light = "#f6c4c3", neutral = "#ecebe6"
  )
}

fig_author <- "Diogo Ribeiro"

#' Open a PNG device sized for Medium (rendered at 2x the column width).
fig_open <- function(path, height = 1000, width = 1600) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  grDevices::png(path, width = width, height = height, res = 200,
                 type = "cairo", bg = fig_colours()$surface,
                 family = "sans")
}

#' Shared par() settings: recessive axes, no box, horizontal labels.
fig_par <- function(mar = c(4.2, 4.8, 4.6, 1.6), oma = c(1.6, 0, 0, 0),
                    ...) {
  k <- fig_colours()
  graphics::par(mar = mar, oma = oma, mgp = c(2.7, 0.6, 0), tcl = 0,
                las = 1, bty = "n", col.axis = k$muted, col.lab = k$secondary,
                fg = k$axis, cex.axis = 0.85, cex.lab = 0.9, xaxs = "i",
                yaxs = "i", ...)
}

#' Title and subtitle written into the top margin of the current panel.
fig_title <- function(title, subtitle = NULL, line = 2.5, outer = FALSE) {
  k <- fig_colours()
  adj <- if (outer) 0.02 else 0
  graphics::mtext(title, side = 3, line = line, adj = adj, cex = 1.08,
                  col = k$ink, outer = outer)
  if (!is.null(subtitle)) {
    graphics::mtext(subtitle, side = 3, line = line - 1.25, adj = adj,
                    cex = 0.8, col = k$secondary, outer = outer)
  }
}

fmt_num <- function(x, digits = 3) {
  rounded <- sign(x) * floor(abs(x) * 10^digits + 0.5 + 1e-9) / 10^digits
  formatC(rounded, format = "f", digits = digits)
}

#' Attribution line in the bottom outer margin.
fig_footer <- function(source) {
  graphics::mtext(paste0(fig_author, " \u00b7 ", source), side = 1,
                  line = 0.4, outer = TRUE, adj = 0.01, cex = 0.62,
                  col = fig_colours()$muted)
}

#' Horizontal gridlines plus a baseline along the bottom of the plot region.
fig_grid <- function(y_at, x_at = NULL) {
  k <- fig_colours()
  graphics::abline(h = y_at, col = k$grid, lwd = 1)
  if (!is.null(x_at)) graphics::abline(v = x_at, col = k$grid, lwd = 1)
  usr <- graphics::par("usr")
  graphics::segments(usr[1], usr[3], usr[2], usr[3], col = k$axis, lwd = 1.2)
}

fig_axis <- function(side, at, labels = at, ...) {
  graphics::axis(side, at = at, labels = labels, col = NA,
                 col.ticks = NA, ...)
}

fmt_eur <- function(x, digits = 0) {
  rounded <- floor(abs(x) * 10^digits + 0.5 + 1e-9) / 10^digits
  s <- formatC(rounded, format = "f", digits = digits, big.mark = ",")
  paste0(ifelse(x < 0, "\u2212\u20ac", "\u20ac"), s)
}

fmt_pct <- function(x, digits = 0) {
  paste0(formatC(100 * x, format = "f", digits = digits), "%")
}

fmt_signed <- function(x, digits = 2) {
  paste0(ifelse(x < 0, "\u2212", ""),
         formatC(abs(x), format = "f", digits = digits))
}

#' Text with a white box behind it, so it stays legible over lines.
fig_label <- function(x, y, labels, adj = c(0, 0.5), cex = 0.72,
                      col = fig_colours()$secondary, pad = 0.35, ...) {
  w <- graphics::strwidth(labels, cex = cex)
  h <- graphics::strheight(labels, cex = cex)
  px <- graphics::strwidth("m", cex = cex) * pad
  py <- graphics::strheight("M", cex = cex) * pad
  x0 <- x - adj[1] * w
  y0 <- y - adj[2] * h
  graphics::rect(x0 - px, y0 - py, x0 + w + px, y0 + h + py,
                 col = fig_colours()$surface, border = NA, xpd = NA)
  graphics::text(x, y, labels, adj = adj, cex = cex, col = col, xpd = NA,
                 ...)
}

fig_check_columns <- function(x, required, label) {
  if (!is.data.frame(x) || nrow(x) == 0L || !all(required %in% names(x))) {
    stop(label, " needs columns: ", paste(required, collapse = ", "),
         call. = FALSE)
  }
  invisible(x)
}

# ---------------------------------------------------------------------------
# Figure 1: where food-service waste arises (published UK benchmark).
# ---------------------------------------------------------------------------

#' WRAP (2013) stage shares of UK hospitality and food-service food waste.
wrap_waste_stages <- function() {
  data.frame(
    stage = c("Spoilage", "Preparation", "Customer plates"),
    share = c(0.21, 0.45, 0.34),
    lever = c(
      "forecasting, ordering,\nstorage, stock rotation",
      "recipes, trimming,\nbatch size, production plan",
      "default portion size,\nplate composition, refills"
    ),
    stringsAsFactors = FALSE
  )
}

plot_waste_stages <- function(path) {
  k <- fig_colours()
  x <- wrap_waste_stages()
  fig_open(path, height = 700)
  on.exit(grDevices::dev.off())
  fig_par(mar = c(0.5, 1.2, 4.6, 1.2))
  graphics::plot.new()
  graphics::plot.window(xlim = c(0, 1), ylim = c(0, 1))
  fig_title(
    "Where food waste arises in hospitality and food service",
    "Share of food waste by stage, UK sector: 920,000 tonnes a year, 75% of it avoidable"
  )
  edges <- c(0, cumsum(x$share))
  gap <- 0.004
  fills <- c("#c9c8c1", "#c9c8c1", k$orange)
  ink <- c(k$ink, k$ink, "#ffffff")
  top <- 0.80
  bottom <- 0.50
  for (i in seq_len(nrow(x))) {
    graphics::rect(edges[i] + gap, bottom, edges[i + 1] - gap, top,
                   col = fills[i], border = NA)
    mid <- (edges[i] + edges[i + 1]) / 2
    graphics::text(mid, top + 0.08, x$stage[i], cex = 0.95, col = k$ink,
                   font = 2)
    graphics::text(mid, (top + bottom) / 2, fmt_pct(x$share[i]),
                   cex = 1.3, col = ink[i], font = 2)
    graphics::text(mid, bottom - 0.05, "levers that act here",
                   cex = 0.7, col = k$muted, adj = c(0.5, 1))
    graphics::text(mid, bottom - 0.14, x$lever[i], cex = 0.8,
                   col = k$secondary, adj = c(0.5, 1))
  }
  graphics::text(0.5, 0.03,
                 "A smaller default plate acts directly only on the orange third; spoilage and preparation answer to other levers.",
                 cex = 0.72, col = k$secondary)
  fig_footer("data: WRAP (2013), Overview of waste in the UK hospitality and food service sector")
  invisible(path)
}

# ---------------------------------------------------------------------------
# Figure 2: URBAN-WASTE selection funnel and kitchen concentration.
# ---------------------------------------------------------------------------

plot_commercial_selection <- function(overview, kitchens, path) {
  fig_check_columns(overview, c(
    "n_records", "n_records_with_weight",
    "n_records_with_positive_whole_day_count", "n_eligible",
    "n_kitchens_with_label", "n_kitchens_with_eligible"
  ), "Commercial overview")
  fig_check_columns(kitchens, c("kitchen", "eligible_records"),
                    "Commercial kitchen coverage")
  k <- fig_colours()
  fig_open(path, height = 960)
  on.exit(grDevices::dev.off())
  fig_par(mar = c(3.0, 1.2, 3.0, 1.4), oma = c(1.6, 0, 4.0, 0))
  graphics::layout(matrix(1:2, nrow = 1), widths = c(1, 1.1))

  steps <- c(overview$n_records, overview$n_records_with_weight,
             overview$n_records_with_positive_whole_day_count,
             overview$n_eligible)
  labels <- c("all source records", "waste weight recorded",
              "positive whole-day portion count",
              "both, plus kitchen and timestamp")
  ypos <- rev(seq_along(steps))
  graphics::plot.new()
  graphics::plot.window(xlim = c(0, max(steps) * 1.02),
                        ylim = c(0.45, length(steps) + 0.75))
  graphics::rect(0, ypos - 0.26, steps, ypos + 0.14,
                 col = c(rep(k$blue_light, length(steps) - 1L), k$blue),
                 border = NA)
  graphics::text(0, ypos + 0.22, labels, adj = c(0, 0), cex = 0.78,
                 col = k$secondary)
  value_text <- paste0(format(steps, big.mark = ","), " (",
                       fmt_pct(steps / steps[1], 1), ")")
  for (i in seq_along(steps)) {
    inside <- steps[i] > max(steps) * 0.45
    graphics::text(if (inside) steps[i] - max(steps) * 0.015 else
                     steps[i] + max(steps) * 0.015,
                   ypos[i] - 0.06, value_text[i],
                   adj = c(if (inside) 1 else 0, 0.5), cex = 0.78,
                   col = if (inside || i < length(steps)) k$ink else k$ink,
                   font = 2)
  }
  graphics::mtext("Records left after each condition", side = 3,
                  line = 1.6, adj = 0, cex = 0.85, col = k$ink)

  graphics::par(mar = c(3.0, 3.6, 3.0, 1.4))
  counts <- sort(kitchens$eligible_records, decreasing = TRUE)
  n <- length(counts)
  ymax <- max(counts) * 1.12
  graphics::plot.new()
  graphics::plot.window(xlim = c(0.4, n + 0.6), ylim = c(0, ymax))
  yat <- pretty(c(0, max(counts)))
  yat <- yat[yat <= ymax]
  fig_grid(yat)
  top_five <- seq_len(n) <= 5L
  graphics::rect(seq_len(n) - 0.36, 0, seq_len(n) + 0.36, counts,
                 col = ifelse(top_five, k$blue, k$blue_light), border = NA)
  fig_axis(2, at = yat)
  graphics::mtext("kitchens, ranked by eligible records", side = 1,
                  line = 1.0, cex = 0.75, col = k$secondary)
  share1 <- counts[1] / sum(counts)
  share5 <- sum(utils::head(counts, 5L)) / sum(counts)
  zero <- sum(counts == 0)
  graphics::text(n + 0.6, max(counts) * 0.97,
                 paste0("largest kitchen: ", fmt_pct(share1, 1),
                        " of eligible rows\nfive largest (dark): ",
                        fmt_pct(share5, 1), "\nkitchens with none: ", zero,
                        " of ", n),
                 adj = c(1, 1), cex = 0.78, col = k$secondary)
  graphics::mtext("Eligible records per kitchen", side = 3, line = 1.6,
                  adj = 0, cex = 0.85, col = k$ink)

  graphics::mtext("Which hotel and restaurant records can carry a waste-per-portion ratio?",
                  side = 3, line = 2.4, outer = TRUE, adj = 0.02,
                  cex = 1.08, col = k$ink)
  graphics::mtext("URBAN-WASTE food tracking records; a ratio needs a recorded weight and a positive whole-day portion count",
                  side = 3, line = 1.15, outer = TRUE, adj = 0.02,
                  cex = 0.8, col = k$secondary)
  fig_footer("data: Eriksson & Lindgren (2019), URBAN-WASTE Food Tracking Device dataset, Zenodo 3355300")
  invisible(path)
}

# ---------------------------------------------------------------------------
# Figure 3: how far the provisional ratio moves when one kitchen is removed.
# ---------------------------------------------------------------------------

plot_commercial_ratio_fragility <- function(overview, loo, sensitivity, path) {
  fig_check_columns(overview, c("provisional_source_units_per_portion",
                                "median_source_units_per_row_portion"),
                    "Commercial overview")
  fig_check_columns(loo, c("provisional_ratio_without_kitchen",
                           "eligible_records_removed"),
                    "Leave-one-kitchen-out table")
  fig_check_columns(sensitivity, c("scenario", "provisional_ratio_of_sums"),
                    "Ratio sensitivity")
  k <- fig_colours()
  pooled <- overview$provisional_source_units_per_portion
  median_row <- overview$median_source_units_per_row_portion
  trimmed <- sensitivity$provisional_ratio_of_sums[
    sensitivity$scenario == "below_95pct"]
  ratio <- loo$provisional_ratio_without_kitchen
  ord <- order(ratio)
  ratio <- ratio[ord]
  removed <- loo$eligible_records_removed[ord]
  n <- length(ratio)

  fig_open(path, height = 1040)
  on.exit(grDevices::dev.off())
  fig_par(mar = c(4.4, 2.0, 5.0, 1.6))
  xr <- range(c(ratio, pooled, median_row, trimmed))
  pad <- diff(xr) * 0.08
  xlim <- c(xr[1] - pad, xr[2] + pad * 4.5)
  graphics::plot.new()
  graphics::plot.window(xlim = xlim, ylim = c(0.3, n + 5.0))
  xat <- pretty(xr)
  xat <- xat[xat >= xlim[1] & xat <= xlim[2]]
  graphics::segments(xat, 0.3, xat, n + 0.7, col = k$grid)
  refs <- c(pooled, trimmed, median_row)
  ref_labels <- c("pooled ratio, all eligible rows",
                  "rows above the 95th percentile removed",
                  "median of the row-level ratios")
  ref_cols <- c(k$ink, k$yellow, k$aqua)
  ref_lty <- c(1, 2, 2)
  for (i in seq_along(refs)) {
    graphics::segments(refs[i], 0.3, refs[i], n + 0.7, col = ref_cols[i],
                       lwd = 1.6, lty = ref_lty[i])
  }
  graphics::segments(pooled, seq_len(n), ratio, seq_len(n), col = k$axis,
                     lwd = 1.2)
  far <- abs(ratio - pooled) > 0.1 * pooled
  nudge <- diff(xlim) * 0.012
  for (i in which(far)) {
    right <- ratio[i] > pooled
    fig_label(ratio[i] + if (right) nudge else -nudge, i,
              paste0("without a kitchen holding ", removed[i], " rows"),
              adj = c(if (right) 0 else 1, 0.5), cex = 0.68)
  }
  graphics::points(ratio, seq_len(n), pch = 21, cex = 1.2,
                   bg = ifelse(far, k$blue, k$blue_light),
                   col = k$surface, lwd = 1.2)
  step <- if (length(xat) > 1L) diff(xat)[1] else 0.01
  fig_axis(1, at = xat, labels = formatC(xat, format = "f",
                                         digits = max(2, -floor(log10(step)))))
  graphics::mtext("pooled ratio with one kitchen left out (unverified weight units per reported portion)",
                  side = 1, line = 2.2, cex = 0.78, col = k$secondary)
  legend_y <- n + c(4.3, 3.15, 2.0)
  x0 <- xlim[1] + diff(xlim) * 0.01
  for (i in seq_along(refs)) {
    graphics::segments(x0, legend_y[i], x0 + diff(xlim) * 0.04, legend_y[i],
                       col = ref_cols[i], lwd = 1.6, lty = ref_lty[i])
    graphics::text(x0 + diff(xlim) * 0.05, legend_y[i],
                   paste0(ref_labels[i], ": ", fmt_num(refs[i], 3)),
                   adj = c(0, 0.5), cex = 0.72, col = k$secondary)
  }
  graphics::text(xlim[2], legend_y[1],
                 paste0("range across ", n, " omissions: ",
                        fmt_num(min(ratio), 3), " to ", fmt_num(max(ratio), 3)),
                 adj = c(1, 0.5), cex = 0.72, col = k$ink)
  fig_title("The summary number depends on which kitchens are in the file",
            "Pooled waste per whole-day portion, recomputed with each eligible kitchen left out in turn (one row each)")
  fig_footer("data: Eriksson & Lindgren (2019), URBAN-WASTE dataset, Zenodo 3355300 \u00b7 weight unit not confirmed by the publisher")
  invisible(path)
}

# ---------------------------------------------------------------------------
# Figure 4: Finnish school, daily waste per diner with the day-136 conflict.
# ---------------------------------------------------------------------------

plot_finland_daily <- function(daily, path) {
  fig_check_columns(daily, c(
    "day_index", "diners", "plate_kg", "plate_g_diner",
    "kitchen_serving_kg", "kitchen_serving_g_diner"
  ), "Finnish daily table")
  k <- fig_colours()
  if (!"phase" %in% names(daily)) {
    daily$phase <- ifelse(daily$day_index <= 135L, "baseline",
                          "intervention")
  }
  flags <- find_finland_discrepancies(daily)
  panels <- list(
    list(field = "plate_g_diner", stage = "plate", label = "Plate waste"),
    list(field = "kitchen_serving_g_diner", stage = "kitchen_serving",
         label = "Kitchen and serving waste")
  )
  ymax <- max(c(daily$plate_g_diner, daily$kitchen_serving_g_diner,
                flags$mass_derived_g_diner), na.rm = TRUE) * 1.12
  fig_open(path, height = 1200)
  on.exit(grDevices::dev.off())
  fig_par(mar = c(1.2, 4.6, 1.6, 1.6), oma = c(4.4, 0, 4.6, 0))
  graphics::layout(matrix(1:2, ncol = 1))
  first_intervention <- min(daily$day_index[daily$phase == "intervention"])
  last_day <- max(daily$day_index)
  for (j in seq_along(panels)) {
    p <- panels[[j]]
    graphics::plot.new()
    graphics::plot.window(xlim = c(0, last_day + 1), ylim = c(0, ymax))
    graphics::rect(first_intervention - 0.5, 0, last_day + 1, ymax,
                   col = "#fdf0e9", border = NA)
    yat <- pretty(c(0, ymax * 0.92))
    yat <- yat[yat < ymax]
    fig_grid(yat)
    y <- daily[[p$field]]
    base <- daily$phase == "baseline"
    graphics::lines(daily$day_index[base], y[base], col = k$axis, lwd = 1)
    graphics::points(daily$day_index[base], y[base], pch = 16, cex = 0.45,
                     col = k$muted)
    graphics::lines(daily$day_index[!base], y[!base], col = k$orange,
                    lwd = 1.4)
    graphics::points(daily$day_index[!base], y[!base], pch = 16,
                     cex = 0.55, col = k$orange)
    fig_axis(2, at = yat)
    graphics::mtext("g per diner", side = 2, line = 2.5, las = 0,
                    cex = 0.72, col = k$secondary)
    fig_label(2, ymax * 0.97, p$label, adj = c(0, 1), cex = 0.88,
              col = k$ink, font = 2)
    if (j == 1L) {
      graphics::text((first_intervention + last_day) / 2, ymax * 0.97,
                     "intervention\n(last 20 days)", adj = c(0.5, 1),
                     cex = 0.7, col = k$secondary)
    }
    f <- flags[flags$stage == p$stage, , drop = FALSE]
    for (i in seq_len(nrow(f))) {
      d <- f$day_index[i]
      lo <- min(f$published_g_diner[i], f$mass_derived_g_diner[i])
      hi <- max(f$published_g_diner[i], f$mass_derived_g_diner[i])
      graphics::segments(d, lo, d, hi, col = k$red, lwd = 1.6)
      graphics::points(d, f$published_g_diner[i], pch = 21, bg = k$red,
                       col = k$surface, cex = 1.3, lwd = 1.2)
      graphics::points(d, f$mass_derived_g_diner[i], pch = 21,
                       bg = k$surface, col = k$red, lwd = 1.8, cex = 1.3)
      ly <- ymax * if (j == 1L) 0.80 else 0.80
      graphics::segments(d, hi, d - 6, ly, col = k$secondary, lwd = 0.8)
      fig_label(d - 6.5, ly,
                paste0("day ", d, ": published ",
                       fmt_num(f$published_g_diner[i], 2),
                       " g per diner; recorded kg \u00f7 diners gives ",
                       fmt_num(f$mass_derived_g_diner[i], 2), " g"),
                adj = c(1, 0.5), cex = 0.68)
    }
  }
  xat <- c(1, seq(25, 150, by = 25))
  fig_axis(1, at = xat)
  graphics::mtext("school day", side = 1, line = 2.0, cex = 0.78,
                  col = k$secondary)
  graphics::mtext("A Finnish secondary-school canteen, 155 school days",
                  side = 3, line = 2.4, outer = TRUE, adj = 0.02,
                  cex = 1.08, col = k$ink)
  graphics::mtext("Published daily waste per diner; on flagged days, filled red = published rate, hollow = rate implied by recorded kg and diners",
                  side = 3, line = 1.15, outer = TRUE, adj = 0.02,
                  cex = 0.78, col = k$secondary)
  graphics::mtext(paste0(fig_author, " \u00b7 data: Kuusisto, Latva & Wil\u00e9n (2026), Food waste dataset: secondary school in Finland 2024\u20132025, Zenodo 20825233"),
                  side = 1, line = 3.4, outer = TRUE, adj = 0.01,
                  cex = 0.62, col = k$muted)
  invisible(path)
}

# ---------------------------------------------------------------------------
# Figure 5: three readings of the same Finnish workbook.
# ---------------------------------------------------------------------------

plot_finland_sensitivity <- function(sensitivity, path) {
  fig_check_columns(sensitivity, c("stage", "interpretation",
                                   "percent_difference"),
                    "Finnish sensitivity table")
  k <- fig_colours()
  readings <- c(
    reported_per_diner = "published per-diner rates",
    exclude_discrepant_days = "discordant day left out",
    derive_all_rates_from_mass = "rates recomputed from kg and diners"
  )
  stages <- c(plate = "Plate waste",
              kitchen_serving = "Kitchen and serving waste")
  cols <- c(k$blue, k$aqua, k$orange)
  fig_open(path, height = 820)
  on.exit(grDevices::dev.off())
  fig_par(mar = c(4.2, 13.4, 5.0, 2.0))
  values <- sensitivity$percent_difference
  xlim <- c(min(-65, floor(min(values) / 10) * 10), 0)
  rows <- list()
  y <- 0
  ylab <- numeric()
  ytxt <- character()
  for (s in names(stages)) {
    y <- y + 0.8
    ylab <- c(ylab, y)
    ytxt <- c(ytxt, stages[[s]])
    for (r in names(readings)) {
      y <- y + 1
      v <- sensitivity$percent_difference[sensitivity$stage == s &
                                            sensitivity$interpretation == r]
      rows[[length(rows) + 1L]] <- data.frame(y = y, value = v,
                                              reading = r, stage = s)
    }
    y <- y + 0.5
  }
  d <- do.call(rbind, rows)
  graphics::plot.new()
  graphics::plot.window(xlim = xlim, ylim = c(y + 0.2, 0.2))
  xat <- seq(xlim[1], 0, by = 10)
  graphics::abline(v = xat, col = k$grid)
  graphics::abline(v = 0, col = k$axis, lwd = 1.2)
  graphics::segments(0, d$y, d$value, d$y, col = k$axis, lwd = 1.2)
  graphics::points(d$value, d$y, pch = 21, cex = 1.5,
                   bg = cols[match(d$reading, names(readings))],
                   col = k$surface, lwd = 1.2)
  graphics::text(d$value, d$y,
                 paste0(fmt_signed(d$value), "%   "),
                 adj = 1, cex = 0.72, col = k$secondary)
  fig_axis(1, at = xat, labels = paste0(fmt_signed(xat, 0), "%"))
  graphics::mtext("change against the same menus in the 135 baseline days",
                  side = 1, line = 2.1, cex = 0.78, col = k$secondary)
  fig_axis(2, at = d$y, labels = readings[d$reading], cex.axis = 0.75,
           col.axis = k$secondary)
  graphics::text(graphics::par("usr")[1], ylab, ytxt, adj = c(1.04, 0.5),
                 xpd = NA, cex = 0.85, col = k$ink, font = 2)
  fig_title("Three readings of one workbook",
            "Menu-matched change in waste per diner during the 20 intervention days")
  fig_footer("data: Kuusisto, Latva & Wil\u00e9n (2026), Zenodo 20825233 \u00b7 descriptive comparison, not a causal effect")
  invisible(path)
}

# ---------------------------------------------------------------------------
# Figure 6: monthly contribution change against customer loss.
# ---------------------------------------------------------------------------

plot_contribution_vs_loss <- function(assumptions, path) {
  k <- fig_colours()
  loss <- seq(0, 0.06, by = 0.0005)
  delta <- vapply(loss, function(l) {
    a <- assumptions
    a$demand_change <- -l
    evaluate_scenario(a)$incremental_contribution_eur
  }, numeric(1))
  marks <- c(0, 0.01, 0.02, 0.05)
  mark_delta <- delta[match(round(marks, 4), round(loss, 4))]
  l_star <- -break_even_demand_change(assumptions)

  fig_open(path, height = 960)
  on.exit(grDevices::dev.off())
  fig_par(mar = c(4.4, 5.4, 5.0, 2.0))
  ylim <- c(-1000, 750)
  graphics::plot.new()
  graphics::plot.window(xlim = c(0, 0.06), ylim = ylim)
  graphics::rect(0, 0, 0.06, ylim[2], col = "#f2f7fd", border = NA)
  graphics::rect(0, ylim[1], 0.06, 0, col = "#fdf2f2", border = NA)
  yat <- seq(-1000, 750, by = 250)
  graphics::abline(h = yat, col = k$grid)
  graphics::abline(h = 0, col = k$axis, lwd = 1.4)
  graphics::lines(loss, delta, col = k$blue, lwd = 2.4)
  graphics::segments(l_star, ylim[1], l_star, 0, col = k$secondary, lty = 2)
  graphics::points(l_star, 0, pch = 21, bg = k$surface, col = k$ink,
                   cex = 1.3, lwd = 1.5)
  fig_label(l_star - 0.0008, -90,
            paste0("break-even at ", fmt_pct(l_star, 2), " of meals lost"),
            adj = c(1, 1), cex = 0.78, col = k$ink)
  graphics::points(marks, mark_delta, pch = 21, bg = k$blue,
                   col = k$surface, cex = 1.35, lwd = 1.2)
  for (i in seq_along(marks)) {
    fig_label(marks[i] + 0.0009, mark_delta[i] + 45,
              fmt_eur(mark_delta[i], 2), adj = c(0, 0), cex = 0.76)
  }
  fig_axis(1, at = seq(0, 0.06, by = 0.01),
           labels = fmt_pct(seq(0, 0.06, by = 0.01)))
  fig_axis(2, at = yat, labels = fmt_eur(yat))
  graphics::mtext("share of monthly meals lost after the change",
                  side = 1, line = 2.2, cex = 0.8, col = k$secondary)
  graphics::mtext("change in monthly contribution", side = 2, line = 4.1,
                  las = 0, cex = 0.8, col = k$secondary)
  graphics::text(0.059, 700, "gain", adj = c(1, 1), cex = 0.75,
                 col = k$muted)
  graphics::text(0.059, -950, "loss", adj = c(1, 0), cex = 0.75,
                 col = k$muted)
  fig_title("A 15% smaller plate pays for itself only if almost nobody leaves",
            "Illustrative restaurant: 2,500 meals a month at \u20ac15, \u20ac0.255 saved per retained meal, \u20ac150 a month to implement")
  fig_footer("hypothetical parameters, not observed restaurant accounts \u00b7 R/economics.R")
  invisible(path)
}

# ---------------------------------------------------------------------------
# Figure 7: one-at-a-time ranges versus the joint corner bounds.
# ---------------------------------------------------------------------------

profit_tornado <- function(assumptions, bounds) {
  validate_profit_uncertainty(bounds)
  base <- evaluate_scenario(assumptions)$incremental_contribution_eur
  vary <- function(field, value) {
    a <- assumptions
    if (field == "customer_loss") {
      a$demand_change <- -value
    } else {
      a[[field]] <- value
    }
    evaluate_scenario(a)$incremental_contribution_eur
  }
  labels <- c(
    affected_ingredient_share = "share of ingredient cost affected",
    portion_reduction = "portion reduction",
    purchase_savings_realisation = "saving realised in purchasing",
    customer_loss = "customers lost",
    monthly_implementation_eur = "monthly implementation cost"
  )
  rows <- lapply(names(labels), function(field) {
    values <- bounds[[field]]
    out <- c(vary(field, values[1]), vary(field, values[2]))
    data.frame(parameter = field, label = labels[[field]],
               low_input = values[1], high_input = values[2],
               delta_at_low_input = out[1], delta_at_high_input = out[2],
               delta_min = min(out), delta_max = max(out),
               stringsAsFactors = FALSE)
  })
  result <- do.call(rbind, rows)
  result$width <- result$delta_max - result$delta_min
  result <- result[order(result$width), , drop = FALSE]
  rownames(result) <- NULL
  attr(result, "baseline_delta") <- base
  attr(result, "joint") <- profit_delta_bounds(assumptions, bounds)
  result
}

plot_profit_tornado <- function(assumptions, bounds, path) {
  k <- fig_colours()
  t <- profit_tornado(assumptions, bounds)
  base <- attr(t, "baseline_delta")
  joint <- attr(t, "joint")
  fmt_input <- function(field, x) {
    if (field == "monthly_implementation_eur") fmt_eur(x) else fmt_pct(x)
  }
  fig_open(path, height = 960)
  on.exit(grDevices::dev.off())
  fig_par(mar = c(4.4, 12.4, 1.0, 1.4), oma = c(1.6, 0, 4.2, 0))
  n <- nrow(t)
  ylim <- c(0.4, n + 1.9)
  xlim <- c(-1650, 1700)
  graphics::plot.new()
  graphics::plot.window(xlim = xlim, ylim = ylim)
  xat <- seq(-1500, 1500, by = 500)
  graphics::segments(xat, 0.4, xat, n + 1.7, col = k$grid)
  graphics::segments(0, 0.4, 0, n + 1.7, col = k$axis, lwd = 1.4)
  yj <- n + 1.25
  graphics::rect(joint$contribution_change_min_eur, yj - 0.3,
                 joint$contribution_change_max_eur, yj + 0.3,
                 col = k$neutral, border = k$secondary, lwd = 1)
  graphics::text(joint$contribution_change_min_eur, yj,
                 paste0(fmt_eur(joint$contribution_change_min_eur, 2), " "),
                 adj = 1, cex = 0.7, col = k$secondary)
  graphics::text(joint$contribution_change_max_eur, yj,
                 paste0(" ", fmt_eur(joint$contribution_change_max_eur, 2)),
                 adj = 0, cex = 0.7, col = k$secondary)
  for (i in seq_len(n)) {
    lo <- t$delta_min[i]
    hi <- t$delta_max[i]
    graphics::rect(min(lo, base), i - 0.3, base, i + 0.3, col = k$red,
                   border = NA)
    graphics::rect(base, i - 0.3, max(hi, base), i + 0.3, col = k$blue,
                   border = NA)
    worse <- if (t$delta_at_low_input[i] < t$delta_at_high_input[i]) {
      t$low_input[i]
    } else {
      t$high_input[i]
    }
    better <- if (worse == t$low_input[i]) t$high_input[i] else t$low_input[i]
    graphics::text(lo, i, paste0(fmt_input(t$parameter[i], worse), " "),
                   adj = 1, cex = 0.7, col = k$secondary)
    graphics::text(hi, i, paste0(" ", fmt_input(t$parameter[i], better)),
                   adj = 0, cex = 0.7, col = k$secondary)
  }
  graphics::segments(base, 0.6, base, n + 0.45, col = k$ink, lwd = 1.2)
  graphics::text(base, n + 0.62, paste0("baseline ", fmt_eur(base, 2)),
                 adj = c(0.5, 0.5), cex = 0.68, col = k$ink)
  fig_axis(1, at = xat, labels = fmt_eur(xat))
  fig_axis(2, at = c(seq_len(n), yj),
           labels = c(t$label, "all five at once"),
           cex.axis = 0.78, col.axis = k$secondary)
  graphics::mtext("change in monthly contribution", side = 1, line = 2.2,
                  cex = 0.8, col = k$secondary)
  fig_title("Customer loss moves the result more than any cost input",
            "One input varied across its assumed range, the others held at the baseline with no customers lost",
            line = 2.4, outer = TRUE)
  fig_footer("assumed ranges, not confidence intervals \u00b7 R/profit_bounds.R")
  invisible(path)
}

# ---------------------------------------------------------------------------
# Figure 8: where the sign of the result is settled by the assumptions.
# ---------------------------------------------------------------------------

plot_feasibility_map <- function(assumptions, bounds, path) {
  k <- fig_colours()
  dr <- 0.0125
  dl <- 0.0025
  grid <- profit_frontier_grid(
    assumptions, bounds,
    portion_reductions = seq(0, 0.30, by = dr),
    customer_losses = seq(0, 0.10, by = dl)
  )
  cls <- c(gain_under_all_assumptions = k$blue_light,
           sign_depends_on_assumptions = k$neutral,
           loss_under_all_assumptions = k$red_light)
  fig_open(path, height = 1000)
  on.exit(grDevices::dev.off())
  fig_par(mar = c(4.4, 4.8, 5.0, 1.8))
  graphics::plot.new()
  graphics::plot.window(xlim = c(-dr / 2, 0.30 + dr / 2),
                        ylim = c(-dl / 2, 0.10 + dl / 2))
  graphics::rect(grid$portion_reduction - dr / 2,
                 grid$customer_loss - dl / 2,
                 grid$portion_reduction + dr / 2,
                 grid$customer_loss + dl / 2,
                 col = cls[grid$classification], border = NA)
  r <- seq(0.005, 0.30, by = 0.0025)
  curve <- quantity_elasticity_threshold(assumptions, r)
  keep <- curve$break_even_customer_loss >= 0
  graphics::lines(r[keep], curve$break_even_customer_loss[keep],
                  col = k$ink, lwd = 2)
  at15 <- curve[abs(curve$portion_reduction - 0.15) < 1e-9, ]
  graphics::points(0.15, at15$break_even_customer_loss, pch = 21,
                   bg = k$ink, col = k$surface, cex = 1.3)
  fig_label(0.153, at15$break_even_customer_loss - 0.004,
            paste0("baseline costs: break-even line\n(15% cut tolerates ",
                   fmt_pct(at15$break_even_customer_loss, 2), ")"),
            adj = c(0, 1), cex = 0.72, col = k$ink)
  graphics::text(0.012, 0.094, "loss under every cost assumption",
                 adj = c(0, 1), cex = 0.82, col = "#8f2a29", font = 2)
  graphics::text(0.295, 0.016, "gain under every\ncost assumption",
                 adj = c(1, 0), cex = 0.82, col = "#184f95", font = 2)
  graphics::text(0.245, 0.068, "sign depends on\nthe cost assumptions",
                 adj = c(0.5, 0.5), cex = 0.82, col = k$secondary, font = 2)
  fig_axis(1, at = seq(0, 0.30, by = 0.05),
           labels = fmt_pct(seq(0, 0.30, by = 0.05)))
  fig_axis(2, at = seq(0, 0.10, by = 0.02),
           labels = fmt_pct(seq(0, 0.10, by = 0.02)))
  graphics::mtext("reduction in the default portion", side = 1, line = 2.2,
                  cex = 0.8, col = k$secondary)
  graphics::mtext("share of meals lost", side = 2, line = 3.3, las = 0,
                  cex = 0.8, col = k$secondary)
  fig_title("Most of the plausible region has no settled answer",
            "Cost share 25\u201355%, purchasing pass-through 50\u201395%, implementation \u20ac75\u2013250 a month, varied jointly")
  fig_footer("assumed ranges, not estimates \u00b7 R/profit_bounds.R, R/quantity_response.R")
  invisible(path)
}

# ---------------------------------------------------------------------------
# Figure 9: break-even quantity elasticity against published elasticities.
# ---------------------------------------------------------------------------

plot_elasticity_threshold <- function(assumptions, path) {
  k <- fig_colours()
  r <- seq(0.02, 0.30, by = 0.0025)
  with_i <- quantity_elasticity_threshold(assumptions, r, TRUE)
  without_i <- quantity_elasticity_threshold(assumptions, r, FALSE)
  limit <- quantity_elasticity_limit(assumptions)
  b <- quantity_response_benchmarks()
  fig_open(path, height = 1060)
  on.exit(grDevices::dev.off())
  fig_par(mar = c(4.4, 4.8, 5.0, 1.8))
  ylim <- c(0, 0.9)
  graphics::plot.new()
  graphics::plot.window(xlim = c(0.02, 0.30), ylim = ylim)
  yat <- seq(0, 0.9, by = 0.1)
  graphics::abline(h = yat, col = k$grid)
  graphics::polygon(c(r, rev(r)),
                    c(pmax(with_i$break_even_quantity_elasticity, 0),
                      rep(0, length(r))),
                    col = "#e3eefb", border = NA)
  usr <- graphics::par("usr")
  graphics::segments(usr[1], 0, usr[2], 0, col = k$axis, lwd = 1.2)
  band <- b[b$elasticity_high > b$elasticity_low, , drop = FALSE]
  point <- b[b$elasticity_high == b$elasticity_low, , drop = FALSE]
  for (i in seq_len(nrow(band))) {
    graphics::rect(usr[1], band$elasticity_low[i], usr[2],
                   band$elasticity_high[i], col = "#fbe1d6", border = NA)
    graphics::text(0.025, band$elasticity_high[i] - 0.012,
                   paste0("Food away from home, price elasticity, systematic review: ",
                          formatC(band$elasticity_low[i], format = "f",
                                  digits = 1), "\u2013",
                          formatC(band$elasticity_high[i], format = "f",
                                  digits = 1)),
                   adj = c(0, 1), cex = 0.72, col = k$secondary)
  }
  for (i in seq_len(nrow(point))) {
    is_quantity <- point$measures[i] == "quantity"
    graphics::segments(usr[1], point$elasticity_low[i], usr[2],
                       point$elasticity_low[i], col = k$orange, lwd = 1.6,
                       lty = if (is_quantity) 1 else 2)
    label <- if (is_quantity) {
      "Ice cream, package-size elasticity (Chicago scanner data): "
    } else {
      "Ice cream, price elasticity (same study): "
    }
    fig_label(if (is_quantity) 0.297 else 0.025,
              point$elasticity_low[i] + if (is_quantity) -0.022 else 0.022,
              paste0(label, formatC(point$elasticity_low[i], format = "f",
                                    digits = 2)),
              adj = c(if (is_quantity) 1 else 0, 0.5), cex = 0.72)
  }
  graphics::lines(r, without_i$break_even_quantity_elasticity,
                  col = k$blue, lwd = 1.8, lty = 2)
  graphics::lines(r, with_i$break_even_quantity_elasticity, col = k$blue,
                  lwd = 2.6)
  at15 <- with_i[abs(with_i$portion_reduction - 0.15) < 1e-9, ]
  graphics::points(0.15, at15$break_even_quantity_elasticity, pch = 21,
                   bg = k$blue, col = k$surface, cex = 1.4, lwd = 1.2)
  graphics::segments(0.15, at15$break_even_quantity_elasticity + 0.015,
                     0.165, 0.30, col = k$secondary, lwd = 0.8)
  fig_label(0.166, 0.30,
            paste0("a 15% cut breaks even at an elasticity of ",
                   fmt_num(at15$break_even_quantity_elasticity, 3)),
            adj = c(0, 0.5), cex = 0.74, col = k$ink)
  graphics::text(0.06, 0.04, "profitable below the solid line", adj = c(0, 0),
                 cex = 0.72, col = "#184f95")
  lx <- 0.118
  ly <- c(0.62, 0.57)
  graphics::segments(lx, ly[1], lx + 0.015, ly[1], col = k$blue, lwd = 2.6)
  graphics::segments(lx, ly[2], lx + 0.015, ly[2], col = k$blue, lwd = 1.8,
                     lty = 2)
  graphics::text(lx + 0.018, ly[1],
                 "break-even, with \u20ac150 a month to implement",
                 adj = c(0, 0.5), cex = 0.72, col = k$secondary)
  graphics::text(lx + 0.018, ly[2],
                 paste0("break-even, no implementation cost (small-cut limit ",
                        fmt_num(limit, 3), ")"),
                 adj = c(0, 0.5), cex = 0.72, col = k$secondary)
  fig_axis(1, at = seq(0.05, 0.30, by = 0.05),
           labels = fmt_pct(seq(0.05, 0.30, by = 0.05)))
  fig_axis(2, at = yat, labels = formatC(yat, format = "f", digits = 1))
  graphics::mtext("reduction in the default portion at an unchanged price",
                  side = 1, line = 2.2, cex = 0.8, col = k$secondary)
  graphics::mtext("elasticity of meals sold to portion size", side = 2,
                  line = 3.2, las = 0, cex = 0.8, col = k$secondary)
  fig_title("How insensitive to portion size customers would need to be",
            "Largest elasticity at which the illustrative restaurant breaks even, against published estimates from other markets")
  fig_footer("benchmarks: \u00c7ak\u0131r & Balagtas (2014); Andreyeva, Long & Brownell (2010) \u00b7 none of them measures restaurant diners")
  invisible(path)
}

# ---------------------------------------------------------------------------
# Figure 10: three menu designs across assumed customer losses.
# ---------------------------------------------------------------------------

plot_menu_designs <- function(assumptions, options, path) {
  k <- fig_colours()
  loss <- seq(0, 0.05, by = 0.001)
  curve <- function(option) {
    vapply(loss, function(l) {
      o <- option
      o$customer_loss <- l
      evaluate_menu_design(assumptions, o)$monthly_contribution_change_eur
    }, numeric(1))
  }
  same <- curve(options$smaller_same_price)
  refill25 <- curve(options$smaller_free_refill)
  refill50 <- {
    o <- options$smaller_free_refill
    o$refill_uptake <- 0.50
    curve(o)
  }
  discount <- curve(options$smaller_discount)
  series <- list(
    list(y = same, col = k$blue, lty = 1,
         label = "smaller, same price"),
    list(y = refill25, col = k$aqua, lty = 1,
         label = "smaller + free refill, 25% take one"),
    list(y = refill50, col = k$aqua, lty = 2,
         label = "smaller + free refill, 50% take one"),
    list(y = discount, col = k$orange, lty = 1,
         label = "smaller, 5% cheaper")
  )
  fig_open(path, height = 1000)
  on.exit(grDevices::dev.off())
  fig_par(mar = c(4.4, 5.4, 5.0, 12.6))
  ylim <- c(-3000, 750)
  graphics::plot.new()
  graphics::plot.window(xlim = c(0, 0.05), ylim = ylim)
  yat <- seq(-3000, 500, by = 500)
  graphics::abline(h = yat, col = k$grid)
  graphics::abline(h = 0, col = k$axis, lwd = 1.4)
  for (s in series) {
    graphics::lines(loss, s$y, col = s$col, lwd = 2.4, lty = s$lty)
  }
  usr <- graphics::par("usr")
  ends <- vapply(series, function(s) utils::tail(s$y, 1), numeric(1))
  label_y <- ends
  ord <- order(label_y, decreasing = TRUE)
  min_gap <- 430
  for (j in seq_along(ord)[-1]) {
    prev <- label_y[ord[j - 1]]
    if (prev - label_y[ord[j]] < min_gap) label_y[ord[j]] <- prev - min_gap
  }
  for (i in seq_along(series)) {
    graphics::segments(usr[2], ends[i], usr[2] + 0.0012, label_y[i],
                       col = series[[i]]$col, lwd = 1.2, xpd = NA)
    graphics::text(usr[2] + 0.0014, label_y[i],
                   paste0(series[[i]]$label, "\n", fmt_eur(ends[i]), " at 5%"),
                   adj = c(0, 0.5), xpd = NA, cex = 0.68, col = k$secondary)
  }
  starts <- vapply(series, function(s) s$y[1], numeric(1))
  graphics::points(rep(0, length(series)), starts, pch = 21,
                   bg = vapply(series, function(s) s$col, character(1)),
                   col = k$surface, cex = 1.2, lwd = 1.2, xpd = NA)
  fig_axis(1, at = seq(0, 0.05, by = 0.01),
           labels = fmt_pct(seq(0, 0.05, by = 0.01)))
  fig_axis(2, at = yat, labels = fmt_eur(yat))
  graphics::mtext("share of monthly meals lost after the change",
                  side = 1, line = 2.2, cex = 0.8, col = k$secondary)
  graphics::mtext("change in monthly contribution", side = 2, line = 4.1,
                  las = 0, cex = 0.8, col = k$secondary)
  fig_title("A discount gives away more than the smaller plate saves",
            "Same 15% portion cut under three menu designs; refills restore the full saving and cost \u20ac0.20 each to serve")
  fig_footer("hypothetical parameters \u00b7 R/menu_designs.R")
  invisible(path)
}

# ---------------------------------------------------------------------------
# Registry used by scripts/make-figures.R and the article test.
# ---------------------------------------------------------------------------

article_figure_files <- function() {
  c(
    waste_stages = "fig01_waste_stages.png",
    commercial_selection = "fig02_commercial_selection.png",
    commercial_fragility = "fig03_commercial_ratio_fragility.png",
    finland_daily = "fig04_finland_daily.png",
    finland_sensitivity = "fig05_finland_sensitivity.png",
    contribution_vs_loss = "fig06_contribution_vs_loss.png",
    profit_tornado = "fig07_profit_tornado.png",
    feasibility_map = "fig08_feasibility_map.png",
    elasticity_threshold = "fig09_elasticity_threshold.png",
    menu_designs = "fig10_menu_designs.png"
  )
}

#' Draw every figure whose inputs are available.
#'
#' @param out_dir Destination folder for PNG files.
#' @param outputs_dir Folder holding the empirical CSV outputs.
#' @return Named logical vector: TRUE where a figure was written.
make_article_figures <- function(out_dir, outputs_dir = "outputs") {
  files <- file.path(out_dir, article_figure_files())
  names(files) <- names(article_figure_files())
  a <- example_assumptions()
  bounds <- example_profit_uncertainty()
  written <- stats::setNames(rep(FALSE, length(files)), names(files))

  plot_waste_stages(files[["waste_stages"]])
  plot_contribution_vs_loss(a, files[["contribution_vs_loss"]])
  plot_profit_tornado(a, bounds, files[["profit_tornado"]])
  plot_feasibility_map(a, bounds, files[["feasibility_map"]])
  plot_elasticity_threshold(a, files[["elasticity_threshold"]])
  plot_menu_designs(a, example_menu_options(), files[["menu_designs"]])
  written[c("waste_stages", "contribution_vs_loss", "profit_tornado",
            "feasibility_map", "elasticity_threshold", "menu_designs")] <- TRUE

  read <- function(name) {
    path <- file.path(outputs_dir, name)
    if (file.exists(path)) {
      utils::read.csv(path, stringsAsFactors = FALSE)
    } else {
      NULL
    }
  }
  overview <- read("commercial_source_overview.csv")
  kitchens <- read("commercial_kitchen_coverage.csv")
  loo <- read("commercial_heterogeneity_leave_one_kitchen_out.csv")
  ratio_sens <- read("commercial_ratio_sensitivity.csv")
  daily <- read("finland_daily.csv")
  finland_sens <- read("finland_sensitivity.csv")

  if (!is.null(overview) && !is.null(kitchens) &&
      "eligible_records" %in% names(kitchens)) {
    plot_commercial_selection(overview, kitchens,
                              files[["commercial_selection"]])
    written[["commercial_selection"]] <- TRUE
  }
  if (!is.null(overview) && !is.null(loo) && !is.null(ratio_sens)) {
    plot_commercial_ratio_fragility(overview, loo, ratio_sens,
                                    files[["commercial_fragility"]])
    written[["commercial_fragility"]] <- TRUE
  }
  if (!is.null(daily)) {
    plot_finland_daily(daily, files[["finland_daily"]])
    written[["finland_daily"]] <- TRUE
  }
  if (!is.null(finland_sens)) {
    plot_finland_sensitivity(finland_sens, files[["finland_sensitivity"]])
    written[["finland_sensitivity"]] <- TRUE
  }
  written
}
