# Offline fixture tests: these observations are synthetic, not empirical results.
source("R/finland.R")
set.seed(129L)
n <- 155L
diners <- rep(100L, n)
menu <- rep(seq_len(25L), length.out = n)
pw <- rep(10, n)
ksw <- rep(3, n)
pw[136:155] <- 8
ksw[136:155] <- 2.5
fixture <- data.frame(
  "PW" = pw,
  "PW (g/diner)" = 1000 * pw / diners,
  "KSW" = ksw,
  "KSW (g/diner)" = 1000 * ksw / diners,
  "DINERS" = diners,
  "MENU" = menu,
  check.names = FALSE
)
daily <- standardise_finland(fixture)
stopifnot(nrow(daily) == 155L, identical(daily$day_index, seq_len(n)))
result <- analyse_finland(daily)
stopifnot(nrow(result$period) == 4L)
stopifnot(nrow(result$menu_matched) == 2L)
stopifnot(all(result$diagnostics$n_discrepant == 0L))
stopifnot(all(result$menu_matched$intervention_days_menu_matched == 20L))
stopifnot(abs(result$menu_matched$difference_g_diner[1] + 20) < 1e-10)
stopifnot(abs(result$menu_matched$difference_g_diner[2] + 5) < 1e-10)
error_expected <- function(x) {
  stopifnot(inherits(tryCatch(force(x), error = identity), "error"))
}
bad <- fixture
bad$DINERS[1] <- 0
error_expected(standardise_finland(bad))
bad <- fixture
bad$MENU[1] <- 26
error_expected(standardise_finland(bad))
bad <- fixture
bad[1, "PW"] <- -1
error_expected(standardise_finland(bad))
bad <- fixture
names(bad)[1] <- "unrecognised-column"
error_expected(standardise_finland(bad))
error_expected(analyse_finland(daily[-1, ]))
stopifnot(identical(parse_nonnegative(c("1,25", "2.5"), "test"),
                    c(1.25, 2.5)))
error_expected(parse_nonnegative("1,200.40", "test"))
message("All Finnish source analysis fixture tests passed.")
