# Reproducible economic scenario plus optional empirical schema audit.
# Invoke from restaurant-portion-economics/ via Rscript scripts/run-analysis.R
source("R/economics.R")
dir.create("outputs", showWarnings = FALSE, recursive = TRUE)

assumptions <- example_assumptions()
baseline <- evaluate_scenario(assumptions)
utils::write.csv(baseline, "outputs/scenario.csv", row.names = FALSE)
utils::write.csv(scenario_grid(assumptions),
                 "outputs/sensitivity.csv", row.names = FALSE)
utils::write.csv(
  data.frame(
    break_even_demand_change = break_even_demand_change(assumptions),
    assumed_monthly_meals = assumptions$meals_per_month,
    price_eur = assumptions$price_eur,
    ingredient_cost_eur = assumptions$ingredient_cost_eur,
    other_variable_cost_eur = assumptions$other_variable_cost_eur,
    affected_ingredient_share = assumptions$affected_ingredient_share,
    realisation = assumptions$purchase_savings_realisation,
    implementation_eur = assumptions$monthly_implementation_eur
  ), "outputs/threshold.csv", row.names = FALSE
)
message("Hypothetical baseline incremental contribution EUR/month: ",
        round(baseline$incremental_contribution_eur, 2))
message("Break-even demand change: ",
        round(100 * break_even_demand_change(assumptions), 2), "%")

# Column schema and missingness only: no guesswork on unidentified units.
audit_table <- function(x, source, file) {
  stopifnot(is.data.frame(x), is.character(source), is.character(file))
  data.frame(
    source = source,
    file = basename(file),
    column = names(x),
    type = vapply(x, function(v) paste(class(v), collapse = "/"),
                  character(1)),
    missing = vapply(x, function(v) sum(is.na(v)), integer(1)),
    rows = nrow(x),
    stringsAsFactors = FALSE
  )
}

# CSV from URBAN-WASTE: compare all locally downloaded CSV candidates,
# retaining original column names; do not assume an encoding or mass unit.
csvs <- list.files("data/raw/urbanwaste", pattern = "\\.csv$",
                   recursive = TRUE, full.names = TRUE)
if (length(csvs) > 0L) {
  schemas <- lapply(csvs, function(path) {
    x <- tryCatch(
      utils::read.csv(path, check.names = FALSE, stringsAsFactors = FALSE,
                      fileEncoding = "UTF-8-BOM"),
      error = function(e) {
        warning("Failed to parse ", path, ": ", conditionMessage(e))
        NULL
      }
    )
    if (is.null(x)) return(NULL)
    audit_table(x, "urbanwaste", path)
  })
  schemas <- Filter(Negate(is.null), schemas)
  if (length(schemas) > 0L) {
    utils::write.csv(do.call(rbind, schemas),
                     "outputs/urbanwaste_schema.csv", row.names = FALSE)
  }
}

# Finnish source: audit sheets but do not label variables prematurely.
excels <- list.files("data/raw/finland", pattern = "\\.(xlsx|xls)$",
                     recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
if (length(excels) > 0L) {
  if (!requireNamespace("readxl", quietly = TRUE)) {
    warning("Install readxl to inspect the Finnish Excel source.")
  } else {
    schemas <- list()
    for (path in excels) {
      sheets <- readxl::excel_sheets(path)
      for (sheet in sheets) {
        x <- as.data.frame(readxl::read_excel(path, sheet = sheet))
        audit <- audit_table(x, paste0("finland:", sheet), path)
        schemas[[length(schemas) + 1L]] <- audit
      }
    }
    if (length(schemas) > 0L) {
      utils::write.csv(do.call(rbind, schemas),
                       "outputs/finland_schema.csv", row.names = FALSE)
    }
  }
}
message("Analysis complete. Outputs in outputs/.")
