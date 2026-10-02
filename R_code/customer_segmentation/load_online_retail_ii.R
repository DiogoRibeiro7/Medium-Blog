# Reproducible loader for the Online Retail II consumer analyses.
#
# Source identity:
#   repo: DiogoRibeiro7/data
#   commit: e90eed21a474c0a54ab9bf658c01dbd778bf58f1
#   path: datasets/online-retail-ii/raw/online_retail_II.xlsx
#   SHA-256: bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980

load_online_retail_ii <- function() {
  if (!requireNamespace("readxl", quietly = TRUE)) {
    stop("Package 'readxl' is required. Install it with install.packages('readxl').")
  }

  local_file <- file.path(".cache", "online-retail-ii", "online_retail_II.xlsx")
  fetcher <- file.path("..", "..", "scripts", "fetch_online_retail_ii.py")

  status <- system2("python", c(fetcher, "--output", local_file))
  if (!identical(status, 0L)) {
    stop("Unable to fetch or verify the canonical Online Retail II workbook.")
  }

  data <- readxl::read_excel(local_file, sheet = "Year 2010-2011")

  # Historical Online Retail II field names used by the workbook differ from
  # the normalized names expected by the R article scripts.
  names(data)[names(data) == "Invoice"] <- "InvoiceNo"
  names(data)[names(data) == "Price"] <- "UnitPrice"
  names(data)[names(data) == "Customer ID"] <- "CustomerID"

  data
}
