# Download source archives through Zenodo metadata with checksum validation.
# Usage: Rscript scripts/download-data.R  (from project directory)
# Dependencies: base R, internet access to Zenodo.

records <- c(urbanwaste = "3355300", finland = "20825233")
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

download_record <- function(name, record_id) {
  stopifnot(is.character(name), length(name) == 1L,
            is.character(record_id), grepl("^[0-9]+$", record_id))
  api <- sprintf("https://zenodo.org/api/records/%s", record_id)
  metadata_path <- tempfile(fileext = ".json")
  on.exit(unlink(metadata_path), add = TRUE)
  status <- tryCatch(
    utils::download.file(api, metadata_path, mode = "wb", quiet = TRUE),
    error = function(e) stop("Could not access Zenodo record ", record_id,
                              ": ", conditionMessage(e), call. = FALSE)
  )
  if (!identical(status, 0L)) stop("Metadata download failed: ", api)
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Install jsonlite to read the Zenodo record metadata.",
         call. = FALSE)
  }
  metadata <- jsonlite::fromJSON(metadata_path, simplifyVector = FALSE)
  entries <- metadata$files
  if (length(entries) == 0L) stop("No files in record: ", record_id)
  files <- lapply(entries, function(entry) {
    list(
      filename = entry$key,
      url = entry$links$self,
      checksum = entry$checksum
    )
  })
  # Keep original filenames, with explicit naming and traversal checks.
  for (entry in files) {
    filename <- entry$filename
    if (!is.character(filename) || length(filename) != 1L ||
        grepl("[/\\\\]", filename) || grepl("^\\.", filename)) {
      stop("Unsafe or unexpected Zenodo filename", call. = FALSE)
    }
    folder <- file.path("data/raw", name)
    dir.create(folder, recursive = TRUE, showWarnings = FALSE)
    destination <- file.path(folder, filename)
    checksum <- entry$checksum
    if (!is.character(checksum) || length(checksum) != 1L ||
        !grepl("^md5:[a-fA-F0-9]{32}$", checksum)) {
      stop("Missing publisher MD5 for ", filename, call. = FALSE)
    }
    expected <- tolower(sub("^md5:", "", checksum))
    existing_ok <- file.exists(destination) &&
      identical(unname(tools::md5sum(destination)), expected)
    if (!existing_ok) {
      temp_path <- tempfile(tmpdir = folder)
      status <- tryCatch(
        utils::download.file(entry$url, temp_path, mode = "wb", quiet = TRUE),
        error = function(e) stop("Download failed: ", filename, ": ",
                                  conditionMessage(e), call. = FALSE)
      )
      if (!identical(status, 0L) ||
          !identical(unname(tools::md5sum(temp_path)), expected)) {
        unlink(temp_path)
        stop("Download or checksum failure for ", filename, call. = FALSE)
      }
      if (!file.rename(temp_path, destination)) {
        unlink(temp_path)
        stop("Could not move downloaded file: ", filename, call. = FALSE)
      }
    }
    message(name, ": ", filename, " (MD5 verified)")
  }
}

for (name in names(records)) download_record(name, records[[name]])
