# Source manifest for externally downloaded Zenodo data.
# Published MD5 digests are checked against the original source files.
sources <- data.frame(
  source = c("finland_school", "urbanwaste_restaurants_hotels"),
  path = c(
    "data/raw/finland/Food waste data school Finland 2024-2025.xlsx",
    "data/raw/urbanwaste/Food Tracking Device dataset.csv"
  ),
  zenodo_record = c("20825233", "3355300"),
  md5_expected = c(
    "3f7e610fedc1810f7e8bc1bd47103ee7",
    "a88a1d7b09abd17064087e62d1704797"
  ),
  stringsAsFactors = FALSE
)
if (!all(file.exists(sources$path))) {
  stop("Expected publisher files missing: run scripts/download-data.R first.")
}
sources$md5_actual <- unname(tools::md5sum(sources$path))
sources$bytes <- as.numeric(file.info(sources$path)$size)
sources$checksum_matches <- sources$md5_expected == sources$md5_actual
if (!all(sources$checksum_matches)) {
  stop("Publisher checksum mismatch; source validation aborted.")
}
dir.create("outputs", showWarnings = FALSE)
utils::write.csv(sources, "outputs/zenodo_source_manifest.csv",
                 row.names = FALSE)
message("Zenodo original source checksums verified.")
