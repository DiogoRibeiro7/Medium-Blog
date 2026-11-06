# Assemble the Medium article from privately maintained editorial sections.
# Run from restaurant-portion-economics/ as:
# Rscript publication/build-article.R /path/to/research-articles-medium
#
# The public repository retains this reusable assembler, but never writes
# an assembled manuscript to its own historical compatibility-notice path.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L || !nzchar(args[[1L]])) {
  stop("Provide the private research-articles-medium checkout as one argument.")
}
private_root <- normalizePath(args[[1L]], mustWork = TRUE)
publication_root <- file.path(
  private_root, "restaurant-portion-economics", "publication"
)
section_names <- c(
  "00-question.md", "01-food-system.md", "02-evidence.md",
  "03-mathematics.md", "04-uncertainty.md", "05-menu-design.md",
  "06-business-and-conclusion.md"
)
sources <- file.path(publication_root, "sections", section_names)
if (!all(file.exists(sources))) {
  stop("Cannot assemble article: missing private source section(s): ",
       paste(section_names[!file.exists(sources)], collapse = ", "))
}
read_section <- function(path) {
  trimws(paste(readLines(path, warn = FALSE, encoding = "UTF-8"),
               collapse = "\n"))
}
article_text <- paste(vapply(sources, read_section, character(1L)),
                      collapse = "\n\n")
output_path <- file.path(publication_root, "medium", "article.md")
dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
writeLines(enc2utf8(paste0(article_text, "\n")),
           output_path, useBytes = TRUE)
message("Built private Medium article from ", length(sources), " source sections.")
