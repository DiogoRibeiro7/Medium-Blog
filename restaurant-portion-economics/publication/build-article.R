# Assemble the copy-ready Medium file from manually reviewed sections.
# Run from restaurant-portion-economics/
root <- "publication"
sources <- file.path(root, "sections", c(
  "00-question.md", "01-food-system.md", "02-evidence.md",
  "03-mathematics.md", "04-uncertainty.md", "05-menu-design.md",
  "06-business-and-conclusion.md"
))
if (!all(file.exists(sources))) {
  stop("Cannot build article: one or more source sections are missing.")
}
read_section <- function(path) {
  trimws(paste(readLines(path, warn = FALSE, encoding = "UTF-8"),
               collapse = "\n"))
}
text <- paste(vapply(sources, read_section, character(1)),
              collapse = "\n\n")
dir.create(file.path(root, "medium"), showWarnings = FALSE, recursive = TRUE)
writeLines(enc2utf8(paste0(text, "\n")),
           file.path(root, "medium", "article.md"), useBytes = TRUE)
message("Built Medium article from ", length(sources), " source sections.")
