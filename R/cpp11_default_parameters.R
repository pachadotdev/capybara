# This file needs to be manually updated when the C++ code is changed.
# i.e., if cpp11.R is changed, this file needs to be updated.

parse_int <- function(x) {
  if (!is.integer(x)) x <- as.integer(x)
  x
}

parse_int_mat <- function(x) {
  if (!(storage.mode(x) == "integer")) storage.mode(x) <- "integer"
  x
}
