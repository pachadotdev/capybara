#' @title Get index list
#' @description Generates an auxiliary list of indexes to project out the fixed
#'  effects (on C++ side the outputs are 0-indexed)
#' @param k_vars Fixed effects
#' @param data Data frame
#' @noRd
get_index_list_ <- function(k_vars, data) {
  indexes <- seq.int(1L, nrow(data))
  lapply(k_vars, function(X, indexes, data) {
    split(indexes, data[[X]])
  }, indexes = indexes, data = data)
}
