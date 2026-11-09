#' Add fixed effects to a data frame
#'
#' @param data A data.frame, data.table, or tibble with columns named after the
#'   fixed-effect dimensions.
#' @param object A \code{felm} or \code{feglm} object fitted with
#'   \code{control = fit_control(return_fe = TRUE)} (the default).
#' @param which Character vector of fixed-effect dimensions to add. The default
#'   (\code{NULL}) adds all dimensions.
#' @param col_names Character vector of output column names, in the same order
#'   as \code{which}. A named vector may instead map dimension names to output
#'   names. The default appends \code{"_fe"} to each dimension name.
#' @param transform Function applied to each full vector of fixed-effect
#'   coefficients before matching levels. The default is \code{identity}; use
#'   \code{exp} for exponentiated coefficients. It must return a numeric vector
#'   of the same length.
#' @details Levels are matched by name, not row position. Missing or unestimated
#'   levels receive \code{NA}. Existing output columns are replaced, but
#'   fixed-effect dimension columns cannot be overwritten. The model's
#'   normalization is retained; fixed effects are not renormalized.
#'   Only base R operations are used.
#' @examples
#' ross2004_subset <- ross2004[ross2004$year == 1999, ]
#' fit <- felm(ltrade ~ ldist | ctry1, ross2004_subset)
#' augmented <- add_fe_to_df(ross2004_subset, fit)
#' head(augmented$ctry1_fe)
#'
#' augmented <- add_fe_to_df(
#'   ross2004_subset, fit, col_names = c(ctry1 = "exporter_fe"), transform = exp
#' )
#' @return A table of the same class and row order as \code{data}, with one
#'   additional column per selected dimension. The input is not modified.
#' @export
add_fe_to_df <- function(
  data,
  object,
  which = NULL,
  col_names = NULL,
  transform = identity
) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data.frame, data.table, or tibble")
  }
  if (anyDuplicated(names(data))) {
    stop("`data` must have unique column names")
  }
  fe <- fixed_effects(object)
  if (is.null(which)) {
    which <- names(fe)
  }
  if (
    !is.character(which) || anyNA(which) || anyDuplicated(which) ||
      any(!which %in% names(fe))
  ) {
    stop("`which` must name distinct fixed-effect dimensions in the model")
  }
  if (!length(which)) {
    return(data)
  }
  missing_cols <- setdiff(which, names(data))
  if (length(missing_cols)) {
    stop(
      "`data` is missing fixed-effect columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }
  if (is.null(col_names)) {
    col_names <- paste0(which, "_fe")
  }
  if (
    !is.character(col_names) || length(col_names) != length(which) ||
      anyNA(col_names) || any(!nzchar(col_names)) || anyDuplicated(col_names)
  ) {
    stop("`col_names` must contain one distinct, non-empty name per dimension")
  }
  if (!is.null(names(col_names))) {
    if (
      anyNA(names(col_names)) || anyDuplicated(names(col_names)) ||
        !setequal(names(col_names), which)
    ) {
      stop(
        "Names of `col_names` must match the selected fixed-effect dimensions"
      )
    }
    col_names <- unname(col_names[which])
  }
  if (any(col_names %in% names(fe))) {
    stop("Output columns cannot overwrite fixed-effect dimension columns")
  }
  if (!is.function(transform)) {
    stop("`transform` must be a function")
  }
  columns <- lapply(which, function(dimension) {
    coefficients <- fe[[dimension]]
    values <- transform(coefficients)
    if (
      !is.numeric(values) || !is.null(dim(values)) ||
        length(values) != length(coefficients)
    ) {
      stop("`transform` must return a numeric vector of the same length")
    }
    unname(values[match(as.character(data[[dimension]]), names(coefficients))])
  })
  data[, col_names] <- columns
  data
}
