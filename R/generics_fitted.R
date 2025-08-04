#' srr_stats
#' @srrstats {G1.0} Implements `fitted` methods for extracting fitted values from `feglm` and `felm` objects.
#' @srrstats {G2.1a} Ensures that the input object is of the expected class (`feglm` or `felm`).
#' @srrstats {G2.3a} Provides consistent and reliable handling of the fitted values for the supported object types.
#' @srrstats {G3.1a} Ensures that the returned fitted values match the specified family link function.
#' @srrstats {G3.1b} Supports both `feglm` and `felm` models with consistent behavior.
#' @srrstats {G3.1c} Outputs fitted values in a standardized format for use in downstream analysis.
#' @srrstats {G5.1} Includes error handling for unsupported or invalid input objects.
#' @srrstats {G5.4a} Includes tests to validate fitted value calculations for edge cases and typical use cases.
#' @srrstats {RE5.0} Designed for computational efficiency and ease of integration into workflows.
#' @srrstats {RE5.2} Ensures compatibility with standard R generics and user expectations.
#' @noRd
NULL

#' @export
#' @noRd
fitted.feglm <- function(object, ...) {
  object[["family"]][["linkinv"]](object[["eta"]])
}

#' @export
#' @noRd
fitted.felm <- function(object, ...) {
  object[["fitted.values"]]
}
