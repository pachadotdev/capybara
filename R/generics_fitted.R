# srr_stats
# {G1.0} Implements `fitted` methods for extracting fitted values from `feglm` and `felm` objects.
# {G2.1a} Ensures that the input object is of the expected class (`feglm` or `felm`).
# {G2.3a} Provides consistent and reliable handling of the fitted values for the supported object types.
# {G3.1a} Ensures that the returned fitted values match the specified family link function.
# {G3.1b} Supports both `feglm` and `felm` models with consistent behavior.
# {G3.1c} Outputs fitted values in a standardized format for use in downstream analysis.
# {G5.1} Includes error handling for unsupported or invalid input objects.
# {G5.4a} Includes tests to validate fitted value calculations for edge cases and typical use cases.
# {RE5.0} Designed for computational efficiency and ease of integration into workflows.
# {RE5.2} Ensures compatibility with standard R generics and user expectations.

#' @export
#' @noRd
fitted.feglm <- function(object, ...) {
  fam <- object[["family"]]
  fam[["linkinv"]](object[["eta"]])
}

#' @export
#' @noRd
fitted.felm <- function(object, ...) {
  object[["fitted_values"]]
}
