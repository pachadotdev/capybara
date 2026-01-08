#' srr_stats
#' @srrstats {G1.0} Defines `coef` methods for extracting coefficients from various model objects.
#' @srrstats {G2.1a} Ensures that the input object is of the expected class (`apes`, `feglm`, or `felm`).
#' @srrstats {G3.1a} Outputs coefficients in a consistent format for interpretability.
#' @srrstats {G3.1b} Supports multiple model object types, maintaining a standardized interface.
#' @srrstats {G3.1c} Provides access to summary statistics (`coefficients`) where applicable.
#' @srrstats {G5.1} Includes robust error handling for unsupported or invalid input objects.
#' @srrstats {G5.4a} Includes tests for extracting coefficients from simple and complex model objects.
#' @srrstats {RE4.2} Returns coefficients via a standard method for feglm-type objects and derived classes (i.e., felm, apes, etc).
#' @srrstats {RE5.0} Enables seamless integration with downstream analysis workflows.
#' @srrstats {RE5.2} Maintains computational efficiency in coefficient extraction.
#' @noRd
NULL

#' @export
#' @noRd
coef.apes <- function(object, ...) {
  object[["delta"]]
}

#' @export
#' @noRd
coef.feglm <- function(object, ...) {
  ct <- object[["coef_table"]]
  setNames(ct[, 1], rownames(ct))
}

#' @export
#' @noRd
coef.felm <- function(object, ...) {
  ct <- object[["coef_table"]]
  setNames(ct[, 1], rownames(ct))
}

#' @export
#' @noRd
coef.summary.apes <- function(object, ...) {
  # Use pre-computed coefficient table from C++
  coefficients <- object[["vcov_table"]]
  if (is.null(coefficients)) {
    # Fallback: compute on-the-fly for backward compatibility
    delta <- object[["delta"]]
    se <- sqrt(diag(object[["vcov"]]))
    coefficients <- cbind(
      Estimate = delta,
      `Std. Error` = se,
      `z value` = delta / se,
      `Pr(>|z|)` = 2.0 * pnorm(-abs(delta / se))
    )
    rownames(coefficients) <- names(delta)
  }
  coefficients
}

#' @export
#' @noRd
coef.summary.feglm <- function(object, ...) {
  # coef_table already has row/column names from the model fitting

  object[["coef_table"]]
}

#' @export
#' @noRd
coef.summary.felm <- function(object, ...) {
  # coef_table already has row/column names from the model fitting
  object[["coef_table"]]
}
