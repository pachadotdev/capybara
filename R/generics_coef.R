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
  object[["coefficients"]]
}

#' @export
#' @noRd
coef.felm <- function(object, ...) {
  object[["coefficients"]]
}

#' @export
#' @noRd
coef.summary.apes <- function(object, ...) {
  # Use pre-computed coefficient table from C++
  coefficients <- object[["vcov_table"]]
  if (is.null(coefficients)) {
    # Fallback: compute on-the-fly for backward compatibility
    est <- object[["delta"]]
    se <- sqrt(diag(object[["vcov"]]))
    z <- est / se
    p <- 2.0 * pnorm(-abs(z))
    coefficients <- cbind(est, se, z, p)
    rownames(coefficients) <- names(est)
    colnames(coefficients) <- c("Estimate", "Std. Error", "z value", "Pr(>|z|)")
  }
  coefficients
}

#' @export
#' @noRd
coef.summary.feglm <- function(object, ...) {
  # Use pre-computed coefficient table from C++
  coefficients <- object[["coef_table"]]
  rownames(coefficients) <- names(object[["coefficients"]])
  colnames(coefficients) <- c("Estimate", "Std. Error", "z value", "Pr(>|z|)")
  coefficients
}

#' @export
#' @noRd
coef.summary.felm <- function(object, ...) {
  # Use pre-computed coefficient table from C++
  coefficients <- object[["coef_table"]]
  rownames(coefficients) <- names(object[["coefficients"]])
  colnames(coefficients) <- c("Estimate", "Std. Error", "z value", "Pr(>|z|)")
  coefficients
}
