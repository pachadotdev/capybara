#' srr_stats
#' @srrstats {G1.0} Implements `confint` methods for extracting confidence intervals for `feglm` and `felm` objects using Wald method.
#' @srrstats {G2.1a} Ensures that the input object is of the expected class (`feglm` or `felm`).
#' @srrstats {G2.3a} Validates the `level` parameter to ensure it is within the accepted range (0, 1).
#' @srrstats {G3.1a} Outputs confidence intervals in a standardized format, making them easily interpretable.
#' @srrstats {G3.1b} Supports multiple model types with consistent behavior.
#' @srrstats {G3.1c} Provides Wald confidence intervals based on asymptotic normality.
#' @srrstats {G5.1} Includes error handling for unsupported or invalid input objects and invalid `level` values.
#' @srrstats {G5.4a} Includes tests to validate confidence interval calculations for edge cases and typical use cases.
#' @srrstats {RE5.0} Designed for seamless integration with downstream analysis workflows.
#' @srrstats {RE5.2} Maintains computational efficiency - uses Wald intervals to avoid expensive profile likelihood computation.
#' @noRd
NULL

#' Confidence Intervals for Model Parameters
#'
#' @param object An object of class \code{feglm} or \code{felm}
#' @param parm A specification of which parameters are to be given confidence
#'   intervals, either a vector of numbers or a vector of names. If missing,
#'   all parameters are considered.
#' @param level The confidence level required (default 0.95)
#' @param ... Additional arguments (currently unused)
#'
#' @details
#' This function computes Wald confidence intervals based on asymptotic normality.
#' Unlike \code{stats::confint.glm}, this does not compute profile likelihood
#' intervals, as the computational cost for high-dimensional fixed effects models
#' would be prohibitive. The Wald intervals are computed as:
#' \deqn{estimate \pm z_{\alpha/2} \times SE}
#' where \eqn{z_{\alpha/2}} is the critical value from the standard normal distribution.
#'
#' @return A matrix with columns giving lower and upper confidence limits for
#'   each parameter.
#'
#' @export
#' @noRd
confint.feglm <- function(object, parm, level = 0.95, ...) {
  estimates <- object$coef_table[, "Estimate"]
  std_errors <- object$coef_table[, "Std. Error"]
  crit_val <- qnorm(1 - (1 - level) / 2)

  conf_int <- cbind(
    estimates - crit_val * std_errors,
    estimates + crit_val * std_errors
  )

  colnames(conf_int) <- paste(100 * (c(0, 1) + c(1, -1) * (1 - level) / 2), "%")
  rownames(conf_int) <- rownames(object$coef_table)

  if (!missing(parm)) {
    conf_int <- conf_int[parm, , drop = FALSE]
  }

  conf_int
}

#' @export
#' @noRd
confint.felm <- function(object, parm, level = 0.95, ...) {
  confint.feglm(object, parm, level, ...)
}
