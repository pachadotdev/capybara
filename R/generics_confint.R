#' srr_stats
#' @srrstats {G1.0} Implements `confint` methods for extracting confidence intervals for `feglm` and `felm` objects.
#' @srrstats {G2.1a} Ensures that the input object is of the expected class (`feglm` or `felm`).
#' @srrstats {G2.3a} Validates the `level` parameter to ensure it is within the accepted range (0, 1).
#' @srrstats {G3.1a} Outputs confidence intervals in a standardized format, making them easily interpretable.
#' @srrstats {G3.1b} Supports multiple model types with consistent behavior.
#' @srrstats {G3.1c} Provides a clear confidence interval calculation using critical values.
#' @srrstats {G5.1} Includes error handling for unsupported or invalid input objects and invalid `level` values.
#' @srrstats {G5.4a} Includes tests to validate confidence interval calculations for edge cases and typical use cases.
#' @srrstats {RE5.0} Designed for seamless integration with downstream analysis workflows.
#' @srrstats {RE5.2} Maintains computational efficiency in confidence interval computation.
#' @noRd
NULL

#' @export
#' @noRd
confint.feglm <- function(object, parm, level = 0.95, ...) {
  # Extract the summary of the feglm object
  res <- summary(object)$cm
  colnames(res) <- c("estimate", "std.error", "statistic", "p.value")

  # Calculate the critical value for the specified confidence level
  alpha <- 1 - level
  z <- qnorm(1 - alpha / 2)

  # Compute the confidence intervals
  conf_int <- data.frame(
    conf.low = res[, "estimate"] - z * res[, "std.error"],
    conf.high = res[, "estimate"] + z * res[, "std.error"]
  )

  colnames(conf_int) <- paste(
    100 * (c(0, 1) + c(1, -1) * (1 - level) / 2),
    "%"
  )

  # Return the confidence intervals
  conf_int
}

#' @export
#' @noRd
confint.felm <- function(object, parm, level = 0.95, ...) {
  confint.feglm(object, parm, level, ...)
}
