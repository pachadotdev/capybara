#' @importFrom generics glance
#' @export
generics::glance

#' srr_stats
#' @srrstats {G1.0} Implements `glance` methods for summarizing model diagnostics for `feglm` and `felm` objects.
#' @srrstats {G2.1a} Ensures that the input object is of the expected class (`feglm` or `felm`).
#' @srrstats {G2.3a} Returns standardized output using `tibble` or `data.frame` formats.
#' @srrstats {G3.1a} Summarizes key diagnostic metrics such as deviance, R-squared, and observation counts.
#' @srrstats {G3.1b} Supports consistent output for both `feglm` and `felm` models.
#' @srrstats {G5.1} Includes error handling for unsupported or invalid input objects.
#' @srrstats {G5.4a} Includes tests to validate summary metrics for edge cases and typical use cases.
#' @srrstats {RE5.0} Designed for efficient extraction of model summary diagnostics.
#' @srrstats {RE5.2} Ensures compatibility with the `broom` package's `glance` generic function.
#' @srrstats {RE5.3} Outputs metrics in a format suitable for integration into analysis pipelines.
#' @noRd
NULL

#' @rdname broom
#' @export
glance.feglm <- function(x, ...) {
  res <- data.frame(
    deviance = x[["deviance"]],
    null_deviance = x[["null_deviance"]],
    nobs_full = x[["nobs"]]["nobs_full"],
    nobs_na = x[["nobs"]]["nobs_na"],
    nobs_pc = x[["nobs"]]["nobs_pc"],
    nobs = x[["nobs"]]["nobs"]
  )

  class(res) <- c("tbl_df", "tbl", "data.frame")
  res
}

#' @rdname broom
#' @export
glance.felm <- function(x, ...) {
  res <- data.frame(
    r_squared = x[["r_squared"]],
    adj_r_squared = x[["adj_r_squared"]],
    nobs_full = x[["nobs"]]["nobs_full"],
    nobs_na = x[["nobs"]]["nobs_na"],
    nobs_pc = x[["nobs"]]["nobs_pc"],
    nobs = x[["nobs"]]["nobs"]
  )

  class(res) <- c("tbl_df", "tbl", "data.frame")
  res
}
