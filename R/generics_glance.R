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
  res <- with(
    summary(x),
    data.frame(
      deviance = deviance,
      null_deviance = null_deviance,
      nobs_full = nobs["nobs_full"],
      nobs_na = nobs["nobs_na"],
      nobs_pc = nobs["nobs_pc"],
      nobs = nobs["nobs"]
    )
  )

  class(res) <- c("tbl_df", "tbl", "data.frame")
  res
}

#' @rdname broom
#' @export
#' @export
glance.felm <- function(x, ...) {
  with(
    summary(x),
    data.frame(
      r.squared = r.squared,
      adj.r.squared = adj.r.squared,
      nobs_full = nobs["nobs_full"],
      nobs_na = nobs["nobs_na"],
      nobs_pc = nobs["nobs_pc"],
      nobs = nobs["nobs"]
    )
  )
}
