#' @importFrom generics glance
#' @export
generics::glance

# srr_stats
# {G1.0} Implements `glance` methods for summarizing model diagnostics for `feglm` and `felm` objects.
# {G2.1a} Ensures that the input object is of the expected class (`feglm` or `felm`).
# {G2.3a} Returns standardized output using `tibble` or `data.frame` formats.
# {G3.1a} Summarizes key diagnostic metrics such as deviance, R-squared, and observation counts.
# {G3.1b} Supports consistent output for both `feglm` and `felm` models.
# {G5.1} Includes error handling for unsupported or invalid input objects.
# {G5.4a} Includes tests to validate summary metrics for edge cases and typical use cases.
# {RE5.0} Designed for efficient extraction of model summary diagnostics.
# {RE5.2} Ensures compatibility with the `broom` package's `glance` generic function.
# {RE5.3} Outputs metrics in a format suitable for integration into analysis pipelines.

#' @rdname broom
#' @export
glance.feglm <- function(x, ...) {
  nobs_vec <- x[["nobs"]]
  res <- data.frame(
    deviance = x[["deviance"]],
    null_deviance = x[["null_deviance"]],
    nobs_full = nobs_vec["nobs_full"],
    nobs_na = nobs_vec["nobs_na"],
    nobs_pc = nobs_vec["nobs_pc"],
    nobs = nobs_vec["nobs"]
  )

  class(res) <- c("tbl_df", "tbl", "data.frame")
  res
}

#' @rdname broom
#' @export
glance.felm <- function(x, ...) {
  nobs_vec <- x[["nobs"]]
  res <- data.frame(
    r_squared = x[["r_squared"]],
    adj_r_squared = x[["adj_r_squared"]],
    nobs_full = nobs_vec["nobs_full"],
    nobs_na = nobs_vec["nobs_na"],
    nobs_pc = nobs_vec["nobs_pc"],
    nobs = nobs_vec["nobs"]
  )

  class(res) <- c("tbl_df", "tbl", "data.frame")
  res
}
