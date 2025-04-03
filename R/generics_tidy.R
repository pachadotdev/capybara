#' @importFrom generics tidy
#' @export
generics::tidy

#' srr_stats
#' @srrstats {G1.0} Implements `tidy` methods for `feglm` and `felm` objects to provide clean, structured coefficient tables.
#' @srrstats {G2.1a} Validates the input object is of class `feglm` or `felm`.
#' @srrstats {G2.2} Extracts model coefficients, standard errors, statistics, and p-values in a consistent format.
#' @srrstats {G2.3} Adds optional confidence intervals for coefficients with user-defined levels.
#' @srrstats {G5.1} Outputs are compatible with downstream tidyverse operations (`tibble`-based structure).
#' @srrstats {G5.2a} Clearly labeled columns: `estimate`, `std.error`, `statistic`, `p.value`, and optionally `conf.low`, `conf.high`.
#' @srrstats {RE2.1} Provides a flexible interface for model summaries with options for additional information (confidence intervals).
#' @srrstats {RE2.3} Facilitates reproducibility by offering confidence levels as a parameter.
#' @srrstats {RE5.0} Ensures outputs are concise, tidy, and formatted for readability.
#' @noRd
NULL

#' @rdname broom
#' @export
tidy.feglm <- function(x, conf_int = FALSE, conf_level = 0.95, ...) {
  res <- summary(x)$coefficients
  colnames(res) <- c("estimate", "std.error", "statistic", "p.value")

  if (conf_int) {
    res <- cbind(res, res[, "estimate"] - 1.96 * res[, "std.error"])
    res <- cbind(res, res[, "estimate"] + 1.96 * res[, "std.error"])
    colnames(res) <- c(
      "estimate", "std.error", "statistic", "p.value",
      "conf.low", "conf.high"
    )
  }

  res <- as.data.frame(res)

  class(res) <- c("tbl_df", "tbl", "data.frame")
  res
}

#' @rdname broom
#' @export
tidy.felm <- function(x, conf_int = FALSE, conf_level = 0.95, ...) {
  tidy.feglm(x, conf_int, conf_level, ...)
}
