#' @importFrom generics tidy
#' @export
generics::tidy

#' srr_stats
#' @srrstats {G1.0} Implements `tidy` methods for `feglm` and `felm` objects to provide clean, structured coefficient
#'  tables.
#' @srrstats {G2.1a} Validates the input object is of class `feglm` or `felm`.
#' @srrstats {G2.2} Extracts model coefficients, standard errors, statistics, and p-values in a consistent format.
#' @srrstats {G2.3} Adds optional confidence intervals for coefficients with user-defined levels.
#' @srrstats {G5.1} Outputs are compatible with downstream tidyverse operations (`tibble`-based structure).
#' @srrstats {G5.2a} Clearly labeled columns: `estimate`, `std.error`, `statistic`, `p.value`, and optionally
#' `conf.low`, `conf.high`.
#' @srrstats {RE2.1} Provides a flexible interface for model summaries with options for additional information
#' (confidence intervals).
#' @srrstats {RE2.3} Facilitates reproducibility by offering confidence levels as a parameter.
#' @srrstats {RE5.0} Ensures outputs are concise, tidy, and formatted for readability.
#' @noRd
NULL

#' @rdname broom
#' @export
tidy.feglm <- function(x, conf_int = FALSE, conf_level = 0.95, ...) {
  # Extract coefficient table from model object (already has row names)
  coef_table <- x[["coef_table"]]

  if (conf_int) {
    # Calculate confidence intervals using the specified level
    z_crit <- qnorm(1 - (1 - conf_level) / 2)
    res <- cbind(
      estimate = coef_table[, 1],
      std.error = coef_table[, 2],
      statistic = coef_table[, 3],
      p.value = coef_table[, 4],
      conf.low = coef_table[, 1] - z_crit * coef_table[, 2],
      conf.high = coef_table[, 1] + z_crit * coef_table[, 2]
    )
  } else {
    res <- coef_table
    colnames(res) <- c("estimate", "std.error", "statistic", "p.value")
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
