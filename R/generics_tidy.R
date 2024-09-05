#' @importFrom generics tidy
#' @export
generics::tidy

#' @title Tidy method for 'feglm' objects
#' @description Integration with the 'broom' package
#' @export
#' @noRd
tidy.feglm <- function(x, conf.int = FALSE, conf.level = 0.95, ...) {
  res <- summary(x)$cm
  colnames(res) <- c("estimate", "std.error", "statistic", "p.value")

  if (conf.int) {
    res <- cbind(res, res[ , "estimate"] - 1.96 * res[ , "std.error"])
    res <- cbind(res, res[ , "estimate"] + 1.96 * res[ , "std.error"])
    colnames(res) <- c("estimate", "std.error", "statistic", "p.value",
      "conf.low", "conf.high")
  }

  res <- as.data.frame(res)

  class(res) <- c("tbl_df", "tbl", "data.frame")
  res
}

#' @title Tidy method for 'felm' objects
#' @description Integration with the 'broom' package
#' @export
#' @noRd
tidy.felm <- function(x, conf.int = FALSE, conf.level = 0.95, ...) {
  tidy.feglm(x, conf.int, conf.level, ...)
}
