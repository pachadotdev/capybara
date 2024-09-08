#' @importFrom generics tidy
#' @export
generics::tidy

#' @rdname broom
#' @export
tidy.feglm <- function(x, conf_int = FALSE, conf_level = 0.95, ...) {
  res <- summary(x)$cm
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
