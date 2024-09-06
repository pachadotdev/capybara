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
  conf.int <- data.frame(
    conf.low = res[, "estimate"] - z * res[, "std.error"],
    conf.high = res[, "estimate"] + z * res[, "std.error"]
  )

  colnames(conf.int) <- paste(100 * (c(0, 1) + c(1, -1) * (1 - level) / 2),
    "%")

  # Return the confidence intervals
  conf.int
}

#' @export
#' @noRd
confint.felm <- function(object, parm, level = 0.95, ...) {
  confint.feglm(object, parm, level, ...)
}
