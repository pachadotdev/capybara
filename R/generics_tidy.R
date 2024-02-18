#' @importFrom generics tidy
#' @export
generics::tidy

#' @export
#' @noRd
tidy.feglm <- function(x, conf.int = FALSE, conf.level = 0.95, ...) {
  result <- summary(x)$cm %>%
    as_tibble(rownames = "term") %>%
    rename(
      estimate = Estimate,
      std.error = `Std. Error`,
      statistic = `z value`,
      p.value = `Pr(>|z|)`
    )

  if (conf.int) {
    result <- result %>%
      mutate(
        conf.low = estimate - 1.96 * std.error,
        conf.high = estimate + 1.96 * std.error
      )
  }

  result
}

#' @export
#' @noRd
tidy.felm <- function(x, conf.int = FALSE, conf.level = 0.95, ...) {
  tidy.feglm(x, conf.int, conf.level, ...)
}
