#' @importFrom generics glance
#' @export
generics::glance

#' @export
#' @noRd
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

#' @export
#' @noRd
glance.felm <- function(x, ...) {
  with(
    summary(x),
    tibble(
      r.squared = r.squared,
      adj.r.squared = adj.r.squared,
      nobs_full = nobs["nobs_full"],
      nobs_na = nobs["nobs_na"],
      nobs_pc = nobs["nobs_pc"],
      nobs = nobs["nobs"]
    )
  )
}
