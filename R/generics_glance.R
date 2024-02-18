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
      null.deviance = null.deviance,
      nobs.full = nobs["nobs.full"],
      nobs.na = nobs["nobs.na"],
      nobs.pc = nobs["nobs.pc"],
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
      nobs.full = nobs["nobs.full"],
      nobs.na = nobs["nobs.na"],
      nobs.pc = nobs["nobs.pc"],
      nobs = nobs["nobs"]
    )
  )
}
