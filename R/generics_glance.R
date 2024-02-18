#' @importFrom generics glance
#' @export
generics::glance

#' @export
#' @noRd
glance.feglm <- function(x, ...) {
  with(
    summary(x),
    tibble(
      deviance = deviance,
      null.deviance = null.deviance,
      nobs.full = nobs["nobs.full"],
      nobs.na = nobs["nobs.na"],
      nobs.pc = nobs["nobs.pc"],
      nobs = nobs["nobs"]
    )
  )
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
