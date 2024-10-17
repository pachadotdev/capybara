#' @export
#' @srrstats {RE4.2} *Model coefficients (via `coef()` / `coefficients()`)*
#' @noRd
coef.apes <- function(object, ...) {
  object[["delta"]]
}

#' @export
#' @srrstats {RE4.2} *Model coefficients (via `coef()` / `coefficients()`)*
#' @noRd
coef.feglm <- function(object, ...) {
  object[["coefficients"]]
}

#' @export
#' @srrstats {RE4.2} *Model coefficients (via `coef()` / `coefficients()`)*
#' @noRd
coef.felm <- function(object, ...) {
  object[["coefficients"]]
}

#' @export
#' @srrstats {RE4.2} *Model coefficients (via `coef()` / `coefficients()`)*
#' @noRd
coef.summary.apes <- function(object, ...) {
  object[["cm"]]
}

#' @export
#' @srrstats {RE4.2} *Model coefficients (via `coef()` / `coefficients()`)*
#' @noRd
coef.summary.feglm <- function(object, ...) {
  object[["cm"]]
}

#' @export
#' @srrstats {RE4.2} *Model coefficients (via `coef()` / `coefficients()`)*
#' @noRd
coef.summary.felm <- function(object, ...) {
  object[["cm"]]
}
