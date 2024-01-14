#' @export
#' @noRd
coef.apes <- function(object, ...) {
  object[["delta"]]
}

#' @export
#' @noRd
coef.feglm <- function(object, ...) {
  object[["coefficients"]]
}

#' @export
#' @noRd
coef.felm <- function(object, ...) {
  object[["coefficients"]]
}

#' @export
#' @noRd
coef.summary.apes <- function(object, ...) {
  object[["cm"]]
}

#' @export
#' @noRd
coef.summary.feglm <- function(object, ...) {
  object[["cm"]]
}

#' @export
#' @noRd
coef.summary.felm <- function(object, ...) {
  object[["cm"]]
}
