#' @export
#' @noRd
fitted.feglm <- function(object, ...) {
  object[["family"]][["linkinv"]](object[["eta"]])
}

#' @export
#' @noRd
fitted.felm <- function(object, ...) {
  object[["fitted.values"]]
}
