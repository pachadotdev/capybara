#' @title Extract \code{feglm} fitted values
#' @description \code{\link{fitted.feglm}} is a generic function which extracts
#'  fitted values from an object returned by \code{\link{feglm}}.
#' @param object an object of class \code{"feglm"}.
#' @param ... other arguments.
#' @return The function \code{\link{fitted.feglm}} returns a vector of fitted
#'  values.
#' @seealso \code{\link{feglm}}
#' @export
fitted.feglm <- function(object, ...) {
  object[["family"]][["linkinv"]](object[["eta"]])
}

#' @title Extract \code{felm} fitted values
#' @description \code{\link{fitted.felm}} is a generic function which extracts
#' fitted values from an object returned by \code{\link{feglm}}.
#' @inherit fitted.feglm
#' @return The function \code{\link{fitted.felm}} returns a vector of fitted
#'  values.
#' @seealso \code{\link{felm}}
#' @export
fitted.felm <- function(object, ...) {
  object[["fitted.values"]]
}
