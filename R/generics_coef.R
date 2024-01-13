#' @title Extract estimates of average partial effects
#' @description \code{\link{coef.apes}} is a generic function which extracts
#'  estimates of the average partial effects from objects returned by
#'  \code{\link{apes}}.
#' @param object an object of class \code{"apes"}.
#' @param ... other arguments.
#' @return The function \code{\link{coef.apes}} returns a named vector of
#'  estimates of the average partial effects.
#' @seealso \code{\link{apes}}
#' @export
coef.apes <- function(object, ...) {
  object[["delta"]]
}

#' @title Extract estimates of structural parameters
#' @description \code{\link{coef.feglm}} is a generic function which extracts
#'  estimates of the structural parameters from objects returned by
#'  \code{\link{feglm}}.
#' @param object an object of class \code{"feglm"}.
#' @param ... other arguments.
#' @return The function \code{\link{coef.feglm}} returns a named vector of
#'  estimates of the structural parameters.
#' @seealso \code{\link{feglm}}
#' @export
coef.feglm <- function(object, ...) {
  object[["coefficients"]]
}

#' @title Extract estimates of structural parameters
#' @description \code{\link{coef.felm}} is a generic function which extracts
#'  estimates of the structural parameters from objects returned by
#'  \code{\link{felm}}.
#' @inherit coef.feglm
#' @return The function \code{\link{coef.felm}} returns a named vector of
#'  estimates of the structural parameters.
#' @seealso \code{\link{felm}}
#' @export
coef.felm <- function(object, ...) {
  object[["coefficients"]]
}

#' @title Extract coefficient matrix for average partial effects
#' @description \code{\link{coef.summary.apes}} is a generic function which
#'  extracts a coefficient matrix for average partial effects from objects
#'  returned by \code{\link{apes}}.
#' @param object an object of class \code{"summary.apes"}.
#' @param ... other arguments.
#' @return The function \code{\link{coef.summary.apes}} returns a named matrix
#'  of estimates related to the average partial effects.
#' @seealso \code{\link{apes}}
#' @export
coef.summary.apes <- function(object, ...) {
  object[["cm"]]
}

#' @title Extract coefficient matrix for structural parameters
#' @description \code{\link{coef.summary.feglm}} is a generic function which
#'  extracts a coefficient matrix for structural parameters from objects
#'  returned by \code{\link{feglm}}.
#' @param object an object of class \code{"summary.feglm"}.
#' @param ... other arguments.
#' @return The function \code{\link{coef.summary.feglm}} returns a named matrix
#'  of estimates related to the structural parameters.
#' @seealso \code{\link{feglm}}
#' @export
coef.summary.feglm <- function(object, ...) {
  object[["cm"]]
}

#' @title Extract coefficient matrix for structural parameters
#' @description generic function which extracts a coefficient matrix for
#'  structural parameters from objects returned by \code{\link{feglm}}.
#' @inherit coef.summary.feglm
#' @return The function \code{\link{coef.summary.feglm}} returns a named matrix
#'  of estimates related to the structural parameters.
#' @seealso \code{\link{felm}}
#' @export
coef.summary.felm <- function(object, ...) {
  object[["cm"]]
}
