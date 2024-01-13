#' @title Predict method for \code{feglm} fits
#' @description \code{\link{predict.feglm}} is a generic function which obtains
#'  predictions from an object returned by \code{\link{feglm}}.
#' @param object an object of class \code{"feglm"}.
#' @param type the type of prediction required. \code{"link"} is on the scale of
#'  the linear predictor whereas \code{"response"} is on the scale of the
#'  response variable. Default is \code{"link"}.
#' @param ... other arguments.
#' @return The function \code{\link{predict.feglm}} returns a vector of
#'  predictions.
#' @seealso \code{\link{feglm}}
#' @export
predict.feglm <- function(object, type = c("link", "response"), ...) {
  # Check validity of 'type'
  type <- match.arg(type)

  # Compute requested type of prediction
  x <- object[["eta"]]
  if (type == "response") {
    x <- object[["family"]][["linkinv"]](x)
  }

  # Return prediction
  x
}

#' @title Predict method for \code{felm} fits
#' @inherit predict.feglm
#' @export
predict.felm <- function(object, ...) {
  object[["fitted.values"]]
}
