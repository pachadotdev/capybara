#' @title Predict method for 'feglm' objects
#' @description Similar to the 'predict' method for 'glm' objects
#' @export
#' @noRd
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

#' @title Predict method for 'felm' objects
#' @description Similar to the 'predict' method for 'lm' objects
#' @export
#' @noRd
predict.felm <- function(object, ...) {
  object[["fitted.values"]]
}
