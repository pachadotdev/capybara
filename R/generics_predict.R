#' srr_stats (tests)
#' @srrstats {G2.3} For univariate character input:
#' @srrstats {G2.3a} Use `match.arg()` or equivalent where applicable to only
#'  permit expected values.
#' @srrstats {G2.3b} Either: use `tolower()` or equivalent to ensure input of
#'  character parameters is not case dependent; or explicitly document that
#'  parameters are strictly case-sensitive.
#' @srrstats {RE4.9} Modelled values of response variables.
#' @srrstats {RE4.12} Where appropriate, functions used to transform input data,
#'  and associated inverse transform functions.
#' @srrstats {RE4.13} Predictor variables, and associated "metadata" where
#'  applicable. (via `confint()`)
#' @srrstats {RE4.18} Regression Software may also implement `summary` methods
#'  for model objects, and in particular should implement distinct `summary`
#'  methods for any cases in which calculation of summary statistics is
#'  computationally non-trivial (for example, for bootstrapped estimates of
#'  confidence intervals).
#' @noRd
NULL

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
