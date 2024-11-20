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
predict.feglm <- function(object, newdata = NULL, type = c("link", "response"), ...) {
  type <- match.arg(type)

  if (!is.null(newdata)) {
    check_data_(newdata)
    
    data <- NA # just to avoid global variable warning
    lhs <- NA
    nobs_na <- NA
    nobs_full <- NA
    model_frame_(newdata, object$formula, NULL)
    check_response_(data, lhs, object$family)
    k_vars <- attr(terms(object$formula, rhs = 2L), "term.labels")
    data <- transform_fe_(data, object$formula, k_vars)

    x <- NA
    nms_sp <- NA
    p <- NA
    model_response_(data, object$formula)

    fes <- fixed_effects(object)
    fes2 <- list()

    for (name in names(fes)) {
      # # match the FE rownames and replace each level in the data with the FE
      fe <- fes[[name]]
      fes2[[name]] <- fe[match(data[[name]], rownames(fe)), ]
    }

    eta <- x %*% object$coefficients + Reduce("+", fes2)
  } else {
    eta <- object[["eta"]]
  }

  if (type == "response") {
    eta <- object[["family"]][["linkinv"]](eta)
  }

  as.numeric(eta)
}

#' @title Predict method for 'felm' objects
#' @description Similar to the 'predict' method for 'lm' objects
#' @export
#' @noRd
predict.felm <- function(object, newdata = NULL, type = c("response", "terms"), ...) {
  type <- match.arg(type)

  if (!is.null(newdata)) {
    check_data_(newdata)

    data <- NA # just to avoid global variable warning
    lhs <- NA
    nobs_na <- NA
    nobs_full <- NA
    model_frame_(newdata, object$formula, NULL)
    k_vars <- attr(terms(object$formula, rhs = 2L), "term.labels")
    data <- transform_fe_(data, object$formula, k_vars)

    x <- NA
    nms_sp <- NA
    p <- NA
    model_response_(data, object$formula)

    fes <- fixed_effects(object)
    fes2 <- list()

    for (name in names(fes)) {
      # # match the FE rownames and replace each level in the data with the FE
      fe <- fes[[name]]
      fes2[[name]] <- fe[match(data[[name]], rownames(fe)), ]
    }

    yhat <- x %*% object$coefficients + Reduce("+", fes2)
  } else {
    yhat <- object[["fitted.values"]]
  }

  as.numeric(yhat)
}
