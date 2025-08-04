#' srr_stats
#' @srrstats {G1.0} Implements `predict` methods for `feglm` and `felm` objects, similar to base R methods.
#' @srrstats {G2.1a} Ensures input objects are of the expected class (`feglm` or `felm`).
#' @srrstats {G2.3a} Provides options for output type (`link`, `response`, or `terms`) via a standardized `type` argument.
#' @srrstats {G2.3b} Handles missing or invalid new data gracefully with appropriate checks and error messages.
#' @srrstats {G3.1a} Computes predicted values for both new and existing data sets.
#' @srrstats {G3.1b} Supports fixed-effects predictions by accounting for levels in the data.
#' @srrstats {G3.4a} Includes an option for type-specific predictions (e.g., `link` vs. `response`).
#' @srrstats {G5.2a} Tests include validation of predictions against known values and edge cases.
#' @srrstats {G5.4a} Outputs predictions in a format compatible with standard R workflows.
#' @srrstats {RE4.9} The predicted values for the model data or new data are returned as a vector with `predict()`.
#' @srrstats {RE4.16} The fixed effects are passed to the `predict()` function to add the group-specific effects to the predictions.
#' @srrstats {RE5.0} Ensures computational efficiency in handling both `feglm` and `felm` prediction workflows.
#' @srrstats {RE5.2} Integrates seamlessly with user-provided data for generating predictions.
#' @srrstats {RE5.3} Provides predictable and consistent output types for downstream analysis.
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

    X <- NA
    nms_sp <- NA
    p <- NA
    model_response_(data, object$formula)

    fes <- object[["fixed.effects"]]
    fes2 <- list()

    for (name in names(fes)) {
      fe <- fes[[name]]

      fe_values <- fe
      fe_names <- names(fe_values)

      # Match values and handle missing levels
      data_values <- data[[name]]
      matched_values <- fe_values[match(data_values, fe_names)]
      matched_values[is.na(matched_values)] <- 0 # Set missing levels to 0
      fes2[[name]] <- matched_values
    }

    eta <- X %*% object$coefficients + Reduce("+", fes2)
  } else {
    eta <- object[["eta"]]
  }

  if (type == "response") {
    eta <- object[["family"]][["linkinv"]](eta)
  }

  # Convert to vector and assign names
  eta <- as.vector(eta)
  names(eta) <- seq_along(eta)

  eta
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
    fe_names <- attr(terms(object$formula, rhs = 2L), "term.labels")
    data <- transform_fe_(data, object$formula, fe_names)

    X <- NA
    nms_sp <- NA
    p <- NA
    model_response_(data, object$formula)

    fes <- object[["fixed.effects"]]
    fes2 <- list()

    for (name in names(fes)) {
      fe <- fes[[name]]

      fe_values <- fe
      fe_names <- names(fe_values)

      # Match values and handle missing levels
      data_values <- data[[name]]
      matched_values <- fe_values[match(data_values, fe_names)]
      matched_values[is.na(matched_values)] <- 0 # Set missing levels to 0
      fes2[[name]] <- matched_values
    }

    # Replace NA coefficients with 0 for prediction
    coef0 <- object$coefficients
    coef0[is.na(coef0)] <- 0

    y <- X %*% coef0 + Reduce("+", fes2)
  } else {
    y <- object[["fitted.values"]]
  }

  # Ensure y is a vector and assign names
  y <- as.vector(y)
  names(y) <- seq_along(y)

  y
}
