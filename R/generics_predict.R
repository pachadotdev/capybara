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

  # Initialize variables that will be assigned by helper functions
    data <- x <- NULL

    # These variables will be assigned by the helper functions
    model_frame_(newdata, object$formula, NULL)
    # Extract fixed effects variables using proper pipe parsing
    k_vars <- parse_formula_pipe_(object$formula)
    data <- transform_fe_(data, object$formula, k_vars)

    # x, nms_sp, p are assigned by model_response_
    model_response_(data, object$formula)

    # Try to use cached fixed effects first (most efficient)
    if (!is.null(object[["__cached_fixed_effects__"]])) {
      fes <- object[["__cached_fixed_effects__"]]
    } else {
      # Compute and cache fixed effects for future use
      fes <- tryCatch({
        result <- fixed_effects(object)
        # Cache the result for future predictions
        object[["__cached_fixed_effects__"]] <- result
        result
      }, error = function(e) {
        warning("Could not compute fixed effects: ", e$message, 
                ". Predictions may be less accurate.")
        list()
      })
    }

    # Compute fixed effects contributions for new data with improved error handling
    fes2 <- list()
    missing_levels <- character(0)
    
    if (length(fes) > 0) {
      for (name in names(fes)) {
        fe <- fes[[name]]
        data_values <- data[[name]]
        
        if (is.matrix(fe)) {
          fe_names <- rownames(fe)
          fe_values <- fe[, 1]
          names(fe_values) <- fe_names
        } else {
          fe_values <- fe
          fe_names <- names(fe_values)
        }
        
        # Check for missing levels in new data
        missing_idx <- !data_values %in% fe_names
        if (any(missing_idx)) {
          missing_vals <- unique(data_values[missing_idx])
          missing_levels <- c(missing_levels, paste0(name, ":", missing_vals))
        }
        
        # Match values and handle missing levels
        matched_values <- fe_values[match(data_values, fe_names)]
        matched_values[is.na(matched_values)] <- 0  # Set missing levels to 0
        fes2[[name]] <- matched_values
      }
    }
    
    # Warn about missing levels
    if (length(missing_levels) > 0) {
      warning("New data contains levels not seen in training data: ", 
              paste(missing_levels, collapse = ", "), 
              ". These levels will be treated as having zero fixed effect.")
    }

    # Replace NA coefficients with 0 for prediction
    coef0 <- object$coefficients
    coef0[is.na(coef0)] <- 0
    
    # Compute linear predictor (x is assigned by model_response_)
    eta <- x %*% coef0
    if (length(fes2) > 0) {
      eta <- eta + Reduce("+", fes2)
    }
  } else {
    # Use stored values from the fitted model
    if (type == "response") {
      eta <- object[["fitted.values"]]  # Already on response scale
    } else {
      eta <- object[["eta"]]  # Linear predictor (link scale)
    }
  }

  # Apply inverse link if response scale is requested
  if (type == "response" && !is.null(newdata)) {
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

    fes <- object[["fixed.effects"]]
    fes2 <- list()

    for (name in names(fes)) {
      fe <- fes[[name]]
      
      fe_values <- fe
      fe_names <- names(fe_values)
      
      # Match values and handle missing levels
      data_values <- data[[name]]
      matched_values <- fe_values[match(data_values, fe_names)]
      # matched_values[is.na(matched_values)] <- 0  # Set missing levels to 0
      fes2[[name]] <- matched_values
    }

    # Replace NA coefficients with 0 for prediction
    coef0 <- object$coefficients
    coef0[is.na(coef0)] <- 0

    return(as.numeric(x %*% coef0 + Reduce("+", fes2)))
  } else {
    return(object[["fitted.values"]])
  }
}
