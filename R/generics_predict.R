#' srr_stats
#' @srrstats {G1.0} Implements `predict` methods for `feglm` and `felm` objects, similar to base R methods.
#' @srrstats {G2.1a} Ensures input objects are of the expected class (`feglm` or `felm`).
#' @srrstats {G2.3a} Provides options for output type (`link`, `response`, or `terms`) via a standardized `type`
#'  argument.
#' @srrstats {G2.3b} Handles missing or invalid new data gracefully with appropriate checks and error messages.
#' @srrstats {G3.1a} Computes predicted values for both new and existing data sets.
#' @srrstats {G3.1b} Supports fixed-effects predictions by accounting for levels in the data.
#' @srrstats {G3.4a} Includes an option for type-specific predictions (e.g., `link` vs. `response`).
#' @srrstats {G5.2a} Tests include validation of predictions against known values and edge cases.
#' @srrstats {G5.4a} Outputs predictions in a format compatible with standard R workflows.
#' @srrstats {RE4.9} The predicted values for the model data or new data are returned as a vector with `predict()`.
#' @srrstats {RE4.16} The fixed effects are passed to the `predict()` function to add the group-specific effects to the
#'  predictions.
#' @srrstats {RE5.0} Ensures computational efficiency in handling both `feglm` and `felm` prediction workflows.
#' @srrstats {RE5.2} Integrates seamlessly with user-provided data for generating predictions.
#' @srrstats {RE5.3} Provides predictable and consistent output types for downstream analysis.
#' @noRd
NULL

#' @title Predict method for 'feglm' objects
#' @description Similar to the 'predict' method for 'glm' objects but returns response predictions as default.
#' @export
#' @noRd
predict.feglm <- function(
  object,
  newdata = NULL,
  type = c("response", "link"),
  ...
) {
  type <- match.arg(type)

  if (!is.null(newdata)) {
    check_data_(newdata)

    # For prediction, we only need the RHS variables, not the response
    # Extract predictor variables and fixed effects
    formula <- object$formula
    k_vars <- attr(terms(formula, rhs = 2L), "term.labels")

    # Get all variables needed (predictors + fixed effects)
    pred_vars <- attr(terms(formula, rhs = 1L), "term.labels")
    all_vars <- c(pred_vars, k_vars)

    # Extract and prepare data
    data <- newdata[, all_vars, drop = FALSE]
    
    # Keep track of original row indices for NA handling
    original_rows <- seq_len(nrow(data))
    
    # Remove rows with NA in any required variable
    complete_cases <- complete.cases(data)
    if (!all(complete_cases)) {
      data <- data[complete_cases, , drop = FALSE]
      original_rows <- original_rows[complete_cases]
    }

    # Transform fixed effects to factors
    data <- transform_fe_(data, formula, k_vars)

    # Create design matrix from predictors only
    if (length(pred_vars) > 0) {
      pred_formula <- reformulate(pred_vars, response = NULL)
      X <- model.matrix(pred_formula, data = data)[, -1, drop = FALSE]
      nms_sp <- colnames(X)
    } else {
      X <- matrix(0, nrow = nrow(data), ncol = 0)
      nms_sp <- character(0)
    }

    # Check if model has an intercept (no fixed effects case)
    # When there are no FEs, C++ adds an intercept, so coef_table includes "(Intercept)"
    coef_table <- object$coef_table
    has_intercept <- "(Intercept)" %in% rownames(coef_table)
    if (has_intercept) {
      # Add intercept column to X
      X <- cbind(1, X)
    }

    fes <- object[["fixed_effects"]]
    fes_names <- names(fes)

    # Fixed effects are normalized (first level = 0) in get_alpha()
    # C++ fitted values use these normalized FEs via accumulate_fixed_effects()
    # So predictions should also use normalized FEs directly
    fes2 <- setNames(
      lapply(fes_names, function(name) {
        # Match values and handle missing levels
        matched_values <- fes[[name]][match(data[[name]], names(fes[[name]]))]
        matched_values[is.na(matched_values)] <- 0 # Set missing levels to 0
        matched_values
      }),
      fes_names
    )

    # Replace NA coefficients with 0 for prediction
    coef0 <- coef_table[, 1]
    coef0[is.na(coef0)] <- 0

    if (length(fes) > 0) {
      eta <- X %*% coef0 + Reduce("+", fes2)
    } else {
      eta <- X %*% coef0
    }

    # Add offset if present in the model
    # Re-evaluate the offset on newdata using the original offset specification
    if (!is.null(object[["offset_spec"]])) {
      offset_spec <- object[["offset_spec"]]
      if (inherits(offset_spec, "formula")) {
        # Evaluate offset formula on newdata
        offset_vars <- attr(terms(offset_spec, data = newdata), "term.labels")
        offset_newdata <- eval(parse(text = offset_vars), envir = newdata)
      } else if (is.numeric(offset_spec)) {
        # If offset was originally a vector, we need it to match newdata length
        if (length(offset_spec) == nrow(data)) {
          offset_newdata <- offset_spec[rownames(data)]
        } else {
          stop(
            "Cannot apply numeric offset to newdata: length mismatch.",
            call. = FALSE
          )
        }
      } else {
        offset_newdata <- rep(0.0, nrow(data))
      }
      eta <- eta + offset_newdata
    }
    
    # Handle NA values - create result vector with NAs where appropriate
    result_eta <- rep(NA_real_, nrow(newdata))
    result_eta[original_rows] <- as.vector(eta)
    eta <- result_eta
  } else {
    eta <- object[["eta"]]
  }

  if (type == "response") {
    fam <- object[["family"]]
    eta <- fam[["linkinv"]](eta)
  }

  # Convert to vector and assign names
  eta <- as.vector(eta)
  # Prefer row names from the prediction data (or original object), fall back to sequential
  if (!is.null(newdata)) {
    rn <- rownames(newdata) # Use original newdata rownames, not filtered data
  } else {
    rn <- rownames(object$data)
  }
  if (!is.null(rn)) {
    names(eta) <- rn
  } else {
    names(eta) <- seq_along(eta)
  }

  eta
}

#' @title Predict method for 'felm' objects
#' @description Similar to the 'predict' method for 'lm' objects
#' @export
#' @noRd
predict.felm <- function(
  object,
  newdata = NULL,
  type = c("response", "terms"),
  ...
) {
  type <- match.arg(type)

  if (!is.null(newdata)) {
    check_data_(newdata)

    # For prediction, we only need the RHS variables, not the response
    # Extract predictor variables and fixed effects
    formula <- object$formula

    # Check if model has fixed effects (rhs = 2)
    has_fe <- length(formula)[2] >= 2

    if (has_fe) {
      fe_names <- attr(terms(formula, rhs = 2L), "term.labels")
    } else {
      fe_names <- character(0)
    }

    # Get all variables needed (predictors + fixed effects)
    pred_vars <- attr(terms(formula, rhs = 1L), "term.labels")
    all_vars <- c(pred_vars, fe_names)

    # Extract and prepare data
    data <- newdata[, all_vars, drop = FALSE]
    
    # Keep track of original row indices for NA handling
    original_rows <- seq_len(nrow(data))
    
    # Remove rows with NA in any required variable
    complete_cases <- complete.cases(data)
    if (!all(complete_cases)) {
      data <- data[complete_cases, , drop = FALSE]
      original_rows <- original_rows[complete_cases]
    }

    # Transform fixed effects to factors
    data <- transform_fe_(data, formula, fe_names)

    # Create design matrix from predictors only
    if (length(pred_vars) > 0) {
      pred_formula <- reformulate(pred_vars, response = NULL)
      X <- model.matrix(pred_formula, data = data)[, -1, drop = FALSE]
    } else {
      X <- matrix(0, nrow = nrow(data), ncol = 0)
    }

    # Check if model has an intercept (no fixed effects case)
    # When there are no FEs, C++ adds an intercept, so coef_table includes "(Intercept)"
    coef_table <- object$coef_table
    has_intercept <- "(Intercept)" %in% rownames(coef_table)
    if (has_intercept) {
      # Add intercept column to X
      X <- cbind(1, X)
    }

    fes <- object[["fixed_effects"]]
    fes_names <- names(fes)

    # Denormalize fixed effects for prediction
    # The FEs were normalized so fe1[1] = 0
    # To match lm() predictions, we don't need to denormalize
    # because lm() also uses one level as reference (sets it to 0)
    # The normalization is already compatible with lm()'s reference level approach

    fes2 <- setNames(
      lapply(fes_names, function(name) {
        # Match values and handle missing levels
        matched_values <- fes[[name]][match(data[[name]], names(fes[[name]]))]
        matched_values[is.na(matched_values)] <- 0 # Set missing levels to 0
        matched_values
      }),
      fes_names
    )

    # Replace NA coefficients with 0 for prediction
    coef0 <- coef_table[, 1]
    coef0[is.na(coef0)] <- 0

    if (length(fes) > 0) {
      y <- X %*% coef0 + Reduce("+", fes2)
    } else {
      y <- X %*% coef0
    }

    # Add offset if present (felm typically doesn't use offsets, but handle for consistency)
    if (!is.null(object[["offset_spec"]])) {
      offset_spec <- object[["offset_spec"]]
      if (inherits(offset_spec, "formula")) {
        offset_vars <- attr(terms(offset_spec, data = newdata), "term.labels")
        offset_newdata <- eval(parse(text = offset_vars), envir = newdata)
        y <- y + offset_newdata
      }
    }
    
    # Handle NA values - create result vector with NAs where appropriate
    result <- rep(NA_real_, nrow(newdata))
    result[original_rows] <- as.vector(y)
    y <- result
  } else {
    y <- object[["fitted_values"]]
  }

  # Ensure y is a vector and assign names
  y <- as.vector(y)
  # Prefer row names from the prediction data (or original object), fall back to sequential
  if (!is.null(newdata)) {
    rn <- rownames(newdata) # Use original newdata rownames, not filtered data
  } else {
    rn <- rownames(object$data)
  }
  if (!is.null(rn)) {
    names(y) <- rn
  } else {
    names(y) <- seq_along(y)
  }

  y
}
