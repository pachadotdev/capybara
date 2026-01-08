#' srr_stats
#' @srrstats {G1.0} Implements generalized linear models with high-dimensional fixed effects.
#' @srrstats {G2.1a} Ensures the input `formula` is correctly specified and includes fixed effects.
#' @srrstats {G2.1b} Validates that the input `data` is non-empty and of class `data.frame`.
#' @srrstats {G2.3a} Uses structured checks for parameters like `weights`, `control`, and starting values.
#' @srrstats {G2.4} Handles missing or perfectly classified data by appropriately excluding them.
#' @srrstats {G2.5} Ensures numerical stability and convergence for large datasets and complex models.
#' @srrstats {G3.1a} Provides robust support for a range of family functions like `gaussian`, `poisson`, and `binomial`.
#' @srrstats {G5.0} Ensures that identical input data and parameter settings consistently produce the same outputs,
#'  supporting reproducible workflows.
#' @srrstats {G5.1} Includes complete output elements (coefficients, deviance, etc.) for reproducibility.
#' @srrstats {G5.2a} Generates unique and descriptive error messages for invalid configurations or inputs.
#' @srrstats {G5.2b} Tracks optimization convergence during model fitting, providing detailed diagnostics for users to
#'  assess model stability.
#' @srrstats {G5.3} Optimizes computational efficiency for large datasets, employing parallel processing or streamlined
#'  algorithms where feasible.
#' @srrstats {G5.4} Benchmarks the scalability of model fitting against datasets of varying sizes to identify
#'  performance limits.
#' @srrstats {G5.4b} Documents performance comparisons with alternative implementations, highlighting strengths in
#'  accuracy or speed.
#' @srrstats {G5.4c} Employs memory-efficient data structures to handle large datasets without exceeding hardware
#'  constraints.
#' @srrstats {G5.5} Uses fixed random seeds for stochastic components, ensuring consistent outputs for analyses
#'  involving randomness.
#' @srrstats {G5.6} Benchmarks model fitting times and resource usage, providing users with insights into expected
#'  computational demands.
#' @srrstats {G5.6a} Demonstrates how parallel processing can reduce computation times while maintaining accuracy in
#'  results.
#' @srrstats {G5.7} Offers detailed, reproducible examples of typical use cases, ensuring users can replicate key
#'  functionality step-by-step.
#' @srrstats {G5.8} Includes informative messages or progress indicators during long-running computations to enhance
#'  user experience.
#' @srrstats {G5.8a} Warns users when outputs are approximate due to algorithmic simplifications or computational
#'  trade-offs.
#' @srrstats {G5.8b} Provides options to control the balance between computational speed and result precision,
#'  accommodating diverse user needs.
#' @srrstats {G5.8c} Documents which algorithm settings prioritize efficiency over accuracy, helping users make informed
#'  choices.
#' @srrstats {G5.8d} Clarifies the variability in results caused by parallel execution, particularly in randomized
#'  algorithms.
#' @srrstats {G5.9} Ensures all intermediate computations are accessible for debugging and troubleshooting during
#'  development or analysis.
#' @srrstats {G5.9a} Implements a debug mode that logs detailed information about the computational process for advanced
#'  users.
#' @srrstats {G5.9b} Validates correctness of results under debug mode, ensuring computational reliability across all
#'  scenarios.
#' @srrstats {RE1.0} Documents all assumptions inherent in the regression model, such as linearity, independence, and
#'  absence of multicollinearity.
#' @srrstats {RE1.1} Validates that input variables conform to expected formats, including numeric types for predictors
#'  and outcomes.
#' @srrstats {RE1.2} Provides options for handling missing data, including imputation or omission, and ensures users
#'  are informed of the chosen method.
#' @srrstats {RE1.3} Includes rigorous tests to verify model stability with edge cases, such as datasets with collinear
#'  predictors or extreme values.
#' @srrstats {RE1.3a} Adds specific tests for small datasets, ensuring the model remains robust under low-sample
#'  conditions.
#' @srrstats {RE1.4} Implements diagnostic checks to verify the assumptions of independence and homoscedasticity,
#'  essential for valid inference.
#' @srrstats {RE2.0} Labels all regression outputs, such as coefficients and standard errors, to ensure clarity and
#'  interpretability.
#' @srrstats {RE2.4} Quantifies uncertainty in regression coefficients using confidence intervals.
#' @srrstats {RE2.4a} Rejects perfect collinearity between independent variables.
#' @srrstats {RE2.4b} Rejects perfect collinearity between dependent and independent variables.
#' @srrstats {RE4.0} This returns a model-type object that is essentially a list with specific components and
#'  attributes.
#' @srrstats {RE4.1} Identifies outliers and influential data points that may unduly impact regression results, offering
#'  visualization tools.
#' @srrstats {RE4.6} Includes standard metrics such as R-squared and RMSE to help users evaluate model performance.
#' @srrstats {RE4.7} Tests sensitivity to hyperparameter choices in regularized or complex regression models.
#' @srrstats {RE4.14} Uses simulated datasets to test the reproducibility and robustness of regression results.
#' @srrstats {RE5.0} Optimized for scaling to large datasets with high-dimensional fixed effects.
#' @srrstats {RE5.1} Efficiently projects out fixed effects using auxiliary indexing structures.
#' @srrstats {RE5.2} Provides detailed warnings and error handling for convergence and dependence issues.
#' @srrstats {RE5.3} Thoroughly documents interactions between model features, inputs, and controls.
#' @srrstats {RE7.4} Provides comprehensive examples that demonstrate proper usage of the regression functions,
#'  covering input preparation, function execution, and result interpretation.
#' @noRd
NULL

#' @title GLM fitting with high-dimensional k-way fixed effects
#'
#' @description \link{feglm} can be used to fit generalized linear models with many high-dimensional fixed effects. The
#'  term fixed effect means having one intercept for each level in each category.
#'
#' @param formula an object of class \code{"formula"}: a symbolic description of the model to be fitted. \code{formula}
#'  must be of type \code{response ~ slopes | fixed_effects | cluster}.
#' @param data an object of class \code{"data.frame"} containing the variables in the model. The expected input is a
#'  dataset with the variables specified in \code{formula} and a number of rows at least equal to the number of variables
#'  in the model.
#' @param family the link function to be used in the model. Similar to \link[stats]{glm.fit} this has to be the result
#'  of a call to a family function. Default is \code{gaussian()}. See \link[stats]{family} for details of family
#'  functions.
#' @param weights an optional string with the name of the prior weights variable in \code{data}.
#' @param beta_start an optional vector of starting values for the structural parameters in the linear predictor.
#'  Default is \eqn{\boldsymbol{\beta} = \mathbf{0}}{\beta = 0}.
#' @param eta_start an optional vector of starting values for the linear predictor.
#' @param offset an optional formula or numeric vector specifying an a priori known component to be included in the
#'  linear predictor. If a formula, it should be of the form \code{~ log(variable)}.
#' @param control a named list of parameters for controlling the fitting process. See \link{fit_control} for details.
#'
#' @details If \link{feglm} does not converge this is often a sign of linear dependence between one or more
#'  regressors and a fixed effects category. In this case, you should carefully inspect your model specification.
#'
#' @return A named list of class \code{"feglm"}. The list contains the following fifteen elements:
#'  \item{coefficients}{a named vector of the estimated coefficients}
#'  \item{eta}{a vector of the linear predictor}
#'  \item{weights}{a vector of the weights used in the estimation}
#'  \item{hessian}{a matrix with the numerical second derivatives}
#'  \item{deviance}{the deviance of the model}
#'  \item{null_deviance}{the null deviance of the model}
#'  \item{conv}{a logical indicating whether the model converged}
#'  \item{iter}{the number of iterations needed to converge}
#'  \item{nobs}{a named vector with the number of observations used in the estimation indicating the dropped and
#'   perfectly predicted observations}
#'  \item{fe_levels}{a named vector with the number of levels in each fixed effects}
#'  \item{nms_fe}{a list with the names of the fixed effects variables}
#'  \item{formula}{the formula used in the model}
#'  \item{data}{the data used in the model after dropping non-contributing
#'   observations}
#'  \item{family}{the family used in the model}
#'  \item{control}{the control list used in the model}
#'
#' @references Gaure, S. (2013). "OLS with Multiple High Dimensional Category Variables". Computational Statistics and
#'  Data Analysis, 66.
#'
#' @references Marschner, I. (2011). "glm2: Fitting generalized linear models with convergence problems". The R Journal,
#'  3(2).
#'
#' @references Stammann, A., F. Heiss, and D. McFadden (2016). "Estimating Fixed Effects Logit Models with Large Panel
#'  Data". Working paper.
#'
#' @references Stammann, A. (2018). "Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional
#'  k-Way Fixed Effects". ArXiv e-prints.
#'
#' @examples
#' # Model without clustering - uses inverse Hessian for vcov
#' mod <- feglm(mpg ~ wt | cyl, mtcars, family = poisson(link = "log"))
#' summary(mod)
#'
#' # Model with clustering - uses sandwich vcov automatically
#' mod <- feglm(mpg ~ wt | cyl | am, mtcars, family = poisson(link = "log"))
#' summary(mod)
#'
#' @export
feglm <- function(
  formula = NULL,
  data = NULL,
  family = gaussian(),
  weights = NULL,
  beta_start = NULL,
  eta_start = NULL,
  offset = NULL,
  control = NULL
) {
  # t0 <- Sys.time()

  # Check validity of formula ----
  check_formula_(formula)

  # Check validity of data ----
  check_data_(data)

  # Check validity of family ----
  check_family_(family)

  # Check validity of control + Extract control list ----
  check_control_(control)

  # Extract offset before data filtering ----
  offset_vec_original <- NULL

  if (!is.null(offset)) {
    if (inherits(offset, "formula")) {
      # Offset provided as formula (e.g., ~ log(variable))
      offset_vars <- attr(terms(offset, data = data), "term.labels")
      if (length(offset_vars) != 1L) {
        stop("Offset formula must specify exactly one term.", call. = FALSE)
      }
      # Evaluate the offset expression in the context of the data
      offset_vec_original <- eval(parse(text = offset_vars), envir = data)
    } else if (is.numeric(offset)) {
      # Offset provided as numeric vector
      offset_vec_original <- offset
      if (length(offset_vec_original) != nrow(data)) {
        stop(
          "Length of offset must equal number of observations.",
          call. = FALSE
        )
      }
    } else {
      stop(
        "Offset must be NULL, a formula, or a numeric vector.",
        call. = FALSE
      )
    }
    names(offset_vec_original) <- rownames(data)
  }

  # Generate model.frame
  lhs <- NA # just to avoid global variable warning
  nobs_na <- NA
  nobs_full <- NA
  weights_vec <- NA
  weights_col <- NA
  model_frame_(data, formula, weights)

  # Ensure that model response is in line with the chosen model ----
  check_response_(data, lhs, family)

  # Get names of the fixed effects variables ----
  fe_vars <- check_fe_(formula, data)

  # Generate temporary variable ----
  tmp_var <- temp_var_(data)

  # Drop observations that do not contribute to the log likelihood ----
  data <- drop_by_link_type_(data, lhs, family, tmp_var, fe_vars, control)

  # Transform fixed effects and clusters to factors ----
  data <- transform_fe_(data, formula, fe_vars)
  nt <- nrow(data)

  # Extract model response and regressor matrix ----
  nms_sp <- NA
  p <- NA
  model_response_(data, formula)

  # Warm-start (opt-in): reuse previous eta if available and sizes match ----
  if (isTRUE(getOption("capybara.warm_start", FALSE))) {
    form_key <- paste(deparse(formula), collapse = "")
    prev <- cache_get_starts_(form_key)
    if (is.null(beta_start) && is.null(eta_start) && !is.null(prev)) {
      if (length(prev$eta) == nrow(X)) {
        eta_start <- prev$eta
      }
    }
  }

  # Extract weights if required ----
  if (is.null(weights)) {
    wt <- rep(1.0, nt)
  } else if (!all(is.na(weights_vec))) {
    # Weights provided as vector
    if (length(weights_vec) != nrow(data)) {
      stop(
        "Length of weights vector must equal number of observations.",
        call. = FALSE
      )
    }
    wt <- weights_vec
  } else if (!all(is.na(weights_col))) {
    # Weights provided as formula - use the extracted column name
    wt <- data[[weights_col]]
  } else {
    # Weights provided as column name
    wt <- data[[weights]]
  }

  # Check validity of weights ----
  check_weights_(wt)

  # Extract offset if required ----
  # Subset the pre-computed offset vector to match the filtered data
  if (is.null(offset)) {
    offset_vec <- rep(0.0, nt)
  } else {
    # Use row names to subset the offset vector to match filtered data
    offset_vec <- offset_vec_original[rownames(data)]
    if (length(offset_vec) != nt) {
      stop(
        "Length of offset does not match number of observations after filtering.",
        call. = FALSE
      )
    }
  }

  # Compute and check starting guesses ----
  start_guesses_(beta_start, eta_start, y, X, beta, nt, wt, p, family)

  # Get names and number of levels in each fixed effects category ----
  if (length(fe_vars) > 0) {
    fe_levels <- vapply(lapply(data[fe_vars], levels), length, integer(1))
    # Generate auxiliary list of indexes for different sub panels ----
    FEs <- get_index_list_(fe_vars, data)
  } else {
    # No fixed effects - create empty list
    fe_levels <- integer(0)
    FEs <- list()
  }

  # Set names on the FEs to ensure they're passed to C++
  names(FEs) <- fe_vars

  # Extract cluster variable from formula (third part) ----
  cl_vars_temp <- suppressWarnings(attr(
    terms(formula, rhs = 3L),
    "term.labels"
  ))
  if (length(cl_vars_temp) >= 1L) {
    # Get cluster index list (similar to FEs but only one level)
    cl_list <- get_index_list_(cl_vars_temp[1L], data)[[1L]]
  } else {
    cl_list <- list()
  }

  # Fit generalized linear model ----
  if (is.integer(y)) {
    y <- as.numeric(y)
  }

  # t1 <- Sys.time()

  fit <- feglm_fit_(
    beta,
    eta,
    y,
    X,
    wt,
    offset_vec,
    0.0,
    family[["family"]],
    control,
    FEs,
    cl_list
  )

  # t2 <- Sys.time()

  nobs <- nobs_(nobs_full, nobs_na, y, fit[["fitted_values"]])

  # Cache starts for potential warm-start in repeated calls (opt-in) ----
  if (isTRUE(getOption("capybara.warm_start", FALSE))) {
    if (!is.null(fit$coef_table) && !is.null(fit$eta)) {
      if (ncol(fit$coef_table) >= 1L && length(fit$eta) == nrow(X)) {
        cache_set_starts_(
          paste(deparse(formula), collapse = ""),
          fit$coef_table[, 1],
          fit$eta
        )
      }
    }
  }

  y <- NULL
  X <- NULL
  eta <- NULL

  # Add names to coef_table, hessian, T(X) (if provided), and fitted values ----
  # When there are no fixed effects, C++ adds an intercept column
  if (length(fe_vars) == 0) {
    nms_sp <- c("(Intercept)", nms_sp)
  }
  dimnames(fit[["coef_table"]]) <- list(
    nms_sp,
    c("Estimate", "Std. Error", "z value", "Pr(>|z|)")
  )
  if (control[["keep_tx"]]) {
    colnames(fit[["tx"]]) <- nms_sp
  }
  non_na_nms_sp <- nms_sp[!is.na(fit[["coef_table"]][, 1])]
  dimnames(fit[["hessian"]]) <- list(non_na_nms_sp, non_na_nms_sp)
  dimnames(fit[["vcov"]]) <- list(non_na_nms_sp, non_na_nms_sp)
  # Preserve row names from the data when possible to match base R prediction naming
  if (!is.null(rownames(data))) {
    names(fit[["fitted_values"]]) <- rownames(data)
  } else {
    names(fit[["fitted_values"]]) <- seq_along(fit[["fitted_values"]])
  }

  # Add separation info if present ----
  if (isTRUE(fit$has_separation)) {
    warning(
      "Separation detected in Poisson model. ",
      "Some observations are perfectly predicted and may need to be removed. ",
      "Consider refitting the model after excluding separated observations."
    )
    fit[["separated_obs"]] <- fit$separated_obs
    fit[["separation_certificate"]] <- fit$separation_certificate
  }

  # Add to fit list ----
  fit[["nobs"]] <- nobs
  fit[["fe_levels"]] <- fe_levels
  fit[["nms_fe"]] <- if (length(fe_vars) > 0) {
    lapply(data[fe_vars], levels)
  } else {
    list()
  }
  fit[["formula"]] <- formula
  fit[["data"]] <- data
  fit[["family"]] <- family
  fit[["control"]] <- control
  fit[["offset"]] <- offset_vec
  fit[["offset_spec"]] <- offset

  # t3 <- Sys.time()

  # print(sprintf(
  #   "feglm fit completed in %.2f seconds (data prep: %.2f s, model fit: %.2f s).",
  #   as.numeric(difftime(t3, t0, units = "secs")),
  #   as.numeric(difftime(t1, t0, units = "secs")),
  #   as.numeric(difftime(t2, t1, units = "secs"))
  # ))

  # Return result list ----
  structure(fit, class = "feglm")
}
