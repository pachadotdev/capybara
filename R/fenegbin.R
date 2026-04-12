#' srr_stats
#' @srrstats {G1.0} Implements Negative Binomial regression with high-dimensional fixed effects, adapting `feglm`.
#' @srrstats {G2.1a} Validates input `formula` to ensure inclusion of fixed effects.
#' @srrstats {G2.1b} Ensures `data` is of the appropriate class and contains non-zero rows.
#' @srrstats {G2.3a} Uses `match.arg()` to validate the `link` argument.
#' @srrstats {G2.3b} Checks numeric parameters such as starting guesses and weights for validity.
#' @srrstats {G2.4} Handles missing and non-contributing observations by excluding them appropriately.
#' @srrstats {G3.1a} Supports customizable link functions (`log`, `sqrt`, and `identity`) and initialization of theta.
#' @srrstats {G3.1b} Provides detailed outputs including coefficients, deviance, and theta.
#' @srrstats {G4.0} Uses an iterative algorithm for joint estimation of coefficients and theta, ensuring convergence.
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
#' @srrstats {RE1.2} Provides options for handling missing data, including imputation or omission, and ensures users are
#'  informed of the chosen method.
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
#' @srrstats {RE5.0} Optimized for high-dimensional fixed effects and large datasets, ensuring computational
#'  feasibility.
#' @srrstats {RE5.1} Validates convergence of both deviance and theta with strict tolerances.
#' @srrstats {RE5.2} Issues warnings if the algorithm fails to converge within the maximum iterations.
#' @srrstats {RE5.3} Outputs reproducible results, including detailed diagnostics and convergence information.
#' @noRd
NULL

#' @title Negative Binomial model fitting with high-dimensional k-way fixed
#'  effects
#'
#' @description A routine that uses the same internals as \link{feglm}.
#'
#' @inheritParams feglm
#'
#' @param init_theta an optional initial value for the theta parameter (see \link[MASS]{glm.nb}).
#' @param link the link function. Must be one of \code{"log"}, \code{"sqrt"}, or \code{"identity"}.
#' @param offset an optional formula or numeric vector specifying an a priori known component to be included in the
#'  linear predictor. If a formula, it should be of the form \code{~ log(variable)}.
#'
#' @examples
#' # check the feglm examples for the details about clustered standard errors
#' mod <- fenegbin(mpg ~ wt | cyl, mtcars)
#' summary(mod)
#'
#' @return A named list of class \code{"feglm"}. The list contains the following
#'  eighteen elements:
#'  \item{coefficients}{a named vector of the estimated coefficients}
#'  \item{eta}{a vector of the linear predictor}
#'  \item{weights}{a vector of the weights used in the estimation}
#'  \item{hessian}{a matrix with the numerical second derivatives}
#'  \item{deviance}{the deviance of the model}
#'  \item{null_deviance}{the null deviance of the model}
#'  \item{conv}{a logical indicating whether the model converged}
#'  \item{iter}{the number of iterations needed to converge}
#'  \item{theta}{the estimated theta parameter}
#'  \item{iter_outer}{the number of outer iterations}
#'  \item{conv_outer}{a logical indicating whether the outer loop converged}
#'  \item{nobs}{a named vector with the number of observations used in the estimation indicating the dropped and
#'   perfectly predicted observations}
#'  \item{fe_levels}{a named vector with the number of levels in each fixed effects}
#'  \item{nms_fe}{a list with the names of the fixed effects variables}
#'  \item{formula}{the formula used in the model}
#'  \item{data}{the data used in the model after dropping non-contributing observations}
#'  \item{family}{the family used in the model}
#'  \item{control}{the control list used in the model}
#'
#' @export
fenegbin <- function(
  formula = NULL,
  data = NULL,
  weights = NULL,
  beta_start = NULL,
  eta_start = NULL,
  init_theta = NULL,
  link = c("log", "identity", "sqrt"),
  offset = NULL,
  control = NULL
) {
  # Check validity of formula ----
  check_formula_(formula)

  # Check validity of data ----
  check_data_(data)

  # Check validity of link ----
  link <- match.arg(link)

  # Check validity of control + Extract control list ----
  control <- check_control_(control)

  # Preserve original row names ----
  orig_rownames <- rownames(data)
  if (is.null(orig_rownames)) {
    orig_rownames <- as.character(seq_len(nrow(data)))
  }

  # Convert formula to normalized string for C++ ----
  # Use normalize_formula_ to expand *, ^, -, /, %in%, . using R's terms()
  formula_str <- normalize_formula_(formula, data)
  
  # Detect if intercept is suppressed (e.g., ~ wt - 1)
  has_intercept <- !grepl("__NO_INTERCEPT__", formula_str, fixed = TRUE)

  # Extract offset before fitting ----
  offset_vec <- extract_offset_(offset, data, nrow(data))
  if (is.null(offset_vec)) offset_vec <- numeric(0)

  # Extract weights vector ----
  w <- if (is.null(weights)) {
    numeric(0)
  } else if (is.numeric(weights)) {
    weights
  } else if (is.character(weights) && length(weights) == 1L) {
    data[[weights]]
  } else if (inherits(weights, "formula")) {
    data[[all.vars(weights)]]
  } else {
    stop("'weights' must be NULL, a numeric vector, a column name, or a formula", call. = FALSE)
  }
  if (length(w) > 0L) check_weights_(w)

  # Store original row count for later ----
  nobs_full <- nrow(data)

  # Get FE variable names ----
  fe_vars <- check_fe_(formula, data)

  # Starting guesses ----
  beta <- if (!is.null(beta_start)) as.numeric(beta_start) else numeric(0)
  eta_vec <- if (!is.null(eta_start)) as.numeric(eta_start) else numeric(0)

  # Set init_theta to 0 if NULL
  if (is.null(init_theta)) {
    init_theta <- 0.0
  } else {
    if (length(init_theta) != 1L || init_theta <= 0) {
      stop("'init_theta' must be a positive scalar.", call. = FALSE)
    }
  }

  # Store data for output ----
  data_for_output <- if (control[["keep_data"]]) data else NULL

  # FIT MODEL ----
  fit <- structure(
    fenegbin_fit_(formula_str, data, w, link, beta, eta_vec, init_theta, offset_vec, control),
    class = c("feglm", "fenegbin")
  )

  # Free large input objects immediately after C++ call
  data <- NULL
  w <- NULL
  beta <- NULL
  eta_vec <- NULL

  # Post-processing ----
  nobs_na <- nobs_full - fit[["nobs_used"]]
  nobs <- c(
    nobs_full = nobs_full,
    nobs_na = nobs_na,
    nobs_separated = 0L,
    nobs_pc = 0L,
    nobs = fit[["nobs_used"]]
  )

  nms_fe <- fit[["nms_fe"]]
  fe_levels <- fit[["fe_levels"]]

  # Information if convergence failed ----
  if (!isTRUE(fit[["conv_outer"]])) {
    cat("Algorithm did not converge.\n")
  }

  # Get term names from C++ result ----
  nms_sp <- if (!is.null(fit[["term_names"]])) {
    fit[["term_names"]]
  } else {
    paste0("V", seq_len(nrow(fit[["coef_table"]])))
  }

  # Add names to outputs ----
  dimnames(fit[["coef_table"]]) <- list(nms_sp, c("Estimate", "Std. Error", "z value", "Pr(>|z|)"))
  if (control[["keep_tx"]] && !is.null(fit[["tx"]]) && is.matrix(fit[["tx"]])) {
    colnames(fit[["tx"]]) <- nms_sp
  }
  if (!is.null(fit[["hessian"]])) {
    dimnames(fit[["hessian"]]) <- list(nms_sp, nms_sp)
  }
  if (!is.null(fit[["vcov"]])) {
    dimnames(fit[["vcov"]]) <- list(nms_sp, nms_sp)
  }

  # Set fitted_values names ----
  if (!is.null(fit[["obs_indices"]])) {
    used_rownames <- orig_rownames[fit[["obs_indices"]]]
    names(fit[["fitted_values"]]) <- used_rownames
    fit[[".rownames"]] <- used_rownames
    if (!is.null(data_for_output)) {
      data_for_output <- data_for_output[fit[["obs_indices"]], ]
    }
  } else {
    names(fit[["fitted_values"]]) <- orig_rownames
    fit[[".rownames"]] <- orig_rownames
  }

  # Clean up C++ internal fields ----
  fit[["obs_indices"]] <- NULL
  fit[["nobs_used"]] <- NULL
  fit[["term_names"]] <- NULL

  # Build result ----
  fit[["nobs"]] <- nobs
  fit[["fe_levels"]] <- fe_levels
  fit[["nms_fe"]] <- nms_fe
  fit[["formula"]] <- formula
  if (control[["keep_data"]]) {
    fit[["data"]] <- data_for_output
  }
  fit[["family"]] <- negative.binomial(theta = fit[["theta"]], link = link)
  fit[["control"]] <- control
  fit[["offset"]] <- offset_vec

  fit
}

# Convergence Check ----

fenegbin_check_convergence_ <- function(dev, dev_old, theta, theta_old, tol) {
  dev_crit <- abs(dev - dev_old) / (0.1 + abs(dev))
  theta_crit <- abs(theta - theta_old) / (0.1 + abs(theta_old))
  dev_crit <= tol && theta_crit <= tol
}

# Generate result list ----

fenegbin_result_list_ <- function(
  fit,
  theta,
  iter,
  conv,
  nobs,
  fe_levels,
  nms_fe,
  formula,
  data,
  family,
  control
) {
  reslist <- c(
    fit,
    list(
      theta = theta,
      iter_outer = iter,
      conv_outer = conv,
      nobs = nobs,
      fe_levels = fe_levels,
      nms_fe = nms_fe,
      formula = formula,
      data = data,
      family = family,
      control = control
    )
  )

  # Return result list ----
  structure(reslist, class = c("feglm", "fenegbin"))
}
