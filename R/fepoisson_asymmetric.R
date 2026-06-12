# srr_stats
# {G1.0} Implements Poisson regression with high-dimensional fixed effects via `feglm`.
# {G2.1a} Validates input `formula` to ensure correct specification of fixed effects.
# {G2.1b} Ensures `data` is appropriately formatted and contains sufficient observations.
# {G2.3a} Uses internally validated arguments (`control` and starting guesses) for consistency.
# {G3.1a} Supports canonical log link function for Poisson family.
# {G3.1b} Provides detailed outputs including coefficients, deviance, and convergence diagnostics.
# {G5.0} Ensures that identical input data and parameter settings consistently produce the same outputs, supporting reproducible workflows.
# {G5.1} Includes complete output elements (coefficients, deviance, etc.) for reproducibility.
# {G5.2a} Generates unique and descriptive error messages for invalid configurations or inputs.
# {G5.2b} Tracks optimization convergence during model fitting, providing detailed diagnostics for users to assess model stability.
# {G5.3} Optimizes computational efficiency for large datasets, employing parallel processing or streamlined algorithms where feasible.
# {G5.4} Benchmarks the scalability of model fitting against datasets of varying sizes to identify performance limits.
# {G5.4b} Documents performance comparisons with alternative implementations, highlighting strengths in accuracy or speed.
# {G5.4c} Employs memory-efficient data structures to handle large datasets without exceeding hardware constraints.
# {G5.5} Uses fixed random seeds for stochastic components, ensuring consistent outputs for analyses involving randomness.
# {G5.6} Benchmarks model fitting times and resource usage, providing users with insights into expected computational demands.
# {G5.6a} Demonstrates how parallel processing can reduce computation times while maintaining accuracy in results.
# {G5.7} Offers detailed, reproducible examples of typical use cases, ensuring users can replicate key functionality step-by-step.
# {G5.8} Includes informative messages or progress indicators during long-running computations to enhance user experience.
# {G5.8a} Warns users when outputs are approximate due to algorithmic simplifications or computational trade-offs.
# {G5.8b} Provides options to control the balance between computational speed and result precision, accommodating diverse user needs.
# {G5.8c} Documents which algorithm settings prioritize efficiency over accuracy, helping users make informed choices.
# {G5.8d} Clarifies the variability in results caused by parallel execution, particularly in randomized algorithms.
# {G5.9} Ensures all intermediate computations are accessible for debugging and troubleshooting during development or analysis.
# {G5.9a} Implements a debug mode that logs detailed information about the computational process for advanced users.
# {G5.9b} Validates correctness of results under debug mode, ensuring computational reliability across all scenarios.
# {RE1.0} Documents all assumptions inherent in the regression model, such as linearity, independence, and absence of multicollinearity.
# {RE1.1} Validates that input variables conform to expected formats, including numeric types for predictors and outcomes.
# {RE1.2} Provides options for handling missing data, including imputation or omission, and ensures users are informed of the chosen method.
# {RE1.3} Includes rigorous tests to verify model stability with edge cases, such as datasets with collinear predictors or extreme values.
# {RE1.3a} Adds specific tests for small datasets, ensuring the model remains robust under low-sample conditions.
# {RE1.4} Implements diagnostic checks to verify the assumptions of independence and homoscedasticity, essential for valid inference.
# {RE2.0} Labels all regression outputs, such as coefficients and standard errors, to ensure clarity and interpretability.
# {RE2.4} Quantifies uncertainty in regression coefficients using confidence intervals.
# {RE2.4a} Rejects perfect collinearity between independent variables.
# {RE2.4b} Rejects perfect collinearity between dependent and independent variables.
# {RE4.0} This returns a model-type object that is essentially a list with specific components and attributes.
# {RE4.1} Identifies outliers and influential data points that may unduly impact regression results, offering visualization tools.
# {RE4.6} Includes standard metrics such as R-squared and RMSE to help users evaluate model performance.
# {RE4.7} Tests sensitivity to hyperparameter choices in regularized or complex regression models.
# {RE4.14} Uses simulated datasets to test the reproducibility and robustness of regression results.
# {RE5.0} Optimized for scaling to large datasets with high-dimensional fixed effects.
# {RE5.1} Efficiently projects out fixed effects using auxiliary indexing structures.
# {RE5.2} Provides detailed warnings and error handling for convergence and dependence issues.
# {RE5.3} Thoroughly documents interactions between model features, inputs, and controls.
# {RE7.4} Provides comprehensive examples that demonstrate proper usage of the regression functions, covering input preparation, function execution, and result interpretation.

#' @title Asymmetric Poisson Pseudo-Maximum Likelihood (APPML) Estimation
#'
#' @description Fits an asymmetric Poisson pseudo-maximum likelihood model with high-dimensional fixed effects
#'  using expectile regression. This approach extends standard PPML by allowing different weights for positive
#'  and negative residuals, enabling estimation of conditional expectiles rather than the conditional mean.
#'
#' @inheritParams feglm
#'
#' @details
#' The APPML estimator minimizes an asymmetric loss function based on expectiles. For a given expectile \eqn{\tau},
#' observations with negative residuals receive weight \eqn{\tau} while observations with positive residuals
#' receive weight \eqn{1 - \tau}. The algorithm iteratively:
#' \enumerate{
#'   \item Computes residuals from the current fit
#'   \item Updates weights as \eqn{w_i = |\tau - \mathbf{1}(r_i < 0)|}
#'   \item Re-fits the weighted Poisson model
#'   \item Checks convergence using \eqn{(b - b_{old})' V^{-1} (b - b_{old}) < \epsilon}
#' }
#'
#' The expectile parameter is specified via \code{control = fit_control(expectile = ...)}. When
#' \code{expectile = 0.5}, the estimator is equivalent to standard PPML. Values below 0.5 estimate
#' lower conditional expectiles (more sensitive to small values), while values above 0.5 estimate
#' upper conditional expectiles (more sensitive to large values).
#'
#' @return A named list of class \code{"feglm"} containing:
#'  \item{coefficients}{named vector of estimated coefficients}
#'  \item{vcov}{variance-covariance matrix of coefficients}
#'  \item{eta}{linear predictor}
#'  \item{fitted_values}{fitted values from the final iteration}
#'  \item{residuals}{residuals from the final fit}
#'  \item{weights}{observation weights used in final fit}
#'  \item{appml_weights}{asymmetric weights used in APPML algorithm}
#'  \item{deviance}{the deviance of the model}
#'  \item{null_deviance}{the null deviance of the model}
#'  \item{conv}{logical indicating whether inner GLM converged}
#'  \item{conv_outer}{logical indicating whether APPML outer loop converged}
#'  \item{iter}{number of inner iterations}
#'  \item{iter_outer}{number of outer APPML iterations}
#'  \item{expectile}{the expectile value used}
#'  \item{objective_function}{final value of the convergence criterion}
#'  \item{negative_residuals_share}{proportion of negative residuals in final fit}
#'  \item{nobs}{a named vector with the number of observations}
#'  \item{fe_levels}{a named vector with the number of levels in each fixed effect}
#'  \item{nms_fe}{a list with the names of the fixed effects variables}
#'  \item{formula}{the formula used in the model}
#'  \item{family}{the family used in the model (Poisson)}
#'  \item{control}{the control list used in the model}
#'
#' @references
#' Newey, W. K., & Powell, J. L. (1987). Asymmetric least squares estimation and testing.
#'   \emph{Econometrica}, 55(4), 819-847.
#'
#' @examples
#' ross2004_subset <- ross2004[ross2004$year == 1999, ]
#' ross2004_subset <- ross2004_subset[ross2004_subset$ltrade >
#'   quantile(ross2004_subset$ltrade, 0.75), ]
#'
#' # Lower expectile (10th) - more weight on negative residuals
#' fit10 <- fepoisson_asymmetric(
#'   ltrade ~ ldist | ctry1, ross2004_subset,
#'   control = fit_control(expectile = 0.1)
#' )
#'
#' summary(fit10)
#'
#' @seealso \link{fepoisson}, \link{feglm}, \link{fit_control}
#'
#' @export
fepoisson_asymmetric <- function(
  formula = NULL,
  data = NULL,
  weights = NULL,
  beta_start = NULL,
  eta_start = NULL,
  offset = NULL,
  control = NULL
) {
  # Check validity of formula ----
  check_formula_(formula)

  # Check validity of data ----
  check_data_(data)

  # Check validity of control + Extract control list ----
  control <- check_control_(control)

  # Check validity of expectile ----
  check_expectile_(control[["expectile"]])

  # Determine needed columns (validates they exist) ----
  cols_info <- get_needed_cols_(formula, data, weights, offset)

  # Preserve original row names ----
  orig_rownames <- rownames(data)
  needs_rowname_conversion <- is.null(orig_rownames)

  # Convert formula to normalized string for C++ ----
  formula_str <- normalize_formula_(formula, data)

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

  if (length(w) > 0L) {
    check_weights_(w)
  }
  if (is.integer(w)) {
    w <- as.double(w)
  }

  # Store original row count for later ----
  nobs_full <- nrow(data)

  # Get FE variable names ----
  fe_vars <- check_fe_(formula, data)

  # Starting guesses ----
  beta <- if (!is.null(beta_start)) as.numeric(beta_start) else numeric(0)
  eta_vec <- if (!is.null(eta_start)) as.numeric(eta_start) else numeric(0)

  # Store data for output ----
  data_for_output <- if (control[["keep_data"]]) data else NULL

  # FIT MODEL ----
  fit <- fepoisson_asymmetric_fit_(
    formula_str, data, w, beta, eta_vec, offset_vec, control
  )

  # Free large input objects immediately after C++ call
  data <- NULL
  w <- NULL
  beta <- NULL
  eta_vec <- NULL

  # Post-processing ----
  num_separated <- if (isTRUE(fit[["has_separation"]])) {
    fit[["num_separated"]]
  } else {
    0L
  }
  # nobs_used is the working sample (after NA removal AND separation exclusion)
  nobs_na <- nobs_full - fit[["nobs_used"]] - num_separated
  nobs <- c(
    nobs_full = nobs_full,
    nobs_na = nobs_na,
    nobs_separated = num_separated,
    nobs_pc = 0L,
    nobs = fit[["nobs_used"]]
  )

  nms_fe <- fit[["nms_fe"]]
  fe_levels <- fit[["fe_levels"]]

  # Information if convergence failed ----
  if (!isTRUE(fit[["conv_outer"]])) {
    warning("Algorithm did not converge.\n")
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
  if (!is.null(fit[["hessian"]]) && nrow(fit[["hessian"]]) == length(nms_sp)) {
    dimnames(fit[["hessian"]]) <- list(nms_sp, nms_sp)
  }
  if (!is.null(fit[["vcov"]]) && nrow(fit[["vcov"]]) == length(nms_sp)) {
    dimnames(fit[["vcov"]]) <- list(nms_sp, nms_sp)
  }

  # Set fitted_values names ----
  if (!is.null(fit[["obs_indices"]])) {
    if (needs_rowname_conversion) {
      orig_rownames <- as.character(seq_len(nobs_full))
    }
    used_rownames <- orig_rownames[fit[["obs_indices"]]]
    names(fit[["fitted_values"]]) <- used_rownames
    names(fit[["residuals"]]) <- used_rownames
    names(fit[["appml_weights"]]) <- used_rownames
    fit[[".rownames"]] <- used_rownames
    if (!is.null(data_for_output)) {
      data_for_output <- data_for_output[fit[["obs_indices"]], ]
    }
  } else {
    if (needs_rowname_conversion) {
      orig_rownames <- as.character(seq_len(nobs_full))
    }
    names(fit[["fitted_values"]]) <- orig_rownames
    names(fit[["residuals"]]) <- orig_rownames
    names(fit[["appml_weights"]]) <- orig_rownames
    fit[[".rownames"]] <- orig_rownames
  }

  # Clean up C++ internal fields ----
  fit[["obs_indices"]] <- NULL
  fit[["nobs_used"]] <- NULL
  fit[["term_names"]] <- NULL
  fit[["has_separation"]] <- NULL
  fit[["num_separated"]] <- NULL

  # Build result ----
  fit[["nobs"]] <- nobs
  fit[["fe_levels"]] <- fe_levels
  fit[["nms_fe"]] <- nms_fe
  fit[["formula"]] <- formula
  if (control[["keep_data"]]) {
    fit[["data"]] <- data_for_output
  }
  fit[["family"]] <- poisson()
  fit[["control"]] <- control
  fit[["offset"]] <- offset_vec

  structure(fit, class = c("feglm", "fepoisson_asymmetric"))
}

# Expectile validity check ----

check_expectile_ <- function(expectile) {
  if (is.null(expectile) || expectile <= 0 || expectile >= 1) {
    stop(
      "'expectile' must be specified in 'control' and between 0 and 1 ",
      "(exclusive). Use: control = fit_control(expectile = 0.5)",
      call. = FALSE
    )
  }
  invisible(TRUE)
}
