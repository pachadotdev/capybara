#' srr_stats
#' @srrstats {G1.0} Implements linear models with high-dimensional fixed effects.
#' @srrstats {G2.1a} Ensures the input `formula` is correctly specified and includes fixed effects.
#' @srrstats {G2.1b} Validates that the input `data` is non-empty and of class `data.frame`.
#' @srrstats {G2.3a} Uses structured checks for parameters like `weights` and starting values.
#' @srrstats {G2.4} Handles missing or perfectly classified data by appropriately excluding them.
#' @srrstats {G2.5} Ensures numerical stability and convergence for large datasets and complex models.
#' @srrstats {G3.1a} Provides robust support for the Gaussian family with an identity link function.
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
#' @srrstats {RE4.0} This returns a model-type object that is essentially a list with specific components and attributes.
#' @srrstats {RE4.1} Identifies outliers and influential data points that may unduly impact regression results, offering
#'  visualization tools.
#' @srrstats {RE4.6} Includes standard metrics such as R-squared and RMSE to help users evaluate model performance.
#' @srrstats {RE4.7} Tests sensitivity to hyperparameter choices in regularized or complex regression models.
#' @srrstats {RE4.14} Uses simulated datasets to test the reproducibility and robustness of regression results.
#' @srrstats {RE5.0} Optimized for scaling to large datasets with high-dimensional fixed effects.
#' @srrstats {RE5.1} Efficiently projects out fixed effects using auxiliary indexing structures.
#' @srrstats {RE5.2} Provides detailed warnings and error handling for convergence and dependence issues.
#' @srrstats {RE5.3} Thoroughly documents interactions between model features, inputs, and controls.
#' @srrstats {RE7.4} Provides comprehensive examples that demonstrate proper usage of the regression functions, covering
#'  input preparation, function execution, and result interpretation.
#' @noRd
NULL

#' @title LM fitting with high-dimensional k-way fixed effects
#'
#' @description \link{feglm} can be used to fit linear models with many high-dimensional fixed effects. The estimation
#'  procedure is based on unconditional maximum likelihood and can be interpreted as a \dQuote{weighted demeaning}
#'  approach.
#'
#' @inheritParams feglm
#'
#' @return A named list of class \code{"felm"}. The list contains the following
#'  eleven elements:
#'  \item{coefficients}{a named vector of the estimated coefficients}
#'  \item{fitted_values}{a vector of the estimated dependent variable}
#'  \item{weights}{a vector of the weights used in the estimation}
#'  \item{hessian}{a matrix with the numerical second derivatives}
#'  \item{null_deviance}{the null deviance of the model}
#'  \item{nobs}{a named vector with the number of observations used in the estimation indicating the dropped and
#'    perfectly predicted observations}
#'  \item{fe_levels}{a named vector with the number of levels in each fixed effect}
#'  \item{nms_fe}{a list with the names of the fixed effects variables}
#'  \item{formula}{the formula used in the model}
#'  \item{data}{the data used in the model after dropping non-contributing observations}
#'  \item{control}{the control list used in the model}
#'
#' @references Gaure, S. (2013). "OLS with Multiple High Dimensional Category Variables". Computational Statistics and
#'  Data Analysis, 66.
#' @references Marschner, I. (2011). "glm2: Fitting generalized linear models with convergence problems". The R Journal,
#'  3(2).
#' @references Stammann, A., F. Heiss, and D. McFadden (2016). "Estimating Fixed Effects Logit Models with Large Panel
#'  Data". Working paper.
#' @references Stammann, A. (2018). "Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional
#'  k-Way Fixed Effects". ArXiv e-prints.
#'
#' @examples
#' # Model with fixed effects
#' mod <- felm(mpg ~ wt | cyl, mtcars)
#' summary(mod)
#'
#' # Model without fixed effects but with clustered standard errors
#' # Note: Use 0 to indicate no fixed effects when specifying clusters
#' mod <- felm(mpg ~ wt | 0 | cyl, mtcars)
#' summary(mod)
#'
#' @export
felm <- function(
  formula = NULL,
  data = NULL,
  weights = NULL,
  vcov = NULL,
  control = NULL
) {
  # Check validity of formula ----
  check_formula_(formula)

  # Check validity of data ----
  check_data_(data)

  # Check validity of control + Extract control list ----
  check_control_(control)

  # Process vcov argument ----
  vcov_result <- process_vcov_(vcov, control)
  vcov_label <- vcov_result$vcov_label
  control <- vcov_result$control

  # Determine needed columns (validates they exist) ----
  cols_info <- get_needed_cols_(formula, data, weights)

  # Preserve original row names ----
  orig_rownames <- rownames(data)
  needs_rowname_conversion <- is.null(orig_rownames)

  # Convert formula to normalized string for C++ ----
  # Use normalize_formula_ to expand *, ^, -, /, %in%, . using R's terms()
  formula_str <- normalize_formula_(formula, data)
  
  # Detect if intercept is suppressed (e.g., ~ wt - 1)
  has_intercept <- !grepl("__NO_INTERCEPT__", formula_str, fixed = TRUE)

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

  # Store data for output if needed ----
  data_for_output <- if (control[["keep_data"]]) data else NULL

  # FIT MODEL ----
  fit <- felm_fit_(formula_str, data, w, control)

  # Free data for GC
  data <- NULL
  w <- NULL

  # Get FE and cluster variable names from formula ----
  vars <- get_fe_cl_vars_(formula)
  fe_vars <- vars$fe_vars
  cl_vars <- vars$cl_vars
  vcov_label <- vcov_result$vcov_label

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

  # Get term names from C++ result ----
  nms_sp <- if (!is.null(fit[["term_names"]])) {
    fit[["term_names"]]
  } else {
    paste0("V", seq_len(ncol(fit[["coef_table"]])))
  }

  # Add names to outputs ----
  # Add intercept name only if: no FE, and intercept is not suppressed (- 1)
  if (length(fe_vars) == 0L && has_intercept) {
    nms_sp <- c("(Intercept)", nms_sp)
  }
  dimnames(fit[["coef_table"]]) <- list(nms_sp, c("Estimate", "Std. Error", "z value", "Pr(>|z|)"))
  if (!is.null(fit[["hessian"]])) {
    dimnames(fit[["hessian"]]) <- list(nms_sp, nms_sp)
  }
  if (!is.null(fit[["vcov"]])) {
    dimnames(fit[["vcov"]]) <- list(nms_sp, nms_sp)
  }
  if (control[["keep_tx"]] && !is.null(fit[["tx"]]) && is.matrix(fit[["tx"]])) {
    colnames(fit[["tx"]]) <- nms_sp
  }

  # Set fitted_values names ----
  if (!is.null(fit[["obs_indices"]])) {
    if (needs_rowname_conversion) {
      orig_rownames <- as.character(seq_len(nobs_full))
    }
    used_rownames <- orig_rownames[fit[["obs_indices"]]]
    names(fit[["fitted_values"]]) <- used_rownames
    fit[[".rownames"]] <- used_rownames
    if (!is.null(data_for_output)) {
      data_for_output <- data_for_output[fit[["obs_indices"]], ]
    }
  } else {
    if (needs_rowname_conversion) {
      orig_rownames <- as.character(seq_len(nobs_full))
    }
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
  fit[["control"]] <- control
  fit[["vcov_type"]] <- if (!is.null(vcov_label)) {
    vcov_label
  } else {
    if (length(cl_vars) > 0L) {
      if (!is.null(control$vcov_type)) control$vcov_type else "cluster"
    } else {
      "iid"
    }
  }

  structure(fit, class = "felm")
}
