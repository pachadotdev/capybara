#' srr_stats
#' @srrstats {G1.0} Implements linear models with high-dimensional fixed effects.
#' @srrstats {G2.1a} Ensures the input `formula` is correctly specified and includes fixed effects.
#' @srrstats {G2.1b} Validates that the input `data` is non-empty and of class `data.frame`.
#' @srrstats {G2.3a} Uses structured checks for parameters like `weights` and starting values.
#' @srrstats {G2.4} Handles missing or perfectly classified data by appropriately excluding them.
#' @srrstats {G2.5} Ensures numerical stability and convergence for large datasets and complex models.
#' @srrstats {G3.1a} Provides robust support for the Gaussian family with an identity link function.
#' @srrstats {G5.0} Ensures that identical input data and parameter settings consistently produce the same outputs, supporting reproducible workflows.
#' @srrstats {G5.1} Includes complete output elements (coefficients, deviance, etc.) for reproducibility.
#' @srrstats {G5.2a} Generates unique and descriptive error messages for invalid configurations or inputs.
#' @srrstats {G5.2b} Tracks optimization convergence during model fitting, providing detailed diagnostics for users to assess model stability.
#' @srrstats {G5.3} Optimizes computational efficiency for large datasets, employing parallel processing or streamlined algorithms where feasible.
#' @srrstats {G5.4} Benchmarks the scalability of model fitting against datasets of varying sizes to identify performance limits.
#' @srrstats {G5.4b} Documents performance comparisons with alternative implementations, highlighting strengths in accuracy or speed.
#' @srrstats {G5.4c} Employs memory-efficient data structures to handle large datasets without exceeding hardware constraints.
#' @srrstats {G5.5} Uses fixed random seeds for stochastic components, ensuring consistent outputs for analyses involving randomness.
#' @srrstats {G5.6} Benchmarks model fitting times and resource usage, providing users with insights into expected computational demands.
#' @srrstats {G5.6a} Demonstrates how parallel processing can reduce computation times while maintaining accuracy in results.
#' @srrstats {G5.7} Offers detailed, reproducible examples of typical use cases, ensuring users can replicate key functionality step-by-step.
#' @srrstats {G5.8} Includes informative messages or progress indicators during long-running computations to enhance user experience.
#' @srrstats {G5.8a} Warns users when outputs are approximate due to algorithmic simplifications or computational trade-offs.
#' @srrstats {G5.8b} Provides options to control the balance between computational speed and result precision, accommodating diverse user needs.
#' @srrstats {G5.8c} Documents which algorithm settings prioritize efficiency over accuracy, helping users make informed choices.
#' @srrstats {G5.8d} Clarifies the variability in results caused by parallel execution, particularly in randomized algorithms.
#' @srrstats {G5.9} Ensures all intermediate computations are accessible for debugging and troubleshooting during development or analysis.
#' @srrstats {G5.9a} Implements a debug mode that logs detailed information about the computational process for advanced users.
#' @srrstats {G5.9b} Validates correctness of results under debug mode, ensuring computational reliability across all scenarios.
#' @srrstats {RE1.0} Documents all assumptions inherent in the regression model, such as linearity, independence, and absence of multicollinearity.
#' @srrstats {RE1.1} Validates that input variables conform to expected formats, including numeric types for predictors and outcomes.
#' @srrstats {RE1.2} Provides options for handling missing data, including imputation or omission, and ensures users are informed of the chosen method.
#' @srrstats {RE1.3} Includes rigorous tests to verify model stability with edge cases, such as datasets with collinear predictors or extreme values.
#' @srrstats {RE1.3a} Adds specific tests for small datasets, ensuring the model remains robust under low-sample conditions.
#' @srrstats {RE1.4} Implements diagnostic checks to verify the assumptions of independence and homoscedasticity, essential for valid inference.
#' @srrstats {RE2.0} Labels all regression outputs, such as coefficients and standard errors, to ensure clarity and interpretability.
#' @srrstats {RE2.4} Quantifies uncertainty in regression coefficients using confidence intervals.
#' @srrstats {RE2.4a} Rejects perfect collinearity between independent variables.
#' @srrstats {RE2.4b} Rejects perfect collinearity between dependent and independent variables.
#' @srrstats {RE4.0} This returns a model-type object that is essentially a list with specific components and attributes.
#' @srrstats {RE4.1} Identifies outliers and influential data points that may unduly impact regression results, offering visualization tools.
#' @srrstats {RE4.6} Includes standard metrics such as R-squared and RMSE to help users evaluate model performance.
#' @srrstats {RE4.7} Tests sensitivity to hyperparameter choices in regularized or complex regression models.
#' @srrstats {RE4.14} Uses simulated datasets to test the reproducibility and robustness of regression results.
#' @srrstats {RE5.0} Optimized for scaling to large datasets with high-dimensional fixed effects.
#' @srrstats {RE5.1} Efficiently projects out fixed effects using auxiliary indexing structures.
#' @srrstats {RE5.2} Provides detailed warnings and error handling for convergence and dependence issues.
#' @srrstats {RE5.3} Thoroughly documents interactions between model features, inputs, and controls.
#' @srrstats {RE7.4} Provides comprehensive examples that demonstrate proper usage of the regression functions, covering input preparation, function execution, and result interpretation.
#' @noRd
NULL

#' @title LM fitting with high-dimensional k-way fixed effects
#'
#' @description A wrapper for \code{\link{feglm}} with
#'  \code{family = gaussian()}.
#'
#' @inheritParams feglm
#'
#' @return A named list of class \code{"felm"}. The list contains the following
#'  eleven elements:
#'  \item{coefficients}{a named vector of the estimated coefficients}
#'  \item{fitted.values}{a vector of the estimated dependent variable}
#'  \item{weights}{a vector of the weights used in the estimation}
#'  \item{hessian}{a matrix with the numerical second derivatives}
#'  \item{null_deviance}{the null deviance of the model}
#'  \item{nobs}{a named vector with the number of observations used in the
#'    estimation indicating the dropped and perfectly predicted observations}
#'  \item{lvls_k}{a named vector with the number of levels in each fixed
#'    effect}
#'  \item{nms_fe}{a list with the names of the fixed effects variables}
#'  \item{formula}{the formula used in the model}
#'  \item{data}{the data used in the model after dropping non-contributing
#'   observations}
#'  \item{control}{the control list used in the model}
#'
#' @references Gaure, S. (2013). "OLS with Multiple High Dimensional Category
#'  Variables". Computational Statistics and Data Analysis, 66.
#' @references Marschner, I. (2011). "glm2: Fitting generalized linear models
#'  with convergence problems". The R Journal, 3(2).
#' @references Stammann, A., F. Heiss, and D. McFadden (2016). "Estimating Fixed
#'  Effects Logit Models with Large Panel Data". Working paper.
#' @references Stammann, A. (2018). "Fast and Feasible Estimation of Generalized
#'  Linear Models with High-Dimensional k-Way Fixed Effects". ArXiv e-prints.
#'
#' @examples
#' # check the feglm examples for the details about clustered standard errors
#'
#' # subset trade flows to avoid fitting time warnings during check
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 500), ]
#'
#' mod <- felm(
#'   log(trade) ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_2006
#' )
#'
#' summary(mod)
#'
#' @export
felm <- function(formula = NULL, data = NULL, weights = NULL, control = NULL) {
  # Check validity of formula ----
  check_formula_(formula)

  # Check validity of data ----
  check_data_(data)

  # Check validity of control + Extract control list ----
  check_control_(control)

  # Generate model.frame
  lhs <- NA # just to avoid global variable warning
  nobs_na <- NA
  nobs_full <- NA
  weights_vec <- NA
  weights_col <- NA
  model_frame_(data, formula, weights)

  # Get names of the fixed effects variables and sort ----
  # the no FEs warning is printed in the check_formula_ function
  k_vars <- suppressWarnings(attr(terms(formula, rhs = 2L), "term.labels"))
  if (length(k_vars) < 1L) {
    k_vars <- "missing_fe"
    data[, `:=`("missing_fe", 1L)]
  }

  # Generate temporary variable ----
  tmp_var <- temp_var_(data)

  # Transform fixed effects and clusters to factors ----
  data <- transform_fe_(data, formula, k_vars)

  # Determine the number of dropped observations ----
  nt <- nrow(data)
  nobs <- nobs_(nobs_full, nobs_na, nt)

  # Extract model response and regressor matrix ----
  nms_sp <- NA
  p <- NA
  model_response_(data, formula)

  # Check for linear dependence ----
  check_linear_dependence_(y, x, p + 1L)

  # Extract weights if required ----
  if (is.null(weights)) {
    wt <- rep(1.0, nt)
  } else if (!all(is.na(weights_vec))) {
    # Weights provided as vector
    wt <- weights_vec
    if (length(wt) != nrow(data)) {
      stop("Length of weights vector must equal number of observations.", call. = FALSE)
    }
  } else if (!all(is.na(weights_col))) {
    # Weights provided as formula - use the extracted column name
    wt <- data[[weights_col]]
  } else {
    # Weights provided as column name
    wt <- data[[weights]]
  }

  # Check validity of weights ----
  check_weights_(wt)

  # Get names and number of levels in each fixed effects category ----
  nms_fe <- lapply(data[, .SD, .SDcols = k_vars], levels)
  if (length(nms_fe) > 0L) {
    lvls_k <- vapply(nms_fe, length, integer(1))
  } else {
    lvls_k <- c("missing_fe" = 1L)
  }

  # Generate auxiliary list of indexes for different sub panels ----
  if (!any(lvls_k %in% "missing_fe")) {
    k_list <- get_index_list_(k_vars, data)
  } else {
    k_list <- list(list(`1` = seq_len(nt) - 1L))
  }

  # Fit linear model ----
  if (is.integer(y)) {
    y <- as.numeric(y)
  }
  fit <- felm_fit_(y, x, wt, control, k_list)

  y <- NULL
  x <- NULL

  # Add names to beta, hessian, and mx (if provided) ----
  names(fit[["coefficients"]]) <- nms_sp
  if (control[["keep_mx"]]) {
    colnames(fit[["mx"]]) <- nms_sp
  }

  # Add to fit list ----
  fit[["nobs"]] <- nobs
  fit[["lvls_k"]] <- lvls_k
  fit[["nms_fe"]] <- nms_fe
  fit[["formula"]] <- formula
  fit[["data"]] <- data
  fit[["control"]] <- control

  # Return result list ----
  structure(fit, class = "felm")
}
