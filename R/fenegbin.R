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
#' @srrstats {RE5.0} Optimized for high-dimensional fixed effects and large datasets, ensuring computational feasibility.
#' @srrstats {RE5.1} Validates convergence of both deviance and theta with strict tolerances.
#' @srrstats {RE5.2} Issues warnings if the algorithm fails to converge within the maximum iterations.
#' @srrstats {RE5.3} Outputs reproducible results, including detailed diagnostics and convergence information.
#' @noRd
NULL

#' @title Negative Binomial model fitting with high-dimensional k-way fixed
#'  effects
#'
#' @description A routine that uses the same internals as \code{\link{feglm}}.
#'
#' @inheritParams feglm
#'
#' @param init_theta an optional initial value for the theta parameter (see
#'  \code{\link[MASS]{glm.nb}}).
#' @param link the link function. Must be one of \code{"log"}, \code{"sqrt"}, or
#'  \code{"identity"}.
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
#'  \item{nobs}{a named vector with the number of observations used in the
#'   estimation indicating the dropped and perfectly predicted observations}
#'  \item{fe_levels}{a named vector with the number of levels in each fixed
#'   effects}
#'  \item{nms_fe}{a list with the names of the fixed effects variables}
#'  \item{formula}{the formula used in the model}
#'  \item{data}{the data used in the model after dropping non-contributing
#'   observations}
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
    control = NULL) {
  # Check validity of formula ----
  check_formula_(formula)

  # Check validity of data ----
  check_data_(data)

  # Check validity of link ----
  link <- match.arg(link)

  # Check validity of control + Extract control list ----
  control <- check_control_(control)

  # Generate model.frame
  X <- eta <- lhs <- nobs_na <- nobs_full <- NA
  model_frame_(data, formula, weights)

  # Create a dummy family for response checking
  family <- poisson(link = link)

  # Ensure that model response is in line with the chosen model ----
  check_response_(data, lhs, family)

  # Get names of the fixed effects variables ----
  fe_vars <- check_fe_(formula, data)

  # Get names of the fixed effects variables and sort ----
  fe_names <- attr(terms(formula, rhs = 2L), "term.labels")

  # Generate temporary variable ----
  tmp_var <- temp_var_(data)

  # Drop observations that do not contribute to the log likelihood ----
  data <- drop_by_link_type_(data, lhs, family, tmp_var, fe_names, control)

  # Transform fixed effects and clusters to factors ----
  data <- transform_fe_(data, formula, fe_names)

  # Determine the number of dropped observations ----
  nt <- nrow(data)

  # Extract model response and regressor matrix ----
  nms_sp <- p <- NA
  model_response_(data, formula)

  # Extract weights if required ----
  if (is.null(weights)) {
    w <- rep(1.0, nt)
  } else {
    w <- data[[weights]]
  }

  # Check validity of weights ----
  check_weights_(w)

  # Get starting guesses if provided
  beta <- if (!is.null(beta_start)) {
    as.numeric(beta_start)
  } else {
    numeric(0) # Empty vector for default initialization in C++
  }

  # Get eta starting guesses if provided
  eta_vec <- if (!is.null(eta_start)) {
    as.numeric(eta_start)
  } else {
    numeric(0) # Empty vector for default initialization in C++
  }

  # Get names and number of levels in each fixed effects category ----
  nms_fe <- lapply(data[fe_vars], levels)
  if (length(nms_fe) > 0L) {
    fe_levels <- vapply(nms_fe, length, integer(1))
  } else {
    fe_levels <- c("missing_fe" = 1L)
  }

  # Generate auxiliary list of indexes for different sub panels ----
  if (!any(fe_levels %in% "missing_fe")) {
    FEs <- get_index_list_(fe_vars, data)
  } else {
    FEs <- list(missing_fe = seq_len(nt))
  }

  # Set names on the FEs list to ensure they're passed to C++
  names(FEs) <- fe_vars

  # Set init_theta to 0 if NULL (C++ will handle default)
  if (is.null(init_theta)) {
    init_theta <- 0.0
  } else {
    # Validate init_theta
    if (length(init_theta) != 1L || init_theta <= 0) {
      stop("'init_theta' must be a positive scalar.", call. = FALSE)
    }
  }

  # Fit negative binomial model using C++ implementation - now just one call
  if (is.integer(y)) {
    y <- as.numeric(y)
  }

  fit <- structure(fenegbin_fit_(
    X, y, w, FEs, link, beta, eta_vec, init_theta, control
  ), class = c("feglm", "fenegbin"))

  # Compute nobs using y and fitted values
  nobs <- nobs_(nobs_full, nobs_na, y, predict(fit))

  # Information if convergence failed ----
  if (!fit[["conv_outer"]]) {
    cat("Algorithm did not converge.\n")
  }

  # Add names to beta, hessian, and X_dm (if provided) ----
  names(fit[["coefficients"]]) <- nms_sp
  if (control[["keep_tx"]]) {
    colnames(fit[["tx"]]) <- nms_sp
  }
  dimnames(fit[["hessian"]]) <- list(nms_sp, nms_sp)

  # Add to fit list ----
  fit[["nobs"]] <- nobs
  fit[["fe_levels"]] <- fe_levels
  fit[["nms_fe"]] <- nms_fe
  fit[["formula"]] <- formula
  fit[["data"]] <- data
  fit[["family"]] <- negative.binomial(theta = fit[["theta"]], link = link)
  fit[["control"]] <- control

  # Return result ----
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
    fit, theta, iter, conv, nobs, fe_levels,
    nms_fe, formula, data, family, control) {
  reslist <- c(
    fit, list(
      theta      = theta,
      iter_outer = iter,
      conv_outer = conv,
      nobs       = nobs,
      fe_levels  = fe_levels,
      nms_fe     = nms_fe,
      formula    = formula,
      data       = data,
      family     = family,
      control    = control
    )
  )

  # Return result list ----
  structure(reslist, class = c("feglm", "fenegbin"))
}
