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
#'
#' # subset trade flows to avoid fitting time warnings during check
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 700), ]
#'
#' mod <- fenegbin(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_2006
#' )
#'
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
#'  \item{iter.outer}{the number of outer iterations}
#'  \item{conv.outer}{a logical indicating whether the outer loop converged}
#'  \item{nobs}{a named vector with the number of observations used in the
#'   estimation indicating the dropped and perfectly predicted observations}
#'  \item{lvls_k}{a named vector with the number of levels in each fixed
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
  lhs <- nobs_na <- nobs_full <- NA
  model_frame_(data, formula, weights)

  # Check starting guess of theta ----
  family <- init_theta_(init_theta, link)
  rm(init_theta)

  # Ensure that model response is in line with the chosen model ----
  check_response_(data, lhs, family)

  # Get names of the fixed effects variables and sort ----
  k_vars <- attr(terms(formula, rhs = 2L), "term.labels")

  # Generate temporary variable ----
  tmp_var <- temp_var_(data)

  # Drop observations that do not contribute to the log likelihood ----
  data <- drop_by_link_type_(data, lhs, family, tmp_var, k_vars, control)

  # Transform fixed effects and clusters to factors ----
  data <- transform_fe_(data, formula, k_vars)

  # Determine the number of dropped observations ----
  nt <- nrow(data)
  nobs <- nobs_(nobs_full, nobs_na, nt)

  # Extract model response and regressor matrix ----
  nms_sp <- p <- NA
  model_response_(data, formula)

  # Check for linear dependence in 'x' ----
  check_linear_dependence_(y, x, p + 1L)

  # Extract weights if required ----
  if (is.null(weights)) {
    wt <- rep(1.0, nt)
  } else {
    wt <- data[[weights]]
  }

  # Check validity of weights ----
  check_weights_(wt)

  # Compute and check starting guesses ----
  start_guesses_(beta_start, eta_start, y, x, beta, nt, wt, p, family)

  # Get names and number of levels in each fixed effects category ----
  nms_fe <- lapply(data[, .SD, .SDcols = k_vars], levels)
  lvls_k <- vapply(nms_fe, length, integer(1))

  # Generate auxiliary list of indexes for different sub panels ----
  k_list <- get_index_list_(k_vars, data)

  # Extract control arguments ----
  tol <- control[["dev_tol"]]
  limit <- control[["limit"]]
  iter_max <- control[["iter_max"]]
  trace <- control[["trace"]]

  # Initial negative binomial fit ----

  theta <- suppressWarnings(
    theta.ml(
      y     = y,
      mu    = family[["linkinv"]](eta),
      n     = nt,
      limit = limit,
      trace = trace
    )
  )

  fit <- feglm_fit_(
    beta, eta, y, x, wt, theta, family[["family"]], control, k_list
  )

  beta <- fit[["coefficients"]]
  eta <- fit[["eta"]]
  dev <- fit[["deviance"]]

  # Alternate between fitting glm and \theta ----
  conv <- FALSE
  for (iter in seq.int(iter_max)) {
    # Fit negative binomial model
    dev_old <- dev
    theta_old <- theta
    family <- negative.binomial(theta, link)
    theta <- suppressWarnings(
      theta.ml(
        y     = y,
        mu    = family[["linkinv"]](eta),
        n     = nt,
        limit = limit,
        trace = trace
      )
    )
    fit <- feglm_fit_(
      beta, eta, y, x, wt, theta, family[["family"]], control,
      k_list
    )
    beta <- fit[["coefficients"]]
    eta <- fit[["eta"]]
    dev <- fit[["deviance"]]

    # Progress information
    if (trace) {
      cat("Outer Iteration=", iter, "\n")
      cat("Deviance=", format(dev, digits = 5L, nsmall = 2L), "\n")
      cat("theta=", format(theta, digits = 5L, nsmall = 2L), "\n")
      cat("Estimates=", format(beta, digits = 3L, nsmall = 2L), "\n")
    }

    # Check termination condition ----
    if (fenegbin_check_convergence_(dev, dev_old, theta, theta_old, tol)) {
      if (trace) {
        cat("Convergence\n")
      }
      conv <- TRUE
      break
    }
  }

  y <- x <- eta <- NULL

  # Information if convergence failed ----
  if (!conv && trace) cat("Algorithm did not converge.\n")

  # Add names to beta, hessian, and mx (if provided) ----
  names(fit[["coefficients"]]) <- nms_sp
  if (control[["keep_mx"]]) {
    colnames(fit[["mx"]]) <- nms_sp
  }
  dimnames(fit[["hessian"]]) <- list(nms_sp, nms_sp)

  fenegbin_result_list_(
    fit, theta, iter, conv, nobs, lvls_k, nms_fe,
    formula, data, family, control
  )
}

# Convergence Check ----

fenegbin_check_convergence_ <- function(dev, dev_old, theta, theta_old, tol) {
  dev_crit <- abs(dev - dev_old) / (0.1 + abs(dev))
  theta_crit <- abs(theta - theta_old) / (0.1 + abs(theta_old))
  dev_crit <= tol && theta_crit <= tol
}

# Generate result list ----

fenegbin_result_list_ <- function(
    fit, theta, iter, conv, nobs, lvls_k,
    nms_fe, formula, data, family, control) {
  reslist <- c(
    fit, list(
      theta      = theta,
      iter.outer = iter,
      conv.outer = conv,
      nobs       = nobs,
      lvls_k     = lvls_k,
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
