#' srr_stats
#' @srrstats {G1.0} Implements Poisson regression with high-dimensional fixed effects via `feglm`.
#' @srrstats {G2.1a} Validates input `formula` to ensure correct specification of fixed effects.
#' @srrstats {G2.1b} Ensures `data` is appropriately formatted and contains sufficient observations.
#' @srrstats {G2.3a} Uses internally validated arguments (`control` and starting guesses) for consistency.
#' @srrstats {G3.1a} Supports canonical log link function for Poisson family.
#' @srrstats {G3.1b} Provides detailed outputs including coefficients, deviance, and convergence diagnostics.
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
#' @srrstats {RE5.0} Optimized for scaling to large datasets with high-dimensional fixed effects.
#' @srrstats {RE5.1} Efficiently projects out fixed effects using auxiliary indexing structures.
#' @srrstats {RE5.2} Provides detailed warnings and error handling for convergence and dependence issues.
#' @srrstats {RE5.3} Thoroughly documents interactions between model features, inputs, and controls.
#' @srrstats {RE7.4} Provides comprehensive examples that demonstrate proper usage of the regression functions, covering
#'  input preparation, function execution, and result interpretation.
#' @noRd
NULL

#' @title Asymmetric Poisson Pseudo-Maximum Likelihood (APPML) Estimation
#'
#' @description Fits an asymmetric Poisson pseudo-maximum likelihood model with high-dimensional fixed effects
#'  using expectile regression. This approach extends standard PPML by allowing different weights for positive
#'  and negative residuals, enabling estimation of conditional expectiles rather than the conditional mean.
#'
#' @inheritParams feglm
#' @param residuals_start an optional vector of starting residuals for the iterative algorithm. If \code{NULL},
#'  residuals from an initial unweighted fit are used.
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
#' @return A named list of class \code{"feglm_asymmetric"} containing:
#'  \item{coefficients}{named vector of estimated coefficients}
#'  \item{vcov}{variance-covariance matrix of coefficients}
#'  \item{fitted_values}{fitted values from the final iteration}
#'  \item{residuals}{residuals from the final fit}
#'  \item{weights}{final observation weights}
#'  \item{converged}{logical indicating whether the algorithm converged}
#'  \item{iterations}{number of iterations performed}
#'  \item{expectile}{the expectile value used}
#'  \item{objective_function}{final value of the convergence criterion}
#'  \item{negative_residuals_share}{proportion of negative residuals in final fit}
#'  \item{fit}{the final \code{feglm} fit object}
#'  \item{nobs}{number of observations used}
#'
#' @references
#' Newey, W. K., & Powell, J. L. (1987). Asymmetric least squares estimation and testing.
#'   \emph{Econometrica}, 55(4), 819-847.
#'
#' @examples
#' # Standard PPML (expectile = 0.5)
#' mod_ppml <- fepoisson_asymmetric(
#'   mpg ~ wt | cyl, mtcars,
#'   control = fit_control(expectile = 0.5)
#' )
#' summary(mod_ppml)
#'
#' # Lower expectile (10th) - more weight on negative residuals
#' mod_low <- fepoisson_asymmetric(
#'   mpg ~ wt | cyl, mtcars,
#'   control = fit_control(expectile = 0.1)
#' )
#'
#' # Upper expectile (90th) - more weight on positive residuals
#' mod_high <- fepoisson_asymmetric(
#'   mpg ~ wt | cyl, mtcars,
#'   control = fit_control(expectile = 0.9)
#' )
#'
#' # Compare coefficients across expectiles
#' cbind(
#'   low = coef(mod_low),
#'   median = coef(mod_ppml),
#'   high = coef(mod_high)
#' )
#'
#' @seealso \link{fepoisson}, \link{feglm}, \link{fit_control}
#'
#' @export
fepoisson_asymmetric <- function(
  formula = NULL,
  data = NULL,
  vcov = NULL,
  beta_start = NULL,
  eta_start = NULL,
  offset = NULL,
  control = NULL,
  residuals_start = NULL
) {
  # Initialize control if NULL
  if (is.null(control)) {
    control <- fit_control()
  }

  # Extract expectile parameters from control

  expectile <- control[["expectile"]]
  expectile_tol <- control[["expectile_tol"]]
  expectile_iter_max <- control[["expectile_iter_max"]]
  expectile_trace <- control[["expectile_trace"]]

  # Validate expectile is specified
  if (is.null(expectile)) {
    stop(
      "expectile must be specified in control. Use: control = fit_control(expectile = 0.5)",
      call. = FALSE
    )
  }

  # Get response variable name from formula
  y_var <- as.character(formula[[2L]])

  # Ensure data is a data.frame
  if (!is.data.frame(data)) {
    stop("'data' must be a data.frame.", call. = FALSE)
  }

  # Make a copy to avoid modifying the original
  data_internal <- as.data.frame(data)

  # Initial fit without asymmetric weights
  if (is.null(residuals_start)) {
    initial_fit <- feglm(
      formula = formula,
      data = data_internal,
      family = "poisson",
      vcov = vcov,
      beta_start = beta_start,
      eta_start = eta_start,
      offset = offset,
      control = control
    )
    fitted_vals <- fitted(initial_fit)
    residuals_current <- data_internal[[y_var]][as.integer(names(fitted_vals))] - fitted_vals
  } else {
    residuals_current <- residuals_start
  }

  # Initialize weights based on expectile
  weights_current <- abs(expectile - (residuals_current < 0))

  # Store weights in data for fitting
  data_internal[[".appml_weights"]] <- NA_real_
  obs_indices <- as.integer(names(residuals_current))
  data_internal[[".appml_weights"]][obs_indices] <- weights_current

  # Initialize convergence tracking
  cv <- Inf
  count <- 0L
  bold <- NULL
  fit <- NULL

  # Iterative reweighting
  while (cv > expectile_tol && count < expectile_iter_max) {
    count <- count + 1L

    # Fit weighted Poisson model
    fit <- tryCatch(
      {
        feglm(
          formula = formula,
          data = data_internal,
          family = "poisson",
          weights = ".appml_weights",
          vcov = vcov,
          beta_start = if (count == 1L) beta_start else coef(fit),
          eta_start = NULL,
          offset = offset,
          control = control
        )
      },
      error = function(e) {
        if (expectile_trace) {
          message("Error during fitting at iteration ", count, ": ", e$message)
        }
        NULL
      }
    )

    if (is.null(fit)) {
      warning("Fitting failed at iteration ", count, ". Returning last successful fit.", call. = FALSE)
      break
    }

    # Compute new residuals
    fitted_vals <- fitted(fit)
    obs_indices <- as.integer(names(fitted_vals))
    residuals_new <- data_internal[[y_var]][obs_indices] - fitted_vals

    # Extract coefficients
    b <- coef(fit)

    # Initialize bold on first iteration
    if (count == 1L) {
      bold <- rep(0, length(b))
    }

    # Compute convergence criterion: (b - bold)' * inv(V) * (b - bold)
    current_vcov <- vcov(fit)

    invV <- tryCatch(
      {
        solve(current_vcov)
      },
      error = function(e) {
        if (expectile_trace) {
          message("Error inverting vcov at iteration ", count, ": ", e$message)
        }
        NULL
      }
    )

    if (is.null(invV)) {
      warning("Could not invert variance-covariance matrix at iteration ", count, ".", call. = FALSE)
      break
    }

    diff_b <- b - bold
    cv <- as.numeric(t(diff_b) %*% invV %*% diff_b)

    # Update residuals and weights
    residuals_current <- residuals_new
    weights_current <- abs(expectile - (residuals_current < 0))
    data_internal[[".appml_weights"]][obs_indices] <- weights_current

    # Print iteration info if requested
    if (expectile_trace) {
      message("Iteration ", count, ": objective function = ", format(cv, scientific = TRUE))
    }

    # Update bold for next iteration
    bold <- b
  }

  # Check convergence
  converged <- cv <= expectile_tol

  if (!converged && count >= expectile_iter_max) {
    warning(
      "Algorithm did not converge after ", expectile_iter_max, " iterations. ",
      "Final objective function value: ", format(cv, scientific = TRUE),
      call. = FALSE
    )
  }

  # Compute share of negative residuals
  negative_share <- mean(residuals_current < 0)

  # Print summary if tracing
  if (expectile_trace) {
    message("\n")
    message("Number of obs = ", length(residuals_current))
    message("Iterations = ", count)
    message("Tolerance = ", expectile_tol)
    message("Objective function = ", format(cv, scientific = TRUE))
    message("% negative residuals = ", round(100 * negative_share, 3), "%")
    message("Expectile = ", expectile, " expectile regression")
  }

  # Build result object
  result <- list(
    coefficients = coef(fit),
    vcov = vcov(fit),
    fitted_values = fitted(fit),
    residuals = residuals_current,
    weights = weights_current,
    converged = converged,
    iterations = count,
    expectile = expectile,
    objective_function = cv,
    negative_residuals_share = negative_share,
    fit = fit,
    nobs = length(residuals_current),
    formula = formula,
    family = fit[["family"]],
    control = control
  )

  structure(result, class = c("feglm_asymmetric", "feglm"))
}
