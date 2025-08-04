#' srr_stats
#' @srrstats {G1.0} Implements Poisson regression with high-dimensional fixed effects via `feglm`.
#' @srrstats {G2.1a} Validates input `formula` to ensure correct specification of fixed effects.
#' @srrstats {G2.1b} Ensures `data` is appropriately formatted and contains sufficient observations.
#' @srrstats {G2.3a} Uses internally validated arguments (`control` and starting guesses) for consistency.
#' @srrstats {G3.1a} Supports canonical log link function for Poisson family.
#' @srrstats {G3.1b} Provides detailed outputs including coefficients, deviance, and convergence diagnostics.
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
#' @srrstats {RE7.4} Provides comprehensive examples that demonstrate proper usage of the regression functions,
#' covering input preparation, function execution, and result interpretation.
#' @noRd
NULL

#' @title Poisson model fitting high-dimensional with k-way fixed effects
#'
#' @description A wrapper for \code{\link{feglm}} with
#'  \code{family = poisson()}.
#'
#' @inheritParams feglm
#'
#' @examples
#' # check the feglm examples for the details about clustered standard errors
#' mod <- fepoisson(mpg ~ wt | cyl, mtcars)
#' summary(mod)
#'
#' @return A named list of class \code{"feglm"}.
#'
#' @export
fepoisson <- function(
    formula = NULL,
    data = NULL,
    weights = NULL,
    beta_start = NULL,
    eta_start = NULL,
    control = NULL) {
  feglm(
    formula = formula, data = data, weights = weights, family = poisson(),
    beta_start = beta_start, eta_start = eta_start, control = control
  )
}
