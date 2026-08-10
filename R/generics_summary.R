# These are dummy functions to follow the summary method
# The C++ structs already return all the necessary components to print the equivalent of glm() + summary()

# srr_stats
# {G1.0} Implements `summary` methods for various model objects (`feglm`, `felm`) to provide detailed post-estimation statistics.
# {G2.1a} Ensures that input objects are of the expected class (`feglm`, `felm`).
# {G2.3} Accurately computes standard errors, z-values, and p-values for model coefficients.
# {G3.1} Includes residual statistics, deviance measures, and (where applicable) R-squared values for Poisson models.
# {G5.2a} Outputs include well-structured coefficient matrices with appropriate column headers and row names.
# {RE2.1} Summary methods ensure compatibility with standard statistical workflows by providing model evaluation metrics.
# {RE2.2} Custom handling of model-specific details like Poisson R-squared and Negative Binomial `theta` values.
# {RE4.11} The deviance, null deviance, R-squared and adjusted R-squared are returned in the summaries.
# {RE4.18} Implemented `summary()` functions specific for GLMs and LMs (i.e., it shows R2 for LMs and Poisson models).
# {RE5.0} Reduces cyclomatic complexity through modular functions for computing summary components.
# {RE5.2} Facilitates interpretability of models by providing a unified and clear summary output format.

#' @title Summary method for fixed effects GLMs
#' @param object 'feglm' object
#' @param ... additional arguments for S3 compliance (unused)
#' @exportS3Method
summary.feglm <- function(object, ...) {
  class(object) <- c("summary.feglm", class(object))
  object
}

#' @title Summary method for fixed effects LMs
#' @param object 'felm' object
#' @param type 'hessian' (no other type works with 'felm')
#' @param ... additional arguments for S3 compliance (unused)
#' @exportS3Method
summary.felm <- function(
  object,
  type = "hessian",
  ...
) {
  class(object) <- c("summary.felm", class(object))
  object
}
