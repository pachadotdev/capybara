# These are dummy functions to follow the summary method
# The C++ structs already return all the necessary components to print the equivalent of glm() + summary()

# srr_stats
# {G1.0} Implements `summary` methods for various model objects (`apes`, `feglm`, `felm`) to provide detailed post-estimation statistics.
# {G2.1a} Ensures that input objects are of the expected class (`apes`, `feglm`, `felm`).
# {G2.3} Accurately computes standard errors, z-values, and p-values for model coefficients.
# {G3.1} Includes residual statistics, deviance measures, and (where applicable) R-squared values for Poisson models.
# {G5.2a} Outputs include well-structured coefficient matrices with appropriate column headers and row names.
# {RE2.1} Summary methods ensure compatibility with standard statistical workflows by providing model evaluation metrics.
# {RE2.2} Custom handling of model-specific details like Poisson R-squared and Negative Binomial `theta` values.
# {RE4.11} The deviance, null deviance, R-squared and adjusted R-squared are returned in the summaries.
# {RE4.18} Implemented `summary()` functions specific for GLMs and LMs (i.e., it shows R2 for LMs and Poisson models).
# {RE5.0} Reduces cyclomatic complexity through modular functions for computing summary components.
# {RE5.2} Facilitates interpretability of models by providing a unified and clear summary output format.

#' @title Summary method for fixed effects APEs
#' @inherit vcov.apes
#' @export
#' @noRd
summary.apes <- function(object, ...) {
  class(object) <- c("summary.apes", class(object))
  object
}

#' @title Summary method for fixed effects GLMs
#' @inherit vcov.feglm
#' @export
#' @noRd
summary.feglm <- function(object, ...) {
  class(object) <- c("summary.feglm", class(object))
  object
}

#' @title Summary method for fixed effects LMs
#' @inherit vcov.felm
#' @export
#' @noRd
summary.felm <- function(
  object,
  type = "hessian",
  ...
) {
  class(object) <- c("summary.felm", class(object))
  object
}
