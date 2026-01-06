#' srr_stats
#' @srrstats {G1.0} Implements `summary` methods for various model objects (`apes`, `feglm`, `felm`) to provide detailed post-estimation statistics.
#' @srrstats {G2.1a} Ensures that input objects are of the expected class (`apes`, `feglm`, `felm`).
#' @srrstats {G2.3} Accurately computes standard errors, z-values, and p-values for model coefficients.
#' @srrstats {G3.1} Includes residual statistics, deviance measures, and (where applicable) pseudo R-squared values for Poisson models.
#' @srrstats {G5.2a} Outputs include well-structured coefficient matrices with appropriate column headers and row names.
#' @srrstats {RE2.1} Summary methods ensure compatibility with standard statistical workflows by providing model evaluation metrics.
#' @srrstats {RE2.2} Custom handling of model-specific details like Poisson pseudo R-squared and Negative Binomial `theta` values.
#' @srrstats {RE4.11} The deviance, null deviance, R-squared and adjusted R-squared are returned in the summaries.
#' @srrstats {RE4.18} Implemented `summary()` functions specific for GLMs and LMs (i.e., it shows R2 for LMs and pseudo R2 for Poisson models).
#' @srrstats {RE5.0} Reduces cyclomatic complexity through modular functions for computing summary components.
#' @srrstats {RE5.2} Facilitates interpretability of models by providing a unified and clear summary output format.
#' @noRd
NULL

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
