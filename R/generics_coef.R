# srr_stats
# {G1.0} Defines `coef` methods for extracting coefficients from various model objects.
# {G2.1a} Ensures that the input object is of the expected class (`apes`, `feglm`, or `felm`).
# {G3.1a} Outputs coefficients in a consistent format for interpretability.
# {G3.1b} Supports multiple model object types, maintaining a standardized interface.
# {G3.1c} Provides access to summary statistics (`coefficients`) where applicable.
# {G5.1} Includes robust error handling for unsupported or invalid input objects.
# {G5.4a} Includes tests for extracting coefficients from simple and complex model objects.
# {RE4.2} Returns coefficients via a standard method for feglm-type objects and derived classes (i.e., felm, apes, etc).
# {RE5.0} Enables seamless integration with downstream analysis workflows.
# {RE5.2} Maintains computational efficiency in coefficient extraction.

#' @title Extract coefficients from a 'capybara_model' object
#' @description Shared by 'feglm' and 'felm' objects (and their subclasses, e.g.
#'  'fenegbin', 'fepoisson_asymmetric'), similar to the 'coef' method for 'glm'/'lm' objects.
#' @param object a 'capybara_model' object
#' @param ... additional arguments for S3 compliance (unused)
#' @exportS3Method
coef.capybara_model <- function(object, ...) {
  ct <- object[["coef_table"]]
  setNames(ct[, 1], rownames(ct))
}

#' @title Extract coefficients from 'summary.feglm' object
#' @description Similar to the 'coef' method for 'summary.glm' objects.
#' @param object 'summary.feglm' object
#' @param ... additional arguments for S3 compliance (unused)
#' @exportS3Method
coef.summary.feglm <- function(object, ...) {
  # coef_table already has row/column names from the model fitting

  object[["coef_table"]]
}

#' @title Extract coefficients from 'summary.felm' object
#' @description Similar to the 'coef' method for 'summary.felm' objects.
#' @param object 'summary.felm' object
#' @param ... additional arguments for S3 compliance (unused)
#' @exportS3Method
coef.summary.felm <- function(object, ...) {
  # coef_table already has row/column names from the model fitting
  object[["coef_table"]]
}
