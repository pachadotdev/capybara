#' srr_stats
#' @srrstats {G1.0} Implements covariance matrix extraction methods for `apes`, `feglm`, and `felm` objects.
#' @srrstats {G2.1a} Validates input objects as instances of `apes`, `feglm`, or `felm`.
#' @srrstats {G2.2} Provides various covariance estimation types including `hessian`, `outer.product`, and `sandwich`.
#' @srrstats {G2.3} Handles cases with or without clustering variables, ensuring flexibility for diverse use cases.
#' @srrstats {G3.0} Handles edge cases such as non-invertible hessians or missing cluster variables gracefully with
#'  informative errors.
#' @srrstats {G4.0} Integrates seamlessly with the modeling pipeline, supporting consistent outputs for downstream
#'  analysis.
#' @srrstats {RE2.1} Ensures compatibility with multiway clustering approaches as proposed by Cameron, Gelbach, and
#'  Miller (2011).
#' @srrstats {RE2.3} Supports computation of robust covariance estimates for generalized linear models and linear
#'  models.
#' @srrstats {RE5.0} Ensures that the output covariance matrix is correctly labeled for interpretability.
#' @srrstats {RE5.2} Provides explicit errors for invalid or missing clustering variables in clustered covariance
#'  computation.
#' @srrstats {RE6.1} Implements efficient matrix operations to handle large-scale data and high-dimensional models.
#' @noRd
NULL

#' @title Covariance matrix for APEs
#'
#' @description Covariance matrix for the estimator of the average partial effects from objects returned by \link{apes}.
#'
#' @param object an object of class \code{"apes"}.
#' @param ... additional arguments.
#'
#' @return A named matrix of covariance estimates.
#'
#' @seealso \link{apes}
#'
#' @export
#'
#' @noRd
vcov.apes <- function(object, ...) {
  object[["vcov"]]
}

#' @title Covariance matrix for GLMs
#'
#' @description Covariance matrix for the estimator of the structural parameters from objects returned by \link{feglm}.
#'  The covariance is computed during model fitting - either the inverse Hessian (default) or the sandwich estimator if
#'  cluster variables are specified in the formula.
#'
#' @param object an object of class \code{"feglm"}.
#' @param ... additional arguments (currently ignored).
#'
#' @return A named matrix of covariance estimates.
#'
#' @references Cameron, C., J. Gelbach, and D. Miller (2011). "Robust Inference With Multiway Clustering". Journal of
#'  Business & Economic Statistics 29(2).
#'
#' @seealso \link{feglm}
#'
#' @examples
#' # Model without clustering - returns inverse Hessian covariance
#' mod <- fepoisson(mpg ~ wt | cyl, mtcars)
#' round(vcov(mod), 5)
#'
#' # Model with clustering - returns sandwich covariance
#' mod_cl <- fepoisson(mpg ~ wt | cyl | am, mtcars)
#' round(vcov(mod_cl), 5)
#'
#' @export
vcov.feglm <- function(object, ...) {
  v <- object[["vcov"]]

  if (is.null(v)) {
    stop("Covariance matrix not found in model object.", call. = FALSE)
  }

  # Add names to match coefficients
  coef_table <- object[["coef_table"]]
  nms <- rownames(coef_table)
  if (!is.null(nms) && length(nms) > 0) {
    dimnames(v) <- list(nms, nms)
  }

  v
}

# Particular case for linear models ----

#' @title Covariance matrix for LMs
#'
#' @description Covariance matrix for the estimator of the structural parameters from objects returned by \link{felm}.
#'  The covariance is computed during model fitting - either the inverse Hessian (default) or the sandwich estimator if
#'  cluster variables are specified in the formula.
#'
#' @param object an object of class \code{"felm"}.
#' @param ... additional arguments (currently ignored).
#'
#' @return A named matrix of covariance estimates.
#'
#' @references Cameron, C., J. Gelbach, and D. Miller (2011). "Robust Inference With Multiway Clustering". Journal of
#'  Business & Economic Statistics 29(2).
#'
#' @seealso \link{felm}
#'
#' @examples
#' # Model without clustering - returns inverse Hessian covariance
#' mod <- felm(log(mpg) ~ log(wt) | cyl, mtcars)
#' round(vcov(mod), 5)
#'
#' # Model with clustering - returns sandwich covariance
#' mod_cl <- felm(log(mpg) ~ log(wt) | cyl | am, mtcars)
#' round(vcov(mod_cl), 5)
#'
#' @export
vcov.felm <- function(object, ...) {
  v <- object[["vcov"]]

  if (is.null(v)) {
    stop("Covariance matrix not found in model object.", call. = FALSE)
  }

  # Add names to match coefficients
  coef_table <- object[["coef_table"]]
  nms <- rownames(coef_table)
  if (!is.null(nms) && length(nms) > 0) {
    dimnames(v) <- list(nms, nms)
  }

  v
}
