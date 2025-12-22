#' @title Check for Statistical Separation in Poisson Models
#'
#' @description Detect statistical separation (perfect predictors) in Poisson
#'  regression models. Separation occurs when a linear combination of regressors
#'  can perfectly predict zero outcomes, causing maximum likelihood estimates to

#'  diverge. This function implements the ReLU and simplex methods from Correia,
#'  Guimarães, and Zylkin (2019).
#'
#' @param y Numeric vector of response values (non-negative integers for count data).
#' @param X Numeric matrix of regressors (design matrix). Can be \code{NULL} or
#'  empty if only checking fixed effects separation.
#' @param w Numeric vector of weights. Default is a vector of ones.
#' @param tol Tolerance for convergence. Default is \code{1e-8}.
#' @param zero_tol Tolerance for treating values as zero. Default is \code{1e-12}.
#' @param max_iter Maximum iterations for ReLU method. Default is \code{1000L}.
#' @param simplex_max_iter Maximum iterations for simplex method. Default is \code{10000L}.
#' @param use_relu Logical, whether to use the ReLU method. Default is \code{TRUE}.
#' @param use_simplex Logical, whether to use the simplex method. Default is \code{TRUE}.
#' @param verbose Logical, whether to print progress information. Default is \code{FALSE}.
#'
#' @return A list with components:
#' \describe{
#'   \item{separated_obs}{Integer vector of row indices (1-based) of separated observations.}
#'   \item{num_separated}{Number of separated observations found.}
#'   \item{converged}{Logical indicating if the algorithm converged.}
#'   \item{iterations}{Number of iterations used (for ReLU method).}
#'   \item{certificate}{Numeric vector giving the separation certificate (z vector), if computed by the ReLU method.}
#' }
#'
#' @details
#' Statistical separation occurs in Poisson and other count data models when
#' there exists a linear combination of regressors that is non-negative for all
#' observations with \eqn{y = 0} and positive for at least one such observation.
#' This causes the MLE to not exist (coefficients diverge to \eqn{\pm\infty}).
#'
#' The function implements two complementary detection methods:
#'
#' \strong{ReLU Method:} An iterative algorithm that solves a sequence of
#' constrained least squares problems using the Rectified Linear Unit (ReLU)
#' activation function. This method is more thorough and provides a certificate
#' of separation.
#'
#' \strong{Simplex Method:} A linear programming approach that checks for
#' separating hyperplanes. This method is faster for initial screening.
#'
#' Both methods are based on the theoretical framework of Correia, Guimarães,
#' and Zylkin (2019), who show that separation detection is equivalent to
#' solving a specific linear programming problem.
#'
#' @references
#' Correia, S., Guimarães, P., and Zylkin, T. (2019). "Verifying the Existence
#' of Maximum Likelihood Estimates for Generalized Linear Models."
#' \url{https://arxiv.org/abs/1903.01633}
#'
#' Correia, S., Guimarães, P., and Zylkin, T. (2020). "Fast Poisson Estimation
#' with High-Dimensional Fixed Effects." \emph{The Stata Journal}, 20(1), 95-115.
#'
#' @examples
#' \dontrun{
#' # Example with separation
#' set.seed(123)
#' n <- 100
#' x <- rnorm(n)
#' # Create separation: y = 0 whenever x > 1
#' y <- rpois(n, exp(1 - x))
#' y[x > 1] <- 0
#'
#' # Check for separation
#' result <- check_separation(y, matrix(x, ncol = 1))
#' print(result$num_separated)
#' }
#'
#' @seealso \code{\link{fepoisson}}, \code{\link{feglm}}
#'
#' @export
check_separation <- function(
    y,
    X = NULL,
    w = NULL,
    tol = 1e-8,
    zero_tol = 1e-12,
    max_iter = 1000L,
    simplex_max_iter = 10000L,
    use_relu = TRUE,
    use_simplex = TRUE,
    verbose = FALSE) {
  # Validate y
  y <- as.numeric(y)
  n <- length(y)

  if (any(is.na(y))) {
    stop("y contains NA values.", call. = FALSE)
  }

  if (any(y < 0)) {
    stop("y must be non-negative.", call. = FALSE)
  }

  # Validate X
  if (is.null(X) || length(X) == 0) {
    X <- matrix(0, nrow = n, ncol = 0)
  } else {
    X <- as.matrix(X)
    if (nrow(X) != n) {
      stop("Number of rows in X must equal length of y.", call. = FALSE)
    }
  }

  # Validate w
  if (is.null(w)) {
    w <- rep(1.0, n)
  } else {
    w <- as.numeric(w)
    if (length(w) != n) {
      stop("Length of w must equal length of y.", call. = FALSE)
    }
    if (any(w < 0)) {
      stop("Weights must be non-negative.", call. = FALSE)
    }
  }

  # Validate parameters
  if (tol <= 0) {
    stop("tol must be positive.", call. = FALSE)
  }
  if (zero_tol <= 0) {
    stop("zero_tol must be positive.", call. = FALSE)
  }
  max_iter <- as.integer(max_iter)
  simplex_max_iter <- as.integer(simplex_max_iter)
  if (max_iter < 1L) {
    stop("max_iter must be at least 1.", call. = FALSE)
  }
  if (simplex_max_iter < 1L) {
    stop("simplex_max_iter must be at least 1.", call. = FALSE)
  }

  use_relu <- as.logical(use_relu)
  use_simplex <- as.logical(use_simplex)
  verbose <- as.logical(verbose)

  if (!use_relu && !use_simplex) {
    stop("At least one of use_relu or use_simplex must be TRUE.", call. = FALSE)
  }

  # Call C++ function
  result <- check_separation_(
    y_r = y,
    X_r = X,
    w_r = w,
    tol = tol,
    zero_tol = zero_tol,
    max_iter = max_iter,
    simplex_max_iter = simplex_max_iter,
    use_relu = use_relu,
    use_simplex = use_simplex,
    verbose = verbose
  )

  # Clean up result
  result$separated_obs <- as.integer(result$separated_obs)
  result$num_separated <- as.integer(result$num_separated)

  class(result) <- "capybara_separation"
  result
}
