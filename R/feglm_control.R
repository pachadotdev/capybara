#' srr_stats
#' @srrstats {G1.0} Implements controls for efficient and numerically stable fitting of generalized linear models with fixed effects.
#' @srrstats {G2.0} Validates numeric input parameters to ensure they meet constraints (e.g., positive tolerance levels).
#' @srrstats {G2.0a} The main function explains that the tolerance must be unidimensional or the function gives an error.
#' @srrstats {G2.1a} Ensures the proper data types for arguments (e.g., logical for `trace`, integer for `iter_max`).
#' @srrstats {G2.3a} Uses argument validation to ensure appropriate ranges for critical parameters (e.g., `iter_max` and `limit` >= 1).
#' @srrstats {G2.14a} Provides informative error messages when tolerance levels or iteration counts are invalid.
#' @srrstats {G2.14b} Provides clear error messages when the data structure is incompatible with the model requirements.
#' @srrstats {G5.2a} Produces unique and descriptive error messages for all validation checks.
#' @srrstats {RE3.0} If the deviance difference between 2 iterations is not less than tolerance after the max number of iterations, it prints a convergence warning.
#' @srrstats {RE5.0} Supports control over algorithmic complexity, such as dropping perfectly separated observations (`drop_pc`) and optional matrix storage (`keep_mx`).
#' @srrstats {G5.4a} Includes robust edge case handling, such as enforcing positive tolerance and iteration counts.
#' @noRd
NULL

#' NA_standards
#' @srrstatsNA {G2.14} Missing observations are dropped, otherwise providing imputation methods would bias the estimation (i.e., replacing all missing values with the median).
#' @noRd
NULL

#' @title Set \code{feglm} Control Parameters
#'
#' @description Set and change parameters used for fitting \code{\link{feglm}}.
#'  Termination conditions are similar to \code{\link[stats]{glm}}.
#'
#' @param dev_tol tolerance level for the first stopping condition of the
#'  maximization routine. The stopping condition is based on the relative change
#'  of the deviance in iteration \eqn{r} and can be expressed as follows:
#'  \eqn{|dev_{r} - dev_{r - 1}| / (0.1 + |dev_{r}|) < tol}{|dev - devold| /
#'  (0.1 + |dev|) < tol}. The default is \code{1.0e-08}.
#' @param center_tol tolerance level for the stopping condition of the centering
#'  algorithm. The stopping condition is based on the relative change of the
#'  centered variable similar to the \code{'lfe'} package. The default is
#'  \code{1.0e-08}.
#' @param iter_max unsigned integer indicating the maximum number of iterations
#'  in the maximization routine. The default is \code{25L}.
#' @param iter_center_max unsigned integer indicating the maximum number of
#'  iterations in the centering algorithm. The default is \code{10000L}.
#' @param iter_inner_max unsigned integer indicating the maximum number of
#'  iterations in the inner loop of the centering algorithm. The default is
#'  \code{50L}.
#' @param iter_interrupt unsigned integer indicating the maximum number of
#' iterations before the algorithm is interrupted. The default is \code{1000L}.
#' @param limit unsigned integer indicating the maximum number of iterations of
#'  \code{\link[MASS]{theta.ml}}. The default is \code{10L}.
#' @param trace logical indicating if output should be produced in each
#'  iteration. Default is \code{FALSE}.
#' @param drop_pc logical indicating to drop observations that are perfectly
#'  classified/separated and hence do not contribute to the log-likelihood. This
#'  option is useful to reduce the computational costs of the maximization
#'  problem and improves the numerical stability of the algorithm. Note that
#'  dropping perfectly separated observations does not affect the estimates.
#'  The default is \code{TRUE}.
#' @param keep_mx logical indicating if the centered regressor matrix should be
#'  stored. The centered regressor matrix is required for some covariance
#'  estimators, bias corrections, and average partial effects. This option saves
#'  some computation time at the cost of memory. The default is \code{TRUE}.
#'
#' @return A named list of control parameters.
#'
#' @examples
#' feglm_control(0.05, 0.05, 10L, 10L, TRUE, TRUE, TRUE)
#'
#' @seealso \code{\link{feglm}}
#'
#' @export
feglm_control <- function(
    dev_tol = 1.0e-06,
    center_tol = 1.0e-06,
    iter_max = 25L,
    iter_center_max = 10000L,
    iter_inner_max = 50L,
    iter_interrupt = 1000L,
    limit = 10L,
    trace = FALSE,
    drop_pc = TRUE,
    keep_mx = FALSE) {
  # Check validity of tolerance parameters
  if (dev_tol <= 0.0 || center_tol <= 0.0) {
    stop(
      "All tolerance parameters should be greater than zero.",
      call. = FALSE
    )
  }

  # Check validity of 'iter_max'
  iter_max <- as.integer(iter_max)
  if (iter_max < 1L) {
    stop(
      "Maximum number of iterations should be at least one.",
      call. = FALSE
    )
  }

  # Check validity of 'iter_center_max'
  iter_center_max <- as.integer(iter_center_max)
  if (iter_center_max < 1L) {
    stop(
      "Maximum number of iterations for centering should be at least one.",
      call. = FALSE
    )
  }

  # Check validity of 'iter_inner_max'
  iter_inner_max <- as.integer(iter_inner_max)
  if (iter_inner_max < 1L) {
    stop(
      "Maximum number of iterations for inner loop should be at least one.",
      call. = FALSE
    )
  }

  # Check validity of 'iter_interrupt'
  iter_interrupt <- as.integer(iter_interrupt)
  if (iter_interrupt < 1L) {
    stop(
      "Maximum number of iterations for interrupt should be at least one.",
      call. = FALSE
    )
  }

  # Check validity of 'limit'
  limit <- as.integer(limit)
  if (limit < 1L) {
    stop("Maximum number of iterations should be at least one.", call. = FALSE)
  }

  # Return list with control parameters
  list(
    dev_tol = dev_tol,
    center_tol = center_tol,
    iter_max = iter_max,
    iter_center_max = iter_center_max,
    iter_inner_max = iter_inner_max,
    iter_interrupt = iter_interrupt,
    limit = limit,
    trace = as.logical(trace),
    drop_pc = as.logical(drop_pc),
    keep_mx = as.logical(keep_mx)
  )
}
