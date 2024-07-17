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
#' @seealso \code{\link{feglm}}
feglm_control <- function(
    dev_tol = 1.0e-08,
    center_tol = 1.0e-08,
    iter_max = 25L,
    limit = 10L,
    trace = FALSE,
    drop_pc = TRUE,
    keep_mx = TRUE) {
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

  # Check validity of 'limit'
  limit <- as.integer(limit)
  if (limit < 1L) {
    stop("Maximum number of iterations should be at least one.", call. = FALSE)
  }

  # Return list with control parameters
  list(
    dev_tol    = dev_tol,
    center_tol = center_tol,
    iter_max   = iter_max,
    limit      = limit,
    trace      = as.logical(trace),
    drop_pc    = as.logical(drop_pc),
    keep_mx    = as.logical(keep_mx)
  )
}
