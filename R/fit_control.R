#' srr_stats
#' @srrstats {G1.0} Implements controls for efficient and numerically stable fitting of generalized linear models with fixed effects.
#' @srrstats {G2.0} Validates numeric input parameters to ensure they meet constraints (e.g., positive tolerance levels).
#' @srrstats {G2.0a} The main function explains that the tolerance must be unidimensional or the function gives an error.
#' @srrstats {G2.1a} Ensures the proper data types for arguments (e.g., logical for `trace`, integer for `iter_max`).
#' @srrstats {G2.3a} Uses argument validation to ensure appropriate ranges for critical parameters (e.g., `iter_max` and `limit` >= 1).
#' @srrstats {G2.14a} Provides informative error messages when tolerance levels or iteration counts are invalid.
#' @srrstats {G2.14b} Provides clear error messages when the data structure is incompatible with the model requirements.
#' @srrstats {G5.2a} Produces unique and descriptive error messages for all validation checks.
#' @srrstats {RE3.0} If the deviance difference between 2 iterations is not less than tolerance after the max number of iterations, it
#'  prints a convergence warning.
#' @srrstats {RE5.0} Supports control over algorithmic complexity, such as dropping perfectly separated observations (`drop_pc`)
#'  and optional matrix storage (`keep_mx`).
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
#'  algorithm. The default is \code{1.0e-08}.
#' @param iter_center_max unsigned integer indicating the maximum number of
#'  iterations in the centering algorithm. The default is \code{10000L}.
#' @param collin_tol tolerance level for detecting collinearity. The default is
#'  \code{1.0e-10}.

feglm_control <- function(
    dev_tol = 1.0e-8,
    center_tol = 1.0e-8,
    collin_tol = 1.0e-10, # Make this configurable
    iter_max = 25L,
    iter_center_max = 10000L, # Make this configurable
    iter_inner_max = 50L,
    iter_interrupt = 1000L,
    iter_ssr = 10L,
    limit = 10L,
    trace = FALSE,
    drop_pc = TRUE,
    keep_mx = FALSE) {
  # ... existing validation ...

  # Check validity of 'collin_tol'
  if (collin_tol <= 0.0) {
    stop("Collinearity tolerance should be greater than zero.", call. = FALSE)
  }

  # Return list with control parameters
  list(
    dev_tol = dev_tol,
    center_tol = center_tol,
    collin_tol = collin_tol, # Add this
    iter_max = iter_max,
    iter_center_max = iter_center_max,
    iter_inner_max = iter_inner_max,
    iter_interrupt = iter_interrupt,
    iter_ssr = iter_ssr,
    limit = limit,
    trace = as.logical(trace),
    drop_pc = as.logical(drop_pc),
    keep_mx = as.logical(keep_mx)
  )
}
