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
#' @srrstats {RE5.0} Supports control over algorithmic complexity, such as dropping perfectly separated observations (`drop_pc`) and optional matrix storage (`keep_tx`).
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
#' @param collin_tol tolerance level for detecting collinearity. The default is
#'  \code{1.0e-07}. 
#' @param alpha_tol tolerance for fixed effects (alpha) convergence.
#'  The default is \code{1.0e-06}.
#' @param step_halving_factor numeric indicating the factor by which the step
#'  size is halved to iterate towards convergence. This is used to control the
#'  step size during optimization. The default is \code{0.5}.
#' @param iter_max unsigned integer indicating the maximum number of iterations
#'  in the maximization routine. The default is \code{25L}.
#' @param iter_center_max unsigned integer indicating the maximum number of
#'  iterations in the centering algorithm. The default is \code{10000L}.
#' @param iter_inner_max unsigned integer indicating the maximum number of
#'  iterations in the inner loop of the centering algorithm. The default is
#'  \code{50L}.
#' @param iter_interrupt unsigned integer indicating the maximum number of
#' iterations before the algorithm is interrupted. The default is \code{1000L}.
#' @param iter_ssr unsigned integer indicating the number of iterations
#' to skip before checking if the sum of squared residuals improves. The default
#' is \code{10L}.
#' @param iter_alpha_max maximum iterations for fixed effects computation.
#'  The default is \code{10000L}.
#' @param step_halving_memory numeric memory factor for step-halving algorithm.
#'  Controls how much of the previous iteration is retained. The default is \code{0.9}.
#' @param max_step_halving maximum number of post-convergence step-halving attempts.
#'  The default is \code{2}.
#' @param start_inner_tol starting tolerance for inner solver iterations.
#'  The default is \code{1.0e-04}.
#' @param return_fe logical indicating if the fixed effects should be returned.
#'  This can be useful when fitting general equilibrium models where skipping the
#'  fixed effects for intermediate steps speeds up computation. The default is
#'  \code{TRUE} and only applies to the \code{feglm} class.
#' @param accel_start Integer. Iteration to start conjugate gradient acceleration in centering.
#'  The default is \code{6L}.
#' @param project_tol_factor Factor to multiply center_tol for projection tolerance.
#'  The default is \code{1e-3}.
#' @param grand_accel_tol Tolerance for grand acceleration convergence.
#'  The default is \code{1e-10}.
#' @param project_group_tol Tolerance for individual group projections.
#'  The default is \code{1e-12}.
#' @param irons_tuck_tol Tolerance for Irons-Tuck acceleration.
#'  The default is \code{1e-10}.
#' @param grand_accel_interval Interval for applying grand acceleration.
#'  The default is \code{5L}.
#' @param irons_tuck_interval Interval for applying Irons-Tuck acceleration.
#'  The default is \code{3L}.
#' @param ssr_check_interval Interval for adaptive SSR convergence checks.
#'  The default is \code{40L}.
#' @param convergence_factor Factor for detecting slow convergence.
#'  The default is \code{1.1}.
#' @param tol_multiplier Multiplier for early termination tolerance check.
#'  The default is \code{20.0}.
#' @param keep_tx logical indicating if the centered regressor matrix should be
#'  stored. The centered regressor matrix is required for some covariance
#'  estimators, bias corrections, and average partial effects. This option saves
#'  some computation time at the cost of memory. The default is \code{TRUE}.
#' @param init_theta Initial value for the negative binomial dispersion parameter (theta).
#'  The default is \code{0.0}.
#'
#' @return A named list of control parameters.
#'
#' @examples
#' fit_control(0.05, 0.05, 10L, 10L, TRUE, TRUE, TRUE)
#'
#' @seealso \code{\link{feglm}}
#'
#' @export
fit_control <- function(
    dev_tol = 1.0e-08,
    center_tol = 1.0e-08,
    collin_tol = 1.0e-10,
    step_halving_factor = 0.5,
    alpha_tol = 1.0e-08,
    iter_max = 25L,
    iter_center_max = 10000L,
    iter_inner_max = 50L,
    iter_alpha_max = 10000L,
    iter_interrupt = 1000L,
    iter_ssr = 10L,
    step_halving_memory = 0.9,
    max_step_halving = 2L,
    start_inner_tol = 1.0e-06,
    accel_start = 6L,
    project_tol_factor = 1e-3,
    grand_accel_tol = 1e-10,
    project_group_tol = 1e-12,
    irons_tuck_tol = 1e-10,
    grand_accel_interval = 5L,
    irons_tuck_interval = 3L,
    ssr_check_interval = 40L,
    convergence_factor = 1.1,
    tol_multiplier = 20.0,
    return_fe = TRUE,
    keep_tx = FALSE,
    init_theta = 0.0) {
  # Check validity of tolerance parameters
  if (dev_tol <= 0.0 || center_tol <= 0.0 || collin_tol <= 0.0 ||
      step_halving_factor <= 0.0 || alpha_tol <= 0.0 ||
      project_tol_factor <= 0.0 || grand_accel_tol <= 0.0 ||
      project_group_tol <= 0.0 || irons_tuck_tol <= 0.0) {
    stop(
      "All tolerance parameters should be greater than zero.",
      call. = FALSE
    )
  }

  # Check validity of iter parameters
  iter_max <- as.integer(iter_max)
  iter_center_max <- as.integer(iter_center_max)
  iter_inner_max <- as.integer(iter_inner_max)
  iter_interrupt <- as.integer(iter_interrupt)
  iter_ssr <- as.integer(iter_ssr)
  grand_accel_interval <- as.integer(grand_accel_interval)
  irons_tuck_interval <- as.integer(irons_tuck_interval)
  ssr_check_interval <- as.integer(ssr_check_interval)
  if (iter_max < 1L || iter_center_max < 1L ||
      iter_inner_max < 1L || iter_interrupt < 1L || iter_ssr < 1L ||
      grand_accel_interval < 1L || irons_tuck_interval < 1L || 
      ssr_check_interval < 1L) {
    stop(
      "All iteration and interval parameters should be greater than or equal to one.",
      call. = FALSE
    )
  }

  # Check validity of logical parameters
  return_fe <- as.logical(return_fe)
  keep_tx <- as.logical(keep_tx)
  if (is.na(return_fe) || is.na(keep_tx)) {
    stop(
      "All logical parameters should be TRUE or FALSE.",
      call. = FALSE
    )
  }

  # Check validity of integer parameters for acceleration
  max_step_halving <- as.integer(max_step_halving)
  if (max_step_halving < 0L) {
    stop(
      "max_step_halving should be greater than or equal to zero.",
      call. = FALSE
    )
  }

  # Check validity of numeric parameters for acceleration
  if (step_halving_memory <= 0 || step_halving_memory >= 1) {
    stop(
      "step_halving_memory should be between 0 and 1 (exclusive).",
      call. = FALSE
    )
  }
  
  if (start_inner_tol <= 0) {
    stop(
      "start_inner_tol should be greater than zero.",
      call. = FALSE
    )
  }
  
  if (convergence_factor <= 1.0) {
    stop(
      "convergence_factor should be greater than 1.0.",
      call. = FALSE
    )
  }
  
  if (tol_multiplier <= 0.0) {
    stop(
      "tol_multiplier should be greater than zero.",
      call. = FALSE
    )
  }

  # Validate accel_start
  accel_start <- as.integer(accel_start)
  if (accel_start < 0L) {
    stop("accel_start should be >= 0.", call. = FALSE)
  }

  list(
    dev_tol = dev_tol,
    center_tol = center_tol,
    collin_tol = collin_tol,
    step_halving_factor = step_halving_factor,
    alpha_tol = alpha_tol,
    iter_max = iter_max,
    iter_center_max = iter_center_max,
    iter_inner_max = iter_inner_max,
    iter_alpha_max = iter_alpha_max,
    iter_interrupt = iter_interrupt,
    iter_ssr = iter_ssr,
    step_halving_memory = step_halving_memory,
    max_step_halving = max_step_halving,
    start_inner_tol = start_inner_tol,
    accel_start = accel_start,
    project_tol_factor = project_tol_factor,
    grand_accel_tol = grand_accel_tol,
    project_group_tol = project_group_tol,
    irons_tuck_tol = irons_tuck_tol,
    grand_accel_interval = grand_accel_interval,
    irons_tuck_interval = irons_tuck_interval,
    ssr_check_interval = ssr_check_interval,
    convergence_factor = convergence_factor,
    tol_multiplier = tol_multiplier,
    return_fe = return_fe,
    keep_tx = keep_tx,
    init_theta = init_theta
  )
}
