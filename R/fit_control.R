#' srr_stats
#' @srrstats {G1.0} Implements controls for efficient and numerically stable fitting of generalized linear models with
#'  fixed effects.
#' @srrstats {G2.0} Validates numeric input parameters to ensure they meet constraints (e.g., positive tolerance
#'  levels).
#' @srrstats {G2.0a} The main function explains that the tolerance must be unidimensional or the function gives an
#'  error.
#' @srrstats {G2.1a} Ensures the proper data types for arguments (e.g., logical for `trace`, integer for `iter_max`).
#' @srrstats {G2.3a} Uses argument validation to ensure appropriate ranges for critical parameters (e.g., `iter_max` and
#' `limit` >= 1).
#' @srrstats {G2.14a} Provides informative error messages when tolerance levels or iteration counts are invalid.
#' @srrstats {G2.14b} Provides clear error messages when the data structure is incompatible with the model requirements.
#' @srrstats {G5.2a} Produces unique and descriptive error messages for all validation checks.
#' @srrstats {RE3.0} If the deviance difference between 2 iterations is not less than tolerance after the max number of
#'  iterations, it prints a convergence warning.
#' @srrstats {RE5.0} Supports control over algorithmic complexity, such as dropping perfectly separated observations
#'  (`drop_pc`) and optional matrix storage (`keep_tx`).
#' @srrstats {G5.4a} Includes robust edge case handling, such as enforcing positive tolerance and iteration counts.
#' @noRd
NULL

#' NA_standards
#' @srrstatsNA {G2.14} Missing observations are dropped, otherwise providing imputation methods would bias the
#'  estimation (i.e., replacing all missing values with the median).
#' @noRd
NULL

#' @title Set \code{feglm} Control Parameters
#'
#' @description Set and change parameters used for fitting \link{feglm}, \link{felm}, and \link{fenegbin}. Termination
#'  conditions are similar to \link[stats]{glm}.
#'
#' @param dev_tol tolerance level for the first stopping condition of the maximization routine. The stopping condition
#'  is based on the relative change of the deviance in iteration \eqn{r} and can be expressed as follows: \eqn{|dev_{r}
#'  - dev_{r - 1}| / (0.1 + |dev_{r}|) < tol}{|dev - devold| / (0.1 + |dev|) < tol}. The default is \code{1.0e-08}.
#' @param center_tol tolerance level for the stopping condition of the centering algorithm. The stopping condition is
#'  based on the relative change of the centered variable similar to the \code{'lfe'} package. The default is
#'  \code{1.0e-06}.
#' @param center_tol_loose initial (loose) tolerance for adaptive centering in GLM iterations. During early IRLS
#'  iterations when deviance is changing rapidly, this looser tolerance is used to save computation. As the GLM
#'  converges, the tolerance is tightened towards \code{center_tol}. The default is \code{1.0e-04}.
#' @param collin_tol tolerance level for detecting collinearity. The default is \code{1.0e-07}.
#' @param alpha_tol tolerance for fixed effects (alpha) convergence. The default is \code{1.0e-06}.
#' @param sep_tol tolerance for separation detection. The default is \code{1.0e-08}.
#' @param step_halving_factor numeric indicating the factor by which the step size is halved to iterate towards
#'  convergence. This is used to control the step size during optimization. The default is \code{0.5}.
#' @param iter_max integer indicating the maximum number of iterations in the maximization routine. The default is
#'  \code{25L}.
#' @param iter_center_max integer indicating the maximum number of iterations in the centering algorithm. The default is
#'  \code{10000L}.
#' @param iter_inner_max integer indicating the maximum number of iterations in the inner loop of the centering
#'  algorithm. The default is \code{50L}.
#' @param iter_alpha_max maximum iterations for fixed effects computation. The default is \code{10000L}.
#' @param sep_max_iter maximum iterations for ReLU separation detection algorithm. The default is \code{200L}.
#' @param sep_simplex_max_iter maximum iterations for simplex separation detection algorithm. The default is \code{2000L}.
#' @param sep_zero_tol tolerance for treating values as zero in separation detection. The default is \code{1.0e-12}.
#' @param sep_use_relu logical indicating whether to use the ReLU algorithm for separation detection. The default is \code{TRUE}.
#' @param sep_use_simplex logical indicating whether to use the simplex algorithm for separation detection. The default is \code{TRUE}.
#' @param step_halving_memory numeric memory factor for step-halving algorithm. Controls how much of the previous
#'  iteration is retained. The default is \code{0.9}.
#' @param max_step_halving maximum number of post-convergence step-halving attempts. The default is \code{2}.
#' @param start_inner_tol starting tolerance for inner solver iterations. The default is \code{1.0e-04}.
#' @param grand_acc_period integer indicating the period (in iterations) for grand acceleration in the centering
#'  algorithm. Grand acceleration applies a second-level Irons-Tuck extrapolation on the overall convergence
#'  trajectory. Lower values (e.g., 4-10) may speed up convergence for difficult problems. Set to a very large
#'  value (e.g., 10000) to effectively disable. The default is \code{4L}.
#' @param return_fe logical indicating if the fixed effects should be returned. This can be useful when fitting general
#'  equilibrium models where skipping the fixed effects for intermediate steps speeds up computation. The default is
#'  \code{TRUE} and only applies to the \code{feglm} class.
#' @param keep_tx logical indicating if the centered regressor matrix should be stored. The centered regressor matrix is
#'  required for some covariance estimators, bias corrections, and average partial effects. This option saves
#'  some computation time at the cost of memory. The default is \code{TRUE}.
#' @param check_separation logical indicating whether to perform separation detection for Poisson models. When \code{TRUE}
#'  (default), observations with perfect prediction are automatically detected and excluded from estimation. Set to
#'  \code{FALSE} to skip this check and speed up computation when separation is known not to be an issue. The default
#'  is \code{TRUE}.
#' @param init_theta Initial value for the negative binomial dispersion parameter (theta). The default is \code{0.0}.
#' @param vcov_type Optional character string specifying the type of variance-covariance estimator to be used. It only
#' applies for formulas with a cluster variable like `z ~ x + y | fe | cl`. When \code{NULL} (default), the variance
#' covariance matrix follows a regular sandwich estimator. When set to `"m-estimator"`, uses standard clustered
#' M-estimator sandwich. When set to `"m-estimator-dyadic"`, uses dyadic clustering that accounts for correlation
#' between observations sharing entities. For dyadic clustering, specify two entity columns in the formula like
#' `z ~ x + y | fe | cl1 + cl2`.
#'
#' @return A named list of control parameters.
#'
#' @examples
#' fit_control(0.05, 0.05, 10L, 10L, TRUE, TRUE, TRUE)
#'
#' @seealso \link{feglm}, \link{felm}, and \link{fenegbin}
#'
#' @export
fit_control <- function(
  dev_tol = 1.0e-08,
  center_tol = 1.0e-06,
  center_tol_loose = 1.0e-04,
  collin_tol = 1.0e-10,
  step_halving_factor = 0.5,
  alpha_tol = 1.0e-08,
  iter_max = 25L,
  iter_center_max = 10000L,
  iter_inner_max = 50L,
  iter_alpha_max = 10000L,
  step_halving_memory = 0.9,
  max_step_halving = 2L,
  start_inner_tol = 1.0e-06,
  grand_acc_period = 4L,
  sep_tol = 1.0e-08,
  sep_zero_tol = 1.0e-12,
  sep_max_iter = 200L,
  sep_simplex_max_iter = 2000L,
  sep_use_relu = TRUE,
  sep_use_simplex = TRUE,
  return_fe = TRUE,
  keep_tx = FALSE,
  check_separation = TRUE,
  init_theta = 0.0,
  vcov_type = NULL
) {
  # Check validity of tolerance parameters
  if (
    dev_tol <= 0.0 ||
      center_tol <= 0.0 ||
      center_tol_loose <= 0.0 ||
      collin_tol <= 0.0 ||
      step_halving_factor <= 0.0 ||
      alpha_tol <= 0.0 ||
      sep_tol <= 0.0 ||
      sep_zero_tol <= 0.0
  ) {
    stop(
      "All tolerance parameters should be greater than zero.",
      call. = FALSE
    )
  }

  # Check validity of iter parameters
  iter_max <- as.integer(iter_max)
  iter_center_max <- as.integer(iter_center_max)
  iter_inner_max <- as.integer(iter_inner_max)
  sep_max_iter <- as.integer(sep_max_iter)
  sep_simplex_max_iter <- as.integer(sep_simplex_max_iter)

  if (
    iter_max < 1L ||
      iter_center_max < 1L ||
      iter_inner_max < 1L ||
      sep_max_iter < 1L ||
      sep_simplex_max_iter < 1L
  ) {
    stop(
      "All iteration parameters should be greater than or equal to one.",
      call. = FALSE
    )
  }

  # Check validity of logical parameters
  return_fe <- as.logical(return_fe)
  keep_tx <- as.logical(keep_tx)
  check_separation <- as.logical(check_separation)
  sep_use_relu <- as.logical(sep_use_relu)
  sep_use_simplex <- as.logical(sep_use_simplex)
  if (is.na(return_fe) || is.na(keep_tx) || is.na(check_separation) ||
    is.na(sep_use_relu) || is.na(sep_use_simplex)) {
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

  # Check validity of grand_acc_period
  grand_acc_period <- as.integer(grand_acc_period)
  if (grand_acc_period < 1L) {
    stop(
      "grand_acc_period should be greater than or equal to one.",
      call. = FALSE
    )
  }

  # Check validity of vcov_type
  if (!is.null(vcov_type) && !vcov_type %in% c("m-estimator", "m-estimator-dyadic")) {
    stop(
      "vcov_type should be either NULL, 'm-estimator', or 'm-estimator-dyadic'.",
      call. = FALSE
    )
  }

  list(
    dev_tol = dev_tol,
    center_tol = center_tol,
    center_tol_loose = center_tol_loose,
    collin_tol = collin_tol,
    step_halving_factor = step_halving_factor,
    alpha_tol = alpha_tol,
    iter_max = iter_max,
    iter_center_max = iter_center_max,
    iter_inner_max = iter_inner_max,
    iter_alpha_max = iter_alpha_max,
    step_halving_memory = step_halving_memory,
    max_step_halving = max_step_halving,
    start_inner_tol = start_inner_tol,
    grand_acc_period = grand_acc_period,
    sep_tol = sep_tol,
    sep_zero_tol = sep_zero_tol,
    sep_max_iter = sep_max_iter,
    sep_simplex_max_iter = sep_simplex_max_iter,
    sep_use_relu = sep_use_relu,
    sep_use_simplex = sep_use_simplex,
    return_fe = return_fe,
    keep_tx = keep_tx,
    check_separation = check_separation,
    init_theta = init_theta,
    vcov_type = vcov_type
  )
}
