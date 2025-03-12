#' srr_stats
#' @srrstats {G1.0} Implements recovery of fixed effects for models estimated with high-dimensional k-way fixed effects.
#' @srrstats {G2.1a} Ensures that the input object is of the expected class (`felm` or `feglm`).
#' @srrstats {G2.2} Checks for valid tolerance levels (`alpha_tol`) to control convergence.
#' @srrstats {G3.1a} Outputs include named vectors of estimated fixed effects for interpretability.
#' @srrstats {G3.3} Handles multiple high-dimensional fixed effect categories by iterative computation.
#' @srrstats {G5.1} Provides robust error handling for missing or invalid input objects.
#' @srrstats {G5.2a} Issues unique error messages for invalid input or class mismatches.
#' @srrstats {RE5.0} Optimized for computational efficiency in high-dimensional fixed effects recovery.
#' @srrstats {RE5.1} Includes iterative solving of normal equations for high-dimensional datasets.
#' @srrstats {RE5.2} Ensures numerical stability with specified tolerance thresholds.
#' @noRd
NULL

#' @title Recover the estimates of the fixed effects after fitting (G)LMs
#'
#' @description The system might not have a unique solution since we do not take
#'  collinearity into account. If the solution is not unique, an estimable
#'  function has to be applied to our solution to get meaningful estimates of
#'  the fixed effects.
#'
#' @param object an object of class \code{"feglm"}.
#' @param control a list of control parameters. If \code{NULL}, the default
#'  control parameters are used.
#'
#' @return A named list containing named vectors of estimated fixed effects.
#'
#' @references Stammann, A. (2018). "Fast and Feasible Estimation of Generalized
#'  Linear Models with High-Dimensional k-way Fixed Effects". ArXiv e-prints.
#' @references Gaure, S. (n. d.). "Multicollinearity, identification, and
#'  estimable functions". Unpublished.
#'
#' @seealso \code{\link{felm}}, \code{\link{feglm}}
#'
#' @examples
#' # check the feglm examples for the details about clustered standard errors
#'
#' # subset trade flows to avoid fitting time warnings during check
#' set.seed(123)
#' trade_2006 <- trade_panel[trade_panel$year == 2006, ]
#' trade_2006 <- trade_2006[sample(nrow(trade_2006), 500), ]
#'
#' mod <- fepoisson(
#'   trade ~ log_dist + lang + cntg + clny | exp_year + imp_year,
#'   trade_2006
#' )
#'
#' fixed_effects(mod)
#'
#' @export
fixed_effects <- function(object = NULL, control = NULL) {
  # Check validity of 'object' ----
  if (is.null(object)) {
    stop("'object' has to be specified.", call. = FALSE)
  } else if (isFALSE(inherits(object, "felm")) &&
    isFALSE(inherits(object, "feglm"))) {
    stop(
      "'fixed_effects' called on a non-'felm' or non-'feglm' object.",
      call. = FALSE
    )
  }

  # Check validity of control + Extract control list ----
  control <- check_control_(control)

  # Extract required quantities from result list ----
  beta <- object[["coefficients"]]
  data <- object[["data"]]
  formula <- object[["formula"]]
  lvls_k <- object[["lvls_k"]]
  nms_fe <- object[["nms_fe"]]
  k_vars <- names(lvls_k)
  k <- length(lvls_k)
  eta <- object[["eta"]]

  # Extract regressor matrix ----
  x <- model.matrix(formula, data, rhs = 1L)[, -1L, drop = FALSE]
  attr(x, "dimnames") <- NULL

  # Generate auxiliary list of indexes for different sub panels ----
  k_list <- get_index_list_(k_vars, data)

  # Recover fixed effects by alternating the solutions of normal equations ----
  if (inherits(object, "feglm")) {
    pie <- eta - x %*% beta
  } else {
    pie <- fitted.values(object) - x %*% beta
  }
  fe_list <- get_alpha_(pie, k_list, control)

  # Assign names to the different fixed effects categories ----
  for (i in seq.int(k)) {
    colnames(fe_list[[i]]) <- k_vars[i]
    rownames(fe_list[[i]]) <- nms_fe[[i]]
  }
  names(fe_list) <- k_vars

  # Return list of estimated fixed effects ----
  fe_list
}
